from __future__ import print_function
import os
import sqlite3
import pandas as pd
import argparse
import sqlalchemy

def merge_summed_regions(source_db_name, destination_db_name, exceptions):
    source_conn = sqlite3.connect(source_db_name)
    src_cur = source_conn.cursor()
    engine = sqlalchemy.create_engine("postgresql+psycopg2://dwm:password@features-test.ct0qrar1ezs6.us-west-1.rds.amazonaws.com:5432/features_test")
    destination_conn = engine.connect()

    df = pd.read_sql_query("SELECT tbl_name,sql FROM sqlite_master WHERE type='table'", source_conn)
    for t_idx in range(0,len(df)):
        table_name = df.loc[t_idx].tbl_name
        if table_name not in exceptions:
            print("merging {}".format(table_name))

            row_count = int(pd.read_sql('SELECT COUNT(*) FROM {table_name}'.format(table_name=table_name), source_conn).values)
            chunksize = 1000000
            number_of_chunks = int(row_count / chunksize)

            for i in range(number_of_chunks + 1):
                print("\tmerging chunk {} of {}".format(i, number_of_chunks))
                query = 'SELECT * FROM {table_name} LIMIT {offset}, {chunksize}'.format(
                    table_name=table_name, offset=i * chunksize, chunksize=chunksize)
                table_df = pd.read_sql_query(query, con=source_conn)
                table_df.to_sql(name=table_name, con=destination_conn, if_exists='append', index=False, chunksize=None)

            # drop the table in the source database
            # src_cur.execute('DROP TABLE IF EXISTS {table_name}'.format(table_name=table_name))

    # if we moved all the tables to the destination, delete the database file. Otherwise, vacuum it 
    # to minimise disk space.
    # if len(exceptions) == 0:
    #     source_conn.close()
    #     os.remove(source_db_name)
    # else:
    #     src_cur.execute('VACUUM')
    #     source_conn.close()

    source_conn.close()
    destination_conn.commit()
    destination_conn.close()

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the search MGF from the instrument database.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The source database name.', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The destination database name.', required=True)
args = parser.parse_args()

merge_summed_regions(args.source_database_name, args.destination_database_name, [])
