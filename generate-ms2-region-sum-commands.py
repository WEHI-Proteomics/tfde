import argparse
import pymysql

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the commands to run MS2 feature region summing in parallel.')
parser.add_argument('-fbs','--feature_batch_size', type=int, default=20, help='The number of features in each batch.', required=False)
parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
args = parser.parse_args()

# Connect to the database
source_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database='timsTOF')
c = source_conn.cursor()

# Find out the range of feature IDs in the database
c.execute("SELECT MIN(feature_id), MAX(feature_id) from features WHERE charge_state >= {}".format(args.minimum_charge_state))
row = c.fetchone()
feature_id_lower = int(row[0])
feature_id_upper = int(row[1])
number_of_features = feature_id_upper - feature_id_lower + 1    # could actually count them but we are going to work in feature ID ranges, not individual feature IDs (which if there are gaps in feature IDs means some batches may have fewer features than the batch size)

# Work out how many batches are needed
number_of_batches = number_of_features / args.feature_batch_size
if (number_of_features % args.feature_batch_size > 0):
    number_of_batches += 1

print("feature range {}-{}, total number of features {}, number of batches {}".format(feature_id_lower, feature_id_upper, number_of_features, number_of_batches))




# Close the database connection
source_conn.close()
