import argparse
import sqlite3

parser = argparse.ArgumentParser(description='Generates the commands to run frame summation in parallel.')
parser.add_argument('-sdb','--source_directory', type=str, help='The name of the source directory.', required=True)
parser.add_argument('-ddb','--destination_directory', type=str, help='The name of the destination directory.', required=True)
parser.add_argument('-bdb','--base_database_name', type=str, help='The base name of the database.', required=True)
parser.add_argument('-fts','--frames_to_sum', type=int, default=5, help='The number of source frames to sum.', required=False)
parser.add_argument('-bs','--batch_size', type=int, default=10000, help='The batch size.', required=False)
args = parser.parse_args()

# Connect to the database file to find out the number of unique frame ids
source_conn = sqlite3.connect("{}\{}".format(args.source_directory, args.base_database_name))
c = source_conn.cursor()
q = c.execute("SELECT COUNT(DISTINCT frame_id) FROM frames")
row = q.fetchone()
number_of_frames_in_source = int(row[0])

NUMBER_OF_SUMMED_FRAMES = number_of_frames_in_source / args.frames_to_sum
NUMBER_OF_BATCHES = NUMBER_OF_SUMMED_FRAMES / args.batch_size
NUMBER_OF_SOURCE_FRAMES_PER_BATCH = args.frames_to_sum * args.batch_size

for i in range(NUMBER_OF_BATCHES):
    start_frame_source = (i*NUMBER_OF_SOURCE_FRAMES_PER_BATCH)+1
    start_frame_destination = i*args.batch_size+1
    end_frame_destination = (i+1)*args.batch_size
    print("python \otf-peak-detect\sum-frames-intensity-descent.py -sdb {}\{} -ddb {}\summed-{}-{}-{} -n {} -bf {} -sf {}".format(args.source_directory, args.base_database_name, args.destination_directory, start_frame_destination, end_frame_destination, args.base_database_name, args.batch_size, start_frame_destination, start_frame_source))
