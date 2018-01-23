import argparse
import pymysql

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the commands to run MS1 peak detection in parallel.')
parser.add_argument('-db','--database_name', type=str, help='The name of the database.', required=True)
parser.add_argument('-fbs','--number_of_batches', type=int, default=12, help='The number of batches.', required=False)
args = parser.parse_args()

# Connect to the database
source_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database="'{}'".format(args.database_name))
src_c = source_conn.cursor()

# Find out the number of summed MS1 frames in the database
src_c.execute("SELECT value FROM summing_info WHERE item=\"frame_upper\"")
row = src_c.fetchone()
number_of_frames = int(row[0])

batch_size = number_of_frames / args.number_of_batches
if (batch_size * args.number_of_batches) < number_of_frames:
    args.number_of_batches += 1

print("number of frames {}, batch size {}, number of batches {}".format(number_of_frames, batch_size, args.number_of_batches))

for i in range(args.number_of_batches):
    first_frame_id = i*batch_size+1
    last_frame_id = first_frame_id + batch_size - 1
    if last frame_id > number_of_frames:
        last_frame_id = number_of_frames

    # print("start \"Summing {}-{}\" python sum-frames-intensity-descent.py -sdb \"{}/{}\" -ddb \"{}/summed-{}-{}-{}\" -n {} -bf {} -sf {}".format(base_feature_id, last_summed_frame_id, args.source_directory, 
    #     args.base_database_name, args.destination_directory, base_feature_id, last_summed_frame_id, args.base_database_name, number_of_features_required_this_batch, 
    #     base_feature_id, base_source_frame_index))
    print("nohup python -u ./peak-detect-ms1.py -db {} -fl {} -fu {} > ../logs/batch-{}-{}-{}.log 2>&1 &".format(args.database_name, first_frame_id, last_frame_id, i, first_frame_id, last_frame_id))

# Close the database connection
source_conn.close()
