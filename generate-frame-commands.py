import argparse
import pymysql

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the commands to run MS1 peak detection in parallel.')
parser.add_argument('-db','--database_name', type=str, help='The name of the database.', required=True)
parser.add_argument('-fbs','--number_of_batches', type=int, default=12, help='The number of batches.', required=False)
parser.add_argument('-hn','--hostname', default='mscypher-004', type=str, help='The hostname of the database.', required=False)
args = parser.parse_args()

# Connect to the database
source_conn = pymysql.connect(host="{}".format(args.hostname), user='root', passwd='password', database="{}".format(args.database_name))
src_c = source_conn.cursor()

# Find out the number of summed MS1 frames in the database
src_c.execute("SELECT value FROM summing_info WHERE item=\"frame_upper\"")
row = src_c.fetchone()
number_of_frames = int(row[0])

# Close the database connection
source_conn.close()

batch_size = number_of_frames / args.number_of_batches
if (batch_size * args.number_of_batches) < number_of_frames:
    args.number_of_batches += 1

print("number of frames {}, batch size {}, number of batches {}".format(number_of_frames, batch_size, args.number_of_batches))
print("")
print("python ./sum-frames-ms1-prep.py -db {}".format(args.database_name))
for i in range(args.number_of_batches):
    first_frame_id = i*batch_size+1
    last_frame_id = first_frame_id + batch_size - 1
    if last_frame_id > number_of_frames:
        last_frame_id = number_of_frames

    # print("start \"Summing {}-{}\" python sum-frames-intensity-descent.py -sdb \"{}/{}\" -ddb \"{}/summed-{}-{}-{}\" -n {} -bf {} -sf {}".format(base_feature_id, last_summed_frame_id, args.source_directory, 
    #     args.base_database_name, args.destination_directory, base_feature_id, last_summed_frame_id, args.base_database_name, number_of_features_required_this_batch, 
    #     base_feature_id, base_source_frame_index))
    # print("nohup python -u ./peak-detect-ms1.py -db {} -fl {} -fu {} > ../logs/{}-peak-batch-{}-{}-{}.log 2>&1 &".format(args.database_name, first_frame_id, last_frame_id, args.database_name, i, first_frame_id, last_frame_id))
    print("qsub -l nodes=1:ppn=12,mem=12gb ./py.sh -F \"./otf-peak-detect/sum-frames-ms1.py -db {} -fl {} -fu {}\"".format(args.database_name, first_frame_id, last_frame_id, args.database_name, i, first_frame_id, last_frame_id))

print("")
print("python ./peak-detect-ms1-prep.py -db {}".format(args.database_name))
for i in range(args.number_of_batches):
    first_frame_id = i*batch_size+1
    last_frame_id = first_frame_id + batch_size - 1
    if last_frame_id > number_of_frames:
        last_frame_id = number_of_frames

    # print("start \"Summing {}-{}\" python sum-frames-intensity-descent.py -sdb \"{}/{}\" -ddb \"{}/summed-{}-{}-{}\" -n {} -bf {} -sf {}".format(base_feature_id, last_summed_frame_id, args.source_directory, 
    #     args.base_database_name, args.destination_directory, base_feature_id, last_summed_frame_id, args.base_database_name, number_of_features_required_this_batch, 
    #     base_feature_id, base_source_frame_index))
    print("nohup python -u ./peak-detect-ms1.py -db {} -fl {} -fu {} > ../logs/{}-peak-batch-{}-{}-{}.log 2>&1 &".format(args.database_name, first_frame_id, last_frame_id, args.database_name, i, first_frame_id, last_frame_id))

print("")
print("python ./cluster-detect-ms1-prep.py -db {}".format(args.database_name))
for i in range(args.number_of_batches):
    first_frame_id = i*batch_size+1
    last_frame_id = first_frame_id + batch_size - 1
    if last_frame_id > number_of_frames:
        last_frame_id = number_of_frames

    # print("start \"Summing {}-{}\" python sum-frames-intensity-descent.py -sdb \"{}/{}\" -ddb \"{}/summed-{}-{}-{}\" -n {} -bf {} -sf {}".format(base_feature_id, last_summed_frame_id, args.source_directory, 
    #     args.base_database_name, args.destination_directory, base_feature_id, last_summed_frame_id, args.base_database_name, number_of_features_required_this_batch, 
    #     base_feature_id, base_source_frame_index))
    print("nohup python -u ./cluster-detect-ms1.py -db {} -fl {} -fu {} > ../logs/{}-cluster-batch-{}-{}-{}.log 2>&1 &".format(args.database_name, first_frame_id, last_frame_id, args.database_name, i, first_frame_id, last_frame_id))
