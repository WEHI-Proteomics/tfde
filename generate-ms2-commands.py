import argparse
import pymysql

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the commands to run MS2 peak detection in parallel.')
parser.add_argument('-db','--database_name', type=str, help='The name of the database.', required=True)
parser.add_argument('-fbs','--number_of_batches', type=int, default=12, help='The number of batches.', required=False)
parser.add_argument('-ms2ce','--ms2_collision_energy', type=float, help='Collision energy used for MS2.', required=True)
args = parser.parse_args()

# Connect to the database
source_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database="{}".format(args.database_name))
src_c = source_conn.cursor()

# Find out the number of MS1 features in the database
src_c.execute("SELECT MAX(feature_id) FROM features")
row = src_c.fetchone()
number_of_features = int(row[0])

# Close the database connection
source_conn.close()

batch_size = number_of_features / args.number_of_batches
if (batch_size * args.number_of_batches) < number_of_features:
    args.number_of_batches += 1

print("number of features {}, batch size {}, number of batches {}".format(number_of_features, batch_size, args.number_of_batches))
print("")
print("python ./feature-region-ms2-sum-frames-prep.py -db {}".format(args.database_name))

for i in range(args.number_of_batches):
    first_feature_id = i*batch_size+1
    last_feature_id = first_feature_id + batch_size - 1
    if last_feature_id > number_of_features:
        last_feature_id = number_of_features

    # print("start \"Summing {}-{}\" python sum-frames-intensity-descent.py -sdb \"{}/{}\" -ddb \"{}/summed-{}-{}-{}\" -n {} -bf {} -sf {}".format(base_feature_id, last_summed_frame_id, args.source_directory, 
    #     args.base_database_name, args.destination_directory, base_feature_id, last_summed_frame_id, args.base_database_name, number_of_features_required_this_batch, 
    #     base_feature_id, base_source_frame_index))
    print("nohup python -u ./feature-region-ms2-sum-frames.py -db {} -fl {} -fu {} -ms2ce {} > ../logs/{}-ms2-feature-region-sum-batch-{}-{}-{}.log 2>&1 &".format(args.database_name, first_feature_id, last_feature_id, args.ms2_collision_energy, args.database_name, i, first_feature_id, last_feature_id))
