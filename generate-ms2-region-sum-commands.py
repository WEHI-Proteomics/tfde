import argparse
import pymysql

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the commands to run MS2 feature region summing in parallel.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the (converted but not summed) source database, for reading MS2 frames.', required=True)
parser.add_argument('-fbs','--feature_batch_size', type=int, default=200, help='The number of features in each batch.', required=False)
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

for i in range(number_of_batches):
    base_feature_id = i*args.feature_batch_size+1
    if (base_feature_id+args.feature_batch_size-1 > number_of_features):
        number_of_features_required_this_batch = number_of_features-base_feature_id+1
    else:
        number_of_features_required_this_batch = args.feature_batch_size

    last_feature_id = base_feature_id+number_of_features_required_this_batch-1

    # print("start \"Summing {}-{}\" python sum-frames-intensity-descent.py -sdb \"{}/{}\" -ddb \"{}/summed-{}-{}-{}\" -n {} -bf {} -sf {}".format(base_feature_id, last_summed_frame_id, args.source_directory, 
    #     args.base_database_name, args.destination_directory, base_feature_id, last_summed_frame_id, args.base_database_name, number_of_features_required_this_batch, 
    #     base_feature_id, base_source_frame_index))
    print("nohup python -u ./feature-region-ms2-sum-frames.py -sdb \"{}\" -fl {} -fu {} > ../logs/batch-{}-{}-{}.log 2>&1 &".format(args.source_database_name, base_feature_id, last_feature_id, i, base_feature_id, last_feature_id))



# Close the database connection
source_conn.close()
