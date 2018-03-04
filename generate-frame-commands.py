import argparse
import sqlite3
import pandas as pd

#
# Run this script in the script directory.
# Don't put ".sqlite" at the end of the database name
#
# python generate-frame-commands.py -db Hela_20A_20R_500 -ce 7 > ../frame-commands.txt
#

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the commands to run MS1 peak detection in parallel.')
parser.add_argument('-db','--database_name', type=str, help='The name of the database.', required=True)
parser.add_argument('-fbs','--number_of_batches', type=int, default=12, help='The number of batches.', required=False)
parser.add_argument('-fts','--frames_to_sum', type=int, default=150, help='The number of MS1 source frames to sum.', required=False)
parser.add_argument('-fso','--frame_summing_offset', type=int, default=25, help='The number of MS1 source frames to shift for each summation.', required=False)
parser.add_argument('-ce','--collision_energy', type=int, help='Collision energy, in eV.', required=True)
args = parser.parse_args()

# Connect to the database
source_conn = sqlite3.connect("../{}.sqlite".format(args.database_name))
src_c = source_conn.cursor()

# Find the complete set of frame ids to be processed
frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(args.collision_energy), source_conn)
frame_ids = tuple(frame_ids_df.values[:,0])
number_of_frames = 1 + int(((len(frame_ids) - args.frames_to_sum) / args.frame_summing_offset))

# Close the database connection
source_conn.close()

batch_size = number_of_frames / args.number_of_batches
if (batch_size * args.number_of_batches) < number_of_frames:
    args.number_of_batches += 1

print("number of frames {}, batch size {}, number of batches {}".format(number_of_frames, batch_size, args.number_of_batches))
print("")
print("python ./otf-peak-detect/sum-frames-ms1-prep.py -sdb {}.sqlite".format(args.database_name))
for i in range(args.number_of_batches):
    first_frame_id = i*batch_size+1
    last_frame_id = first_frame_id + batch_size - 1
    if last_frame_id > number_of_frames:
        last_frame_id = number_of_frames

    print("nohup python -u ./otf-peak-detect/sum-frames-ms1.py -sdb {}.sqlite -ddb {}-{}-{}-{}.sqlite "
        "-fl {} -fu {} > ./logs/sum-frames-ms1-{}-{}.log 2>&1 &".format(
            args.database_name, 
            args.database_name, i, first_frame_id, last_frame_id, 
            first_frame_id, last_frame_id, 
            first_frame_id, last_frame_id))
    # print("qsub -l nodes=1:ppn=12,mem=4gb -F \"./otf-peak-detect/sum-frames-ms1.py -db {} -fl {} -fu {} -ce {}\" ./py.sh".format(args.database_name, first_frame_id, last_frame_id, args.collision_energy))

print("")
print("python ./otf-peak-detect/peak-detect-ms1-prep.py -db {}".format(args.database_name))
for i in range(args.number_of_batches):
    first_frame_id = i*batch_size+1
    last_frame_id = first_frame_id + batch_size - 1
    if last_frame_id > number_of_frames:
        last_frame_id = number_of_frames

    print("nohup python -u ./otf-peak-detect/peak-detect-ms1.py -sdb {}.sqlite -ddb {}-{}-{}-{}.sqlite "
        "-fl {} -fu {} > ./logs/peak-detect-ms1-{}-{}.log 2>&1 &".format(
            args.database_name, 
            args.database_name, i, first_frame_id, last_frame_id, 
            first_frame_id, last_frame_id, 
            first_frame_id, last_frame_id))
    # print("qsub -l nodes=1:ppn=12,mem=4gb -F \"./otf-peak-detect/peak-detect-ms1.py -db {} -fl {} -fu {}\" ./py.sh".format(args.database_name, first_frame_id, last_frame_id))

print("")
print("python ./otf-peak-detect/cluster-detect-ms1-prep.py -db {}".format(args.database_name))
for i in range(args.number_of_batches):
    first_frame_id = i*batch_size+1
    last_frame_id = first_frame_id + batch_size - 1
    if last_frame_id > number_of_frames:
        last_frame_id = number_of_frames

    print("nohup python -u ./otf-peak-detect/cluster-detect-ms1.py -sdb {}.sqlite -ddb {}-{}-{}-{}.sqlite "
        "-fl {} -fu {} > ./logs/cluster-detect-ms1-{}-{}.log 2>&1 &".format(
            args.database_name, 
            args.database_name, i, first_frame_id, last_frame_id, 
            first_frame_id, last_frame_id, 
            first_frame_id, last_frame_id))
    # print("qsub -l nodes=1:ppn=12,mem=4gb -F \"./otf-peak-detect/cluster-detect-ms1.py -db {} -fl {} -fu {}\" ./py.sh".format(args.database_name, first_frame_id, last_frame_id))
