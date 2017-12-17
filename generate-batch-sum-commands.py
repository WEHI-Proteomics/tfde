import argparse
import sqlite3

parser = argparse.ArgumentParser(description='Generates the commands to run frame summation in parallel.')
parser.add_argument('-sdb','--source_directory', type=str, help='The name of the source directory.', required=True)
parser.add_argument('-ddb','--destination_directory', type=str, help='The name of the destination directory.', required=True)
parser.add_argument('-bdb','--base_database_name', type=str, help='The base name of the database.', required=True)
parser.add_argument('-fts','--frames_to_sum', type=int, default=5, help='The number of source frames to sum.', required=False)
parser.add_argument('-ce','--collision_energy', type=int, default=10, help='Collision energy, in eV. Use 10 for MS1, 35 for MS2', required=False)
parser.add_argument('-sfb','--summed_frame_batch_size', type=int, default=10000, help='The number of summed frames in each batch.', required=False)
args = parser.parse_args()

args.source_directory = args.source_directory.replace('\\', '/')
args.destination_directory = args.destination_directory.replace('\\', '/')

# Connect to the database file to find out the number of unique frame ids for this collision energy
source_db = "{}/{}".format(args.source_directory, args.base_database_name)
print("connecting to {}".format(source_db))
source_conn = sqlite3.connect(source_db)
c = source_conn.cursor()
q = c.execute("SELECT COUNT(DISTINCT frame_id) FROM frame_properties WHERE collision_energy={}".format(args.collision_energy))
row = q.fetchone()
number_of_frames_in_source = int(row[0])
print("found {} unique frames for collision energy {}".format(number_of_frames_in_source, args.collision_energy))

number_of_summed_frames = number_of_frames_in_source / args.frames_to_sum
number_of_batches = number_of_summed_frames / args.summed_frame_batch_size
if (number_of_summed_frames % args.summed_frame_batch_size > 0):
    number_of_batches += 1
number_of_source_frames_per_batch = args.frames_to_sum * args.summed_frame_batch_size
print("total summed frames {} in {} batches\n".format(number_of_summed_frames, number_of_batches))

for i in range(number_of_batches):
    base_summed_frame_id = i*args.summed_frame_batch_size+1
    if (base_summed_frame_id+args.summed_frame_batch_size-1 > number_of_summed_frames):
	    number_of_summed_frames_required_this_batch = number_of_summed_frames-base_summed_frame_id+1
    else:
        number_of_summed_frames_required_this_batch = args.summed_frame_batch_size

    last_summed_frame_id = base_summed_frame_id+number_of_summed_frames_required_this_batch-1
    base_source_frame_index = i*number_of_source_frames_per_batch

    print("start \"Summing {}-{}\" python sum-frames-intensity-descent.py -sdb \"{}/{}\" -ddb \"{}/summed-{}-{}-{}\" -n {} -bf {} -sf {}".format(base_summed_frame_id, last_summed_frame_id, args.source_directory, 
    	args.base_database_name, args.destination_directory, base_summed_frame_id, last_summed_frame_id, args.base_database_name, number_of_summed_frames_required_this_batch, 
    	base_summed_frame_id, base_source_frame_index))
