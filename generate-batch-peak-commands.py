import argparse

parser = argparse.ArgumentParser(description='Generates the commands to run peak detection in parallel.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-bs','--batch_size', type=int, default=2000, help='The batch size.', required=False)
parser.add_argument('-fr','--number_of_frames', type=int, default=56780, help='The number of frames.', required=False)
args = parser.parse_args()

NUMBER_OF_BATCHES = args.number_of_frames / args.batch_size

for i in range(NUMBER_OF_BATCHES):
    start_frame = i*args.batch_size+1
    end_frame = (i+1)*args.batch_size
    range_str = "{}-{}".format(start_frame, end_frame)
    print("nohup python -u ./otf-peak-detect/peak-detect-intensity-descent.py -db {} -fl {} -fu {} -ts /efs/{}.sqlite >/efs/{}.log &".format(args.database_name, start_frame, end_frame, range_str, range_str))
