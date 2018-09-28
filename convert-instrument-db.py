import sys
import numpy as np
import pandas as pd
import timsdata
import sqlite3
import os
import argparse
import time
import csv
import os.path

# Usage: python convert-instrument-db.py -sdb "S:\data\Projects\ProtemicsLab\Bruker timsTOF\databases\20170714_SN34_UPS2_yeast200ng_AIF15_Slot1-39_01_728.d" -ddb "S:\data\Projects\ProtemicsLab\Bruker timsTOF\converted\20170714_SN34_UPS2_yeast200ng_AIF15_Slot1-39_01_728.sqlite"

COLLISION_ENERGY_MS1_SET_PROPERTY_NAME = "Collision_Energy_Set"
COLLISION_ENERGY_PROPERTY_NAME = "Collision_Energy_Act"
FRAME_RATE_PROPERTY_NAME = "Digitizer_AcquisitionTime_Set"
TARGET_MASS_START_PROPERTY_NAME = "Mode_TargetMassStart"
TARGET_MASS_END_PROPERTY_NAME = "Mode_TargetMassEnd"

# frame array indices
FRAME_ID_IDX = 0
FRAME_NUMSCAN_IDX = 1

# frame collision energy indices
FRAME_ID_IDX = 0
FRAME_COLLISION_ENERGY_IDX = 1

parser = argparse.ArgumentParser(description='Convert the Bruker database to a detection database.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
parser.add_argument('-bs','--batch_size', type=int, default=10000, help='The size of the frames to be written to the database.', required=False)
parser.add_argument('-nf','--number_of_frames', type=int, help='The number of frames to convert.', required=False)
args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

analysis_dir = args.source_database_name
if sys.version_info.major == 2:
    analysis_dir = unicode(analysis_dir)

td = timsdata.TimsData(analysis_dir)
source_conn = td.conn

# Get the frame information
print("Loading the frames information")
frames_df = pd.read_sql_query("select Id,NumScans from Frames order by Id ASC;", source_conn)
frames_v = frames_df.values

frame_count = len(frames_v)
max_frame_id = np.max(frames_v[:,FRAME_ID_IDX])
min_frame_id = np.min(frames_v[:,FRAME_ID_IDX])
print("Analysis has {} frames. Frame IDs {}-{}".format(frame_count, min_frame_id, max_frame_id))

# Get the raw frame period
df = pd.read_sql_query("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(FRAME_RATE_PROPERTY_NAME), source_conn)
property_id = df.loc[0].Id
df = pd.read_sql_query("SELECT Value FROM FrameProperties WHERE Property={}".format(property_id), source_conn)
raw_frame_period_in_msec = df.loc[0].Value

# Get the mass range
df = pd.read_sql_query("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(TARGET_MASS_START_PROPERTY_NAME), source_conn)
property_id = df.loc[0].Id
df = pd.read_sql_query("SELECT Value FROM GroupProperties WHERE Property={}".format(property_id), source_conn)
mz_lower = df.loc[0].Value

df = pd.read_sql_query("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(TARGET_MASS_END_PROPERTY_NAME), source_conn)
property_id = df.loc[0].Id
df = pd.read_sql_query("SELECT Value FROM GroupProperties WHERE Property={}".format(property_id), source_conn)
mz_upper = df.loc[0].Value

df = pd.read_sql_query("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(COLLISION_ENERGY_MS1_SET_PROPERTY_NAME), source_conn)
property_id = df.loc[0].Id
df = pd.read_sql_query("SELECT Value FROM GroupProperties WHERE Property={}".format(property_id), source_conn)
ms1_collision_energy = df.loc[0].Value

# Get the collision energy property values
q = source_conn.execute("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(COLLISION_ENERGY_PROPERTY_NAME))
collision_energy_property_id = q.fetchone()[0]

print("Loading the collision energy property values")
collision_energies_df = pd.read_sql_query("SELECT Frame,Value FROM FrameProperties WHERE Property={}".format(collision_energy_property_id), source_conn)
collision_energies_v = collision_energies_df.values

# remove the destination database if it remains from a previous run - it's faster to recreate it
if os.path.isfile(args.destination_database_name):
    os.remove(args.destination_database_name)

# Connect to the destination database
dest_conn = sqlite3.connect(args.destination_database_name)
dest_c = dest_conn.cursor()

# Create the tables
print("Setting up tables and indexes")

dest_c.execute("DROP TABLE IF EXISTS frames")
dest_c.execute("DROP TABLE IF EXISTS frame_properties")
dest_c.execute("DROP TABLE IF EXISTS convert_info")

dest_c.execute("CREATE TABLE frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER, raw_frame_point TEXT)")
dest_c.execute("CREATE TABLE frame_properties (frame_id INTEGER, collision_energy REAL)")
dest_c.execute("CREATE TABLE convert_info (item TEXT, value TEXT)")

points = []
frame_properties = []

start_run = time.time()
peak_id = 0 # set the peak ID to be zero for now
max_scans = 0

frame_count = 0

print("Converting...")
for frame in frames_v:
    frame_id = frame[FRAME_ID_IDX]
    num_scans = frame[FRAME_NUMSCAN_IDX]
    pointId = 0

    if num_scans > max_scans:
        max_scans = num_scans

    for scan_line, scan in enumerate(td.readScans(frame_id, 0, num_scans)):
        index = np.array(scan[0], dtype=np.float64)
        mz_values = td.indexToMz(frame_id, index)
        if len(mz_values) > 0:
            intensity_values = scan[1]
            for i in range(0, len(intensity_values)):   # step through the intensity readings (i.e. points) on this scan line
                pointId += 1
                points.append((int(frame_id), int(pointId), float(mz_values[i]), int(scan_line), int(intensity_values[i]), int(peak_id), "{}|{}".format(int(frame_id), int(pointId))))

    frame_count += 1

    # Check whether we've done a chunk to write out to the database
    if (frame_id % args.batch_size) == 0:
        dest_c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?)", points)
        dest_conn.commit()
        print("{} frames converted...".format(frame_count))
        del points[:]

    if (args.number_of_frames is not None) and (frame_count >= args.number_of_frames):
        break

# Write what we have left
if len(points) > 0:
    dest_c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?)", points)
    dest_conn.commit()
    print("{} frames converted...".format(frame_count))
    del points[:]

for collision_energy in collision_energies_v:
    frame_properties.append((int(collision_energy[FRAME_ID_IDX]), float(collision_energy[FRAME_COLLISION_ENERGY_IDX])))
    if (args.number_of_frames is not None) and (len(frame_properties) == args.number_of_frames):
        break

print("Writing frame properties")
dest_c.executemany("INSERT INTO frame_properties VALUES (?, ?)", frame_properties)
dest_conn.commit()

stop_run = time.time()

info.append(("source_frame_lower", int(min_frame_id)))
info.append(("source_frame_upper", int(frame_count)))
info.append(("source_frame_count", int(frame_count)))
info.append(("num_scans", int(max_scans)))
info.append(("raw_frame_period_in_msec", float(raw_frame_period_in_msec)))
info.append(("mz_lower", float(mz_lower)))
info.append(("mz_upper", float(mz_upper)))
info.append(("ms1_collision_energy", float(ms1_collision_energy)))
info.append(("run processing time (sec)", float(stop_run-start_run)))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))

print("{} info: {}".format(parser.prog, info))

dest_c.executemany("INSERT INTO convert_info VALUES (?, ?)", info)

# Commit changes and close the connection
dest_conn.commit()
dest_conn.close()
