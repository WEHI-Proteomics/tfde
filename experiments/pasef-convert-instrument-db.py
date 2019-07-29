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

parser = argparse.ArgumentParser(description='Convert the Bruker database to a detection database.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
args = parser.parse_args()

analysis_dir = args.source_database_name
if sys.version_info.major == 2:
    analysis_dir = unicode(analysis_dir)

td = timsdata.TimsData(analysis_dir)
source_conn = td.conn

# Get the frame information
print("Loading the frames information")
frames_df = pd.read_sql_query("select Id,NumScans,Time from Frames order by Id ASC;", source_conn)

# determine the frame_id range
frame_lower = int(frames_df.Id.min())
frame_upper = int(frames_df.Id.max())

# Get the mz range to analyse
df = pd.read_sql_query("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(TARGET_MASS_START_PROPERTY_NAME), source_conn)
property_id = df.iloc[0].Id
df = pd.read_sql_query("SELECT Value FROM GroupProperties WHERE Property={}".format(property_id), source_conn)
mz_lower = df.iloc[0].Value

df = pd.read_sql_query("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(TARGET_MASS_END_PROPERTY_NAME), source_conn)
property_id = df.iloc[0].Id
df = pd.read_sql_query("SELECT Value FROM GroupProperties WHERE Property={}".format(property_id), source_conn)
mz_upper = df.iloc[0].Value

# Get the collision energy used for ms1, to help distinguish ms1 and ms2 frames
df = pd.read_sql_query("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(COLLISION_ENERGY_MS1_SET_PROPERTY_NAME), source_conn)
property_id = df.iloc[0].Id
df = pd.read_sql_query("SELECT Value FROM GroupProperties WHERE Property={}".format(property_id), source_conn)
ms1_collision_energy = df.iloc[0].Value
print("ms1 collision energy: {}".format(ms1_collision_energy))

# Get the collision energy property values
q = source_conn.execute("SELECT Id FROM PropertyDefinitions WHERE PermanentName=\"{}\"".format(COLLISION_ENERGY_PROPERTY_NAME))
collision_energy_property_id = q.fetchone()[0]

print("Loading the collision energy property values for each frame")
collision_energies_df = pd.read_sql_query("SELECT Frame,Value FROM FrameProperties WHERE Property={}".format(collision_energy_property_id), source_conn)
collision_energies_df.rename(columns={"Value":"collision_energy"}, inplace=True)
frames_df = pd.merge(frames_df, collision_energies_df, how='left', left_on=['Id'], right_on=['Frame'])

points = []

start_run = time.time()
peak_id = 0 # set the peak ID to be zero for now
max_scans = 0

print("Converting...")
for idx in range(len(frames_df)):
    frame_id = int(frames_df.iloc[idx].Id)
    num_scans = int(frames_df.iloc[idx].NumScans)
    retention_time_secs = float(frames_df.iloc[idx].Time)
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
                points.append((int(frame_id), int(pointId), float(mz_values[i]), int(scan_line), int(intensity_values[i]), int(peak_id), "{}|{}".format(int(frame_id), int(pointId)), retention_time_secs))

print("Writing frame properties")
frames_df.rename(columns={"Id":"frame_id", "Time":"retention_time_secs"}, inplace=True)
print(frames_df.memory_usage())

stop_run = time.time()
