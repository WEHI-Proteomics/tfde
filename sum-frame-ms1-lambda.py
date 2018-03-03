import sys
import logging
import rds_config
import pymysql
import pandas as pd
import time

#rds settings
rds_host  = "dwm-instance.cy0jiebct3t0.ap-southeast-2.rds.amazonaws.com"
name = rds_config.db_username
password = rds_config.db_password
db_name = rds_config.db_name


logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    source_conn = pymysql.connect(rds_host, user=name, passwd=password, db=db_name, connect_timeout=5)
except:
    logger.error("ERROR: Unexpected error: Could not connect to MariaDB instance.")
    sys.exit()

logger.info("SUCCESS: Connection to RDS mysql instance succeeded")

def handler(event, context):
    """
    This function sums an ms1 frame
    """

    with source_conn.cursor() as src_c:
        src_c.execute("SELECT value from convert_info where item=\"num_scans\"")
        row = src_c.fetchone()
        scan_lower = 1
        scan_upper = int(row[0])
        logger.info("scan range {}-{}".format(scan_lower, scan_upper))

        # Store the arguments as metadata in the database for later reference
        summing_info = []

        start_run = time.time()

        # Find the complete set of frame ids to be processed
        frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(args.collision_energy), source_conn)
        frame_ids = tuple(frame_ids_df.values[:,0])
        number_of_summed_frames = 1 + int(((len(frame_ids) - args.frames_to_sum) / args.frame_summing_offset))

        summedFrameId = 2500  # the summed frame id (1-based) as a parameter

        baseFrameIdsIndex = (summedFrameId-1)*args.frame_summing_offset
        frameIdsToSum = frame_ids[baseFrameIdsIndex:baseFrameIdsIndex+args.frames_to_sum]
        logger.info("Processing {} frames ({}) to create summed frame {}".format(len(frameIdsToSum), frameIdsToSum, summedFrameId))
        frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} order by frame_id, mz, scan asc;".format(frameIdsToSum), source_conn)
        frame_v = frame_df.values

        pointId = 1
        points = []
        for scan in range(scan_lower, scan_upper+1):
            points_v = frame_v[np.where(frame_v[:,2] == scan)]
            points_to_process = len(points_v)
            while len(points_v) > 0:
                max_intensity_index = np.argmax(points_v[:,3])
                point_mz = points_v[max_intensity_index, 1]
                # print("m/z {}, intensity {}".format(point_mz, points_v[max_intensity_index, 3]))
                delta_mz = standard_deviation(point_mz) * 4.0
                # Find all the points in this point's std dev window
                nearby_point_indices = np.where((points_v[:,1] >= point_mz-delta_mz) & (points_v[:,1] <= point_mz+delta_mz))[0]
                nearby_points = points_v[nearby_point_indices]
                # How many distinct frames do the points come from?
                unique_frames = np.unique(nearby_points[:,0])
                if len(unique_frames) >= args.noise_threshold:
                    # find the total intensity and centroid m/z
                    centroid_intensity = nearby_points[:,3].sum()
                    centroid_mz = peakutils.centroid(nearby_points[:,1], nearby_points[:,3])
                    points.append((int(summedFrameId), int(pointId), float(centroid_mz), int(scan), int(round(centroid_intensity)), 0))
                    pointId += 1

                # remove the points we've processed
                points_v = np.delete(points_v, nearby_point_indices, 0)

        if len(points) > 0:
            src_c.executemany("INSERT INTO summed_frames VALUES (%s, %s, %s, %s, %s, %s)", points)
            src_c.executemany("INSERT INTO elution_profile VALUES (%s, %s)", [(summedFrameId, sum(zip(*points)[4]))])
        else:
            logger.info("no points for summed frame id {}".format(summedFrameId))
            src_c.executemany("INSERT INTO elution_profile VALUES (%s, %s)", [(summedFrameId, 0)])

        stop_run = time.time()
        logger.info("{} seconds to process run".format(stop_run-start_run))

        summing_info.append(("scan_lower", scan_lower))
        summing_info.append(("scan_upper", scan_upper))

        summing_info.append(("run processing time (sec)", stop_run-start_run))
        summing_info.append(("processed", time.ctime()))

        summing_info_entry = []
        summing_info_entry.append(("summed frame {}".format(summedFrameId), ' '.join(str(e) for e in summing_info)))

        src_c.executemany("INSERT INTO summing_info VALUES (%s, %s)", summing_info_entry)
        source_conn.commit()
    

    return "Processed frame {} in {} seconds".format(summedFrameId, stop_run-start_run)
