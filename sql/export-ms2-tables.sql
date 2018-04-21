.mode csv
.headers on
.once "S:/data/Projects/ProtemicsLab/Bruker timsTOF/experiments/ms2-feature/summed-ms2-regions.csv"
select * from summed_ms2_regions;
.once "S:/data/Projects/ProtemicsLab/Bruker timsTOF/experiments/ms2-feature/ms1-features.csv"
select * from features where feature_id=1;
.once "S:/data/Projects/ProtemicsLab/Bruker timsTOF/experiments/ms2-feature/clusters.csv"
select frame_id,cluster_id,charge_state,base_isotope_peak_id,base_peak_max_point_mz,base_peak_max_point_scan,fit_error,intensity_sum,feature_id,mz_lower,mz_upper,scan_lower,scan_upper from clusters where feature_id=1;
.once "S:/data/Projects/ProtemicsLab/Bruker timsTOF/experiments/ms2-feature/ms2-peaks.csv"
select * from ms2_peaks where feature_id=1;
.once "S:/data/Projects/ProtemicsLab/Bruker timsTOF/experiments/ms2-feature/summed-ms1-regions.csv"
select * from summed_ms1_regions;
