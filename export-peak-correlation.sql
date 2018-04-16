.mode csv
.headers on
.once "S:/data/Projects/ProtemicsLab/Bruker timsTOF/experiments/ms2-feature/correlation/summed_ms1_regions.csv"
select * from summed_ms1_regions order by feature_id;
.once "S:/data/Projects/ProtemicsLab/Bruker timsTOF/experiments/ms2-feature/correlation/summed_ms2_regions.csv"
select * from summed_ms2_regions order by feature_id;
.once "S:/data/Projects/ProtemicsLab/Bruker timsTOF/experiments/ms2-feature/correlation/peak_correlation.csv"
select * from peak_correlation order by feature_id,base_peak_id,correlation;
