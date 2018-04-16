.mode csv
.headers on
.once "summed_ms1_regions.csv"
select * from summed_ms1_regions order by feature_id;
.once "summed_ms2_regions.csv"
select * from summed_ms2_regions order by feature_id;
.once "peak_correlation.csv"
select * from peak_correlation order by feature_id,base_peak_id,correlation;
