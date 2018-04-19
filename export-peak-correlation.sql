.mode csv
.headers on

-- find the top-correlating peaks
-- select * from peak_correlation a where a.ROWID IN ( SELECT b.ROWID FROM peak_correlation b WHERE b.feature_id = a.feature_id ORDER by correlation DESC LIMIT 2);

-- get the points from the ms1 summed feature regions for the positive-correlating ms2 peaks
.once "summed_ms1_regions.csv"
select * from summed_ms1_regions where (feature_id,peak_id) in (select feature_id,base_peak_id from peak_correlation where correlation > 0);

-- get the points from the ms2 summed feature regions for the positive-correlating ms2 peaks
.once "summed_ms2_regions.csv"
select * from summed_ms2_regions where (feature_id,peak_id) in (select feature_id,ms2_peak_id from peak_correlation where correlation > 0);

.once "peak_correlation.csv"
select * from peak_correlation where correlation > 0;
