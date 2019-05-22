





    # load the pre-binned ms2 values
    binned_ms2_df = pd.read_pickle(args.prebinned_ms2)
    raw_scratch_df = binned_ms2_df.copy() # take a copy because we're going to delete stuff

    # do intensity descent to find the peaks
    ms2_peaks_l = []
    while len(raw_scratch_df) > 0:
        # find the most intense point
        peak_df = raw_scratch_df.loc[raw_scratch_df.summed_intensity.idxmax()]
        peak_mz = peak_df.mz_centroid
        peak_mz_lower = peak_mz - args.ms2_peak_delta
        peak_mz_upper = peak_mz + args.ms2_peak_delta

        # get all the raw points within this m/z region
        peak_raw_points_df = raw_scratch_df[(raw_scratch_df.mz_centroid >= peak_mz_lower) & (raw_scratch_df.mz_centroid <= peak_mz_upper)]
        if len(peak_raw_points_df) > 0:
            mz_centroid = peakutils.centroid(peak_raw_points_df.mz_centroid, peak_raw_points_df.summed_intensity)
            summed_intensity = peak_raw_points_df.summed_intensity.sum()
            ms2_peaks_l.append((mz_centroid, summed_intensity))

            # remove the raw points assigned to this peak
            raw_scratch_df = raw_scratch_df[~raw_scratch_df.isin(peak_raw_points_df)].dropna(how = 'all')

    ms2_peaks_df = pd.DataFrame(ms2_peaks_l, columns=['mz','intensity'])

    # deconvolute the peaks
    ms2_deconvoluted_peaks, _ = deconvolute_peaks(ms2_peaks_l, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(10.0), truncate_after=0.95)

    ms2_deconvoluted_peaks_l = []
    for peak in ms2_deconvoluted_peaks:
        # discard a monoisotopic peak that has either of the first two peaks as placeholders (indicated by intensity of 1)
        if ((len(peak.envelope) >= 3) and (peak.envelope[0][1] > 1) and (peak.envelope[1][1] > 1)):
            ms2_deconvoluted_peaks_l.append((round(peak.mz, 4), int(peak.charge), peak.neutral_mass, int(peak.intensity), peak.score, peak.signal_to_noise))

    ms2_deconvoluted_peaks_df = pd.DataFrame(ms2_deconvoluted_peaks_l, columns=['mz','charge','neutral_mass','intensity','score','SN'])
    # 'neutral mass' is the zero charge M, so we add the proton mass to get M+H (the monoisotopic mass)
    ms2_deconvoluted_peaks_df['m_plus_h'] = ms2_deconvoluted_peaks_df.neutral_mass + PROTON_MASS

    print("\t\twindow {}, building the MGF".format(window_number))

    # append the monoisotopic and the ms2 fragments to the list for MGF creation
    pairs_df = ms2_deconvoluted_peaks_df[['mz', 'intensity']].copy().sort_values(by=['intensity'], ascending=False)
    spectra = []
    spectrum = {}
    spectrum["m/z array"] = pairs_df.mz.values
    spectrum["intensity array"] = pairs_df.intensity.values
    params = {}
    params["TITLE"] = "RawFile: {} Index: 0 precursor: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {}".format(os.path.basename(CONVERTED_DATABASE_NAME).split('.')[0], precursor_id, feature_charge, feature_intensity, feature_id, round(feature_rt_apex,2))
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(feature_monoisotopic_mass,6), feature_intensity)
    params["CHARGE"] = "{}+".format(feature_charge)
    params["RTINSECONDS"] = "{}".format(round(feature_rt_apex,2))
    params["SCANS"] = "{}".format(int(feature_rt_apex))
    spectrum["params"] = params
    spectra.append(spectrum)

    # add it to the list of spectra
    mgf_spectra.append(spectra)
