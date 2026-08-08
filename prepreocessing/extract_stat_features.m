function features = extract_stat_features(s)

%% Extraction

features.mean = mean(s);
features.std = std(s);
features.cv = features.std / features.mean;     % coefficients of variation
features.max = max(s);
features.min = min(s);
features.rms = sqrt(mean(s.^2));
features.crest = features.max / features.rms;   % crest factor
features.percentile10 = prctile(s, 10);
features.percentile25 = prctile(s, 25);    
features.percentile75 = prctile(s, 75);    
features.percentile90 = prctile(s, 90);
features.interquartile = features.percentile75 - features.percentile25;
features.skewness = skewness(s);
features.kurtosis = kurtosis(s);
features.zcr = ZCR(s);
features.feat1 = median(s)/(mean(s)+0.00001);
features.feat2 = prctile(s,95)/(max(s)+0.00001);
features.feat3 = prctile(s,5)/(min(s)+0.00001);

features = struct2cell(features);

end