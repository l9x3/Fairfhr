% add feature names
        spectral_prefix = {'spectral_'};
        spectral_names = {'centroid', 'crest', 'decrease', 'entropy', 'flatness', 'kurtosis', 'knewness', 'slope', 'spread'};
        spectral_names = strcat(spectral_prefix,spectral_names);
        
        statistics_names = {'mean', 'std', 'cv', 'max', 'min', 'rms', 'crest', 'p10', 'p25', 'p75', 'p90', 'iqr', ...
            'skewness', 'kurtosis', 'zcr', 'median/mean', 'p95/max', 'p5/min'};
        
        feature_suffix = horzcat(statistics_names,spectral_names);
        
        feature_names = {};
        for i = 1 : size(data_wins,2)
            if i == 1
                feature_prefix = {'raw_filt_'};
            elseif i == 2
                feature_prefix = {'pitched_'};
            else
                feature_prefix = strcat({'IMF'},num2str(i-2),{'_'});
            end
            
            assembled_names = strcat(feature_prefix,feature_suffix);
            feature_names = horzcat(feature_names,assembled_names);
        end
        
        feature_names = horzcat(feature_names,{'label'});
        
        pitch_names = {};
        pitch_extra_names = {'plp_mean','mfcc_mean','plp_std','mfcc_std'};
        for i = 1 : 13
            feature_prefix = strcat({'raw_filt_'},num2str(i),{'_'});
            pitch_names = horzcat(pitch_names,feature_prefix);
        end
        
        psycho_names1 = {};
        for i = 1 : length(pitch_extra_names)
            feature_prefix = strcat(pitch_names,pitch_extra_names(i));
            psycho_names1 = horzcat(psycho_names1,feature_prefix);
        end
        
        pitch_names = {};
        pitch_extra_names = {'plp_mean','mfcc_mean','plp_std','mfcc_std'};
        for i = 1 : 13
            feature_prefix = strcat({'pitched_'},num2str(i),{'_'});
            pitch_names = horzcat(pitch_names,feature_prefix);
        end
        
        psycho_names2 = {};
        for i = 1 : length(pitch_extra_names)
            feature_prefix = strcat(pitch_names,pitch_extra_names(i));
            psycho_names2 = horzcat(psycho_names2,feature_prefix);
        end
            
          
        feature_names = [feature_names(1:27) psycho_names1 feature_names(28:54) psycho_names2 feature_names(55:end)];