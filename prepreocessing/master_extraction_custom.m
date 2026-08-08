clear all

%%%%%%%%%%%%% GENERATE DATASET FROM CUSTOM RECORDINGS %%%%%%%%%%%%

% add rastamat
addpath(genpath('rastamat'));

addpath(genpath('audio_dataset'));
addpath(genpath('labels_dataset'));

mkdir('second_heartbeats');
mkdir('second_heartbeats_imf');

% read audio files and peak locations
audio_files = dir('audio_dataset/*.wav*');
peak_files = dir('labels_dataset/*.mat');

dataset = [];
label_cnt = 1;
filter1_cnt = 1;
filter2_cnt = 1;
overall_cnt = 1;

results = zeros(31,4);
resultsNum = zeros(31,4);

overall_dataset = [];
labels_filt = [];

% audio data splitting - there's none if 1
splitMax = 1;

% load pitch shifter simulink model
load_system('pitchShifter.slx')

%% iterate through recordings
for k = [27:28] % change accordingly by file idx
    
    audio_file = audio_files(k).name;
    peak_file = peak_files(k).name;
    
    % downsample everything from 44100 Hz to 4000 Hz
    peak_locs = load(peak_file,'lk');
    peak_locs = peak_locs.lk;
    
    [y,fs] = audioread(audio_file);
    y_orig = y;
    fs_orig = fs;
    
    fs_new = 4000;
    y = resample(y,fs_new,fs);
    
    % align Doppler and mic with -60 ms shift
    delay = -0.06;
    peak_locs = peak_locs + round(delay * fs);
    peak_locs = round(peak_locs*fs_new/fs);
    peak_locs_all = peak_locs;
    
    fs = fs_new;
    
    % window size is 150 ms and hop size is 75 ms - CHANGE ACCORDINGLY
    win_size = 0.2*fs;
    check_win_size = 0.5*fs;
    hop_size = round(0.05*fs);
    
    % IMFs are extracted from 50-1000 Hz spectrum
    [b,a] = butter(6,[ 50/(fs/2) 1000/(fs/2)]);
    y_imf = filtfilt(b,a,y);
    
    % best guess for raw HB data is from 50-150 Hz spectrum
    non_filt_y = y;
    [b,a] = butter(6,[50/(fs/2) 150/(fs/2)]);
    y = filtfilt(b,a,y);
    
    % calculate pitched signal from filtered original
    input = y;
    t = 1/fs:1/fs:length(input)/fs;
    t = t';
    options = simset('SrcWorkspace','current');
    set_param('pitchShifter', 'StopTime', num2str(t(end)))
    sim('pitchShifter',[],options)
    pitched = simout(0.04*fs:end);
    pitched(end+1:length(y)) = 0;
    
    %% iterate through splits - if applicable (default = 1)
    for splitNum = 1:splitMax
        
        dataset = [];
        label_cnt = 1;
        
        % only focus on the part of recording if splitMax > 1
        peak_locs = peak_locs_all;
        peak_locs(peak_locs>round((splitNum)/splitMax*length(y))) = [];
        peak_locs(peak_locs<round((splitNum-1)/splitMax*length(y))) = [];
        if length(peak_locs)<2
            continue;
        end
        
        % add filtered raw data and pitched data to the analysis
        data = [y pitched];
        
        splitString = split(audio_file,'.');
        rec_name = [splitString{1} '_chunk.' splitString{2}];
        
        long_cnt = 0;
        frame_counter = 0;
        
        %% iterate through each segment
        for i = 2*fs+1 : hop_size : length(y)-2*fs
            win = y(i:i+win_size-1);
            
            overall_cnt = overall_cnt + 1;
            
            long_cnt = long_cnt+1;
            
            frame_counter = frame_counter + 1;
            
            % if there's no heartbeat label in the vicinity, just continue
            check_win = [i:i+check_win_size-1];
            Y = peak_locs;
            locations = ismember(Y,check_win);
            positions = find(locations);
            if (isempty(positions))
                filter1_cnt = filter1_cnt + 1;
                continue;
            end
            
            % check frequency ratio in order to find poorly placed mic
            % recordings
            check_freq_win = non_filt_y(i - fs/2 : i + fs/2);
            freqs_temp = fft(check_freq_win);
            freqs = abs(freqs_temp(1:fs/2)).^2;
            freq_ratio = sum(freqs(1:1*100)) / sum(freqs);
            if freq_ratio < 0.99
                filter2_cnt = filter2_cnt + 1;
                continue;
            end
            
            % check HB position in the analysis window - add a bit of
            % padding on the sides (20%) to remove bordering positives and
            % negatives from the dataset
            X = [i-win_size/5:i+win_size-1+win_size/5];
            locations = ismember(Y,X);
            positions = find(locations);
            if (isempty(positions))
                label(label_cnt) = -1;
            else
                label(label_cnt) = (Y(positions(1)) - X(1)) / (X(end) - X (1));
            end
            
            % chunk raw_filtered and pitched from the overall recordings
            data_wins = data(i:i+win_size-1,:);
            feat_row = [];
            
            % calculate IMFs
            emd_sift_tolerance = 0.2;
            emd_win = emd(y_imf(i-fs:i+win_size-1+fs,:),'MaxNumIMF',2,'SiftRelativeTolerance', emd_sift_tolerance);
            emd_win = emd_win(fs+1:end-fs,:);
            
            % add IMFs to raw_filtered + pitched chunks
            data_wins = [data_wins emd_win];
            
            % iterate through data (raw_filtered + pitched + IMFs) and extract features
            for data_win = 1 : size(data_wins,2)
                
                % normalize chunks
                chunk = data_wins(:,data_win);
                chunk = chunk/(rms(chunk));

                chunk_size = length(chunk);
                
                % extract spectral features
                centroid = spectralCentroid(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                crest = spectralCrest(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                decrease = spectralDecrease(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                entropy = spectralEntropy(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                flatness = spectralFlatness(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                kurtosis = spectralKurtosis(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                skewness = spectralSkewness(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                slope = spectralSlope(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                spread = spectralSpread(chunk,fs,'Window',hamming(chunk_size,'periodic'));
                
                % append spectral features
                spectral_df = [centroid crest decrease entropy flatness kurtosis skewness slope spread];
                
                % extract statistical features
                stat_df = extract_stat_features(chunk)';
                
                % mfcc and plp feature extraction - only on raw and pitched
                if data_win <= 2
                    [mfcc_coeffs] = melfcc(win,4000,'maxfreq', 2000);
                    [cep2, spec2,ignore,lpcas] = rastaplp(win, fs, 0, 12);
                    mfcc_mean = mean(lpcas');
                    delta_mean = mean(mfcc_coeffs');
                    mfcc_std = std(lpcas');
                    delta_std = std(mfcc_coeffs');
                    mirFeats = [mfcc_mean delta_mean mfcc_std delta_std];
                else
                    mirFeats = [];
                end
                
                % append features together
                feat_row = [feat_row;  cell2mat(stat_df'); spectral_df'; mirFeats'];
            end
            
            % add feature row along with the label
            dataset = [dataset; [feat_row' label(end)]];
            
            % increment label cnt
            label_cnt = label_cnt + 1;
        end
            
        % remove examples that are have labels close to window edges
        for i = 1 : size(dataset,1)
            if dataset(i,end) >= 0
                if dataset(i,end) >= 0.2 && dataset(i,end) <= 0.8
                    dataset(i,end) = 1;
                else
                    dataset(i,end) = 0;
                end
            end
        end
        dataset(dataset(:,end) == 0,:) = [];
        
        overall_dataset = [overall_dataset; dataset];
        
    end
end

%% normalize overall dataset
feat_mean = mean(overall_dataset(:,1:end-1),1);
feat_std = std(overall_dataset(:,1:end-1),1);
overall_dataset(:,1:end-1) = (overall_dataset(:,1:end-1) - feat_mean) ./ feat_std;

%% run feature naming script
name_features;

%% save dataset
T = array2table(overall_dataset, ...
    'VariableNames',feature_names);
writetable(T,'dataset.csv');