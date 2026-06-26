close all 
clear all

%% Loading and calling the function to have the DICOM series

fully_sampled = display_dicom_series('/Users/burcu/Downloads/MATLAB/AMRI/4321_COMPRESSED_SENSING/20190522_1.3.46.670589.11.42151.5.0.14852.2019052212421305004/MR/00201_3DT1_full');
cs_applied_4 = display_dicom_series('/Users/burcu/Downloads/MATLAB/AMRI/4321_COMPRESSED_SENSING/20190522_1.3.46.670589.11.42151.5.0.14852.2019052212421305004/MR/00301_3DT1_CS_4');
cs_applied_2 = display_dicom_series('/Users/burcu/Downloads/MATLAB/AMRI/4321_COMPRESSED_SENSING/20190522_1.3.46.670589.11.42151.5.0.14852.2019052212421305004/MR/00401_3DT1_CS_2');

%% Display the image stacks without windowing
% 
% figure(1); 
% montage(fully_sampled, []);
% title('DICOM Series without Windowing (Fully Sampled)');
% 
% figure(2); 
% montage(cs_applied_4, []);
% title('DICOM Series without Windowing (CS Applied, 4)');
% 
% figure(3); 
% montage(cs_applied_2, []);
% title('DICOM Series without Windowing (CS Applied, 2)');

%% Calling the function to window the DICOM series

fully_sampled_windowed = window_dicom_series(fully_sampled);
cs_applied_4_windowed = window_dicom_series(cs_applied_4);
cs_applied_2_windowed = window_dicom_series(cs_applied_2);


%% Display the image stack with windowing

figure(4); 
montage(fully_sampled_windowed, []);
title('DICOM Series (Fully Sampled)');
pause

figure(5); 
montage(cs_applied_4_windowed, []);
title('DICOM Series (CS Applied, 4 Times Undersampled)');
pause

figure(6); 
montage(cs_applied_2_windowed, []);
title('DICOM Series (CS Applied, 2 Times Undersampled)');
pause

%% Regions with pure noise

figure(7);
montage(fully_sampled_windowed(30:45,90:105,1,60), []);

figure(8);
montage(cs_applied_4_windowed(30:45,90:105,1,60), []);

figure(9);
montage(cs_applied_2_windowed(30:45,90:105,1,60), []);

%% SNR calculations

var_fully_sampled = VarianceFinder(fully_sampled_windowed);
noise_region_fully_sampled = fully_sampled_windowed(30:45,90:105,1,60);
noise_var_fully_sampled = 120*VarianceFinder(noise_region_fully_sampled);
SNR_fully_sampled = SNRfinder(var_fully_sampled,noise_var_fully_sampled);

var_cs_applied_4 = VarianceFinder(cs_applied_4_windowed);
noise_region_cs_applied_4 = cs_applied_4_windowed(30:45,90:105,1,60);
noise_var_cs_applied_4 = 120*VarianceFinder(noise_region_cs_applied_4);
SNR_cs_applied_4 = SNRfinder(var_cs_applied_4,noise_var_cs_applied_4);

var_cs_applied_2 = VarianceFinder(cs_applied_2_windowed);
noise_region_cs_applied_2= cs_applied_2_windowed(30:45,90:105,1,60);
noise_var_cs_applied_2 = 120*VarianceFinder(noise_region_cs_applied_2);
SNR_cs_applied_2 = SNRfinder(var_cs_applied_2,noise_var_cs_applied_2);

%% ML Application

%Taking the pre-trained neural network
net = denoisingNetwork('DnCNN');

%Denoising the image via pre-trained neural network
DNN_fully_sampled = denoiseImage(fully_sampled_windowed,net);
DNN_cs_applied_4 = denoiseImage(cs_applied_4_windowed,net);
DNN_cs_applied_2 = denoiseImage(cs_applied_2_windowed,net);

%% Display the image stack with DNN

figure(10); 
montage(DNN_fully_sampled, []);
title('DICOM Series (Fully Sampled), dCNN Applied');
pause

figure(11); 
montage(DNN_cs_applied_4, []);
title('DICOM Series (CS Applied, 4 Times Undersaple), dCNN Applied');
pause

figure(12); 
montage(DNN_cs_applied_2, []);
title('DICOM Series (CS Applied, 2 Times Undersaple), dCNN Applied');
pause

%% SNR calculations

var_DNN_fully_sampled = VarianceFinder(DNN_fully_sampled);
noise_region_DNN_fully_sampled = DNN_fully_sampled(30:45,90:105,1,60);
noise_var_DNN_fully_sampled = 120*VarianceFinder(noise_region_DNN_fully_sampled);
SNR_DNN_fully_sampled = SNRfinder(var_DNN_fully_sampled,noise_var_DNN_fully_sampled);

var_DNN_cs_applied_4 = VarianceFinder(DNN_cs_applied_4);
noise_region_DNN_cs_applied_4 = DNN_cs_applied_4(30:45,90:105,1,60);
noise_var_DNN_cs_applied_4 = 120*VarianceFinder(noise_region_DNN_cs_applied_4);
SNR_DNN_cs_applied_4 = SNRfinder(var_DNN_cs_applied_4,noise_var_DNN_cs_applied_4);

var_DNN_cs_applied_2 = VarianceFinder(DNN_cs_applied_2);
noise_region_DNN_cs_applied_2 = DNN_cs_applied_2(30:45,90:105,1,60);
noise_var_DNN_cs_applied_2 = 120*VarianceFinder(noise_region_DNN_cs_applied_2);
SNR_DNN_cs_applied_2 = SNRfinder(var_DNN_cs_applied_2,noise_var_DNN_cs_applied_2);


%% Functions

function [X] = display_dicom_series(folder_path)
    cd(folder_path);

    % Preallocate the 160-by-160-by-1-by-120 image array.
    X = repmat(int16(0), [160, 160, 1, 120]);

    % Read the series of images.
    for p = 1:120
        filename = sprintf('%05d.dcm', p); % Adjust the format here to ensure leading zeros
        X(:, :, 1, p) = dicomread(filename);
    end
end

function [X_display] = window_dicom_series(X)
    % Windowing: find the global minimum and maximum pixel values across all images
    min_pixel_value = double(min(X(:)));
    max_pixel_value = double(max(X(:)));

    % Set the window level to the middle of the range, and the width to the range
    window_center = (max_pixel_value + min_pixel_value) / 2;
    window_width = max_pixel_value - min_pixel_value;

    % Apply the windowing for display
    X_windowed = double(X) - (window_center - (window_width / 2));
    X_windowed = (X_windowed / window_width) * 159;
    X_windowed(X_windowed < 0) = 0;
    X_windowed(X_windowed > 159) = 159;

    % Convert to uint8 for display purposes
    X_display = uint8(X_windowed);
end

function varian_square = VarianceFinder(region)

    %Type conversion
    region = double(region);

    %Getting the size of the region
    N1 = size(region,1);
    N2 = size(region,2);

    %Finding the average of the region
    avg = sum(region,'all')/(N1*N2);

    summation = 0;

    %Iterating through the region for finding the variance V
    for i = 1:N1
        for j = 1:N2
            summation = summation+(region(i,j)-avg)^2; 
        end
    end

    %Finding the variance V of the flat region
    varian_square = summation/(N1*N2);

end

function SNR = SNRfinder(var_signal,var_noise)
    SNR = 10*log10(var_signal/var_noise);
end