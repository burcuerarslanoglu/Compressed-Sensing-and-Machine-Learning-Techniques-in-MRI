% Preallocate the 256-by-256-by-1-by-20 image array.
X = repmat(int16(0), [256, 256, 1, 18]);

% Read the series of images.
for p = 1:18
    filename = sprintf('%05d.dcm', p); % Adjust the format here to ensure leading zeros
    X(:, :, 1, p) = dicomread(filename);
end

% Display the image stack.
montage(X, [])
 
% Windowing: find the global minimum and maximum pixel values across all images
min_pixel_value = double(min(X(:)));
max_pixel_value = double(max(X(:)));

% Set the window level to the middle of the range, and the width to the range
window_center = (max_pixel_value + min_pixel_value) / 2;
window_width = max_pixel_value - min_pixel_value;

% Apply the windowing for display
X_windowed = double(X) - (window_center - (window_width / 2));
X_windowed = (X_windowed / window_width) * 255;
X_windowed(X_windowed < 0) = 0;
X_windowed(X_windowed > 255) = 255;

% Convert to uint8 for display purposes
X_display = uint8(X_windowed);

% Display the image stack.
montage(X_display, []);