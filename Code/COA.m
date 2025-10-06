% Select the input image folder
image_folder = uigetdir(); % Opens a dialog to select the folder containing images

% Get list of all PNG images in the folder
image_files = dir(fullfile(image_folder, '*.png'));

% Define output directory in Downloads
downloads_folder = fullfile(getenv('USERPROFILE'), 'Downloads'); % Windows
% downloads_folder = fullfile(getenv('HOME'), 'Downloads'); % macOS/Linux
output_folder = fullfile(downloads_folder, 'Extracted_Masked_Orientations');

% Create the output folder if it doesn’t exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Loop through each image file and process it
for k = 1:length(image_files)
    % Read the image
    image_name = image_files(k).name;
    image_path = fullfile(image_folder, image_name);
    I = imread(image_path);
    
    % Image preprocessing
    se = strel('disk', 5);
    background = imopen(I, se);
    I2 = I - background;
    I3 = imadjust(I2);
    
    % Adaptive binarization
    bw = imbinarize(I3); 
    bw = bwareaopen(bw, 50); % Keep smaller objects
    
    % Watershed segmentation
    D = -bwdist(~bw);
    D(~bw) = -Inf;
    L = watershed(D);

    % Extract ellipse properties
    stats = regionprops('table', L, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation');

    % Filter out very large regions (to remove unwanted ellipses)
    image_size_threshold = min(size(I)) * 0.75;
    valid_idx = stats.MajorAxisLength < image_size_threshold;
    stats = stats(valid_idx, :);

    % **NEW: Remove ellipses with integer orientations**
    stats = stats(stats.Orientation ~= round(stats.Orientation), :);

    % Save extracted data to CSV (x-coordinates and orientations)
    csv_filename = fullfile(output_folder, [erase(image_name, '.png'), '_extracted.csv']);
    writematrix([stats.Centroid(:,1), stats.Orientation], csv_filename);

    % Display progress in command window
    fprintf('Processed: %s\nSaved CSV: %s\n', image_name, csv_filename);

    % Plot and save the processed image with ellipses
    figure('Visible', 'off'); % Hide figure while processing
    hold on
    axis equal % Keeps correct aspect ratio
    xlim([0, size(I,2)]);
    ylim([0, size(I,1)]);
    set(gca, 'YDir', 'reverse'); % Keep orientation consistent
    title(['Processed Image: ', image_name])
    for i = 1:height(stats)
        draw_ellipse(stats.MajorAxisLength(i)/2, stats.MinorAxisLength(i)/2, ...
            stats.Orientation(i), stats.Centroid(i,1), stats.Centroid(i,2), 'r');
    end
    hold off

    % Save processed image with ellipses
    processed_image_filename = fullfile(output_folder, [erase(image_name, '.png'), '_processed.png']);
    saveas(gcf, processed_image_filename);
    close(gcf); % Close figure to free memory
end

fprintf('All images processed! Results saved in: %s\n', output_folder);

%% Function to Draw Ellipse
function draw_ellipse(a, b, theta, x0, y0, color)
    t = linspace(0, 2*pi, 100);
    X = a * cos(t);
    Y = b * sin(t);
    R = [cosd(theta), -sind(theta); sind(theta), cosd(theta)];
    rotated = R * [X; Y];
    plot(rotated(1, :) + x0, rotated(2, :) + y0, color, 'LineWidth', 2);
end
