function [areas,...
          perimeters,...
          irregularity_index,...
          eqv_diameter,...
          convex_area,...
          solidity,...
          contrast,...
          correlation,...
          energy,...
          homogeneity,...
          entropy] = process_nodule(input_image_name)
      
      
input_image = imread(input_image_name);
input_image = imresize(input_image, [800 1200]);

%if the input image is RGB, then convert it to gray scale
if size(input_image,3) == 3
    input_image = rgb2gray(input_image);
end

% contrast stretching
local_contrast_image = imadjust(input_image, stretchlim(input_image));

%crop abdominal area
half_width = size(input_image, 2) / 2;
image_height = size(input_image, 1);

for row = image_height : -1 : (3*image_height/4)
    previous_batch_intensity = 0;
    for i = 0:1:4
        previous_batch_intensity = previous_batch_intensity + input_image(row,1+i);   
    end
    previous_batch_intensity = previous_batch_intensity/5;
        
    for col = 6:5:half_width
        
        %compute average pixel intensity
        avg = 0;
        for i = 0:1:4
            avg = avg + input_image(row,col+i);   
        end
        avg = avg/5;
        
        % compare with previous batch
        if avg - previous_batch_intensity >= 15 % 15 is statistically determined and dependent to the histogram equalization bin count.
            input_image = imcrop(input_image, [1 1 1024 (row-1)]);
            break;
        end
        
        previous_batch_intensity = avg;
        
    end
end

% get the negative image
for row = 1:size(input_image,1)
    for col = 1:size(input_image,2)
        input_image(row,col) = 255 - input_image(row,col);
    end
end

%apply median filter(5x5 mask)
lpf_input = medfilt2(input_image, [5 5]);
figure; imshow(lpf_input);

%apply high boost filter
hbf_lpf_input = imsharpen(lpf_input);

%apply histogram equalization
hist_eq_input = histeq(hbf_lpf_input);

%apply median filter(3x3 mask)
lpf_hist_eq_input = medfilt2(hist_eq_input, [3 3]);

%local contrast enhancement
local_contrast_image = imadjust(lpf_hist_eq_input, stretchlim(lpf_hist_eq_input));

% otsu thresholding 
level = graythresh(local_contrast_image);
input_image_binary = imbinarize(local_contrast_image, level);
inverted_input_image_binary = ~input_image_binary;

se = strel('disk',2);

% erosion
eroded_image = imerode(inverted_input_image_binary,se);
% reconstruction
reconstructed_image = imreconstruct(eroded_image,inverted_input_image_binary);
% dilation
dilated_image = imdilate(reconstructed_image,se);
% conditional dilation
result = imreconstruct(imcomplement(dilated_image),imcomplement(reconstructed_image));
result = imcomplement(result);
% foreground markers generation and manipulation
fgm = imregionalmax(result);
fgm = imdilate(fgm,se);
%connected components labelling and analysis
L = bwlabel(fgm);
figure; imshow(L);

% use filtering with respect to areas to find candidate nodulee
table = regionprops('table',L, 'Area');
len = size(table);
area = table.Area;

for i = 1:len
    if(area(i,1) > 4800)
        L(L == i) = 0;
    end
end

% preallocate memory
areas = zeros(len, 1);
perimeters = zeros(len, 1);
irregularity_index = zeros(len, 1);
eqv_diameter = zeros(len, 1);
convex_area = zeros(len, 1);
solidity = zeros(len, 1);
contrast = zeros(len, 1);
correlation = zeros(len, 1);
energy = zeros(len, 1);
homogeneity = zeros(len, 1);
entropy = zeros(len, 1);


% find all region properties of the labelled region
table = regionprops('table',L, 'ConvexHull', 'BoundingBox', 'Area', 'EquivDiameter', 'Perimeter', 'Solidity');

area = table.Area;
ch = table.ConvexHull;
bbox = table.BoundingBox;
eD = table.EquivDiameter;
p = table.Perimeter;
s = table.Solidity;

% calculate features relevant to possible nodule properties
for i = 1:len
    areas()
end


%figure;imshow(imbinarize(L) & fgm4);
%figure;imshow(imbinarize(L) );


end

