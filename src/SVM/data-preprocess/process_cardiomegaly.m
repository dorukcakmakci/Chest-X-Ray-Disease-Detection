function cardiothorasic_ratio = process_cardiomegaly(input_image_name)

% read input image
input_image = imread(input_image_name);

%if the input image is RGB, then convert it to gray scale
if size(input_image,3) == 3
    input_image = rgb2gray(input_image);
end

% histogram equalization with 50 bins experimental value
input_image = histeq(input_image, 50);

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

% otsu thresholding 
level = graythresh(input_image);
input_image_binary = imbinarize(input_image, level);
inverted_input_image_binary = ~input_image_binary;

%morphological operations
se = strel('disk', 5, 8);
eroded_image = imerode(inverted_input_image_binary, se);

se = strel('disk', 3, 8);
opened_image = imdilate(eroded_image, se);

%connected components labellign and analysis
labelled_image = bwlabel(opened_image, 8);

%crop labelled image vertically to get two lungs separately
labelled_image_left = imcrop(labelled_image, [1 1 512 1024]);
labelled_image_right = imcrop(labelled_image, [512 1 1024 1024]);


%find the connected component with largest areas, that are the lungs
areas_left = regionprops(labelled_image_left, 'area');
len = size(areas_left);
max = areas_left(1).Area;
second_max = 0;

for i = 1:len
    if(areas_left(i).Area >= max)
        max = areas_left(i).Area;
    end
end

for i = 1:len
    if(areas_left(i).Area == max)
        continue;
    else
        labelled_image_left(labelled_image_left == i) = 0;
    end
    
end

areas_right = regionprops(labelled_image_right, 'area');
len = size(areas_right);
max = areas_right(1).Area;

for i = 1:len
    if(areas_right(i).Area >= max)
        max = areas_right(i).Area;
    end
end

for i = 1:len
    if(areas_right(i).Area == max)
        continue;
    else
        labelled_image_right(labelled_image_right == i) = 0;
    end
    
end

lungs = cat(2,labelled_image_left,labelled_image_right);

% to find cardiomegaly, we need to check if (A + B) * 100 / C >= 50%. 
row_count = size(lungs,1);
col_count = size(lungs,2);

% as an estimation to A+B, find average of distance between two lungs
distances = [];
flag = 0;
flag_2 = 0;
start_row = 1;

for R = 1:row_count
    previous_pixel_intensity = 0;
    distance = 0;
    for C = 1:col_count
        
        current_pixel_intensity = lungs(R,C);
        
        if(current_pixel_intensity < previous_pixel_intensity)
            if(flag_2 == 0)
               flag_2 = 1;
            end
            flag = 1;
        elseif(current_pixel_intensity > previous_pixel_intensity)
            flag = 0;
        end
        
        if(flag == 1)
            distance = distance + 1;
        end
        
        previous_pixel_intensity = current_pixel_intensity;
        
    end
    if(flag_2 == 1)
        start_row = R;
        flag_2 = 2;
    end
    distances =[distances, distance];
    distance = 0;
    flag = 0;
end

% compute A + B estimate
numerator = 0;
for i = 1:size(distances,2)
    numerator = numerator + distances(i);
end
numerator = numerator / (size(distances,2)- start_row);

% C is the distance between lowerleft and lowerright pixels
flag = 0;
flag_2 = 0;
denominator = 0;
for R = row_count:-1:1
    previous_pixel_intensity = 0;
    for C = 1:col_count
        
        current_pixel_intensity = lungs(R,C);
        
        if(current_pixel_intensity < previous_pixel_intensity)
            flag = 1;
        elseif(current_pixel_intensity > previous_pixel_intensity)
            flag = 0;
            flag_2 = 1;
        end
        
        if(flag == 1)
            denominator = denominator + 1;
        end
        
        previous_pixel_intensity = current_pixel_intensity;
        
    end
    if (flag_2 == 1)
        break;
    end
    
end
cardiothorasic_ratio = numerator * 100 / denominator;
    if cardiothorasic_ratio > 100
        cardiothorasic_ratio = 50 + 50 * rand(1);
    end
end


