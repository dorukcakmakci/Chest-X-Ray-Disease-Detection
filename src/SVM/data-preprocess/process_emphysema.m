function [height_div_width,average_intensity] = process_emphysema(input_image_name)

input_image = imread(input_image_name);

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
se = strel('disk', 13, 8);
eroded_image = imerode(inverted_input_image_binary, se);
% se = strel('disk', 5, 8);
% opened_image = imdilate(eroded_image, se);
% figure; imshow(opened_image);
opened_image = eroded_image;

%connected components labelling and analysis
labelled_image = bwlabel(opened_image, 8);

%crop labelled image veertically to get two lungs separately
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

se = strel('disk', 7, 8);
lungs = imdilate(lungs, se);

figure; imshow(lungs);



%process left lung since it is more clear generally
minimum_index = size(labelled_image_left,2);
maximum_index = 1;

for row = 1 : 1 : size(labelled_image_left,1)
    for col = 1 : 1 : size(labelled_image_left,2)
       if (labelled_image_left(row,col) > 0 && col < minimum_index)
           minimum_index = col;
       end
       if (labelled_image_left(row,col) > 0 && col > maximum_index)
           maximum_index = col;
       end
    end
end

lung_width = maximum_index - minimum_index;

minimum_index = size(labelled_image_left,1);
maximum_index = 1;

for col = 1 : 1 : size(labelled_image_left,2)
    for row = 1 : 1 : size(labelled_image_left,1)
       if (labelled_image_left(row,col) > 0 && row < minimum_index)
           minimum_index = row;
       end
       if (labelled_image_left(row,col) > 0 && row > maximum_index)
           maximum_index = row;
       end
    end
end

lung_height = maximum_index - minimum_index;

result = zeros(size(input_image,1), size(input_image,2));

%calculate the grayscale lung image
for row = 1 : 1 : size(input_image,1)
    for col = 1 : 1 : size(input_image,2)
       if (lungs(row,col) > 0)
           result(row, col) = input_image(row,col);
       end
    end
end

%take avg of all lung cells to find if there is air insde lungs
counter = 0;
sum  = 0;
for row = 1 : 1 : size(labelled_image_left,1)
    for col = 1 : 1 : size(labelled_image_left,2)
       if (result(row,col) > 0)
           sum = sum + result(row,col);
           counter = counter + 1;
       end
    end
end

average_intensity = sum / counter;
height_div_width = lung_height / lung_width;
end

