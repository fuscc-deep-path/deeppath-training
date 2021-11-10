% filter the white background in needle biospy slide
% Input: needle biospy slide rgb_img and the small area you want to filter
% Output: the binary mask and the tissue without white background
function bw = FilterBackground(rgb_img)
    radius =5;
%     gray = rgb2gray(rgb_img);
    gray = rgb_img(:, :, 2);
    bw = imbinarize(gray);
    
    bw = imfill(~bw, 'holes');  
%     imshow(bw)
    se = strel('disk', 3);
    bw = imerode(bw, se);        
%     imshow(bw)
    se = strel('disk', 5);
    bw = imdilate(bw, se);
%     imshow(bw)
    bw = imfill(bw, 'holes');
    bw = bwareaopen(bw, double(radius*radius), 8);
%     imshow(bw)
% overlap the mask on original image
%     mask = cat(3, bw, bw, bw);
%     mask = uint8(mask);
%     tissue = mask.*rgb_img;
    disp('Filter background from tissues, Success...')
end
