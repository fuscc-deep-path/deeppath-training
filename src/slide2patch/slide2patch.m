%% Slide to patch tiling.
clear;
clc;
addpath(genpath('F:\matlab2018\downloads\openslide-matlab'));
disp('Functions Loading Success...')

% @0 load the AllPatient_PI3K.mat for selecting label (altered or not altered)
% varStruct = load('../AllPatient_PI3K.mat');
% pid_need_use = varStruct.pid_need_use; clear varStruct

% @0 read the slide images
path = 'F:\TNBC_DL\DATA\IMAGE_DATA\FUSCC_TNBC_IMAGES\FUSCC_WSI\';
allfile = dir([path, '*xml']);
filenames  = {allfile.name};

%% Supplementary: check the WSI
%{
tic;
for num = 1:length(filenames)
    namesplit = strsplit(filenames{num}, '.');
    casename = [namesplit{1}, '.ndpi'];
%     if ~exist([path, casename],'file') % judge for whether it is svs-format or not.
%         casename = [namesplit{1}, '.svs'];
%     end
    svs = openslide_open([path, casename]);
    disp((openslide_get_level_count(svs))) % nums is 11, 0 is 40x, 1 is 10x, 2 is 2.5x
    disp(openslide_get_level_downsample(svs, 3))
%     [w,h] = openslide_get_level_dimensions(svs, 6)
%     img= openslide_read_region(svs, w/2, h/4, 5000, 5000, 0); % read the whole image at 10x
%     img = img(:, :, 2:end); % variable 'img' is the case image
    disp(['WSI Loading Success...', casename])
end
toc;
%}

%% parameters setting and tiling
tic;
DEFhwSize = 512;
savelevel = 1; % 0, 1, 2, 3
step = 512;
keepr = 0.75; %% drop_rate; only white region area in a mask patch > w*h*drop_rate can be saved
linecolorvalue = 65280;
savepath = 'F:\TNBC_DL\DATA\IMAGE_DATA\FUSCC_TNBC_IMAGES\FUSCC_PATCH\';
disp('Parameters Setting Success...')

for num = 1:length(filenames)
    namesplit = strsplit(filenames{num}, '.');
    casename = [namesplit{1}, '.ndpi'];
    disp([num2str(num), ' / ', num2str(length(filenames)), '  ------ ',  casename, ' checked!'])

    newsavepath = [savepath, namesplit{1}, '/'];   
    if ~exist(newsavepath, 'dir')
        mkdir(newsavepath);
    else
        continue;
    end
    
    func_WSItilingV3_Box([path,casename], linecolorvalue, DEFhwSize, savelevel, step, keepr, newsavepath);
end
toc;
disp('Finish tiling...')
%% @3 proprecessing the whole image�� getting the tissue part mask
% [mask, ~] = FilterBackground_Biospy(img, height, width); 
% proprecessing the image, filter the background
% mask(1:3200,1:3200) = 0; % filter the blue mirror part
% imshow(imresize(mask, 0.05))
%% %%%%%%%%%%%%% @Optional, show the boundary on image
% Mask2Boundary(img, mask, 'g');
%% @4 getting patches from the whole image
% for i = 1: step: size(img, 1) - height  % height 
%     for j = 1: step: size(img, 2) - width % width
%         region = mask(i : i + height - 1, j : j + width - 1);
%         if sum(sum(region)) < height*width*drop_rate
%             continue;
%         end
        %% %%%%%%%% @optional, show the patch boundarybox on image and mask
%         linesy = cat(2, ones(1,width)*i, [i: i+height], ones(1,width)*(i+height), [i+height:-1:i]);
%         linesx = cat(2, [j:j+width], ones(1,height)*(j+width), [j+width:-1:j], ones(1,height)*j);
%         plot(linesx,linesy, 'b', 'LineWidth',1);
        %% %%%%%%%%           
%         patch = img(i : i + height - 1, j : j + width - 1, :);
% %         imshow(patch); pause;
%         patch_name = [savepath, num2str((i-1)/step+1),'_', num2str((j-1)/step+1), format];
%         imwrite(patch, patch_name);
%     end
% end
% disp(['Patches finished! Folder: ', savepath]);
% prediction = zeros((i-1)/step+1, (j-1)/step+1);
% save MaskPredictionReserved.mat prediction
% imshow(prediction)