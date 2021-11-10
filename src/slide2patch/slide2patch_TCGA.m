%% Slide to patch tiling. TCGA
clear;
clc;
addpath(genpath('F:\matlab2018\downloads\openslide-matlab'));
disp('Functions Loading Success...')

% @0 read the slide images
path = 'F:\TNBC_DL\DATA\IMAGE_DATA\TCGA_TNBC_IMAGES\TCGA_WSI\';
allfile = dir([path, '*xml']);
filenames  = {allfile.name};

%% Supplementary: check the WSI
tic;
level_count = [];
level_downsample1 = []; 
level_downsample2 = []; 
highest_Mag = [];
slide_mpp = [];

for num = 1:length(filenames)
    namesplit = strsplit(filenames{num}, '.');
    casename = [namesplit{1}, '.ndpi'];
    if ~exist([path, casename],'file') % judge for whether it is svs-format or not.
        casename = [namesplit{1}, '.svs'];
    end
    svs = openslide_open([path, casename]);
    disp(['WSI Loading Success...', casename])
%     properties = openslide_get_property_names(svs);
    highestMag = openslide_get_property_value(svs, 'aperio.AppMag');
    mpp = openslide_get_property_value(svs, 'aperio.MPP');
    levelcount = openslide_get_level_count(svs);
    
    leveldownsample1 = openslide_get_level_downsample(svs, 1);
    leveldownsample2 = openslide_get_level_downsample(svs, 2);
    
    highest_Mag = [highest_Mag; highestMag];
    slide_mpp = [slide_mpp; str2double(mpp)];
    level_count = [level_count; levelcount];
    level_downsample1 = [level_downsample1; leveldownsample1];
    level_downsample2 = [level_downsample2; leveldownsample2];
end
check_info = table(filenames', highest_Mag, slide_mpp, level_count, level_downsample1, level_downsample2);
toc;

%% parameters setting and tiling
tic;
savelevel = 0; % 0, 1, 2, 3

keepr = 0.75; %% drop_rate; only white region area in a mask patch > w*h*drop_rate can be saved
linecolorvalue = 65280;
savepath = 'F:\TNBC_DL\DATA\IMAGE_DATA\TCGA_TNBC_IMAGES\TCGA_PATCH\';
disp('Parameters Setting Success...')

for num = 1:length(filenames)
    namesplit = strsplit(filenames{num}, '.');
    casename = [namesplit{1}, '.svs'];

    disp([num2str(num), ' \ ', num2str(length(filenames)), '  ------ ',  casename, ' checked!'])

    newsavepath = [savepath, namesplit{1}, '\'];   
    if ~exist(newsavepath, 'dir')
        mkdir(newsavepath);
    else
        continue;
    end
    
    func_WSItilingV3_Box_TCGA([path,casename], linecolorvalue, savelevel, keepr, newsavepath);
end
toc;
disp('Finish tiling...')

%% %%%%%%%%%%%%% @Optional, show the boundary on image
% Mask2Boundary(img, mask, 'g');