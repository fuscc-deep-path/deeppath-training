%% Chaoyang 20200220 version3 changing for one slide.
% input:  'slidename' of  WSI files
% input:  linevalue is the label line color; eg. '16711680'-blue; '65280'-green; '255'-red
% input:  the DEFh/DEFw/step of patch you want to get;   eg.  200, 200,100
% input:  drop_rate; only white region area in a mask patch > DEFw*DEFh*drop_rate can be saved

function func_WSItilingV3_Box(slidename, linevalue, DEFsize, savelevel, step, dropR, savepath)
    format = '.png'; 
    scale = 1/ (4^savelevel);
    pointer = openslide_open(slidename);
    splitID = strsplit(slidename, {'\', '.'});
    id = splitID{end-1};

    % process the annotation.
    
    if endsWith(slidename, 'ndpi')
        downLevel = 7; % sampling downingLevel is 7 (2^7=128) in case of Out Of Memory
        fact = 2^downLevel; % svs is resampling from ndpi due to big size, so svs is 4^ ndpi is 2^
        xmlpath = replace(slidename, 'ndpi', 'xml');
    else
        downLevel = 2; % sampling downingLevel is 5 (2^5=32) in case of Out Of Memory
        fact = 4^downLevel; % svs is resampling from ndpi due to big size, so svs is 4^ ndpi is 2^
        xmlpath = replace(slidename, 'svs', 'xml');
    end
    
    [color, annotation_info] = GetAnnotation_MultiColor_XML(xmlpath); % get the struct of the xml annotation
    if ~ismember(linevalue, color) % if 'linecolorvalue' are not in 'color', then 'continue'
        error([num2str(linevalue), ' is not in XML! Please check the color coder!'])
        return;
    end
    index = find([annotation_info.linecolor] == linevalue);
    position = {annotation_info(index).X; annotation_info(index).Y}';    

    % loop for each ROI region in a slide 
    for ind = 1: 1: size(position,1)
        disp(['Now is ROI, ', num2str(ind)]);
        P = [position{ind,1}, position{ind, 2}];

        PosStart = int64(min(P,[],1)/fact);
        PosLen = int64(max(P,[],1)/fact) - PosStart;
        LowMROI = openslide_read_region(pointer,PosStart(1),PosStart(2),PosLen(1),PosLen(2)+1, 'level',downLevel);
        tissueMask = FilterBackground(LowMROI(:,:,2:end));
%         mask = uint8(cat(3, tissueMask, tissueMask, tissueMask));
%         overlapTissue = mask.*LowMROI(:,:,2:end);
%         imshow(overlapTissue)
%         imshow(LowMapROI(:,:,2:end);)
%         imshow(tissueMask)
%         pause;

        % slide window for tiling patches
        for i = 1: step/fact: size(tissueMask,1)-(DEFsize/fact)+1
            for j = 1: step/fact: size(tissueMask,2)-(DEFsize/fact)+1
                region = tissueMask(i: i+DEFsize/fact-1, j: j+DEFsize/fact-1);                               
                if sum(sum(region)) < (DEFsize/fact*DEFsize/fact)*dropR
                    continue;
                end
                patch = openslide_read_region(pointer, (PosStart(1) + j - 1)*fact*scale, (PosStart(2) + i - 1)*fact*scale,...
                                                                                                DEFsize*scale, DEFsize*scale,  'level', savelevel);       
                disp([id,'_',num2str(i),'_',num2str(j), format]);
                patch = imresize(patch(:, :, 2:end), 0.5);
                imwrite(patch, [savepath,id,'_',num2str(i),'_',num2str(j),'_',num2str(ind), format]);
            end
        end
    end
end
%