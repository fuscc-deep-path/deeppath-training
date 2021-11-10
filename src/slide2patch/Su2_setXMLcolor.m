%% change linecolor to '65280' in XML 
clear;
clc;
tic;

%%
xmlPath = 'F:\\TNBC_DL\\DATA\\IMAGE_DATA\\TCGA_TNBC_IMAGES\\TCGA_WSI\\';
newPath = 'F:\\TNBC_DL\\DATA\\IMAGE_DATA\\TCGA_TNBC_IMAGES\\TCGA_WSI\\';

% mkdir(newPath)
xmlall = dir([xmlPath,'*xml']);
xmlfiles = {xmlall.name};
%%
for num = 1: length(xmlfiles)
    xDoc = xmlread([xmlPath, xmlfiles{num}]);
    Annotation = xDoc.getElementsByTagName('Annotation'); % read xmlfile and get all annotations
%     for l = 0: 1: Annotation.getLength - 1 % loop all different linecolor Annotation
    item_Annotation = Annotation.item(0);
    linecolor = item_Annotation.getAttribute('LineColor');
    disp(['Previous color:', char(linecolor)])
    item_Annotation.setAttribute('LineColor', '65280');
    linecolor = item_Annotation.getAttribute('LineColor');
    disp(['New      color:', char(linecolor)])
    xmlwrite([newPath,xmlfiles{num}],xDoc);  %����
end
disp('Finish replacing all !')
toc;