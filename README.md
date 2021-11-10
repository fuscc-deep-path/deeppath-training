# Deeppath Training

Pipeline for convolutional neural network (CNN) models to infer molecular features from pathological whole slide images (WSIs) by Fudan University Shanghai Cancer Center.

## Steps
The entire workflow comprises three steps:
1. Tessellating the WSI into image tiles;
2. Classifying the image tiles into five tissue types using a developed tissue type classifier;
3. Tiles sampling and CNN model training, validation and test.

## Requirements
The first step is implemented in MATLAB R2018b with the OpenSlide library.

The other two steps are implemented in python 3.8 with the following packages:
```
joblib==1.0.0
numpy==1.19.1
opencv-python==4.3.0.36
openslide-python==1.1.2
pytorch==1.9.1
sklearn==0.0
tensorboard==2.4.1
tensorboardx==2.1
torchvision==0.10.1
```

## Contact
