# HandDetection_MaskRCNN

This project is a modification of the Mask R-CNN codebase for hand detection.
[https://github.com/tcnshonen/pytorch-mask-rcnn].

# How to run the notebook

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/satyajeetmaharana/HandDetection_MaskRCNN/master

1. You would need to download the preprocessed datasets from this link. 
2. Open the notebook in either mybinder
) or Google Colab (https://colab.research.google.com/github/satyajeetmaharana/HandDetection_MaskRCNN/blob/master/Mask%20R%20CNN%20Model%20for%20Hand%20Detection_Notebook.ipynb)

The Mask R-CNN model is trained with the EgoHands Dataset [http://vision.soic.indiana.edu/projects/egohands/]. 


There are mainly three steps I have followed: 
  (i) Created a dataloader to process the data
  (ii) Built a script for training the model
  (iii) Built an evaluation script to evaluate the model
  
  
## Dataset

The EgoHands dataset contains in total 4800 labeled images from 48 Google Glass videos. The
ground-truth labels consist of polygons with hand segmentations and are provided as Matlab
files. You need to convert them to masks and compute the minimal bounding box. Notice that
some images might not contain any hand at all and you might want to omit these images. There
is no train/val split in the original dataset and you should split it yourself.


## Model

The complete Mask R-CNN architecture is in model.py. However, to better adjust Mask R-CNN
to EgoHands, you are free to modify model.py and use any tricks you learnt from any research literature. A train_model function can be found in model.py. You can use this function to train the model or you can make your own training pipeline if you want.


