###############################################################################
###############################################################################
####### The following code was written by:
####### 
####### Erica Moreira
####### Maja Garbulinska
#######
####### This is the setup.py file where we import all the necessary functions,  
####### define paths to folders that the code uses and as well as .py files 
####### that are to be called by the user. We also included some instructions. 
###############################################################################
###############################################################################

###############################################################################
########################### IMPORT PACKAGES
###############################################################################

import numpy as np
import os
import sys
import shutil
import glob
import matplotlib.pyplot as plt
import torch
import PIL
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nrrd 

from scipy import ndimage
from tifffile import imread
from tifffile import imsave
from glob import glob
from torch.nn import init
from torchvision import transforms
from torch.utils import data
from PIL import Image

###############################################################################
########################## DEFINE PATHS
###############################################################################
#######
####### All the paths defined below are subpaths of the project_path path. 
####### It is required from the user to input the project_path before 
####### running the setup.py file. (This python script) 
#######
####### The folder that the project_path points to contains the following 
####### folders:
#######
####### data       Where data are to be stored
####### scriptsPY  Where all the .py scripts needed are stored. 
####### temp       This folder is used to save temporal labels and images used 
#######            during the training process. This folder should contain a
#######            labels folder and a training folder. 
####### results    Where predictions for all the images are stored after the 
#######            correctly trained model is ready. we have also included a 
#######            piece of code that saves the predictions being done while 
#######            the model is learning. They will be saved in this folder. 
#######            This makes it possible for the user to know what is happening
#######            while the model is being trained for several epochs. 
#######  
###############################################################################

os.chdir(project_path)

## Results will be saved here.
results_path = os.path.join(project_path, "results")

## Path to where the original images are saved. Python will read them all.
image_path = os.path.join(project_path, "data/original/*.*")

## If available, read labels for comparison folder. Python will read them all.
truth_path = os.path.join(project_path, "data/reference/*.*")

## Path to the folder where labels are saved and read during the training 
## and labeling process. This by default will be the labels folder 
## in the temp folder.
labels_path = os.path.join(project_path, "temp/labels")

## Path to the folder where training images are stored temporary to facilitate 
## the training process. This by default will be the training folder in the
## the temp folder.  
training_path = os.path.join(project_path, "temp/training")

## Saving model parameters to here
model_path = os.path.join(project_path, "temp/modeldict")

## folder where python scripts with the model and slicer manipulation are saved. 
scripts_path = os.path.join(project_path, "scriptsPy")

###############################################################################
########################## DEFINE PATHS T0 SPECIFIC SCRIPTS
###############################################################################
#######
####### SCRIPT NAME         FUNCTION
#######
####### modelFunctions      Defines the functions that are needed for this 
#######                     process. 
#######
####### loadFiles           Load the images to slicer and prepares the view for 
#######                     labeling.
#######
####### saveLabels          Pull the labels from slicer and save as an npy.
#######                     object as well as separate 
#######  
####### startModel          Define the model
####### 
####### runModel            Run the model
####### 
####### testModel           Test the model on a set that it has never seen before 
#######                     - the validation set
#######
####### predictLabels       Predict new labels for the training set.
#######
####### loadFilesLabeled    Load back files and their labels predicted by 
#######                     the model. 
#######
####### predictAllLabels    Predict labels for all images
#######
###############################################################################

## Defines the functions that are needed for this process. 
modelFunctions = os.path.join(scripts_path, "modelFunctions.py")

## Load the images to slicer and prepares the view for labeling.
loadFiles = os.path.join(scripts_path, "loadFiles.py")

## Pull the labels from slicer and save as an npy. object as well as separate 
## .tiff files. 
saveLabels = os.path.join(scripts_path, "saveLabels.py")

## Define the model.
startModel = os.path.join(scripts_path, "startModel.py")

## Run the model.
runModel = os.path.join(scripts_path, "runModel.py")

## Test the model on a set that it has never seen before - the validation set
testModel = os.path.join(scripts_path, "testModel.py")

## Predict new labels for the training set.
predictLabels = os.path.join(scripts_path, "predictLabels.py")

## Load back files and their labels predicted by the model. 
loadFilesLabeled = os.path.join(scripts_path, "loadFilesLabeled.py")

## Predict labels for all images
predictAllLabels = os.path.join(scripts_path, "predictAllLabels.py")

###############################################################################
########################## DEFINE ORIGINAL DATA LISTS
###############################################################################
##(label files are set in the TrainTestSplit function)

## List all the files available in the image_path 
X_filenames = np.array(sorted(glob(image_path)))

## List all the files available in the truth_path
z_filenames = np.array(sorted(glob(truth_path)))

###############################################################################
########################## DEFINE PARAMETERS
###############################################################################

## number of epoch at each run 
epochs=100

## number of training images 
numbertrainingimg = 3

## number of validation images 
numbervalidimg = 3

## random seed 
seed=2

###############################################################################
########################## DEFINE MODEL FUNCTIONS
###############################################################################
#######
####### The following script will run the modelFunctions.py and therefore 
####### define all the functions that we need in this process.
####### 
###############################################################################

exec(open(modelFunctions).read())

###############################################################################
########################## GET THE TRAIN AND VALIDATION/TEST SET 
###############################################################################
#######
####### This function will divide the data into training and testing.
####### This function's output has 6 elements:
####### X_train      - list of input training set files
####### y_train      - list of output training set files 
####### X_valid      - list of input validation set files
####### y_valid      - list of output validation set files. These are the true 
#######                labels that are used for comparison with the model 
#######                predictions during the testing process.
####### indeces      - indeces of the training images 
####### indecesvalid - indeces of the validation images 
####### 
###############################################################################

## This function will work even if you do not have reference images for 
## testing. 


[X_train, y_train, X_valid, y_valid, indeces, indecesvalid] \
= TrainTestSplit(numbertrainingimg=numbertrainingimg,
                 numbervalidimg=numbervalidimg,
                 seed=seed, 
                 labelpath=labels_path, 
                 X_filenames=X_filenames, 
                 z_filenames=z_filenames)



###############################################################################
########################## SAMPLE WORKFLOW
###############################################################################
####### 
####### Once you have defined all the paths and needed variables, you can 
####### go ahead and start the segmentation process, by running the following
####### commands in the slicer python interpreter. 
#######
####### - exec(open(loadFiles).read()) - load files to slicer 
####### - label the files as best as you can, the better they are labeled
#######   the faster the model will learn. You can learn more about it from 
#######   the documentation we have provided. 
####### - exec(open(saveLabels).read()) - you can now go ahead and save the 
#######   labels by running this command. 
####### - exec(open(startModel).read()) - define your model, you need to do it 
#######   only once. 
####### - exec(open(runModel).read()) - this command will run the model for 
#######   as many epochs as you specified. You can see the progress of the 
#######   training by looking at the examples of predictions on the training
#######   set that are being saved while the model is learning in the results 
#######   folder. If you think your model could benefit from being train 
#######   for another set of epochs just rerun this command. 
####### - exec(open(testModel).read()) running this is not necessary but can 
#######   be helpful. This command will take your model and compute the 
#######   accuracy on the set of testing images. 
####### - once you have trained the model, you can predict the labels on 
#######   the training set. These labels can be displayed with images in Slicer
#######   by running the following command:
####### - exec(open(loadFilesLabeled).read()) - after running this Slicer will
#######   load in the original images again but with the predicted labels. 
#######   If you think that the model did a good job you can correct the labels
#######   that are wrong to make the performance even better. If the labels are 
#######   still not too good, you can load the images without labels using 
#######   exec(open(loadFiles).read()) and label more. 
####### - Assuming that your model is finished and predicts very well, you 
#######   can predict labels on all the data by running the following:
#######   exec(open(predictAllLabels).read()). The labels will be saved in 
#######   the results folder. 
####### 
###############################################################################



