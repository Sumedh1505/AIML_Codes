# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:50:27 2020

@author: 205557
"""
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import optimizers
from keras.models import Model
from keras.callbacks import History
from keras.applications import vgg16, xception, inception_v3, inception_resnet_v2, nasnet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import pandas as pd
from datetime import datetime as dt
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image
import PIL
import keras
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Dropout
import sys
import os
import argparse
sys.path.append("D:/PROJECTS_ROOT/AIML_utilities/spivision/For_Classification/")
from training import TrainingUtils
from utils import Utility

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def customlayer():
    customlayers=keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    customlayers.add(layers.Dense(100, activation='relu'))
    customlayers.add(BatchNormalization())
    customlayers.add(Dropout(rate= 0.5))
    return customlayers

def custommodel():
	model1=Sequential()
	model1.add(Conv1D(64,kernel_size=(21),activation='relu',input_shape=(64,64)))
	model1.add(MaxPooling1D(pool_size=2))
	model1.add(Flatten())
	model1.add(Dense(2,activation='softmax'))
   return model1


ROOT_PATH = "/home/developer/deep_learning/workspace/ecrash/rotation_prediction/"
DATA_PATH = "/home/developer/deep_learning/data/ecrash/Experiments/rotated_images/"
WEIGHTS_PATH = "/home/developer/deep_learning/PRETRAINED_MODELS/IMAGENET/"
DF_PATH="/home/developer/deep_learning/workspace/ecrash/rotation_prediction/data_df/"
SIM_NUM = 3
MODEL_NAME="nasnet"
DENSE_NEURONS=128
BATCH_SIZE=16
STAGE1_LR=0.001
STAGE2_LR=0.0001
MONITOR=[('val_accuracy','max'),('val_loss','min')]
LEARNING_MONITOR= 'loss'
METRIC=['accuracy']
EPOCHS1=50
EPOCHS2=50
FINETUNE='yes'
NUM_WORKERS=5
CUSTOMLAYER= customlayer()
OUTPUT_ACTIVATION='softmax'
CUSTOMMODEL= custommodel()


def create_dirs():
    """
    This function will be used to create the neccessary data directories,
    which will be further used by the pipeline to save the trained models,
    evaluation results, hyperparameters, training logs etc.
    
        Arguments:
            
            -params  : The list of input params that the user has entered 
                       through the terminal. If nothing is entered, the
                       default values present in the argument parser will
                       be passed as input params.                    
    """
    sim_path = ROOT_PATH + "simulations/" + "SIM_{:02d}/".format(SIM_NUM)
    
    if (os.path.isdir(sim_path) == False):
        os.mkdir(sim_path)
    else:
        print("Please check the SIM NUMBER")
            
    model_path = sim_path + "training_results/"
    eval_path = sim_path + "evaluation_results/"
    weights_path = WEIGHTS_PATH
    source = DATA_PATH
    df_path = DF_PATH
    
    os.mkdir(model_path) if not os.path.isdir(model_path) else None
    os.mkdir(model_path + "stage1/") if not os.path.isdir(model_path + "stage1/") else None
    os.mkdir(model_path + "stage2/") if not os.path.isdir(model_path + "stage2/") else None
         
    os.mkdir(eval_path) if not os.path.isdir(eval_path) else None
    os.mkdir(eval_path + "stage1/") if not os.path.isdir(eval_path + "stage1/") else None
    os.mkdir(eval_path + "stage2/") if not os.path.isdir(eval_path + "stage2/") else None
    input_params = dict()
    input_params['sim'] = SIM_NUM
    input_params['model_name'] = MODEL_NAME
    input_params['dense_neurons'] = DENSE_NEURONS
    input_params['batch_size'] = BATCH_SIZE
    input_params['stage1_lr'] = STAGE1_LR
    input_params['stage2_lr'] = STAGE2_LR 
    input_params['monitor'] = MONITOR
    input_params['metric'] = METRIC
    input_params['learning_monitor']=LEARNING_MONITOR
    input_params['epochs1'] = EPOCHS1
    input_params['epochs2'] = EPOCHS2
    input_params['finetune'] = FINETUNE
    input_params['nworkers'] = NUM_WORKERS
    input_params['custom_layers'] = CUSTOMLAYER
    input_params['custom_model'] = CUSTOMMODEL
    input_params['outputlayer_activation']=OUTPUT_ACTIVATION
    path_dict= dict()
    path_dict['sim_path'] = ROOT_PATH + "simulations/" + "SIM_{:02d}/".format(SIM_NUM)
    path_dict['model_path'] = model_path
    path_dict['eval_path'] = eval_path
    path_dict['weights_path'] =  weights_path
    path_dict['source'] = source
    path_dict['df_path'] = df_path
    return path_dict, input_params 



#Create the directories
path_dict, input_params = create_dirs()

#Create an instance variable of TrainingUtils class
train_utils_obj = TrainingUtils(input_params, path_dict)

#Save the hyperparameters
train_utils_obj.save_params()
PIL.Image.MAX_IMAGE_PIXELS = 933120000
#Start model training and evaluation
train_utils_obj.train()