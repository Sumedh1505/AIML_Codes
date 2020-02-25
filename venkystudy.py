# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 08:03:51 2020

@author: 205557
"""

import sys
sys.path.append("D:/PROJECTS_ROOT/DataServices/Lexis Nexis/ecrash/Modeling/codes/modelingpipeline/")
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import optimizers
from keras.models import Model
from keras.callbacks import History
from keras.models import model_from_json
from keras.applications import vgg16, xception, inception_v3, inception_resnet_v2,nasnet
import pandas as pd
import os
from datetime import datetime as dt
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from evaluation import EvalUtils
from keras.utils import plot_model
from contextlib import redirect_stdout
from time import time
import keras
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Dropout
#from keras.callbacks.tensorboard_v1 import TensorBoard
from utils import Utility
import keras.backend as K
import numpy as np

liby= "D:/PROJECTS_ROOT/DataServices/Lexis Nexis/ecrash/Modeling/codes/modelingpipeline/"


def count_params(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return trainable_count, non_trainable_count
    #print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    #print('Trainable params: {:,}'.format(trainable_count))
    #print('Non-trainable params: {:,}'.format(non_trainable_count))

paramcount_list= []
for model_name in ["vgg16","nasnet","inception_resnet","inceptionv3","xception"]:
    if(model_name=="vgg16"):
        base_model = vgg16.VGG16(weights=None, include_top=False)
    elif(model_name=="inceptionv3"):
        base_model = inception_v3.InceptionV3(weights=None, include_top=False)
    elif(model_name=="resnet50"):
        base_model = resnet.ResNet50(weights=None, include_top=False)
    elif(model_name=="inception_resnet"):
        base_model = inception_resnet_v2.InceptionResNetV2(weights=None, include_top=False)
    elif(model_name=="nasnet"):
        base_model = nasnet.NASNetLarge(weights=None, include_top=False)
    elif(model_name=="xception"):
        base_model = xception.Xception(weights=None, include_top=False)
    print("model name is:",model_name)
    paramcount_list.append((model_name,"base",count_params(base_model)))
    #Adding a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    customlayers= customlayer()
    output_layer = customlayers(x)
    model_stg1 = Model(inputs=base_model.input, outputs=output_layer)
    model_json = model_stg1.to_json()
    modelpath= liby+model_name+".json"
    with open(modelpath, "w") as json_file:
        json_file.write(model_json)
        modelpath= liby+model_name+".pdf"
    plot_model(model_stg1, 
               to_file=modelpath, 
               show_shapes=True, 
               show_layer_names=True)
    paramcount_list.append((model_name,"alllayers",count_params(model_stg1)))
    
df= pd.DataFrame()

modelname=[x[0] for x in paramcount_list]
layers= [x[1] for x in paramcount_list]
trainable=[x[2][0] for x in paramcount_list]
non_trainable=[x[2][1] for x in paramcount_list]


df['trainable']=trainable
df['nontrainable']=non_trainable
df['model']=modelname
df['layers']=layers

df.to_csv(liby+ 'compile.csv')
count_params(model_stg1)
count_params(base_model)

54490937 - 54336736