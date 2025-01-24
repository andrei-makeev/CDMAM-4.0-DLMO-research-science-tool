import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'

import silence_tensorflow.auto # shuts TF/Keras warnings!

import sys
import beepy
import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Concatenate, Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import keras

from skimage.color import gray2rgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tqdm import tqdm

#----- custom data generator for flipping images and adjusting their labels

# class CustomDataGenerator:
    
#     def __init__(self, image_generator, h_flip_prob= 0.5, v_flip_prob= 0.5):
        
#         self.image_generator= image_generator
#         self.h_flip_prob= h_flip_prob
#         self.v_flip_prob= v_flip_prob
        
#     def _flip_with_label(self, image, label):
        
#         if random.random() < self.h_flip_prob:
            
#             image= np.fliplr(image)                          # flip image horizontally
#             label= [label[1], label[0], label[3], label[2]]  # adjust label for horizontal flip
            
#         if random.random() < self.v_flip_prob:
            
#             image= np.flipud(image)                          # flip image vertically
#             label= [label[2], label[3], label[0], label[1]]  # adjust label for vertical flip
            
#         return image, label
    
#     def flow(self, images, labels, batch_size):
        
#         #----- get base generator from Keras ImageDataGenerator
        
#         gen= self.image_generator.flow(images, labels, batch_size= batch_size)
        
#         while True:
            
#             batch_images, batch_labels= next(gen)
            
#             #----- apply augmentation to each image-label pair in the batch
            
#             augmented_images, augmented_labels= [], []
            
#             for img, lbl in zip(batch_images, batch_labels):
                
#                 aug_img, aug_lbl= self._flip_with_label(img, lbl)
#                 augmented_images.append(aug_img)
#                 augmented_labels.append(aug_lbl)
                
#             yield np.array(augmented_images), np.array(augmented_labels)
            
#----- custom callback to gradually unfreeze ResNet18 layers

# class GradualUnfreeze(Callback):
    
#     def __init__(self, model, unfreeze_schedule, strategy, optimizer):
        
#         """
#         args:
#         - model:              model to be fine-tuned
#         - unfreeze_schedule:  dictionary where keys are epochs, and values are stages to unfreeze
#                               number of layers to unfreeze at each interval
#         - strategy:           tf.distribute strategy for multi-GPU training
#         - optimizer:          optimizer to use
#         """        
#         super(GradualUnfreeze, self).__init__()        
#         self.model= model
#         self.unfreeze_schedule= unfreeze_schedule
#         self.strategy= strategy
#         self.optimizer= optimizer
        
#     def on_epoch_begin(self, epoch, logs= None):
        
#         #----- check if we should unfreeze more layers at this epoch
        
#         if epoch in self.unfreeze_schedule:
            
#             stage_to_unfreeze= self.unfreeze_schedule[epoch]
#             print(f"[INFO] Unfreezing layers in {stage_to_unfreeze} at epoch {epoch}")
            
#             #----- unfreeze layers in specified stage
            
#             for layer in self.model.layers:
                
#                 if stage_to_unfreeze in layer.name:
                    
#                     layer.trainable= True
                
#             #----- trainable vs. frozen layer counts
            
#             trainl_layers = [layer.name for layer in model.layers if layer.trainable]
#             print("***trainable layers:", trainl_layers, len(trainl_layers))
            
#             # trainl_count= sum([1 for layer in self.model.layers if layer.trainable])
#             # frozen_count= len(self.model.layers)-trainl_count            
#             # print(f"**trainl layers: {trainl_count}")
#             # print(f"**frozen layers: {frozen_count}")
            
#             current_lr= self.model.optimizer.lr.numpy()
#             self.optimizer.learning_rate= current_lr
            
#             #----- recompile the model after changing trainable status, within strategy.scope
            
#             #with self.strategy.scope():
            
#                 #----- set the current learning rate from scheduler if needed
                
#                 #current_lr= self.model.optimizer.lr.numpy()
#                 #self.optimizer.learning_rate= current_lr
#                 #self.model.compile(optimizer= self.optimizer, loss= 'categorical_crossentropy', metrics= ['accuracy'])

#----- ResNet18 is no more in Keras: model is taken from github.com/qubvel/classification_models

from classification_models.keras import Classifiers
ResNet18, preprocess_input= Classifiers.get('resnet18')

pc_arr= []  # array with PC_4AFC values for cross-validation

CLASSES= ['q0', 'q1', 'q2', 'q3']
NE= 300    # epochs
BS= 32     # batch size
LR= 1e-05  # 0.001, 0.002, 1.0e-04, 2.5e-04

#----- load training data

print('[INFO] loading images...')

data= []
labl= []
diam= []

#----- parse CL arguments

img_path= sys.argv[1]  # path to testing dataset
writ_app= sys.argv[2]  # write or append cross-validation results
f_output= sys.argv[3]  # name of the text file with CV results
mdl_name= sys.argv[4]  # path to the 'baseline' model

#----- open text file for cross-validation results 

if writ_app== 'w':
    
    f_res= open(f_output, 'w')
    
elif writ_app== 'a':
    
    f_res= open(f_output, 'a')
    
print(img_path+'/*.png')  # roi_*.png
for f in tqdm(glob.glob(img_path+'/*.png')):  # roi_*.png
    
    lbl= f[-6:].split('.')[0]  # GT label, such as 'q2'
    dia= f[-18:-7]             # signal diameter and thickness, such as '2.000_0.078' 
    
    #----- CDMAM image data
    
    img= cv2.imread(f, -1)  # read as is
    #im3= gray2rgb(img)     # replicate in three RGB channels
    im3= np.repeat(img[..., np.newaxis], 3, axis= -1)  # convert to RGB-like format        
    im3= im3/im3.max()      # rescale to [0, 1]
    im3= cv2.resize(im3, (224, 224))  # resize to Resnet standard size
    
    data.append(im3)
    labl.append(lbl)
    diam.append(dia)
    
    #print(f, ' ', lbl)
    
print('loaded', len(data), 'images...')

#----- convert data and labels to NumPy arrays

print('[INFO] processing data...')
data= np.array(data, dtype= 'float32')
labl= np.array(labl)
diam= np.array(diam)

#----- perform one-hot encoding on labels

lb= LabelBinarizer()
labl_hot= lb.fit_transform(labl)

#----- let label be a tuple

labl= tuple([diam, labl_hot])

#----- data augmentaion

# image_gen= ImageDataGenerator() #preprocessing_function= gamma_correction,
#                                 #preprocessing_function= norm_imagenet,
#                                 #brightness_range=[0.9, 1.1],
#                                  shear_range= 0.1,
#                                  rotation_range= 5,
#                                  width_shift_range= 0.2,
#                                  height_shift_range= 0.2,
#                                  zoom_range= [0.75, 1.25],
#                                  fill_mode= 'nearest')

# custom_gen= CustomDataGenerator(image_gen, h_flip_prob= 0.5, v_flip_prob= 0.5)

#----- do a 10-fold cross-validation to obtain PC_4AFC mean & std. dev.

kf= KFold(n_splits= 10, shuffle= True, random_state= None)
for train, test in kf.split(data, labl[1]):
    
    trainX, testX, trainY, testY, testZ= data[train], data[test], labl[1][train], labl[1][test], labl[0][test]
    
    # aug= ImageDataGenerator(zoom_range= 0.3,
    #                         width_shift_range= 0.2,
    #                         height_shift_range= 0.2,
    #                         shear_range= 0.15,
    #                         horizontal_flip= True,
    #                         vertical_flip= True,
    #                         fill_mode= 'nearest',
    #                         brightness_range= [0.4, 1.5])
                             
    #----- load pre-existing **baseline** model (which has knowledge of what CDMAM ROIs look like and was trained with 78k images)
    
    model= keras.models.load_model(str(mdl_name))
    print(f'Loading model from: {str(mdl_name)}')
    
    #----- define early stopping criteria
    
    est= EarlyStopping(monitor= 'val_accuracy', patience= 25, restore_best_weights= True, min_delta= 0.001, verbose= 1)
    
    #----- define callbacks
    
    callback_list= [est]
    
    #----- train the network
    
    print('[INFO] training network...')
    
    #H= model.fit(custom_gen.flow(trainX, trainY, batch_size= BS), validation_data= (testX, testY), steps_per_epoch= trainX.shape[0] // BS, epochs= NE, verbose= 0, callbacks= callback_list)
    H= model.fit(trainX, trainY, batch_size= BS, validation_data= (testX, testY), steps_per_epoch= trainX.shape[0] // BS, epochs= NE, verbose= 0, callbacks= callback_list)
    
    #H= model.fit(aug.flow(trainX, trainY, batch_size= BS), validation_data= (testX, testY), steps_per_epoch= trainX.shape[0] // BS, epochs= NE, verbose= 1, callbacks= callback_list)
    #H= model.fit(trainX, trainY, batch_size= BS, validation_data= (valX, valY), steps_per_epoch= trainX.shape[0] // BS, epochs= NE, verbose= 0)  # , callbacks= callback_list)
    #H= model.fit(aug.flow(trainX, trainY, batch_size= BS), validation_data= (valX, valY), steps_per_epoch= trainX.shape[0] // BS, epochs= NE, verbose= 1)
    
    #-----  evaluate the network and show a classification report
    
    print('[INFO] evaluating network...')
    predictions= model.predict(testX, batch_size= BS)
    print(classification_report(testY.argmax(axis= 1), predictions.argmax(axis= 1), target_names= CLASSES))
    
    #----- look at predictions for the testing set and calculate PC_4-AFC
    
    pp= model.predict(testX)
    
    n_hits= 0
    for i in range(len(testX)):
        
        mdl_pos= pp[i].argmax(axis= 0)     # one-hot position (in a 4-element vector) predicted by the model
        lbl_pos= testY[i].argmax(axis= 0)  # one-hot position (in a 4-element vector) in the corresponding GT label
        
        #----- save (append) data needed to plot PC vs. CDMAM detail diameter and thickness to text file
        
        out_str= str(testZ[i].split('_')[0])+' '+str(testZ[i].split('_')[1])+' '+str(mdl_pos)+' '+str(lbl_pos)+'\n'
        
        if mdl_pos== lbl_pos:
            
            n_hits+= 1
            
    pc= n_hits/len(testX)
    
    print('------------------------------------------------')
    print('4-AFC percent correct for the testing set: %.3f' % pc)
    print('------------------------------------------------')
    
    pc_arr.append(pc)
    print(pc_arr)
    
    #beepy.beep(sound= 1)
    
    #----- calculate mean & std. deviation of PC_4AFC
    
    pc_avg= np.mean(pc_arr)
    pc_std= np.std(pc_arr)
    print('------------------------------------------------')
    print('%.3f' % pc_avg, '+/', '%.3f' % pc_std)
    print('------------------------------------------------')
    
res_str= " ".join(["{:.3f}".format(x) for x in pc_arr])

f_res.write(res_str+'\n')
f_res.close()
