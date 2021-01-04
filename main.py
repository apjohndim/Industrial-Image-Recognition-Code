import matplotlib.pyplot as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import cv2
import os
import time   # time1 = time.time(); print('Time taken: {:.1f} seconds'.format(time.time() - time1))
import warnings
import sys
from PIL import Image 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from imutils import paths
import numpy as np
import os
import warnings
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
import numpy
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

sys.path.insert(1, 'C:\\Users\\User\\ZZZ. INDUSTRY\\rEPRODUCTION cODE')
import pandas as pd
import data_loader
import model_maker

from data_loader import load_casting, load_defloc, load_mag, load_tech, load_br, load_el
from model_maker import make_lvgg, make_vgg, make_xception
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import cv2

import seaborn as sns
import math

def gaussian (img_array):

    mean = 0
    vvv = 0.01
    sigma = vvv**0.5
    gaussian = np.random.normal(mean, sigma, (img_array.shape[0],img_array.shape[1]))
    
    noisy_image = np.zeros(img_array.shape, np.float32)
    noisy_image[:, :, 0] = img_array[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img_array[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img_array[:, :, 2] + gaussian
    
    
    
    return noisy_image


def train(epochs,batch_size, model, in_shape, tune, classes, n_split):

    scores = [] #here every fold accuracy will be kept
    f1_scores = []
    recalls = []
    precisions = []
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    test_labels = np.empty(0) #here, every fold labels are kept
    conf_final = np.array([[0,0],[0,0]])
    predictions_all_num = np.empty([0,classes])
    
    omega = 1
    
    # with tf.device('/cpu:0'):
        
    for train_index,test_index in KFold(n_split).split(data):
        trainX,testX=data[train_index],data[test_index]
        trainY,testY=labels[train_index],labels[test_index]
        
        if model == 'lvgg':
            model3 = make_lvgg(in_shape, tune, classes) #in every iteration we retrain the model from the start and not from where it stopped
        elif model == 'xception':
            model3 = make_xception(in_shape, tune, classes)
        elif model == 'vgg':
            model3 = make_vgg(in_shape, 20, classes)
       
        
        if omega == 1:
           model3.summary()
        omega = omega + 1   
        
        print('[INFO] PREPARING FOLD: '+str(omega-1))
        
        history = model3.fit(trainX, trainY, validation_split=0.1, epochs=epochs, batch_size=batch_size)
        
        
        #aug = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        #aug.fit(trainX)
        #history = model3.fit_generator(aug.flow(trainX, trainY,batch_size=batch_size), epochs=epochs, steps_per_epoch=len(trainX)//batch_size)
        
        predict = model3.predict(testX) #for def models functional api
        predict_num = predict
        predict = predict.argmax(axis=-1) #for def models functional api
        
        score = model3.evaluate(testX,testY)
        score = score[1] #keep the accuracy score, not the loss
        scores.append(score) #put the fold score to list
        print ('[INFO] Accuracy ',score)
       
        testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    
        
        if classes == 2:
            recall = recall_score(testY2,predict)
            recalls.append(recall)
            
            precision = precision_score(testY2,predict)
            precisions.append(precision)
        
            oneclass = predict_num[:,1].reshape(-1,1)
            auc = roc_auc_score(testY2, oneclass)
        
            conf = confusion_matrix(testY2, predict) #get the fold conf matrix
            conf_final = conf + conf_final
        
            f1 = f1_score(testY2, predict)
            f1_scores.append(f1)
            
            average_precision = 'n/a'
        else:
            precision = dict()
            recall = dict()
            average_precision = dict()
            for i in range(classes):
                precision[i], recall[i], _ = precision_recall_curve(testY[:, i], predict_num[:, i])
                average_precision[i] = average_precision_score(testY[:, i], predict_num[:, i])

# A "micro-average": quantifying score on all classes jointly
            precision["micro"], recall["micro"], _ = precision_recall_curve(testY.ravel(), predict_num.ravel())
            average_precision["micro"] = average_precision_score(testY, predict_num, average="micro")
        
            
            
        
    
        predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
        predictions_all_num = np.concatenate([predictions_all_num, predict_num])
        testY = testY.argmax(axis=-1)
        test_labels = np.concatenate ([test_labels, testY]) #merge the two np arrays of labels
        
    
    auc = roc_auc_score(labels, predictions_all_num)
    rounded_labels=np.argmax(labels, axis=1)    
    conf_final = confusion_matrix(rounded_labels, predictions_all)
    
    if classes == 2:
        precision_final = precision_score(np.argmax(labels,axis=-1),np.argmax(predictions_all_num,axis=-1))
        recall_final = recall_score(np.argmax(labels,axis=-1),np.argmax(predictions_all_num,axis=-1))
    else:
        precision_final = 'n/a'
        recall_final = 'n/a'
    
    
    scores = np.asarray(scores)
    final_score = np.mean(scores)
    f1sc = np.asarray(f1_scores)
    mean_f1 = np.mean(f1sc)
    
    
    # # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    
    # if 'val_acc' in history.history.keys():
    #     plt.plot(history.history['val_acc'])
    # elif 'val_accuracy' in history.history.keys():
    #     plt.plot(history.history['val_accuracy'])

    # plt.title('Accuracy')
    # plt.ylabel('Accuracy )%)')
    # plt.xlabel('Epoch')
    
    # if 'val_acc' in history.history.keys():
    #     plt.legend(['train', 'validation'], loc='upper left')
    # if 'val_accuracy' in history.history.keys():
    #     plt.legend(['train', 'validation'], loc='upper left')
    # else:
    #     plt.legend(['train'], loc='upper left')
    # plt.grid(False)
    # plt.gca().spines['bottom'].set_color('0.5')
    # plt.gca().spines['top'].set_color('0.5')
    # plt.gca().spines['right'].set_color('0.5')
    # plt.gca().spines['left'].set_color('0.5')
    # plt.savefig('C:\\Users\\User\\ZZZ. INDUSTRY\\accs.png', dpi=300)
    # plt.show()
    
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'])

    # plt.title('Losses')
    # plt.ylabel('Loss (%)')
    # plt.xlabel('Epoch')
    # if 'val_loss' in history.history.keys():
    #     plt.legend(['train', 'validation'], loc='upper left')
    # else:
    #     plt.legend(['train'], loc='upper left')
    # plt.grid(False)
    # plt.gca().spines['bottom'].set_color('0.5')
    # plt.gca().spines['top'].set_color('0.5')
    # plt.gca().spines['right'].set_color('0.5')
    # plt.gca().spines['left'].set_color('0.5')
    # plt.savefig('C:\\Users\\User\\ZZZ. INDUSTRY\\losses.png', dpi=300)
    # plt.show()
    
    # # roc curve
    # fpr = dict()
    # tpr = dict()
    
    # for i in range(classes):
    #     fpr[i], tpr[i], _ = roc_curve(labels[:, i],predictions_all_num[:, i])
    #     plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))
    #     plt.plot(fpr[i], tpr[i], lw=2)
    
    # plt.xlabel("false positive rate")
    # plt.ylabel("true positive rate")
    # plt.legend(loc="best")
    # plt.title("ROC curve")
    # plt.grid(False)
    # plt.gca().spines['bottom'].set_color('0.5')
    # plt.gca().spines['top'].set_color('0.5')
    # plt.gca().spines['right'].set_color('0.5')
    # plt.gca().spines['left'].set_color('0.5')
    # plt.savefig('C:\\Users\\User\\ZZZ. INDUSTRY\\rocurves.png', dpi=300)
    # plt.show()
    
    
    # classes = ['Crease', 'Crescent Gap', 'Inclusion', 'Oil Spot', 'Puncing Hole', 'Rolled Pit', 'Silk Spot', 'Waist Folding', 'Water Spot', 'Welding Line']
    # classes = ['Defect', 'Ok']
    # classes = ['Blowhole', 'Break', 'Crack', 'Fray', 'Free', 'Uneven']
    
    # classes = ['Adapter Square', 'Adapter Trian', 'Box', 'Bracket Pl', 'Bracker Scr', 'Cap', 'Car Rim', 'Clamp B', 'Clamp S',
    #            'Connector Pl', 'Cylinder', 'Engine bearing', 'Cooler r', 'Cooler s', 'Cover', 'Filer', 'Fuse', 'Handle',
    #            'Pump', 'Multi Bracket', 'Punched Rail', 'Screw', 'Screw Bl', 'Star', 'Tee Connector', 'Thread', 'Washer']
    
    # classes = ['Crack', 'Ok']
    
    #classes = ['Ok', 'Defect']
    
    cnf_matrix = confusion_matrix(np.argmax(labels,axis=-1), predictions_all)
    
    
    
    
    #np.set_printoptions(precision=2)

    #conf_img = plot_confusion_matrix(cnf_matrix, classes)
    conf_img = 0
    
    return model3, predictions_all,predictions_all_num,test_labels,labels, auc, conf_final,final_score,mean_f1,precision_final,recall_final,average_precision,cnf_matrix,conf_img,history

import numpy as np
import tensorflow as tf
from tensorflow import keras


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    
   #  fig, ax = plt.subplots(figsize=(15,15))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.xlim(-0.5, len(classes)-0.5) # ADD THIS LINE
    plt.ylim(len(classes)-0.5, -0.5) # ADD THIS LINE
    fig.tight_layout()
    plt.grid(False)
    plt.show()
    fig.savefig('C:\\Users\\User\\ZZZ. INDUSTRY\\cnf.png', dpi=300)
    
    
    
# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import cv2

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    
    
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)



    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


#%% TRAIN - OBTAIN RESULTS

  # GIVE PATH TO IMAGES
# path = 'E:\\Data (Biomarkers phase 2)\\NVLES\\'
#path = 'E:\DATA INDUSTRIAL RECOGNITION\Magnetice Tile Defece'
path = 'E:\\DATA INDUSTRIAL RECOGNITION\\casting product image data for quality inspection'
#path = 'E:\\DATA INDUSTRIAL RECOGNITION\\Defects location for metal surface'
#path = 'E:\\DATA INDUSTRIAL RECOGNITION\\MVTec ITODD\\scenes\\'
#path = 'E:\\DATA INDUSTRIAL RECOGNITION\\Bridge_Crack_Image\\DBCC_Training_Data_Set\\'
#path = 'E:\\DATA INDUSTRIAL RECOGNITION\\httpsgithub.comzae-bayernelpv-dataset\\elpv-dataset-master\\images\\'



  # LOAD IMAGES WITH DIFERRENT FUNCTIONS

#data, labels, labeltemp, image = load_defloc(path)
data, labels, labeltemp, image = load_casting(path)
#data, labels, labeltemp, image = load_mag(path)
#data, labels, labeltemp, image = load_tech(path)
#data, labels, labeltemp, image = load_br(path)
#data, labels, labeltemp, image = load_el(path)

# data, labels, labeltemp, image = load_MPIP(path)


# #
from matplotlib import pyplot as plt
plt.imshow(image, interpolation='nearest')
plt.show()


img = data[1,:,:,:]
# img = img*10
plt.imshow(img, interpolation='nearest')
plt.show()

# 75 is height = first dimension
#in_shape = (142,211,3) # tech loc
#in_shape = (100,100,3) # casting
# in_shape = (32,32,3) # casting
#in_shape = (32,32,3) # el
in_shape = (100,100,3) # el


tune = 1 # SET: 1 FOR TRAINING SCRATCH, 0 FOR OFF THE SHELF, INTEGER FOR TRAINABLE LAYERS (FROM TUNE AND DOWN, THE LAYERS WILL BE TRAINABLE)
# classes = 27 for tech database

#classes = 2
classes = 2
epochs =15
batch_size = 32
n_split = 10



#%% FIT THE MODEL TO THE DATA (FOR PHASE 2)


model = 'vgg'
model3, predictions_all,predictions_all_num,test_labels,labels, auc, conf_final,final_score,mean_f1,precision_final,recall_final,average_precision,cnf_matrix,cnf_img,history = train(epochs,batch_size, model, in_shape, tune, classes, n_split)













#%%
# SAVE AND LOAD MODEL


# model = make_vgg(in_shape,22,classes)
# model.fit(data, labels, epochs=epochs, batch_size=batch_size)

# model.save('C:\\Users\\User\\ZZZ. INDUSTRY\\tmp.h5')

# model = tf.keras.models.load_model('C:\\Users\\User\\ZZZ. INDUSTRY\\tmp.h5')



# #%% 

# ''' GRAD CAM'''


# def Grad_pics (path, save_path, model,w,h):
#     # name = 'ISIC_0015984.jpg'
#     # imagePath = 'E:\\Data (Biomarkers)\\SKIN\\nv\\{}'.format(name)
#     import os
#     from imutils import paths
    
#     imagePaths = sorted(list(paths.list_images(path)))
#     preprocess_input = keras.applications.mobilenet.preprocess_input
    
#     for imagePath in imagePaths[:20]:
#         img_array = preprocess_input(get_img_array(imagePath, size=(h, w)))
        
#         name = os.path.splitext(os.path.basename(imagePath))[0]
#         ext = os.path.splitext(os.path.basename(imagePath))[1]
        
#         last_conv_layer_name = model.layers[19].name
        
#         classifier_layer_names = [model.layers[20].name,model.layers[21].name, model.layers[22].name, model.layers[23].name, model.layers[24].name ]
        
        
#         heatmap = make_gradcam_heatmap(
#             img_array, model, last_conv_layer_name, classifier_layer_names)
        
        
#         heatmap= np.uint8(255 * heatmap)
#         jet = cm.get_cmap("coolwarm")
#         #jet = cm.get_cmap("binary")
#        # jet = cm.get_cmap("winter")
        
#         # We use RGB values of the colormap
#         jet_colors = jet(np.arange(256))[:, :3]
#         jet_heatmap = jet_colors[heatmap]
        
        
#         # img = cv2.imread(imagePath)
#         # img = cv2.resize(img, (w,h))
#         # We create an image with RGB colorized heatmap
        
#         big_heatmap = cv2.resize(jet_heatmap, dsize=(w, h), 
#                                  interpolation=cv2.INTER_CUBIC)
        
#         big_heatmap = keras.preprocessing.image.img_to_array(big_heatmap)
        
#         jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        
        
#         jet_heatmap = jet_heatmap.resize((img_array.shape[2], img_array.shape[1]))
#         jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        
#         # Superimpose the heatmap on original image
#         # superimposed_img = big_heatmap * 200 + img
#         superimposed_img = img_array[0,:,:,:] + (big_heatmap*2)
#         superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        
#         # superimposed_img = keras.preprocessing.image.array_to_img(img_array[0,:,:,:])
        
#         # Save the superimposed image
#         save_path2 = save_path+name+ext
#         superimposed_img.save(save_path2)
        
#         # Display Grad CAM
#         #from IPython.display import Image
#         #display(Image(save_path))


# #%%

# h = in_shape[0]
# w = in_shape[1]

# path = 'E:\\DATA INDUSTRIAL RECOGNITION\\casting product image data for quality inspection\\ok\\'
# save_path = 'C:\\Users\\User\\ZZZ. INDUSTRY\\FEATURES\\'


# #%%
# Grad_pics (path, save_path, model,w,h)


# #%%


# ''' FEATURE MAPS'''

# def feature_maps(path,save_path,model):

#     from matplotlib import pyplot
#     from numpy import expand_dims
#     from keras.preprocessing.image import load_img
#     from keras.preprocessing.image import img_to_array
    
    
#     import os
#     from imutils import paths
    
#     imagePaths = sorted(list(paths.list_images(path)))
#     preprocess_input = keras.applications.mobilenet.preprocess_input
    
#     for imagePath in imagePaths[:1]:
#         name = os.path.splitext(os.path.basename(imagePath))[0]
#         ext = os.path.splitext(os.path.basename(imagePath))[1]
#     # redefine model to output right after the first hidden layer
#         ixs = [2, 5, 9, 13, 17]
#         outputs = [model.layers[i].output for i in ixs]
#         model2 = tf.keras.Model(inputs=model.inputs, outputs=outputs)
#         # load the image with the required shape
        
        
#         img =get_img_array(imagePath, size=(h, w))
#         # expand dimensions so that it represents a single 'sample'
#         # prepare the image (e.g. scale pixel values for the vgg)
#         # img = preprocess_input(img)
#         # get feature map for first hidden layer
#         feature_maps = model2.predict(img)
#         # plot the output from each block
#         square = 3
#         o = 1
#         for fmap in feature_maps:
#         	# plot all 64 maps in an 8x8 squares
#         	ix = 1
#         	for _ in range(square):
#         		for _ in range(square):
#         			# specify subplot and turn of axis
#         			ax = pyplot.subplot(square, square, ix)
#         			ax.set_xticks([])
#         			ax.set_yticks([]); fig = pyplot.gcf();
#         			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gist_gray')
#         			ix += 1
#         	o=o+1 ; p = save_path + name + str(o) + ext ; fig.savefig(p)
#     return imagePath
            
#             #fig.savefig(save_path+name+ext)

# #%%

# pathsaa = feature_maps(path,save_path,model)







