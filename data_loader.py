## MODULE FOR IMPORTING ALL THE DATASETS
print("[INFO] Importing Libraries")
import matplotlib as plt
plt.style.use('ggplot')
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from PIL import Image 
import numpy
import keras
SEED = 50   # set random seed


#imgplot = plt.imshow(img)

def load_casting (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image


def load_br (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.resize(image, (32, 32))/image.max()
        image = cv2.resize(image, (72, 72))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image


def load_defloc (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (205,100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    # labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image



def load_mag (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (120,120))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    # labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image

def load_tech (path):   

    
    import pandas as pd
    import numpy as np
    import os
    labels_file = pd.read_csv('E:\\DATA INDUSTRIAL RECOGNITION\\MVTec ITODD\\scenes\\Book1.csv',dtype=object)
    labels_f = np.array(labels_file)
    
    path = 'E:\\DATA INDUSTRIAL RECOGNITION\\MVTec ITODD\\scenes\\'
    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
    
        
        image = cv2.imread(imagePath)
        if image is None:
            continue
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (211,142))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        
        pos = np.where (labels_f == label[6:])
        try:
            pos1 = pos[0][0]
        except Exception:
            pos1 = pos[0]
        label_2 = labels_f[pos1,1]            
        labels.append(label_2)
        
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    # labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image



def load_el (path):   

    
    import pandas as pd
    import numpy as np
    import os
    
    fname = 'E:\\DATA INDUSTRIAL RECOGNITION\\httpsgithub.comzae-bayernelpv-dataset\\elpv-dataset-master\\labels.csv'
    imported = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=[
                         'path', 'probability', 'type'])
    image_fnames = np.char.decode(imported['path'])
    probs = imported['probability']
    types = np.char.decode(imported['type'])
    
    
    
    
    path = 'E:\\DATA INDUSTRIAL RECOGNITION\\httpsgithub.comzae-bayernelpv-dataset\\elpv-dataset-master\\images\\'
    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
    
        
        image = cv2.imread(imagePath)
        if image is None:
            continue
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100,100))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-1]
        label = 'images/' + label[:-4] + '.png'
        
        pos = np.where (label == image_fnames)
        try:
            pos1 = pos[0][0]
        except Exception:
            pos1 = pos[0]
        label_2 = probs[pos1]
        if float(label_2) > 0:
            label_2 = 1
        labels.append(label_2)
        
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image

def load_break (path):    
    print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for imagePath in imagePaths: #load, resize, normalize, etc
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (175, 115))/image.max()
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
        
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image


