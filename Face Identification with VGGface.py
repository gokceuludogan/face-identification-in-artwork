#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from lbp import lbp_features
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.engine import  Model
from keras.layers import Input, Flatten, Dense
from keras_vggface.vggface import VGGFace
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
import multiprocessing
from sklearn.svm import SVC
import cv2 
from pathlib import Path
from sklearn.metrics import accuracy_score
import pandas as pd

# In[3]:


def get_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    if ALIGN == True:
        img_cv = np.array(img.convert('RGB'))[:, :, ::-1].copy()
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))      
        if len(faces) == 1:
            x, y, w, h = faces[0]
            img = img_cv[y:y + h, x:x + w]
        else:
            img = img_cv
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) # or version=2
    if ALIGN == True:
        from skimage.transform import rescale, resize
        x = resize(x, (1, 224, 224), anti_aliasing=True)
    return (image_path, x)
# In[4]:


def vgg_model():
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        
    hidden_dim = 64

    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(nb_class, activation=(tf.nn.softmax) , name='fc8')(x)
    model = Model(vgg_model.input, out)

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss=["categorical_crossentropy"],
                  metrics=['accuracy'])
    return model


# In[5]:


def categorize(y_train, y_test):
    y_train = to_categorical(y_train, nb_class)
    y_test = to_categorical(y_test, nb_class)
    return y_train, y_test


# In[7]:

#
#def train(model, data, nb_epochs, val_split=0.2):
#    ## Divide into train and validation
#    x_train, x_test, y_train, y_test = data
#    hist = model.fit(x_train[:900], y_train[:900], epochs=nb_epochs, validation_data=(x_train[900:], y_train[900:]), batch_size=8)
#    return hist, model


# In[10]:


#path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/faces_in_artwork/'
#ALIGN = True
#faces_in_artwork = get_dataset(path, get_image, '_vgg_finetune')


# In[12]:

#
#data = train_test_split(faces_in_artwork)
#y_train, y_test  = categorize(data[1], data[3])
#model = vgg_model()
#hist, model = train(model, (data[0], data[1], y_train, y_test), 10)
#y_pred = model.predict(x_test)
#y_pred = np.argmax(y_pred, axis=1)
#y_test = np.argmax(y_test, axis=1)


# In[79]:


def get_vgg_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    if ALIGN == True:
        img_cv = np.array(img.convert('RGB'))[:, :, ::-1].copy()
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))      
        if len(faces) == 1:
            x, y, w, h = faces[0]
            img = img_cv[y:y + h, x:x + w]
        else:
            img = img_cv
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) # or version=2
    if ALIGN == True:
        from skimage.transform import rescale, resize
        x = resize(x, (1, 224, 224), anti_aliasing=True)
    return (img, pretrained_vgg.predict(x))

pretrained_vgg = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max


# In[9]:


def get_dataset(path, feature, file_ext, dump=True):
    dataset = {'image_paths': [], 'targets': [], 'feature': []}
    target_dirs = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    for index, target in tqdm(enumerate(target_dirs)):
        target_path = os.path.join(path, target)
        images = [os.path.join(target_path, img) for img in  os.listdir(target_path)]
        for img in images:
            item = feature(img)
            dataset['image_paths'].append(item[0])
            dataset['targets'].append(index)        
            dataset['feature'].append(item[1])
    if dump == True: 
        with open(PICKLES_PATH / str(path + '_' + file_ext), 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)              
    return dataset     


# In[6]:


def load_dataset(path):
     with open(path, 'rb') as handle:
        return pickle.load(handle)      


# In[11]:


def train_test_split(dataset, split=0.25):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
            dataset['feature'], dataset['targets'], test_size=0.25, random_state=1, stratify=dataset['targets'])
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    return x_train, y_train, x_test, y_test


# In[8]:

def grid_search(model, param_grid, cv, dataset, model_name):
    x_train, y_train, x_test, y_test = dataset
    clf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=4, cv=cv, verbose=1)
    clf.fit(x_train, y_train)   
    best_model = clf.best_estimator_     
    print(clf.best_score_)                                  
    #print(clf.best_estimator_)                            
    #print(clf.score(x_test, y_test)) 
    y_pred = best_model.predict(x_test)
    print('Best model test acc:',  accuracy_score(y_pred, y_test))

    with open(PICKLES_PATH / str(model_name + '.pkl'), 'wb') as f:
        pickle.dump(clf, f)
        
    #print(metrics.classification_report(y_test, y_pred))
    return clf 

# In[8]:

def pipeline(dataset, model_name):
    splitted = train_test_split(dataset)
    svc = SVC()
    Cs = range(-8, 8)
    gammas = range(-5, 2)
    param_grid = {'C': [10 ** c for c in Cs] , 'gamma' : ['auto', 'scale'], 'kernel' : ['linear', 'rbf', 'poly']}
    model = grid_search(svc, param_grid, 6, splitted, model_name)
    X_train, y_train, X_test, y_test = splitted
    result = evaluate_gs( X_train, y_train, X_test, y_test, model_name, model)
    return result    

#%%
def evaluate_gs(X_train, y_train, X_dev, y_dev, model_name, model=None):
    if model == None:
        with open(PICKLES_PATH / str(model_name + '.pkl'), 'rb') as f:
            model = pickle.load(f)

    best_model = model.best_estimator_
    train_acc = accuracy_score(best_model.predict(X_train), y_train)
    dev_acc = accuracy_score(best_model.predict(X_dev), y_dev)
    best_model_idx = model.best_index_
    scores = pd.DataFrame(model.cv_results_)
    cv_mean = scores.loc[best_model_idx, 'mean_test_score']
    cv_std = scores.loc[best_model_idx, 'std_test_score']

    return {'ta':train_acc, 'da':dev_acc, 'cm':cv_mean, 'cs':cv_std}    
# In[18]:

PICKLES_PATH = Path('pickles')
nb_class = 100
path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/real-face-subset/'
real_faces = get_dataset(path, get_vgg_features, '_vgg')


# In[15]:
pipeline(real_faces)


# In[82]:


ALIGN = True
path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/real-face-subset/'
real_faces = get_dataset(path, get_vgg_features, '_vgg_aligned')
pipeline(real_faces)


# In[19]:


path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/faces_in_artwork/'
faces_in_artwork = get_dataset(path, get_vgg_features, '_vgg')


# In[20]:


path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/'
faces_in_artwork = load_dataset(os.path.join(path, 'faces_in_artwork_vgg'))
pipeline(faces_in_artwork)


# In[80]:


get_ipython().run_line_magic('time', '')
path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/faces_in_artwork/'
faces_in_artwork = get_dataset(path, get_vgg_features, '_vgg_aligned')


# In[81]:


pipeline(faces_in_artwork)


# In[37]:


Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
clf = GridSearchCV(estimator=svc_1, param_grid=param_grid, n_jobs=-1)
clf.fit(x_train, y_train)        
print(clf.best_score_)                                  
print(clf.best_estimator_.C)                            
 # Prediction performance on test set is not as good as on train set
y_pred = clf.predict(x_test)
clf.score(x_test, y_test) 

