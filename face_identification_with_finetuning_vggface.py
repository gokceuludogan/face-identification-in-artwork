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
from keras.layers import Input, Flatten, Dense, Dropout
from keras_vggface.vggface import VGGFace
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import multiprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from pathlib import Path
from sklearn.metrics import accuracy_score
import pandas as pd
import cv2

nb_class = 100

def get_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    if ALIGN == True:
        img_cv = np.array(img.convert('RGB'))[:, :, ::-1].copy()
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('gdrive/My Drive/Colab Notebooks/haarcascade_frontalface_default.xml')
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

def vgg_model(hidden_dims, dropout, act_fn, optimizer):
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        
    hidden_dim = 512
    for layer in vgg_model.layers:
        layer.trainable = False
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    for hidden_dim in hidden_dims:
      x = Dense(hidden_dim, activation=act_fn)(x)
      x = Dropout(dropout)(x)
    #x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(nb_class, activation=(tf.nn.softmax) )(x)
    model = Model(vgg_model.input, out)

    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss=["categorical_crossentropy"],
                  metrics=['accuracy'])
    #print(model.summary())
    return model

def categorize(y_train, y_test):
    y_train = to_categorical(y_train, nb_class)
    y_test = to_categorical(y_test, nb_class)
    return y_train, y_test

def train(model, data, nb_epochs, val_split=0.2):
    ## Divide into train and validation
    x_train, x_test, y_train, y_test = data
    hist = model.fit(x_train[:900], y_train[:900], epochs=nb_epochs, validation_data=(x_train[900:], y_train[900:]))
    return hist, model

path = os.path.join(os.getcwd(), 'faces_in_artwork/')
ALIGN = True
faces_in_artwork = get_dataset(path, get_image, '_vgg_finetune')

vggface = KerasClassifier(build_fn=vgg_model, batch_size=256, epochs=100, verbose=0)

param_grid = {'hidden_dims': [[128, 64], [256, 64, 32]], 'dropout': [0.0, 0.5, 0.7], 'act_fn': ['relu'], 'optimizer':['adam']}

#param_grid = {'hidden_dims': [[64], [128]], 'dropout': [0.0, 0.5], 'act_fn': ['relu'], 'optimizer':['adam']}
data = train_test_split(faces_in_artwork)
y_train, y_test  = categorize(data[1], data[3])



# data = train_test_split(faces_in_artwork)
# y_train, y_test  = categorize(data[1], data[3])
# model = vgg_model()
# hist, model = train(model, (data[0], data[1], y_train, y_test), 10)

x_train, y_train, x_test, y_test = (data[0], y_train, data[2],  y_test)
clf = GridSearchCV(estimator=vggface, param_grid=param_grid, n_jobs=1, cv=5, verbose=4)
clf.fit(x_train, y_train)   
best_model = clf.best_estimator_     
print(clf.best_score_)                                  
#print(clf.best_estimator_)                            
#print(clf.score(x_test, y_test))

train_acc = accuracy_score(best_model.predict(x_train), np.argmax(y_train, axis=1))
test_acc = accuracy_score(best_model.predict(x_test), np.argmax(y_test, axis=1))
best_model_idx = clf.best_index_
scores = pd.DataFrame(clf.cv_results_)
cv_mean = scores.loc[best_model_idx, 'mean_test_score']
cv_std = scores.loc[best_model_idx, 'std_test_score']
print(train_acc, test_acc, cv_mean, cv_std)

#param_grid = {'hidden_dims': [[128, 64]], 'dropout': [0.0], 'act_fn': ['relu'], 'optimizer':['adam']}
#epochs = 20
#0.8736059479553904 0.16991643454038996 0.08271375520771099 0.027551409730195823

y_pred = model.predict(data[2])
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(y_test)
print('====')
print(y_pred)