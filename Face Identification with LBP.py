import multiprocessing
import os
from tqdm import tqdm
from lbp import lbp_features,lbp_skimage
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn import metrics
import cv2 
import sys
from sklearn.svm import SVC
from pathlib import Path
from sklearn.metrics import accuracy_score
import pandas as pd

# In[2]:


def check_face_detection(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))      
    if len(faces) == 1:
        pass
        #print('1 face detected', img)   
    elif len(faces) == 0:
        print('no face detected', img)
    else:
        print('{0} faces detected'.format(len(faces)), img)


# In[3]:


def get_images(img_and_impl):
    img, lbp_impl = img_and_impl
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))      
    if len(faces) == 1:
        x, y, w, h = faces[0]
        face = image[y:y + h, x:x + w]
    else:
        face = image
    if lbp_impl == 'own':
        return (img, lbp_features(face))
    else:
        return (img, lbp_skimage(face, 24, 8))


# In[11]:


def get_dataset(path, feature, impl='own', dump=True):
    dataset = {'image_paths': [], 'targets': [], 'feature': []}
    target_dirs = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    for index, target in tqdm(enumerate(target_dirs)):
        target_path = os.path.join(path, target)
        images = [os.path.join(target_path, img) for img in  os.listdir(target_path)]
        with multiprocessing.Pool(processes=12) as pool:
            result = pool.map(feature, [(img, impl) for img in images])
            for item in result:
                dataset['image_paths'].append(item[0])
                dataset['targets'].append(index)        
                dataset['feature'].append(item[1])
    if dump == True: 
        with open(PICKLES_PATH / str(path + '_' + impl), 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)              
    return dataset


# In[5]:
def load_dataset(path):
     with open(path, 'rb') as handle:
        return pickle.load(handle)      
# In[6]:
def train_test_split(dataset, split=0.25):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
            dataset['feature'], dataset['targets'], test_size=0.25, random_state=1, stratify=dataset['targets'])
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    return X_train, y_train, X_test, y_test

# In[7]:


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
    
    
# In[12]:
PICKLES_PATH = Path('pickles')
data_path = 'real-face-subset'
print('Getting dataset with skimage LBP..')
real_face_skimg = get_dataset(data_path, get_images, 'skimage')
# In[55]:

real_face_skimage_result = pipeline(real_face_skimg, 'real_face_skimage')

# In[42]:
print('Getting dataset with own LBP..')
real_face_own = get_dataset(data_path, get_images)

# In[56]:
real_face_own_result = pipeline(real_face_own, 'real_face_own')
# In[58]:

data_path = 'faces_in_artwork'
print('Getting dataset..')
faces_in_artwork_skimg = get_dataset(data_path, get_images, 'skimage')
#%%
faces_in_artwork_skimage_result = pipeline(faces_in_artwork_skimg, 'faces_in_artwork_skimage')

# In[9]:

print('Getting dataset..')
faces_in_artwork_own = get_dataset(data_path, get_images, 'own')
#%%
faces_in_artwork_own_result = pipeline(faces_in_artwork_own, 'faces_in_artwork_own')

