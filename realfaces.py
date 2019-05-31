# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:24:57 2019

@author: gokce
"""
#%%
import os 
from collections import Counter
path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/real-face'

#%%
real_subjects = [os.path.join(path, item) for item in os.listdir(path)]
real_faces = [os.listdir(subject) for subject in real_subjects]

#%%
face_counter = Counter([len(subject) for subject in real_faces])

#%%
artwork_path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/faces_in_artwork'

#%%
subjects = [os.path.join(artwork_path, item) for item in os.listdir(artwork_path) if os.path.isdir(os.path.join(artwork_path, item))]
faces = [os.listdir(subject)for subject in subjects]

#%%
counter = Counter([len(subject) for subject in faces])
print(counter)
total = sum([len(subject) for subject in faces])
avg = total / len(faces)
print(total, avg)

real_face_subset = []
set_of_people = set()
selected = {}
for subject in real_faces:
    for key, value in counter.items():
        #print(key)
        if len(subject) == key and key in face_counter.keys():
            if key in selected:
                if selected[key] < value and subject[0] not in set_of_people:
                    real_face_subset.append(subject)
                    set_of_people.add(subject[0])
                    selected[key] += 1
            else:
                if subject[0] not in set_of_people:
                    real_face_subset.append(subject)
                    set_of_people.add(subject[0])
                    selected[key] = 1
        else:
            if len(subject) + 1 == key and key in face_counter.keys():
                if key in selected:
                    if selected[key] < value and subject[0] not in set_of_people:
                        real_face_subset.append(subject)
                        selected[key] += 1
                        set_of_people.add(subject[0])

                else:
                    if subject[0] not in set_of_people:
                        real_face_subset.append(subject)
                        selected[key] = 1    
                        set_of_people.add(subject[0])

            elif len(subject) + 2 == key and key in face_counter.keys():
                if key in selected:
                    if selected[key] < value and subject[0] not in set_of_people:
                        real_face_subset.append(subject)
                        selected[key] += 1
                        set_of_people.add(subject[0])

                else:
                    if subject[0] not in set_of_people:
                        real_face_subset.append(subject)
                        selected[key] = 1      
                        set_of_people.add(subject[0])

#%%%            
print(len(real_face_subset))
print([item[0] for item in real_face_subset])
print(sum([len(item) for item in real_face_subset]))