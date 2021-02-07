import pprint
import datetime
import requests
import numpy as np
from random import random
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import face_recognition
import time
from numpy.linalg import norm
from mtcnn import MTCNN
from retinaface import RetinaFace
from arcface import ArcFace
from preprocess import detect_face
from arc_face import arc_similarity

# connect to MongoDB to store results in one database
# which named your team_name and authorized with your team_pass
client = MongoClient(host='localhost', port=27017)
db = client.visionism
db.authenticate('visionism', '09016995946')
collection = db.visionism
collection.remove({})

# read the list of probe images
probe_directory = "http://localhost/images/probe/"
probe_images = requests.get(probe_directory + "images.txt").text.split()
print("The number of probe images is equal to " + str(len(probe_images)))

# read the list of gallery images
gallery_directory = "http://localhost/images/gallery/"
gallery_images = requests.get(gallery_directory + "images.txt").text.split()
print("The number of gallery images is equal to " + str(len(gallery_images)))


# create the detector, using default weights
mtcnn_detector = MTCNN()
retina_detector = RetinaFace(quality="normal")
arc_detector = ArcFace.ArcFace()

# K-most similar gallery faces are selected 
K = 1



valid_gallery = []
couldnt_find_face_in_probe = []

for gallery_image in gallery_images:
    print("encoding gallery images" + gallery_image)

    g_image = requests.get(gallery_directory + gallery_image)
    g_image_bytes = BytesIO(g_image.content)
    results , _ = detect_face(g_image_bytes, mtcnn_detector, retina_detector)

    if len(results):
        valid_gallery.append(g_image_bytes)
        print(valid_gallery)
        break
        


for probe_image in probe_images:
    print('examining probe image ' + str(probe_images.index(probe_image)) + ' out of total ' + str(len(probe_images)) + ' images...')
    p_image = requests.get(probe_directory + probe_image)
    p_image_bytes = BytesIO(p_image.content)
    results , _ = detect_face(p_image_bytes, mtcnn_detector, retina_detector)

    if len(results):
        scores = arc_similarity(arc_detector, p_image_bytes, valid_gallery[0] )
        print(scores)
        break

        #TODO: see if we can pass more than one probe to arc_similarity and make the answer as it must be 
        # idx = np.argpartition(scores, -K)
        # k_scores = idx[-K:]

        # print('Probe ' + str(probe_images.index(probe_image)) + ' score list')
        # for i in range(0, len(k_scores)):
        #     print('Gallery ' + str(k_scores[int(i)]) + ', Score ' + str(scores[k_scores[int(i)]]))
    
        # # uncomment to see the k-most similar gallery images to the probe image
        # for index, value in enumerate(idx[-K:]):
        #     known_img = requests.get(gallery_directory + gallery_images[value])
        #     known_img = Image.open(BytesIO(known_img.content))
            # known_img.show()
        # x =  input('Press enter to go to the next probe image')



# print('list of  images in which no faces where found:\n' + str(couldnt_find_face_in_probe))
