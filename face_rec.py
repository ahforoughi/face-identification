import pprint
import requests
import numpy as np
from random import random
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
from mtcnn import MTCNN
from retinaface import RetinaFace
from arcface import ArcFace
from preprocess import detect_face
from preprocess import face_alignment
from arc_face import arc_similarity
import matplotlib.pyplot as plt


# connect to MongoDB to store results in one database
# which named your team_name and authorized with your team_pass
client = MongoClient(host='mongodb', port=27017)
db = client.visionism
db.authenticate('visionism', '09016995946')
collection = db.visionism
collection.remove({})

# read the list of probe images
probe_directory = "http://nginx/images/probe/"
probe_images = requests.get(probe_directory + "images.txt").text.split()
print("The number of probe images is equal to " + str(len(probe_images)))

# read the list of gallery images
gallery_directory = "http://nginx/images/gallery/"
gallery_images = requests.get(gallery_directory + "images.txt").text.split()
print("The number of gallery images is equal to " + str(len(gallery_images)))


# create the detector, using default weights
mtcnn_detector = MTCNN()
retina_detector = RetinaFace(quality="normal")
arc_detector = ArcFace.ArcFace()

# K-most similar gallery faces are selected 
K = 1


# valid gallery stores bytes of images 
valid_gallery = {}
valid_probe = {}
couldnt_find_face_in_probe = []
similarities = []

for gallery_image in gallery_images:
    print("encoding gallery images" + gallery_image)

    g_image = requests.get(gallery_directory + gallery_image)
    g_image_bytes = BytesIO(g_image.content)
    g_image = Image.open(g_image_bytes)

    g_pixels = np.asarray(g_image)
    results , check_mtcnn = detect_face(g_pixels, mtcnn_detector, retina_detector)

    if len(results):
        #TODO: face landmark for retinaface for face alignment
        if check_mtcnn:
            g_pixels = face_alignment(g_pixels, results)
        gallery = np.float32(g_pixels)
        emb_gallery = arc_detector.calc_emb(gallery)
        valid_gallery[gallery_image] = emb_gallery 
        # plt.imshow(aligned_face)
        # plt.show()
        # break
        
        
        
print("size of valid gallery " + str(len(valid_gallery)))  


for probe_image in probe_images:
    print('examining probe image ' + str(probe_images.index(probe_image)) + ' out of total ' + str(len(probe_images)) + ' images...')
    p_image = requests.get(probe_directory + probe_image)
    p_image_bytes = BytesIO(p_image.content)
    p_image = Image.open(p_image_bytes)
    #image = image.convert('RGB')
    # convert to array
    p_pixels = np.asarray(p_image)
    results , check_mtcnn = detect_face(p_pixels, mtcnn_detector, retina_detector)

    if len(results):
        #TODO: see if we can pass more than one probe to arc_similarity and make the answer as it must be 
        valid_probe[probe_image] = p_pixels
        if check_mtcnn:
            p_pixels = face_alignment(p_pixels, results)
 
        for g_key in valid_gallery:

            score = arc_similarity(arc_detector, p_pixels, valid_gallery[g_key])
            #for scaling score between 1 and 100 
            scaled_score = (1/score)*100
            print(f"comparing {probe_image} and {g_key} " + str(scaled_score))
            similarities.append({g_key.split(".")[0] : str(score)})
         #   break
    
        post = {probe_image.split(".")[0] : similarities}
        inserted_id = collection.insert_one(post).inserted_id
        #break


for post in collection.find():
    pprint.pprint(post)

            
            





# print('list of  images in which no faces where found:\n' + str(couldnt_find_face_in_probe))
