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

# ideas: first look for large faces, if not found, look for small faces
# cut the background from the face
# apply some pre-processing to the images in both probe and gallery sets to reduce similarity
# maybe estimate the edges of faces to end up with a small face? what happens to the already-small images in this case?
# guess we don't need this using gray-scale images





# K-most similar gallery faces are selected 
K = 1


def distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a cosine distance
    for each comparison face. The distance tells you how similar the faces are.

    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    distances = np.array(np.zeros(len(face_encodings)))
    if len(face_encodings) == 0:
        return np.empty((0))
    for i in range(0, len(face_encodings)):
        distances[i] = np.dot(face_encodings[i], face_to_compare) / (norm(face_encodings[i]) * norm(face_to_compare))

    return distances

# connect to MongoDB to store results in one database
# which named your team_name and authorized with your team_pass
client = MongoClient(host='localhost', port=27017)
db = client.visionism
db.authenticate('visionism', '09016995946')
collection = db.visionism
collection.remove({})

# read the list of probe images
probe_directory = "http://localhost:10080/images/probe/"
probe_images = requests.get(probe_directory + "images.txt").text.split()
print("The number of probe images is equal to " + str(len(probe_images)))

# read the list of gallery images
gallery_directory = "http://localhost:10080/images/gallery/"
gallery_images = requests.get(gallery_directory + "images.txt").text.split()
print("The number of gallery images is equal to " + str(len(gallery_images)))

known_faces = []
couldnt_find_face_in_probe = []

for gallery_image in gallery_images:
    g_image = requests.get(gallery_directory + gallery_image)
    g_image_bytes = BytesIO(g_image.content)
    g_image = face_recognition.load_image_file(g_image_bytes)

    known_image = face_recognition.face_encodings(g_image)[0]
    known_faces.append(known_image)


for probe_image in probe_images:
    print('examining probe image ' + str(probe_images.index(probe_image)) + ' out of total ' + str(len(probe_images)) + ' images...')
    p_image = requests.get(probe_directory + probe_image, )
    p_image_bytes = BytesIO(p_image.content)
    p_image = face_recognition.load_image_file(p_image_bytes)
    try:
        unknown_face = face_recognition.face_encodings(p_image)[0]
    except:
        print("could not find any face in " + str(probe_image))
        couldnt_find_face_in_probe.append(probe_directory + probe_image)
        continue
    
    unknown_img = Image.open(p_image_bytes)
    # uncomment to see probe image in each iteration 
    # unknown_img.show()

    scores = distance(known_faces, unknown_face)

    idx = np.argpartition(scores, -K)
    k_scores = idx[-K:]

    print('Probe ' + str(probe_images.index(probe_image)) + ' score list')
    for i in range(0, len(k_scores)):
        print('Gallery ' + str(k_scores[int(i)]) + ', Score ' + str(scores[k_scores[int(i)]]))
  
    # uncomment to see the k-most similar gallery images to the probe image
    for index, value in enumerate(idx[-K:]):
        known_img = requests.get(gallery_directory + gallery_images[value])
        known_img = Image.open(BytesIO(known_img.content))
        # known_img.show()
    # x =  input('Press enter to go to the next probe image')



print('list of  images in which no faces where found:\n' + str(couldnt_find_face_in_probe))
