from PIL import Image
import requests
import numpy as np
from random import random
from pymongo import MongoClient
from io import BytesIO
import time
from numpy.linalg import norm
import numpy as np
from mtcnn import MTCNN
import cv2
from retinaface import RetinaFace



print("connecting to mongo...")
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


print("loading image " + gallery_images[1])


def detect_face(url):

    image = requests.get(url)

    image_bytes = BytesIO(image.content)
    image = Image.open(image_bytes)
    #image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    cv2.imshow("image",pixels)

    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image using mtcnn 
    results = detector.detect_faces(pixels)
    print("++++++++mtcn", results)
    if (not results) or results[0]['confidence'] < 0.99:
        print("====== mtcnn not working")
        detector = RetinaFace(quality="normal")
        faces = detector.predict(pixels)
        result_img = detector.draw(pixels,faces)
        cv2.imshow("result", result_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return faces


    bounding_box = results[0]['box']
    keypoints = results[0]['keypoints']

    cv2.rectangle(pixels,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2)

    cv2.circle(pixels,(keypoints['left_eye']), 2, (255,0,0), 2)
    cv2.circle(pixels,(keypoints['right_eye']), 2, (255,0,0), 2)
    cv2.circle(pixels,(keypoints['nose']), 2, (255,0,0), 2)
    cv2.circle(pixels,(keypoints['mouth_left']), 2, (255,0,0), 2)
    cv2.circle(pixels,(keypoints['mouth_right']), 2, (255,0,0), 2)


    #cv2.imwrite("ivan_drawn.jpg", pixels)
    #cv2.namedWindow("pixels")
    cv2.imshow("image2", pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results