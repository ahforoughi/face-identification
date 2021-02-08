from PIL import Image
import requests
import numpy as np
from pymongo import MongoClient
from io import BytesIO
import time
from numpy.linalg import norm
import numpy as np
import cv2


def detect_face(image_bytes, mtcnn_detector, retina_detector):

    image = Image.open(image_bytes)
    #image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)

    # check which model is used
    check_mtcnn = 1

    # detect faces in the image using mtcnn 
    results = mtcnn_detector.detect_faces(pixels)
    #if result in MTCNN do not work well or with lower accuarcy we will use Retina for face detection 
    if (not results) or results[0]['confidence'] < 0.9:
        check_mtcnn = 0
        print("====== mtcnn not working")
        results = retina_detector.predict(pixels)
        # result_img = detector.draw(pixels,faces)
        # cv2.imshow("result", result_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    # This is for visualizing the result of detection uncomment if necessory 

    # bounding_box = results[0]['box']
    # keypoints = results[0]['keypoints']

    # cv2.rectangle(pixels,
    #             (bounding_box[0], bounding_box[1]),
    #             (bounding_box[0]+bounding_box[2],  
    #             bounding_box[1] + bounding_box[3]),
    #             (0,155,255),
    #             2)

    # cv2.circle(pixels, (keypoints['left_eye']), 2, (255, 0,0), 2)
    # cv2.circle(pixels, (keypoints['right_eye']), 2, (255, 0,0), 2)
    # cv2.circle(pixels, (keypoints['nose']), 2, (255,0,0), 2)
    # cv2.circle(pixels, (keypoints['mouth_left']), 2, (255, 0,0), 2)
    # cv2.circle(pixels, (keypoints['mouth_right']), 2, (255, 0,0), 2)


    # #cv2.imwrite("ivan_drawn.jpg", pixels)
    # #cv2.namedWindow("pixels")
    # cv2.imshow("image2", pixels)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return results, check_mtcnn


