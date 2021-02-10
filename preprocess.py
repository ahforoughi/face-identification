from PIL import Image
from matplotlib import image
import numpy as np
from numpy.linalg import norm
import numpy as np
import cv2
import math
from autocrop import Cropper

cropper = Cropper()


def detect_face(pixels, mtcnn_detector, retina_detector):

    # check which model is used
    check_mtcnn = 1

    # detect faces in the image using mtcnn 
    results = mtcnn_detector.detect_faces(pixels)
    # print(results)

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


def EuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def alignment_procedure(img, left_eye, right_eye):

    #this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    #-----------------------
    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    #-----------------------
    #find length of triangle edges

    a = EuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = EuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = EuclideanDistance(np.array(right_eye), np.array(left_eye))

    #-----------------------

    #apply cosine rule

    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree

        #-----------------------
        #rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    #-----------------------

    return img #return img anyway

def face_alignment(img, results):
    detection = results[0]
    keypoints = detection["keypoints"]
    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]

    img = alignment_procedure(img, left_eye, right_eye)
    cropped_array = cropper.crop(img)
    if cropped_array is not None:
        return cropped_array
    #cropped_array = cv2.cvtColor(cropped_array, cv2.COLOR_BGR2RGB)
    return img




