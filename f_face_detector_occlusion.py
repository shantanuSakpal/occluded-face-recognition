from keras.models import load_model
import numpy as np 
import cv2
"""
tensorflow_version --> '2.0.0'
keras_version -------> '2.3.1'
"""

'''
to use:
1. instantiate model
    face_detector = f_face_detector.detector_face_occlusion()
2. enter some image with a face and predict
    boxes_face = face_detector.detect_face(img)

Note: returns the bounding_box where it found faces
'''

class detector_face_occlusion():
    def __init__(self):
        # network architecture
        prototxt_path = "face_detector/deploy.prototxt"
        # weights of the model
        caffemodel_path = "face_detector/weights.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    def detect_face(self,image):
        (h, w) = image.shape[:2]
        # I prepare the image to enter the model
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # I input the image into the model
        self.detector.setInput(blob)
        # I propagate the image forward through the model
        detections = self.detector.forward()
        """"
        detections, has 4 columns that are:
        0 column -->
        1st column -->
        2nd column --> number of detections made by default 200
        3th column --> has 7 subcolumns which are
            4.0 -->
            4.1 -->
            4.2 -->confidence
            4.3 --> x0
            4.4 --> y0
            4.5 --> x1
            4.6 --> y1
        """
        # check the confidence of the 200's predictions
        list_box = []
        for i in range(0, detections.shape[2]):
            # box --> array[x0,y0,x1,y1]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # confidence range --> [0-1]
            confidence = detections[0,0,i,2]
            if confidence >=0.8:
                if len(list_box) == 0:
                    list_box = np.expand_dims(box, axis=0)
                else:
                    list_box = np.vstack((list_box, box))
        return list_box

