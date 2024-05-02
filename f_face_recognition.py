import face_recognition 
import numpy as np
import config as cfg
import f_face_detector_occlusion


face_detector = f_face_detector_occlusion.detector_face_occlusion()
def detect_face(image):
    '''
    Input: image numpy.ndarray, shape=(W,H,3)
    Output: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,each tuple represents a detected face
    If nothing is detected  --> Output: []

    before -->    box_faces = face_recognition.face_locations(image)
    '''
    
    #This novel method returns a list of bounding boxes, where each bounding box represents a detected face.
    list_box = face_detector.detect_face(image)
    try:
        box_faces = [(box[1],box[2],box[3],box[0]) for box in list_box.astype("int")]
    except:
        box_faces = []
    return box_faces


def get_features(img,box):
    '''
    The get_features function  takes an image and a list of bounding boxes as input. Each bounding box in the list represents a detected face in the image. The function then uses the face_encodings function from the face_recognition library to extract the features of each detected face. These features are returned as a list of arrays, where each array represents the features of a face.
    
    Input:
        -img: numpy.ndarray image, shape=(W,H,3)
        -box: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)], each tuple represents a detected face
    Output:
        -features: [array,array,...,array], each array represents the features of a face
    
    '''
    features = face_recognition.face_encodings(img,box)
    return features

def compare_faces(face_encodings,db_features,db_names):
    '''
    The compare_faces function is used to compare a list of face features with a database of known face features.
    
    Input:
        db_features = [array,array,...,array] , Each array represents the characteristics of a face 
        db_names =  array(array,array,...,array) Each array represents the characteristics of a user
    Output:
        -match_name: ['name', 'unknown'] list of names that matched
        If there is no match but there is a person, it returns 'unknown'
    '''
    match_name = []
    names_temp = db_names
    Feats_temp = db_features           

    for face_encoding in face_encodings:
        try:
            dist = face_recognition.face_distance(Feats_temp,face_encoding)
        except:
            dist = face_recognition.face_distance([Feats_temp],face_encoding)
        index = np.argmin(dist)
        if dist[index] <= cfg.threshold:
            match_name = match_name + [names_temp[index]]
        else:
            match_name = match_name + ["unknow"]
    return match_name