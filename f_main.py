import f_face_recognition as rec_face
import traceback
import f_storage as st
import numpy as np
import cv2
import face_recognition

#------------------------ Start the main flow ----------------------------
class rec():
    def __init__(self):
        '''
        -db_names: [name1,name2,...,namen] lista de strings
        -db_features: array(array,array,...,array) cada array representa las caracteriticas de un usuario
        '''
        print("Creating DataBase ...")
        self.db_names, self.db_features = st.load_images_to_database()
        print("DataBase created ...")

    def recognize_face(self,im):
        '''
        Input:
            -imb64: imagen 
        Output:
            res:{'status':  if everything goes well it is 'ok' otherwise it returns the error found
                'faces': [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,each tuple represents a detected face
                'names': ['name', 'unknow'] list of names that matched}       
        '''
        
        try:
            # detect face 
            box_faces = rec_face.detect_face(im)
            # print("box faces",box_faces)

            # conditional in case no face is detected
            if  not box_faces:
                res = {
                    'status':'ok',
                    'faces':[],
                    'names':[]}
                return res
            else:
                #if no database
                if not self.db_names:
                    res = {
                        'status':'ok',
                        'faces':box_faces,
                        'names':['unknown']*len(box_faces)}
                    return res
                else:
                    # (continues) extract features

                    actual_features = rec_face.get_features(im,box_faces)
                    # compare actual_features with those stored in the database                  

                    match_names = rec_face.compare_faces(actual_features,self.db_features,self.db_names)
                    # save
                    res = {
                        'status':'ok',
                        'faces':box_faces,
                        'names':match_names}
                    return res
        except Exception as ex:
            error = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            res = {
                'status':'error: ' + str(error),
                'faces':[],
                'names':[]}
            return res

    def recognize_face_normal(self,im):
        try:
            # detect face 
            box_faces = face_recognition.face_locations(im)
            # print("box faces",box_faces)
            # conditional in case no face is detected
            if  not box_faces:
                res = {
                    'status':'ok',
                    'faces':[],
                    'names':[]}
                return res
            else:
                #if no database
                if not self.db_names:
                    res = {
                        'status':'ok',
                        'faces':box_faces,
                        'names':['unknown']*len(box_faces)}
                    return res
                else:
                    # (continues) extract features

                    actual_features = rec_face.get_features(im,box_faces)
                    # compare actual_features with those stored in the database                  

                    match_names = rec_face.compare_faces(actual_features,self.db_features,self.db_names)
                    # save
                    res = {
                        'status':'ok',
                        'faces':box_faces,
                        'names':match_names}
                    return res
        except Exception as ex:
            error = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            res = {
                'status':'error: ' + str(error),
                'faces':[],
                'names':[]}
            return res

def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        y0,x1,y1,x0 = box[i]
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img

if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument("-im","--path_im",help="path image")
    parse = parse.parse_args()
    
    path_im = parse.path_im
    im = cv2.imread(path_im)
    # instancio detector
    recognizer = rec()
    res = recognizer.recognize_face(im)
    im = bounding_box(im,res["faces"],res["names"])
    cv2.imshow("face recogntion", im)
    cv2.waitKey(0)
    # print(res)