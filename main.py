import f_main
import cv2 
import time
import argparse
import imutils
import face_recognition 
import os

# instacio recognizer
recognizer = f_main.rec()

def normal_face_recognition(frame,index):
    res = recognizer.recognize_face_normal(frame)
    print(res)
        #res = {
        # 'status':'ok',
        # 'faces':box_faces,
        # 'names':match_names
        # }
        
    #bounding_box draws the bounding box and the name of the person recognized
    frame = f_main.bounding_box(frame,res["faces"],res["names"])
    #print the detected name
    if len(res["names"]) > 0:
        print("Person {} is {}".format(index, res["names"][0]))
    else:
        print("Person {} is unknown".format(index))
    # cv2.imshow("our face recognition",frame)
    cv2.waitKey(0)

    
def our_face_recognition(frame, index):
    #recognize_face calculates the facial features and compares them with the database
    res = recognizer.recognize_face(frame)
    # print("res")
        #res = {
        # 'status':'ok',
        # 'faces':box_faces,
        # 'names':match_names
        # }
        
    #bounding_box draws the bounding box and the name of the person recognized
    frame = f_main.bounding_box(frame,res["faces"],res["names"])
    print("Person {} is {}".format(index, res["names"][0]))

    # cv2.imshow("our face recognition",frame)
    cv2.waitKey(0)

def main(parse):
    if parse.input == "webcam":
        cam = cv2.VideoCapture(0)
        while True:
            # read the frame from the camera and send it to the server
            star_time = time.time()
            ret, frame = cam.read()
            #frame = imutils.resize(frame, width=720)

            res = recognizer.recognize_face(frame)
            print(res)
            frame = f_main.bounding_box(frame,res["faces"],res["names"])

            end_time = time.time() - star_time    
            FPS = 1/end_time
            cv2.putText(frame,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow("face_recognition",frame)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break

    elif parse.input == "image":
            # Single image
        if parse.path_im:
            frame = cv2.imread(parse.path_im)
            if frame is None:
                print("Error: Could not read the image file.")
            else:
                frame = imutils.resize(frame, width=720)
                # normal_face_recognition(frame)
                our_face_recognition(frame)

        # Folder of images
        elif parse.folder_path:
            index = 1
            for filename in os.listdir(parse.folder_path):
                image_path = os.path.join(parse.folder_path, filename)
                if os.path.isfile(image_path) and any(image_path.endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
                    frame = cv2.imread(image_path)
                    if frame is not None:
                        frame = imutils.resize(frame, width=720)
                        normal_face_recognition(frame,index)
                        # our_face_recognition(frame, index)
                        index += 1

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="face recognition implementation")
    parse.add_argument("--input", help="webcam or image")
    parse.add_argument("--path_im", help="path of image")
    parse.add_argument("--folder_path", help="path of folder containing images")
    parse = parse.parse_args()
    main(parse)

    
