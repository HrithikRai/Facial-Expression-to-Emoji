import numpy as np
import cv2
from keras.preprocessing import image
import math
from threading import Thread
import time
from pathlib import Path

#face expression recognizer initialization
from keras.models import model_from_json

def area(x):
	return x[2] * x[3]



#run video feed on a different thread
class cameraFeed():
    #init function
    def __init__(self,source=0):
        self.stream = cv2.VideoCapture(source)
        _,self.frame = self.stream.read()
        self.stopped = False
    #start function
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    #update function
    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            _,self.frame = self.stream.read()
    #read function
    def read(self):
        return self.frame
    #stop function
    def stop(self):
        self.stopped = True

def main():
    face_cascade = cv2.CascadeClassifier("cascades/face_cascade.xml")
    model = model_from_json(open("facial_expression_model_structure.json", "r").read())
    model.load_weights('facial_expression_model_weights.h5') #load weights
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    speedup_bool = False
    print("RUNNING LAMEMOJI A.K.A SNAPCHET")    
    c = cameraFeed().start()
    while True:
        if speedup_bool:
            speedup_bool = False
            continue
        speedup_bool = True
        frame = c.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) < 1:
            continue

        face = max(faces,key = area)
        [x,y,w,h] = face
        d_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        d_face = cv2.cvtColor(d_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        d_face = cv2.resize(d_face, (48, 48)) #resize to 48x48
        img_pixels = image.img_to_array(d_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions

        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
        #grab the emoji
        path = "faces/" + emotions[max_index] + ".png"
        emoji = cv2.imread(path,-1)
        if emoji is None:
            continue
        #resize
        emoji = cv2.resize(emoji,(w,h),interpolation = cv2.INTER_AREA)
        # Create the mask for the emoji
        mask = emoji[:,:,3]
        mask_inv = cv2.bitwise_not(mask)
        emoji = emoji[:,:,0:3]
        roi = frame[y:y+h,x:x+w]
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        roi_fg = cv2.bitwise_and(emoji,emoji,mask = mask)

        # join the roi_bg and roi_fg
        dst = cv2.add(roi_bg,roi_fg)
        frame[y:y+h,x:x+w] = dst

        cv2.imshow("face",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            c.stop()
            break

if __name__ == '__main__':
    main()
