#-*- coding: utf-8 -*
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import cv2
import os
import threading

from voiceoutput import TTS
from voiceinput import STT

from emotionword import emotion_classify
from lifeword import life_classify

class Display():
    def __init__(self):
        self.prev_text_curr = " "
        
        self.real_emotion_prev = " "
        self.real_emotion_curr = " "

        self.emotion_prev = " "
        self.emotion_curr = " "

        self.emotion_last_count = 0
        self.emotion_step = 1

        self.robot_talk_prev = " "
        self.robot_talk_curr = " "

        self.i=0
    def show(self):
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Sad", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        cv2.ocl.setUseOpenCL(False)
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            #얼굴선택
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                self.emotion_curr = emotion_dict[maxindex]
                
                if threading.activeCount() == 1 :
                    self.get_real_emotion()
                    t1 = threading.Thread(target = self.stt_tts)
                    t1.start()

                cv2.putText(frame, self.real_emotion_curr, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_real_emotion(self):
        def emotion_step_one():
            text_dict = {0:"너 슬퍼보이는구나. 왜 슬프니?", 1:"너 화가 나 보이는구나. 왜 화가 났니?", 2:"너 기분이 좋아보이는구나. 왜 기쁘니?",}
            if self.real_emotion_prev =="Happy" or self.real_emotion_prev =="Neutral" :
                     if self.real_emotion_curr=="Angry" :
                         self.robot_talk_curr = text_dict[1]
                     elif self.real_emotion_curr=="Sad":
                         self.robot_talk_curr = text_dict[0]
            elif self.real_emotion_prev =="Angry" or self.real_emotion_prev =="Neutral":
                     if self.real_emotion_curr=="Happy" :
                         self.robot_talk_curr = text_dict[2]
                     elif self.real_emotion_curr=="Sad" :
                         self.robot_talk_curr = text_dict[0]
            elif self.real_emotion_prev =="Sad" or self.real_emotion_prev =="Neutral" :
                     if self.real_emotion_curr=="Happy" :
                         self.robot_talk_curr = text_dict[2]
                     elif self.real_emotion_curr=="Angry" :
                         self.robot_talk_curr = text_dict[1]

        def emotion_step_two():
            text_dict_second_reponse = {0:"나도 너가 기분이 좋아서 기분이 좋아", 1:"난 너의 문제가 해결되어 너의 기분이 나아지길 바래", 2:"너가 기분이 조금이라도 좋아진 것 같아 다행이야"}
            if self.real_emotion_curr =="Happy" :
                    if self.prev_text_curr == "Happy" :
                        self.robot_talk_curr = text_dict_second_reponse[0]
                    else :
                        self.robot_talk_curr = text_dict_second_reponse[2]
            else :
                    if self.prev_text_curr == "Happy" :
                        self.robot_talk_curr = ""
                    else :
                        self.robot_talk_curr = text_dict_second_reponse[1]

        if self.emotion_curr == self.emotion_prev:
            self.emotion_last_count = self.emotion_last_count+1
        else:
            self.emotion_last_count = 0

        if self.emotion_last_count > 7 :
            self.real_emotion_curr = self.emotion_curr
            if self.emotion_step==1:
                emotion_step_one()
            else:
                emotion_step_two()

    def stt_tts(self):
        if self.emotion_step==1 and self.robot_talk_curr != self.robot_talk_prev :
            self.prev_text_curr = self.real_emotion_curr
            self.emotion_step = 2
            TTS().text_to_speech(self.robot_talk_curr,self.i)
            user_say = STT().recog()
            
            self.robot_talk_curr = "한번더 다시 말해줄 수 있니?"
            while user_say == None:
                user_say = STT().recog()
                self.i=self.i+1
                TTS().text_to_speech(self.robot_talk_curr,self.i)
            
            emotion_word = emotion_classify(user_say)
            life_word = life_classify(user_say)
            if emotion_word and life_word:
                self.robot_talk_curr = life_word + "이 널" + emotion_word+"했구나"
                self.i=self.i+1
                TTS().text_to_speech(self.robot_talk_curr,self.i)

        elif self.emotion_step==2 and self.robot_talk_curr != self.robot_talk_prev :
            self.emotion_step=1
            self.i=self.i+1
            print("sujin  ", self.robot_talk_curr)
            TTS().text_to_speech(self.robot_talk_curr, self.i)

        
        self.emotion_prev = self.emotion_curr
        self.real_emotion_prev = self.real_emotion_curr
        self.robot_talk_prev = self.robot_talk_curr


model = Sequential()

def create_model():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('model_7.h5')

def main():
    create_model()

    display = Display()
    display.show()

if __name__ == "__main__":
    main()
