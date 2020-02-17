#-*- coding: utf-8 -*
"""
Created on Tue Aug 27 08:11:48 2019

@author: User
"""
import os
from gtts import gTTS
from playsound import playsound
#import sys
#import winsound
#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

class TTS():
    def __init__(self):
        self.path_dir='C:\\Users\\ssuda\\Desktop\\공프기\\Keras'

    def text_to_speech(self, text , i):
#        file_list=os.listdir(self.path_dir)
#        file_list.sort()

        new_file_alphabet = chr(ord('a')+i)
        tts=gTTS(text=text, lang='ko')
        tts.save('hhh'+ new_file_alphabet +'.mp4')
        playsound('hhh' + new_file_alphabet +'.mp4')