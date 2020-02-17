import speech_recognition as sr

r = sr.Recognizer()
m = sr.Microphone()

class STT():
    # __init__(self):

    def recog(self):
        print("A moment of silence, please...")
        with m as source: r.adjust_for_ambient_noise(source)
        print("Say something!")
        with m as source: audio = r.listen(source)
        print("Got it! Now to recognize it...")

        try:
            # recognize speech using Google Speech Recognition
            value = r.recognize_google(audio)
                # we need some special handling here to correctly print unicode characters to standard output
            if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                print(u"You said {}".format(value).encode("utf-8"))
                return value
            else:  # this version 1of Python uses unicode for strings (Python 3+)
                print("You said {}".format(value))
                return value

        except sr.UnknownValueError:
            print("Oops! Didn't catch that")
        except sr.RequestError as e:
            print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
