#########################
# Imports:
import speech_recognition as sr
import pyttsx3 

# Constants:
#########################
recognizer = sr.recognizer

while True:

    try:

        with sr.Microphone() as mic:

            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)

            text = recognizer.recognize_google(audio)
            text - text.lower()

            print(text)

    except sr.UnknownValueError():
        recognizer = sr.Recognizer()