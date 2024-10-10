from pydub import AudioSegment 
import os

path = os.getcwd()
print(path)

audio = AudioSegment.from_mp3("data/audio/say_the_word.mp3")

audio = audio + 6 # this will increase the volume by 6 Db

audio = audio * 2 # this will repeat the clip

audio = audio.fade_in(2000) # this will fade in the audio over 2000 ms

audio.export('mashup.mp3', format=('mp3'))

audio2 = audio.from_mp3('mashup.mp3')
print('done')