# Audio file formats
# .mp3, lossy compression format with information loss
# .flac, compressed format, no data loss
# .wav uncompressed format, large filesize

import wave
import os

# Audio signal params
# number of channels, mono: one, stario: two
# sample width, number of bytes for each sample
# sampling rate, number of samples per second (Hz)
# number of frames, saved in binary
# values of a frame

obj = wave.open('flask-server/data/audio/test.wav', "rb")

print("Number of channels", obj.getnchannels())
print("Sample width", obj.getsampwidth())
print("Frame rate", obj.getframerate())
print("Number of frames", obj.getnframes())
print("paramaters", obj.getparams())

t_audio = obj.getnframes() / obj.getframerate()
print(t_audio)

frames = obj.readframes(-1)
print(type(frames), type(frames[0]))
print(len(frames)/2)

obj.close()

obj_new = wave.open('test_new.wav', 'wb')

obj_new.setnchannels(1)
obj_new.setsampwidth(2)
obj_new.setframerate(16000)

obj_new.writeframes(frames)

obj_new.close()

