#########################
# Imports:
from deepgram import Deepgram
import asyncio
from .deepgram_key import DEEPGRAM_API_KEY
# Constants:
#########################

#########################
# Get Deepgram Data
# Desc: sends audio file to endpoint and gets response data
# Params: filePath (e.g ../../Audio/life-moves-pretty-fast.wav) fileType: 'audio/wav', 'audio/mp3'...
# Return: Data fetched from deepgram endpoint
# Dependant on: N.A.
# MUST BE RUN AS AWAIT
#########################
async def getDeepgramDataNew(file_name, audio_path='data/audio/'):
  
  file_path = audio_path + file_name

  # Initialize the Deepgram SDK
  deepgram = Deepgram(DEEPGRAM_API_KEY)

  # Check whether requested file is local or remote, and prepare source
  if file_path.startswith('http'):
    # filePath is remote
    # Set the source
    source = {
      'url': file_path
    }
  else:
    # filePath is local
    # Open the audio filePath
    audio = open(file_path, 'rb')

    mimetype = 'audio/{}'.format(file_name[-4:])
    # Set the source
    source = {
      'buffer': audio,
      'mimetype': mimetype
    }

  # Send the audio to Deepgram and get the response
  response = await asyncio.create_task(
    deepgram.transcription.prerecorded(
      source,
      {
        'punctuate': False # adds: , ? ! . ect.
      }
    )
  )
  print('Got Data!')
  return response