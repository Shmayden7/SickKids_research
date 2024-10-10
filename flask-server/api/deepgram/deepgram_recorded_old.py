#########################
# Imports:
from deepgram import Deepgram
import asyncio, json, sys
from .deepgram_key import DEEPGRAM_API_KEY
# Constants:
FILE = 'YOUR_FILE_LOCATION'
#########################

# Location of the file you want to transcribe. Should include filename and extension.
# Example of a local file: ../../Audio/life-moves-pretty-fast.wav
# Example of a remote file: https://static.deepgram.com/examples/interview_speech-analytics.wav


# Mimetype for the file you want to transcribe
# Include this line only if transcribing a local file
# Example: 

#########################
# Get Deepgram Data
# Desc: sends audio file to endpoint and gets response data
# Params: filePath (e.g ../../Audio/life-moves-pretty-fast.wav) fileType: 'audio/wav', 'audio/mp3'...
# Return: Data fetched from deepgram endpoint
# Dependant on: N.A.
# MUST BE RUN AS AWAIT
#########################
async def getDeepgramDataOld(file_name: str, audio_path='data/audio/', file_type='wav'):
  
  file_path = audio_path + file_name
  file_type = 'audio/' + file_type

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

    # Set the source
    source = {
      'buffer': audio,
      'mimetype': file_type
    }

  # Send the audio to Deepgram and get the response
  response = await asyncio.create_task(
    deepgram.transcription.prerecorded(
      source,
      {
        'punctuate': True # adds: , ? ! . ect.
      }
    )
  )

  print(json.dumps(response, indent=3))

  # Write only the transcript to the console
  print(response["results"]["channels"][0]["alternatives"][0]["transcript"])

# try:
#   # If running in a Jupyter notebook, Jupyter is already running an event loop, so run main with this line instead:
#   #await main()
#   asyncio.run(getDeepgramData())
# except Exception as e:
#   exception_type, exception_object, exception_traceback = sys.exc_info()
#   line_number = exception_traceback.tb_lineno
#   print(f'line {line_number}: {exception_type} - {e}')