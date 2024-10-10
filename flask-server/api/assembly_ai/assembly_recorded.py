#########################
# Imports:
import time, requests, json
from .assembly_key import ASSEMBLY_AI_API_KEY
from data.constants import word_list
# Constants:
upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcription_endpoint = 'https://api.assemblyai.com/v2/transcript'
headers = {'authorization': ASSEMBLY_AI_API_KEY}
#########################

#########################
# Desc: uploads a select file to Assembly AI endpoint
# Params: file path for your audio file
# Return: the assembly AI url of where your file lives
# Dependant on: N.A.
#########################
def upload(file_path: str):
    def ReadFile(file_path, chunk_size=5242880): # chunking size of ~5 Megabytes, defined by Assembly AI
        with open(file_path, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    print('Uploading {}!'.format(file_path))
    # post request for Assembly AI's endpoint
    # this will return a link of the address of where your audio file lives
    response = requests.post(upload_endpoint, headers=headers, data=ReadFile(file_path))

    # printing the full json object
    audio_url = response.json()['upload_url']
    if(audio_url):
        print('Audio uploaded to Assembly AI')
    else:
        print('There was an issue uploading your file to Assembly AI :(')
    return audio_url

#########################
# Desc: fetched the specific job Id from the Assembly Ai endpoint
# Params: Assembly AI url for the file, additional paramaters for transcription
# Return: transcript_id
# Dependant on: N.A.
#########################
def transcribe(audio_url: str, cfg: dict) -> str:

    audio_intel = { 
        "audio_url": audio_url,
        "language_code": 'en_us',
        "punctuate": cfg['punctuate'],
        "format_text": cfg['format_text'],
        "speaker_labels": cfg['speaker_labels'],
        "auto_highlights": cfg['auto_highlights'],
    }
    if 'audio_start_from' in cfg and 'audio_end_at' in cfg:
        audio_intel.update({
            "audio_start_from": cfg['audio_start_from'],
            "audio_end_at": cfg['audio_end_at'],
        })

    if cfg['use_boost']:
        audio_intel.update({
            "word_boost": word_list[cfg['list_number']],
            "boost_param": cfg['boost_param'],
        })

    if cfg['speakers_expected'] and cfg['speaker_labels']:
        audio_intel.update({
            "speakers_expected": cfg['speakers_expected'],
        })

    try: 
        transcript_response = requests.post(transcription_endpoint, json=audio_intel, headers=headers)
    except IOError as e:
        print('Error, Did not recieve a transcript_response from Assembly AI :(')
        print(e)
    if 'error' in transcript_response.json():
        print('Error with AI Post Transcript: {}'.format(print(transcript_response.json()['error'])))
    else:
        print('Transcript Recieved from Assembly AI')
    transcript_id = transcript_response.json()['id']
    return transcript_id

#########################
# Desc: make get request to the api, wont always return a val, the api may be 'thinking'
# Params: transcript_id 
# Return: a large object full of data generated by assembly AI
# Dependant on: N.A.
#########################
def performPoll(transcript_id: str):
    polling_endpoint = transcription_endpoint + '/' + transcript_id
    polling_response = requests.get(polling_endpoint, headers=headers)
    return polling_response.json()

#########################
# Desc: performs all the logic to post a file and get data, speech-to-TEXT, Bread and butter Fn()
# Params: file path of your audio file
# Return: Data provided by the API
# Dependant on: transcribe(), performPoll()
#########################
def getAssemblyAIData(file_name: str, cfg: dict, audio_path='data/audio/') -> dict:
    full_path = audio_path + file_name
    audio_url = upload(full_path)
    transcript_id = transcribe(audio_url, cfg)
    while True:
        data = performPoll(transcript_id)
        if data['status'] == 'completed':
            print('Fetched Data From Assembly AI!')
            parsed_data = {
                'audio_duration': data['audio_duration'],
                'text': data['text'],
                'words': data['words'],
                'utterances': data['utterances'],
                'auto_highlights_result': data['auto_highlights_result'],
            }
            return parsed_data
        elif data['status'] == 'error':
            print('Error Fetching From Assembly AI: '+ data['error'])
            return data

        print('Waiting 5s, Assembly AI is Thinking...')
        time.sleep(5) 
