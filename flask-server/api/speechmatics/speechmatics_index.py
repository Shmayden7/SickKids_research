#########################
# Imports:
import time, os
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError 

from api.speechmatics.speechmatics_key import AUTH_TOKEN
from analysis.json_helpers import dumpDictToJSON
#########################

def pollSpeechmatics(file_name: str, audio_path='data/audio/') -> dict:
    settings = ConnectionSettings(
        url="https://asr.api.speechmatics.com/v2",
        auth_token=AUTH_TOKEN,
    )

    # Define transcription parameters
    conf = {
        "type": "transcription",
        "transcription_config": { 
            "language": 'en',
            #"operating_point": 'enhanced',
            "punctuation_overrides": {
                "permitted_marks":[],
                "sensitivity": 0
            },
            # Find out more about entity detection here:
            # https://docs.speechmatics.com/features/entities#enable-entity-metadata
            "enable_entities": False,
        },
    }

    # Open the client using a context manager
    with BatchClient(settings) as client:
        try:
            job_id = client.submit_job(
                audio=(audio_path+file_name),
                transcription_config=conf,
            )
            print(f"Uploading {file_name} to Speechmatics, waiting for responce...")

            # Note that in production, you should set up notifications instead of polling. 
            # Notifications are described here: https://docs.speechmatics.com/features-other/notifications
            transcript = client.wait_for_completion(job_id, transcription_format="json-v2") # works with 'json-v2' or 'txt'
            return transcript
        except HTTPStatusError as e:
            print("Invalid Speechmatics API key")
            print(e)

def runSpeechmatics(file_name: str, audio_path='data/audio/', dump_location='data/json/'):
    print('Running Speechmatics!')
    
    timer_start = time.time()
    data = pollSpeechmatics(file_name, audio_path)
    run_time = float("%.2f" % (time.time() - timer_start))
    data.update({
        'run_time': run_time,
        'duration': data['job']['duration'],
    })
    del data['metadata']
    del data['job']
    del data['format']

    print('Speechmatics Classification Complete! Runtime: {}s'.format(run_time))
    dumpDictToJSON(data, '{}.json'.format(file_name[:-4]), dump_location=dump_location)
    print()