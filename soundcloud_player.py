import os
import requests
from io import BytesIO
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from sclib import SoundcloudAPI, Track

CLIENT_ID = os.getenv("SOUNDCLOUD_CLIENT_ID")
CLIENT_SECRET = os.getenv("SOUNDCLOUD_CLIENT_SECRET")

class SoundCloudPlayer:
    def __init__(self, client_id=None):
        self.client_id = client_id or CLIENT_ID
        if not self.client_id:
            raise ValueError("SoundCloud client_id not provided")
        self.api = SoundcloudAPI(client_id=self.client_id)

    def search_track(self, query):
        url = SoundcloudAPI.SEARCH_URL.format(query=requests.utils.quote(query), client_id=self.client_id, limit=1, offset=0)
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Search request failed: {resp.status_code}")
        data = resp.json()
        collection = data.get("collection")
        if not collection:
            raise ValueError("No tracks found for query")
        return Track(obj=collection[0], client=self.api)

    def fetch_audiosegment(self, track):
        stream_url = track.get_stream_url()
        resp = requests.get(stream_url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch track data: {resp.status_code}")
        return AudioSegment.from_file(BytesIO(resp.content), format="mp3")

    def play(self, query, output_device=None):
        track = self.search_track(query)
        segment = self.fetch_audiosegment(track)
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(segment.array_type).max
        sd.play(samples, samplerate=segment.frame_rate, device=output_device)
        sd.wait()

