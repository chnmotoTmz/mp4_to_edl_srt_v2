import subprocess
import whisper
import pydub
import os
import re
from typing import List, Dict

from segment import Segment
from edl_data import EDLData
from srt_data import SRTData


class MP4File:
    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.audio_filepath: str = ""
        self.transcription: str = ""
        self.segments: List[Segment] = []
        self.edl_data: EDLData = EDLData(title=os.path.basename(filepath), fcm="FCM_DEFAULT")  # Default FCM
        self.srt_data: SRTData = SRTData()

    def extract_audio(self) -> None:
        """Extracts audio from the MP4 file."""
        try:
            base_name = os.path.splitext(os.path.basename(self.filepath))[0]
            self.audio_filepath = f"{base_name}.wav"
            command = [
                "ffmpeg",
                "-i",
                self.filepath,
                "-vn",  # Disable video
                "-acodec", "pcm_s16le",  # Use PCM for audio
                "-ar", "44100",  # Set audio sample rate
                self.audio_filepath,
            ]
            subprocess.run(command, check=True)
        except FileNotFoundError:
            print(f"Error: FFmpeg not found.  Check your installation.")
            raise
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {self.filepath}: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    def transcribe(self) -> None:
        """Transcribes the audio file using Whisper."""
        try:
            model = whisper.load_model("base")
            result = model.transcribe(self.audio_filepath)
            self.transcription = result["text"]
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise

    def segment_audio(self, threshold: float = 0.5) -> None:
        """Segments the audio based on a silence threshold."""
        try:
            audio = pydub.AudioSegment.from_wav(self.audio_filepath)
            # Replace with your actual timecode extraction logic
            timecodes = self._extract_timecodes(audio)
            
            for i in range(len(timecodes) - 1):
                start_timecode = timecodes[i]
                end_timecode = timecodes[i+1]
                transcription_segment = self._extract_transcription_segment(i, timecodes)
                self.segments.append(Segment(start_timecode, end_timecode, transcription_segment))
        except Exception as e:
            print(f"Error segmenting audio: {e}")
            raise

    def generate_edl_data(self) -> Dict:
        """Generates EDL data."""
        edl_events = []
        for segment in self.segments:
            edl_events.append(segment.to_dict())
        self.edl_data.events = edl_events
        return self.edl_data.to_dict()

    def generate_srt_data(self) -> Dict:
        """Generates SRT data."""
        srt_segments = []
        for segment in self.segments:
            srt_segments.append(segment.to_srt_dict())
        self.srt_data.segments = srt_segments
        return self.srt_data.to_dict()

    def _extract_timecodes(self, audio):
        # Placeholder - Replace with actual timecode extraction logic
        # Assumes timecodes are in the format HH:MM:SS:FF
        return [f"00:00:00:00", f"00:00:01:00"]  # Placeholder, replace with actual values

    def _extract_transcription_segment(self, index, timecodes):
        # Placeholder - Replace with actual transcription extraction logic
        # Assumes transcription is already extracted and stored
        return "Placeholder Transcription"

