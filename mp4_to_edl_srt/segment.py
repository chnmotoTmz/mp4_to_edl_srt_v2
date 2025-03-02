import re
from typing import List, Dict


class Segment:
    def __init__(self, start_timecode: str, end_timecode: str, transcription: str):
        self.start_timecode: str = start_timecode
        self.end_timecode: str = end_timecode
        self.transcription: str = transcription

    def to_dict(self) -> Dict:
        return {
            "start_timecode": self.start_timecode,
            "end_timecode": self.end_timecode,
            "transcription": self.transcription,
        }

    def to_srt_dict(self) -> Dict:
        """Converts the segment to a dictionary suitable for SRT format."""
        return {
            "start_time": self.start_timecode,
            "end_time": self.end_timecode,
            "text": self.transcription,
        }
