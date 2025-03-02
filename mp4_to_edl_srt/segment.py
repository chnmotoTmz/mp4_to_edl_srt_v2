import re
from typing import Dict


class Segment:
    def __init__(self, start_timecode: str, end_timecode: str, transcription: str):
        self.start_timecode: str = start_timecode  # HH:MM:SS:FF for EDL
        self.end_timecode: str = end_timecode      # HH:MM:SS:FF for EDL
        self.transcription: str = transcription

    def to_dict(self) -> Dict:
        """Converts the segment to a dictionary for EDL."""
        return {
            "source_in": self.start_timecode,
            "source_out": self.end_timecode,
            "transcription": self.transcription,
        }

    def to_srt_dict(self) -> Dict:
        """Converts the segment to a dictionary for SRT with millisecond precision."""
        # Convert HH:MM:SS:FF to HH:MM:SS,MMM
        start_srt = self._convert_to_srt_time(self.start_timecode)
        end_srt = self._convert_to_srt_time(self.end_timecode)
        return {
            "start_time": start_srt,
            "end_time": end_srt,
            "text": self.transcription,
        }

    def _convert_to_srt_time(self, timecode: str) -> str:
        """Converts HH:MM:SS:FF to HH:MM:SS,MMM format (30 fps)."""
        hh, mm, ss, ff = map(int, timecode.split(":"))
        total_seconds = hh * 3600 + mm * 60 + ss + ff / 30.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
