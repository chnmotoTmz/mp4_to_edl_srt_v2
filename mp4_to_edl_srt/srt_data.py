import re
from typing import List, Dict

from segment import Segment


class SRTData:
    """
    Represents SRT data, storing segments for generating SRT output.
    """
    def __init__(self) -> None:
        """Initializes an empty list to store SRT segments."""
        self.segments: List[Dict[str, str]] = []  # Type hint for clarity

    def add_segment(self, segment: Segment) -> None:
        """
        Adds a segment to the SRT data.

        Args:
            segment: The Segment object containing the segment data.
        """
        self.segments.append(segment.to_srt_dict())

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Converts the SRT data to a dictionary.

        Returns:
            A dictionary containing the list of SRT segments.
        """
        return {"segments": self.segments}
        
    def __str__(self) -> str:
        """Formats the SRT data in SRT subtitle format."""
        lines = []
        for i, segment in enumerate(self.segments, start=1):
            lines.append(f"{i}")
            lines.append(f"{segment['start_time']} --> {segment['end_time']}")
            lines.append(segment['text'])
            lines.append("")
        return "\n".join(lines)
