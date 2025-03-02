import re
from typing import List, Dict


class EDLData:
    def __init__(self, title: str = "Untitled", fcm: str = "NON-DROP FRAME"):
        self.title: str = title
        self.fcm: str = fcm
        self.events: List[Dict] = []

    def add_event(self, event: Dict) -> None:
        """Adds an event to the EDL data."""
        self.events.append(event)

    def to_dict(self) -> Dict:
        """Converts the EDL data to a dictionary."""
        return {
            "title": self.title,
            "fcm": self.fcm,
            "events": self.events,
        }

    def __str__(self) -> str:
        """Formats the EDL data in CMX 3600 format."""
        lines = [f"TITLE: {self.title}", f"FCM: {self.fcm}", ""]
        for i, event in enumerate(self.events, start=1):
            event_line = (
                f"{i:03d}  {event['reel_name']:8s} {event['track_type']:4s} {event['transition']:1s}        "
                f"{event['source_in']} {event['source_out']} {event['record_in']} {event['record_out']}"
            )
            lines.append(event_line)
            lines.append(f"* FROM CLIP NAME: {event['clip_name']}")
            lines.append("")
        return "\n".join(lines)
