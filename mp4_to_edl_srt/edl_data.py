import re
from typing import List, Dict


class EDLData:
    def __init__(self, title: str = "Untitled", fcm: str = "FCM_DEFAULT"):
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
