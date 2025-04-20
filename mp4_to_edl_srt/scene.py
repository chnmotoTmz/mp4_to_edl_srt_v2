from typing import Optional, TYPE_CHECKING
from mp4_to_edl_srt.timecode_utils import TimecodeConverter
from dataclasses import dataclass

# Avoid circular import for type hinting
if TYPE_CHECKING:
    from mp4_to_edl_srt.timecode_utils import TimecodeConverter

class Scene:
    """Represents a detected scene within the video."""

    def __init__(self, start_ms: int, end_ms: int, scene_id: int, description: Optional[str] = None, thumbnail_path: Optional[str] = None, scene_good_reason: Optional[str] = None, scene_bad_reason: Optional[str] = None, scene_evaluation_tag: Optional[str] = None):
        """
        Initializes a Scene object.

        Args:
            start_ms: Start time of the scene in milliseconds.
            end_ms: End time of the scene in milliseconds.
            scene_id: Unique identifier for the scene.
            description: Optional textual description of the scene.
            thumbnail_path: Optional path to the representative thumbnail image.
            scene_good_reason: Optional tag indicating a positive aspect of the scene.
            scene_bad_reason: Optional tag indicating a negative aspect of the scene.
            scene_evaluation_tag: The raw evaluation tag returned by the API (e.g., Generic).
        """
        if start_ms >= end_ms:
            raise ValueError("Scene start time must be less than end time.")
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.scene_id = scene_id
        self.description = description if description else f"Scene {scene_id}"
        self.thumbnail_path = thumbnail_path
        self.scene_good_reason = scene_good_reason
        self.scene_bad_reason = scene_bad_reason
        self.scene_evaluation_tag = scene_evaluation_tag

    def get_start_time_edl(self, converter: 'TimecodeConverter') -> str:
        """Returns the start time formatted for EDL using the provided converter."""
        return converter.ms_to_hhmmssff(self.start_ms)

    def get_end_time_edl(self, converter: 'TimecodeConverter') -> str:
        """Returns the end time formatted for EDL using the provided converter."""
        return converter.ms_to_hhmmssff(self.end_ms)

    def get_start_time_srt(self, converter: 'TimecodeConverter') -> str:
        """Returns the start time formatted for SRT using the provided converter."""
        return converter.ms_to_hhmmssmmm(self.start_ms)

    def get_end_time_srt(self, converter: 'TimecodeConverter') -> str:
        """Returns the end time formatted for SRT using the provided converter."""
        return converter.ms_to_hhmmssmmm(self.end_ms)

    def to_dict(self, converter: 'TimecodeConverter') -> dict:
        """Converts the Scene object to a dictionary suitable for JSON serialization."""
        data = {
            "scene_id": self.scene_id,
            "start_timecode": self.get_start_time_edl(converter),
            "end_timecode": self.get_end_time_edl(converter),
            "description": self.description,
            "thumbnail_path": self.thumbnail_path,
            "scene_good_reason": self.scene_good_reason,
            "scene_bad_reason": self.scene_bad_reason,
            "scene_evaluation_tag": self.scene_evaluation_tag
        }
        return data

    def __str__(self) -> str:
        reason_str = ""
        if self.scene_good_reason:
            reason_str += f" [Good: {self.scene_good_reason}]"
        if self.scene_bad_reason:
            reason_str += f" [Bad: {self.scene_bad_reason}]"
        return f"Scene {self.scene_id}: {self.start_ms}ms - {self.end_ms}ms ({self.description}){reason_str}" 