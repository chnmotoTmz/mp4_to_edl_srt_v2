import os
from typing import List, Optional
from mp4_to_edl_srt.segment import Segment # Import Segment
from mp4_to_edl_srt.timecode_utils import TimecodeConverter # Import TimecodeConverter


class EDLData:
    """
    Represents EDL data, generating the EDL formatted string from sorted segments.
    """
    def __init__(self, title: str = "Default Title", fcm: str = "NON-DROP FRAME", converter: Optional[TimecodeConverter] = None):
        """Initializes EDL data with title, FCM, and optionally a TimecodeConverter."""
        self.title = title
        self.fcm = fcm
        # Store segments directly instead of pre-formatted events
        self.segments: List[Segment] = [] 
        # Converter is now required for calculations
        if converter is None:
            raise ValueError("TimecodeConverter instance is required for EDLData")
        self.converter = converter

    def add_segment(self, segment: Segment) -> None:
        """Adds a segment to be processed for EDL generation."""
        self.segments.append(segment)

    def _get_absolute_start_ms(self, seg: Segment) -> int:
        """Helper function to calculate the absolute start time in milliseconds."""
        relative_start_ms = self.converter.hhmmssff_to_ms(seg.start_timecode)
        offset_ms = self.converter.hhmmssff_to_ms(seg.source_timecode_offset or '00:00:00:00')
        return relative_start_ms + offset_ms

    def generate_edl_string(self) -> str:
        """Generates the complete EDL formatted string."""
        if not self.segments:
            return f"TITLE: {self.title}\nFCM: {self.fcm}\n\n* No segments found.\n"

        # Sort segments first by file_index, then by relative start time (in ms)
        # Convert start_timecode (HH:MM:SS:FF) to ms for sorting
        sorted_segments = sorted(
            self.segments, 
            key=lambda seg: (
                seg.file_index if seg.file_index is not None else 0, # Sort by file index (use 0 if None)
                self.converter.hhmmssff_to_ms(seg.start_timecode) # Then by relative start time in ms
            )
        )

        edl_lines = []
        edl_lines.append(f"TITLE: {self.title}")
        edl_lines.append(f"FCM: {self.fcm}")
        edl_lines.append("") # Blank line after header

        record_time_ms = 0 # Start record time at 0 for the sequence

        for i, segment in enumerate(sorted_segments):
            event_num = f"{i + 1:03d}"
            # Use file_index for TAPE name, default to 00 if missing
            reel_name = f"TAPE{segment.file_index:02d}" if segment.file_index is not None else "TAPE00"
            track_type = "AA/V" # Use combined Audio/Video track type
            transition = "C"    # Standard cut transition

            # Calculate absolute source times
            source_in_ms = self._get_absolute_start_ms(segment)
            relative_start_ms = self.converter.hhmmssff_to_ms(segment.start_timecode)
            relative_end_ms = self.converter.hhmmssff_to_ms(segment.end_timecode)
            duration_ms = max(0, relative_end_ms - relative_start_ms)
            source_out_ms = source_in_ms + duration_ms

            source_in_tc = self.converter.ms_to_hhmmssff(source_in_ms)
            source_out_tc = self.converter.ms_to_hhmmssff(source_out_ms)

            # Calculate record times based on sequence
            record_in_ms = record_time_ms
            record_out_ms = record_in_ms + duration_ms

            record_in_tc = self.converter.ms_to_hhmmssff(record_in_ms)
            record_out_tc = self.converter.ms_to_hhmmssff(record_out_ms)

            # Use source filename for clip name, fallback if missing
            clip_name = segment.source_filename or f"Segment_{event_num}"
            
            # Choose comment source (e.g., scene description or transcription)
            # Using scene description if available, else transcription
            comment = segment.scene_description if segment.scene_description else segment.transcription
            # Limit comment length if necessary
            comment = comment[:100] if comment else ""

            # Format the main event line
            event_line = f"{event_num}  {reel_name}   {track_type} {transition}        {source_in_tc} {source_out_tc} {record_in_tc} {record_out_tc}"
            edl_lines.append(event_line)

            # Format the FROM CLIP NAME line
            clip_line = f"* FROM CLIP NAME: {clip_name}"
            edl_lines.append(clip_line)
            
            # Optional: Add comment line if needed (scene desc or transcription)
            # if comment: # COMMENTED OUT: Do not add COMMENT lines to EDL
            #     comment_line = f"* COMMENT: {comment}" 
            #     edl_lines.append(comment_line)
                
            # Add empty line between events for readability (optional)
            edl_lines.append("") 

            # Update record time for the next segment
            record_time_ms = record_out_ms

        return "\n".join(edl_lines)

    # Override __str__ to use the new generation method
    def __str__(self) -> str:
        return self.generate_edl_string()

    # Keep add_event for potential backward compatibility or other uses, but mark as legacy?
    # Or remove it if it's definitely not used by main.py anymore.
    # Let's remove it for now to enforce using segments.
    # def add_event(self, event: dict) -> None:
    #     """(Legacy) Adds a pre-formatted event dictionary."""
    #     self.events.append(event)
