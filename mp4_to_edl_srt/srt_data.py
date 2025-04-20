import os
from typing import List, Dict, Optional
from mp4_to_edl_srt.segment import Segment
from mp4_to_edl_srt.timecode_utils import TimecodeConverter


class SRTData:
    """
    Represents SRT data, storing segments for generating SRT output.
    """
    def __init__(self, converter: TimecodeConverter) -> None:
        """Initializes an empty list to store SRT segments and the converter."""
        self.segments: List[Segment] = []
        self.converter = converter

    def add_segment(self, segment: Segment, source_timecode_offset: Optional[str] = None) -> None:
        """Adds a segment to the SRT data and stores its source offset."""
        segment.source_timecode_offset = source_timecode_offset # Store offset in the segment
        self.segments.append(segment)

    def write_to_file(self, output_path: str) -> None:
        """Writes the SRT data to a file, sorted by absolute start time."""
        # セグメントを開始時間 (ミリ秒) + オフセット (ミリ秒) でソート
        def get_absolute_start_ms(seg: Segment) -> int:
            relative_start_ms = self.converter.hhmmssff_to_ms(seg.start_timecode)
            offset_ms = self.converter.hhmmssff_to_ms(seg.source_timecode_offset or '00:00:00:00')
            return relative_start_ms + offset_ms
            
        sorted_segments = sorted(self.segments, key=get_absolute_start_ms)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            current_srt_time_ms = 0 # Initialize sequence time for SRT
            for i, segment in enumerate(sorted_segments, 1):
                # Calculate duration of the segment
                start_ms = self.converter.hhmmssff_to_ms(segment.start_timecode)
                end_ms = self.converter.hhmmssff_to_ms(segment.end_timecode)
                segment_duration_ms = max(0, end_ms - start_ms) # Ensure non-negative duration
                
                # Calculate SRT timestamps based on sequence time
                srt_start_ms = current_srt_time_ms
                srt_end_ms = srt_start_ms + segment_duration_ms
                
                # Convert milliseconds to SRT format HH:MM:SS,mmm
                start_srt = self.converter.ms_to_srt_time(srt_start_ms)
                end_srt = self.converter.ms_to_srt_time(srt_end_ms)
                
                # Get text (potentially with scene description)
                # Use the existing to_srt_dict logic carefully, or get text directly
                # srt_dict = segment.to_srt_dict() # This recalculates relative time, avoid it here
                text = segment.transcription
                if segment.scene_description:
                    text = f"[{segment.scene_description}] {text}"

                # SRT形式で書き込み
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
                
                # Update the current time for the next segment
                current_srt_time_ms = srt_end_ms
        
        print(f"SRTファイルを保存しました: {output_path}")
        print(f"合計 {len(sorted_segments)} セグメントを書き込みました")
