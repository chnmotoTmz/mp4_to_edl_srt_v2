import os
from typing import List, Dict
from segment import Segment


class SRTData:
    """
    Represents SRT data, storing segments for generating SRT output.
    """
    def __init__(self) -> None:
        """Initializes an empty list to store SRT segments."""
        self.segments: List[Segment] = []

    def add_segment(self, segment: Segment) -> None:
        """Adds a segment to the SRT data."""
        self.segments.append(segment)

    def write_to_file(self, output_path: str) -> None:
        """Writes the SRT data to a file."""
        # セグメントを開始時間でソート
        sorted_segments = sorted(self.segments, key=lambda x: self._timecode_to_seconds(x.start_timecode))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(sorted_segments, 1):
                # SRTのタイムコード形式に変換
                start_srt = self._timecode_to_srt(segment.start_timecode)
                end_srt = self._timecode_to_srt(segment.end_timecode)
                
                # テキストを取得（すでにSegmentクラスで整形済み）
                text = segment.transcription
                
                # SRT形式で書き込み
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
        
        print(f"SRTファイルを保存しました: {output_path}")
        print(f"合計 {len(sorted_segments)} セグメントを書き込みました")

    def _timecode_to_srt(self, timecode: str) -> str:
        """Converts HH:MM:SS:FF to SRT format (HH:MM:SS,mmm)."""
        hh, mm, ss, ff = timecode.split(':')
        # 30fpsを想定して、フレーム数をミリ秒に変換（1フレーム = 33.33ミリ秒）
        ms = int(int(ff) * 1000 / 30)
        return f"{hh}:{mm}:{ss},{ms:03d}"
        
    def _timecode_to_seconds(self, timecode: str) -> float:
        """Converts HH:MM:SS:FF to seconds (30 fps)."""
        hh, mm, ss, ff = map(int, timecode.split(":"))
        return hh * 3600 + mm * 60 + ss + ff / 30.0
