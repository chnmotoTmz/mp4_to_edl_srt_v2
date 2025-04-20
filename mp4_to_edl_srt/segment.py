import re
from typing import Dict, Optional, List, TYPE_CHECKING

# Avoid circular import for type hinting
if TYPE_CHECKING:
    from mp4_to_edl_srt.timecode_utils import TimecodeConverter

class Segment:
    def __init__(self, start_timecode: str, end_timecode: str, transcription: str, scene_id: Optional[int] = None, scene_description: Optional[str] = None, transcription_good_reason: Optional[str] = None, transcription_bad_reason: Optional[str] = None, source_timecode_offset: Optional[str] = None, source_filename: Optional[str] = None, file_index: Optional[int] = None):
        """
        Initializes a segment with start and end timecodes and transcription.

        Args:
            start_timecode: The start timecode in HH:MM:SS:FF format.
            end_timecode: The end timecode in HH:MM:SS:FF format.
            transcription: The transcription text for this segment.
            scene_id: Optional identifier for the scene this segment belongs to.
            scene_description: Optional textual description of the scene.
            transcription_good_reason: Optional tag indicating a positive aspect of the transcription.
            transcription_bad_reason: Optional tag indicating a negative aspect of the transcription.
            source_timecode_offset: Optional timecode offset (HH:MM:SS:FF) of the source MP4 file.
            source_filename: Optional filename of the source MP4 file.
            file_index: Optional index of the source MP4 file (for reel name).
        """
        self.start_timecode = start_timecode
        self.end_timecode = end_timecode
        # 日本語テキストの場合、単語間の不要なスペースを削除
        self.transcription = self._clean_japanese_text(transcription)
        self.scene_id = scene_id
        self.scene_description = scene_description
        self.transcription_good_reason = transcription_good_reason
        self.transcription_bad_reason = transcription_bad_reason
        self.source_timecode_offset = source_timecode_offset # Store the offset
        self.source_filename = source_filename # Store source filename
        self.file_index = file_index           # Store file index

    def _clean_japanese_text(self, text: str) -> str:
        """
        日本語テキストから不要なスペースを削除します。
        英数字の間のスペースは保持します。
        
        Args:
            text: 処理するテキスト
            
        Returns:
            整形されたテキスト
        """
        # テキストが空の場合はそのまま返す
        if not text:
            return text
            
        # 英数字パターン
        alpha_num_pattern = re.compile(r'[a-zA-Z0-9]')
        
        # 文字列を文字のリストに変換
        chars = list(text)
        result = []
        
        # 前の文字が英数字かどうかのフラグ
        prev_is_alpha_num = False
        
        for i, char in enumerate(chars):
            if char == ' ':
                # 前後の文字が英数字の場合のみスペースを保持
                prev_char = chars[i-1] if i > 0 else ''
                next_char = chars[i+1] if i < len(chars) - 1 else ''
                
                if (alpha_num_pattern.match(prev_char) and 
                    alpha_num_pattern.match(next_char)):
                    result.append(char)
                    prev_is_alpha_num = False
            else:
                result.append(char)
                prev_is_alpha_num = alpha_num_pattern.match(char) is not None
                
        return ''.join(result)

    def to_edl_dict(self) -> dict:
        """
        Converts the segment to a dictionary for EDL generation.

        Returns:
            A dictionary containing the segment data for EDL.
        """
        return {
            "start_timecode": self.start_timecode,
            "end_timecode": self.end_timecode,
            "transcription": self.transcription
        }

    def __str__(self) -> str:
        """Returns a string representation of the segment."""
        reason_str = ""
        if self.transcription_good_reason:
            reason_str += f" [Good: {self.transcription_good_reason}]"
        if self.transcription_bad_reason:
            reason_str += f" [Bad: {self.transcription_bad_reason}]"
        scene_info = f" (Scene: {self.scene_id})" if self.scene_id else ""
        offset_info = f" (Offset: {self.source_timecode_offset})" if self.source_timecode_offset else ""
        source_info = f" (Source: {self.source_filename}[{self.file_index}])" if self.source_filename and self.file_index is not None else ""
        return f"{self.start_timecode} - {self.end_timecode}: {self.transcription}{scene_info}{reason_str}{offset_info}{source_info}"

    def to_dict(self) -> Dict:
        """Converts the segment to a dictionary suitable for JSON output."""
        data = {
            "start_timecode": self.start_timecode,
            "end_timecode": self.end_timecode,
            "transcription": self.transcription,
            "scene_id": self.scene_id,
            "scene_description": self.scene_description,
            "transcription_good_reason": self.transcription_good_reason,
            "transcription_bad_reason": self.transcription_bad_reason,
            "source_timecode_offset": self.source_timecode_offset,
            "source_filename": self.source_filename,
            "file_index": self.file_index
        }
        return data

    def to_srt_dict(self) -> Dict:
        """Converts the segment to a dictionary for SRT with millisecond precision."""
        # Convert HH:MM:SS:FF to HH:MM:SS,MMM
        start_srt = self._convert_to_srt_time(self.start_timecode)
        end_srt = self._convert_to_srt_time(self.end_timecode)

        # Combine transcription with scene description if available
        text = self.transcription
        if self.scene_description:
            text = f"[{self.scene_description}] {text}"

        return {
            "start_time": start_srt,
            "end_time": end_srt,
            "text": text,
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
