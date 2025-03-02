import re
from typing import Dict


class Segment:
    def __init__(self, start_timecode: str, end_timecode: str, transcription: str):
        """
        Initializes a segment with start and end timecodes and transcription.

        Args:
            start_timecode: The start timecode in HH:MM:SS:FF format.
            end_timecode: The end timecode in HH:MM:SS:FF format.
            transcription: The transcription text for this segment.
        """
        self.start_timecode = start_timecode
        self.end_timecode = end_timecode
        # 日本語テキストの場合、単語間の不要なスペースを削除
        self.transcription = self._clean_japanese_text(transcription)

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
        return f"{self.start_timecode} - {self.end_timecode}: {self.transcription}"

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
