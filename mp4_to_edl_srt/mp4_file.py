import os
import subprocess
import whisper
import pydub
import re
from typing import List, Dict, Tuple

from segment import Segment
from edl_data import EDLData
from srt_data import SRTData


class MP4File:
    def __init__(self, filepath: str, file_index: int):
        self.filepath: str = os.path.normpath(filepath)
        self.file_index: int = file_index  # For reel name (e.g., TAPE01)
        self.audio_filepath: str = ""
        self.transcription_result: Dict = {}
        self.segments: List[Segment] = []
        self.edl_data: EDLData = EDLData(title="My Video Project", fcm="NON-DROP FRAME")
        self.srt_data: SRTData = SRTData()

    def extract_audio(self) -> None:
        """Extracts audio from the MP4 file using FFmpeg."""
        base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        output_dir = os.path.dirname(self.filepath)
        self.audio_filepath = os.path.join(output_dir, f"{base_name}.wav")
        
        command = [
            "ffmpeg",
            "-i", self.filepath,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-y",  # Overwrite output file if it exists
            self.audio_filepath,
        ]
        try:
            print(f"FFmpegコマンド: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"音声ファイルを抽出しました: {self.audio_filepath}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpegエラー: {e.stderr}")
            raise Exception(f"FFmpeg error: {e.stderr}")

    def transcribe(self) -> None:
        """Transcribes the audio file using Whisper with word-level timestamps."""
        try:
            if not os.path.exists(self.audio_filepath):
                raise FileNotFoundError(f"音声ファイルが見つかりません: {self.audio_filepath}")
                
            print(f"Whisperモデルをロード中...")
            model = whisper.load_model("base")
            
            print(f"文字起こし中: {self.audio_filepath}")
            self.transcription_result = model.transcribe(self.audio_filepath, word_timestamps=True)
            
            print(f"文字起こし完了: {len(self.transcription_result.get('segments', []))}セグメント")
        except Exception as e:
            print(f"文字起こし中にエラーが発生しました: {str(e)}")
            raise Exception(f"Whisper transcription error: {e}")

    def segment_audio(self, threshold: float = 0.5) -> None:
        """Segments the audio based on silence threshold using Whisper timestamps."""
        segments = self.transcription_result.get("segments", [])
        if not segments:
            print("警告: 文字起こし結果にセグメントが含まれていません")
            return

        current_segment = []
        last_end = 0.0

        print(f"音声をセグメント化中 (閾値: {threshold}秒)...")
        
        for segment in segments:
            if "words" not in segment:
                print(f"警告: セグメントに単語情報が含まれていません")
                continue
                
            for word in segment["words"]:
                start_time = word["start"]
                end_time = word["end"]
                if start_time - last_end > threshold and current_segment:
                    # End previous segment and start a new one
                    self._add_segment(current_segment)
                    current_segment = []
                current_segment.append(word)
                last_end = end_time

        if current_segment:
            self._add_segment(current_segment)
            
        print(f"セグメント化完了: {len(self.segments)}セグメント")

    def _add_segment(self, words: List[Dict]) -> None:
        """Creates a Segment from a list of words."""
        start_time = words[0]["start"]
        end_time = words[-1]["end"]
        transcription = " ".join(word["word"] for word in words)
        start_timecode = self._seconds_to_timecode(start_time)
        end_timecode = self._seconds_to_timecode(end_time)
        self.segments.append(Segment(start_timecode, end_timecode, transcription))
        print(f"セグメント追加: {start_timecode} - {end_timecode}")

    def generate_edl_data(self, record_start: str) -> Tuple[EDLData, str]:
        """Generates EDL data with sequential record timecodes."""
        reel_name = f"TAPE{self.file_index:02d}"
        current_record = self._timecode_to_seconds(record_start)

        print(f"EDLデータを生成中 (リール名: {reel_name}, 開始レコード: {record_start})...")
        
        for segment in self.segments:
            duration = self._timecode_to_seconds(segment.end_timecode) - self._timecode_to_seconds(segment.start_timecode)
            record_in = self._seconds_to_timecode(current_record)
            record_out = self._seconds_to_timecode(current_record + duration)
            event = {
                "reel_name": reel_name,
                "track_type": "AA/V",
                "transition": "C",
                "source_in": segment.start_timecode,
                "source_out": segment.end_timecode,
                "record_in": record_in,
                "record_out": record_out,
                "clip_name": os.path.basename(self.filepath),
            }
            self.edl_data.add_event(event)
            current_record += duration

        new_record_start = self._seconds_to_timecode(current_record)
        print(f"EDLデータ生成完了: {len(self.edl_data.events)}イベント, 次の開始レコード: {new_record_start}")
        return self.edl_data, new_record_start

    def generate_srt_data(self) -> SRTData:
        """Generates SRT data."""
        print(f"SRTデータを生成中...")
        for segment in self.segments:
            self.srt_data.add_segment(segment)
        print(f"SRTデータ生成完了: {len(self.srt_data.segments)}セグメント")
        return self.srt_data

    def _seconds_to_timecode(self, seconds: float) -> str:
        """Converts seconds to HH:MM:SS:FF format (30 fps)."""
        total_frames = int(seconds * 30)
        hours = total_frames // (3600 * 30)
        minutes = (total_frames % (3600 * 30)) // (60 * 30)
        secs = (total_frames % (60 * 30)) // 30
        frames = total_frames % 30
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

    def _timecode_to_seconds(self, timecode: str) -> float:
        """Converts HH:MM:SS:FF to seconds (30 fps)."""
        hh, mm, ss, ff = map(int, timecode.split(":"))
        return hh * 3600 + mm * 60 + ss + ff / 30.0

