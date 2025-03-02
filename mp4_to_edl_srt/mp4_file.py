import os
import subprocess
import whisper
import pydub
import re
from typing import List, Dict, Tuple
import warnings
from pydub import AudioSegment
from pydub.silence import split_on_silence

from segment import Segment
from edl_data import EDLData
from srt_data import SRTData

# Whisperの警告を非表示にする
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

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

    def transcribe(self, initial_prompt: str = None) -> None:
        """Transcribes the audio file using Whisper with word-level timestamps."""
        try:
            if not os.path.exists(self.audio_filepath):
                raise FileNotFoundError(f"音声ファイルが見つかりません: {self.audio_filepath}")
                
            print(f"Whisperモデルをロード中...")
            # CPUモードで実行するため、device="cpu"を明示的に指定
            model = whisper.load_model("base", device="cpu")
            
            # 初期プロンプトが指定されていない場合はデフォルト値を使用
            if initial_prompt is None:
                initial_prompt = "日本語での自然な会話。文脈に応じて適切な表現を使用してください。"
                
            print(f"文字起こし中: {self.audio_filepath}")
            print(f"使用する初期プロンプト: {initial_prompt}")
            
            # 環境変数からWhisperパラメータを取得
            temperature = float(os.environ.get("WHISPER_TEMPERATURE", "0.2"))
            beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
            condition_on_previous = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS", "True").lower() == "true"
            
            print(f"Whisperパラメータ:")
            print(f" - Temperature: {temperature}")
            print(f" - Beam Size: {beam_size}")
            print(f" - 文脈考慮: {'有効' if condition_on_previous else '無効'}")
            
            # 音声の前処理（環境変数から設定を取得）
            enable_preprocessing = os.environ.get("ENABLE_AUDIO_PREPROCESSING", "True").lower() == "true"
            
            if enable_preprocessing:
                print(f"音声の前処理を実行中...")
                processed_audio_path = self._preprocess_audio(self.audio_filepath)
                print(f"前処理済み音声ファイル: {processed_audio_path}")
            else:
                print(f"音声の前処理はスキップされました")
                processed_audio_path = self.audio_filepath
            
            # 詳細なパラメータを設定して日本語文字起こしの精度を向上
            self.transcription_result = model.transcribe(
                processed_audio_path,
                language="ja",                           # 日本語を強制的に指定
                task="transcribe",                       # 文字起こしタスク
                verbose=True,                            # 詳細な出力を表示
                initial_prompt=initial_prompt,           # 初期プロンプトを設定
                condition_on_previous_text=condition_on_previous,  # 前のテキストに条件付け（文脈を考慮）
                temperature=temperature,                 # 柔軟性の設定
                best_of=1,                               # 最良の結果を1つ選択
                beam_size=beam_size,                     # ビームサイズの設定
                word_timestamps=True,                    # 単語レベルのタイムスタンプを取得
                fp16=False                               # CPU使用時はFP16を無効化（FP32を使用）
            )
            
            print(f"文字起こし完了: {len(self.transcription_result.get('segments', []))}セグメント")
        except Exception as e:
            print(f"文字起こし中にエラーが発生しました: {str(e)}")
            raise Exception(f"Whisper transcription error: {e}")

    def _preprocess_audio(self, audio_path: str) -> str:
        """音声ファイルの前処理（ノイズ除去と音量調整）を行います。"""
        try:
            print(f"音声ファイルを前処理中: {audio_path}")
            
            # 出力ファイルパスを生成
            base_dir = os.path.dirname(audio_path)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            processed_path = os.path.join(base_dir, f"{base_name}_processed.wav")
            
            # pydubを使用して音声を読み込み
            audio = AudioSegment.from_file(audio_path)
            
            # 音量の正規化（必要に応じて調整）
            print(f"音量を正規化中...")
            normalized_audio = audio.normalize(headroom=1.0)
            
            # ノイズ除去（FFmpegを使用）
            print(f"ノイズ除去フィルターを適用中...")
            normalized_audio.export("temp_normalized.wav", format="wav")
            
            # FFmpegのノイズ除去フィルターを適用
            command = [
                "ffmpeg",
                "-i", "temp_normalized.wav",
                "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25",  # ハイパス、ローパス、ノイズ除去フィルター
                "-y",  # 既存ファイルを上書き
                processed_path
            ]
            
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            # 一時ファイルを削除
            if os.path.exists("temp_normalized.wav"):
                os.remove("temp_normalized.wav")
                
            print(f"音声前処理完了: {processed_path}")
            return processed_path
            
        except Exception as e:
            print(f"音声前処理中にエラーが発生しました: {str(e)}")
            print(f"前処理をスキップして元の音声ファイルを使用します。")
            return audio_path

    def segment_audio(self, threshold: float = 0.5) -> None:
        """Segments the audio based on silence threshold using Whisper timestamps."""
        segments = self.transcription_result.get("segments", [])
        if not segments:
            print("警告: 文字起こし結果にセグメントが含まれていません")
            return

        # Whisperのセグメントをそのまま使用する
        print(f"音声をセグメント化中...")
        
        # タイムコードでソート
        sorted_segments = sorted(segments, key=lambda x: x.get("start", 0))
        
        for i, segment in enumerate(sorted_segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            # 空のセグメントをスキップ
            if not text:
                continue
                
            # 極端に短いセグメントをスキップ（0.2秒未満）
            if end_time - start_time < 0.2:
                continue
                
            # タイムコードの逆転がないか確認
            if i > 0 and start_time < sorted_segments[i-1].get("end", 0):
                # 前のセグメントの終了時間より新しい開始時間を設定
                start_time = sorted_segments[i-1].get("end", 0) + 0.1
                
            start_timecode = self._seconds_to_timecode(start_time)
            end_timecode = self._seconds_to_timecode(end_time)
            
            # セグメントを追加
            self.segments.append(Segment(start_timecode, end_timecode, text))
            print(f"セグメント追加: {start_timecode} - {end_timecode} | {text[:30]}...")

    def _add_segment(self, words: List[Dict]) -> None:
        """Creates a Segment from a list of words."""
        start_time = words[0]["start"]
        end_time = words[-1]["end"]
        # 単語間のスペースを調整（日本語の場合は不要）
        language = self.transcription_result.get("language", "")
        if language in ["ja", "zh", "ko"]:
            # 日本語、中国語、韓国語の場合はスペースを削除
            transcription = "".join(word["word"].strip() for word in words)
        else:
            # その他の言語の場合は単語間にスペースを入れる
            transcription = " ".join(word["word"].strip() for word in words)
            
        start_timecode = self._seconds_to_timecode(start_time)
        end_timecode = self._seconds_to_timecode(end_time)
        self.segments.append(Segment(start_timecode, end_timecode, transcription))
        print(f"セグメント追加: {start_timecode} - {end_timecode}")

    def generate_edl_data(self, record_start: str) -> Tuple[EDLData, str]:
        """Generates EDL data with sequential record timecodes."""
        reel_name = f"TAPE{self.file_index:02d}"
        current_record = self._timecode_to_seconds(record_start)

        print(f"EDLデータを生成中 (リール名: {reel_name}, 開始レコード: {record_start})...")
        
        # EDLイベントのリストをクリア
        self.edl_data = EDLData(title="My Video Project", fcm="NON-DROP FRAME")
        
        # EDLイベントとそのレコードタイムコードを保存するリスト
        self.edl_events_with_timecode = []
        
        for segment in self.segments:
            duration = self._timecode_to_seconds(segment.end_timecode) - self._timecode_to_seconds(segment.start_timecode)
            record_in = self._seconds_to_timecode(current_record)
            record_out = self._seconds_to_timecode(current_record + duration)
            
            # ビデオとオーディオを含むイベント（DaVinci Resolveで正しく認識される形式）
            main_event = {
                "reel_name": reel_name,
                "track_type": "AA/V",  # オーディオとビデオの両方
                "transition": "C",
                "source_in": segment.start_timecode,
                "source_out": segment.end_timecode,
                "record_in": record_in,
                "record_out": record_out,
                "clip_name": os.path.basename(self.filepath),
            }
            self.edl_data.add_event(main_event)
            
            # EDLイベントとそのレコードタイムコードを保存（SRT生成用）
            self.edl_events_with_timecode.append({
                "segment": segment,
                "record_in": record_in,
                "record_out": record_out
            })
            
            current_record += duration

        new_record_start = self._seconds_to_timecode(current_record)
        print(f"EDLデータ生成完了: {len(self.edl_data.events)}イベント, 次の開始レコード: {new_record_start}")
        return self.edl_data, new_record_start

    def generate_srt_data(self) -> SRTData:
        """Generates SRT data based on EDL record timecodes."""
        print(f"SRTデータを生成中...")
        
        # SRTデータを初期化
        self.srt_data = SRTData()
        
        # EDLのレコードタイムコードに基づいてSRTセグメントを生成
        if hasattr(self, 'edl_events_with_timecode') and self.edl_events_with_timecode:
            print(f"EDLのレコードタイムコードに基づいてSRTデータを生成します")
            
            for event in self.edl_events_with_timecode:
                segment = event["segment"]
                record_in = event["record_in"]
                record_out = event["record_out"]
                
                # EDLのレコードタイムコードを使用して新しいセグメントを作成
                srt_segment = Segment(
                    record_in,
                    record_out,
                    segment.transcription
                )
                
                self.srt_data.add_segment(srt_segment)
        else:
            # EDLデータがない場合は元のセグメントを使用
            print(f"警告: EDLデータが見つかりません。元のセグメントを使用します。")
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

