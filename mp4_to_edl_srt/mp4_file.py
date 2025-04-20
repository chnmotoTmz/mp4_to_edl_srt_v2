import os
import subprocess
import whisper
import pydub
import re
import json
import torch  # torchを明示的にインポート
from typing import List, Dict, Tuple, Optional, Any, Callable
import warnings
from pydub import AudioSegment
from pydub.silence import split_on_silence
from datetime import datetime
import logging
import traceback
from dataclasses import dataclass
from mp4_to_edl_srt.api_client import GeminiClient
import tempfile
import shutil # For cleaning up temporary directory

# faster-whisperのサポートを追加（インストールされていない場合はスキップ）
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("faster-whisperライブラリが利用可能です。高速モードが使用できます。")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("faster-whisperライブラリが見つかりません。標準モードで実行します。")
    print("高速モードを使用するには次のコマンドを実行してください: pip install faster-whisper")

from mp4_to_edl_srt.segment import Segment
from mp4_to_edl_srt.edl_data import EDLData
from mp4_to_edl_srt.srt_data import SRTData
from mp4_to_edl_srt.scene import Scene
from mp4_to_edl_srt.scene_analysis import SceneAnalyzer
from mp4_to_edl_srt.timecode_utils import TimecodeConverter

# Whisperの警告を非表示にする
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

logger = logging.getLogger(__name__)

class ConfigManager:
    """アプリケーション設定を管理するクラス"""
    
    DEFAULT_CONFIG = {
        "whisper": {
            "model": "large-v3",
            "language": "auto",
            "initial_prompt": "日本語での自然な会話。",
            "temperature": 0.0,
            "beam_size": 5,
            "condition_on_previous": True
        },
        "audio": {
            "sample_rate": 44100,
            "channels": 2,
            "format": "wav",
            "preprocessing": False
        },
        "segmentation": {
            "threshold": 0.5,
            "min_segment_length": 0.2,
            "max_segment_length": 30.0,
            "merge_threshold": 0.3
        },
        "edl": {
            "title": "MP4 to EDL Project",
            "fcm": "NON-DROP FRAME",
            "use_timecode_offset": True
        },
        "paths": {
            "last_input_folder": "",
            "last_output_folder": ""
        },
        "gui": {
            "theme": "default",
            "font_size": 10,
            "window_width": 800,
            "window_height": 900
        },
        "scene_analysis": {
            "enabled": False,
            "frame_analysis_rate": 30
        }
    }
    
    def __init__(self, config_path="config.json"):
        """設定を初期化"""
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """設定ファイルを読み込む"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    
                # デフォルト設定をベースに、ロードした設定で上書き
                config = self.DEFAULT_CONFIG.copy()
                self._update_nested_dict(config, loaded_config)
                return config
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            # 設定ファイルがない場合はデフォルト設定を保存
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def _update_nested_dict(self, d, u):
        """ネストされた辞書を再帰的に更新"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def save_config(self):
        """現在の設定をファイルに保存"""
        self._save_config(self.config)
    
    def _save_config(self, config):
        """設定をJSONファイルに保存"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"設定ファイル保存エラー: {e}")
    
    def get(self, section, key=None):
        """設定値を取得"""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section, key, value):
        """設定値を設定"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def update_section(self, section, values):
        """セクション全体を更新"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)

class MP4File:
    def __init__(self, filepath: str, file_index: int, converter: TimecodeConverter, scene_analyzer: Optional[SceneAnalyzer] = None):
        self.filepath: str = os.path.normpath(filepath)
        self.file_index: int = file_index  # For reel name (e.g., TAPE01)
        self.converter = converter # Store the TimecodeConverter
        self.scene_analyzer = scene_analyzer # Store the SceneAnalyzer (optional)
        self.audio_filepath: str = ""
        self.transcription_result: Dict = {}
        self.segments: List[Segment] = []
        self.detected_scenes: List[Scene] = [] # List to store detected scenes
        self.edl_data: EDLData = EDLData(title="My Video Project", fcm="NON-DROP FRAME", converter=self.converter) # Pass the converter instance when initializing EDLData
        self.srt_data: SRTData = SRTData(converter=self.converter) # Pass converter to SRTData
        self.creation_time: Optional[str] = None
        self.timecode_offset: Optional[str] = "00:00:00:00" # Default offset
        self.duration: Optional[float] = None  # 動画の長さ（秒単位）
        
        # ファイルのメタデータを抽出
        self.extract_metadata()

        # --- Add GeminiClient initialization --- 
        # Get GeminiClient from SceneAnalyzer if available
        self.gemini_client: Optional[GeminiClient] = scene_analyzer.gemini_client if scene_analyzer and hasattr(scene_analyzer, 'gemini_client') else None
        if self.gemini_client:
             logger.debug("MP4File initialized with GeminiClient.")
        else:
             logger.debug("MP4File initialized without GeminiClient (SceneAnalyzer might be missing or lack the client).")
        # --- End GeminiClient initialization ---

    def extract_metadata(self) -> None:
        """MP4ファイルからメタデータ（作成時間やタイムコード）を抽出します。"""
        try:
            print(f"ファイルのメタデータを抽出中: {self.filepath}")
            
            # FFprobeを使用してメタデータを抽出
            command = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                self.filepath
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            metadata = json.loads(result.stdout)
            
            # 動画の長さを取得
            if "format" in metadata and "duration" in metadata["format"]:
                self.duration = float(metadata["format"]["duration"])
                print(f"ビデオの長さ: {self.duration} 秒 ({self._seconds_to_timecode(self.duration)})")
            
            # creation_timeを探す
            if "format" in metadata and "tags" in metadata["format"]:
                tags = metadata["format"]["tags"]
                if "creation_time" in tags:
                    self.creation_time = tags["creation_time"]
                    print(f"ファイル作成時間: {self.creation_time}")
                    
                    # creation_timeからタイムコードオフセットを計算
                    try:
                        # ISO 8601形式の日時文字列をパース (例: 2023-01-01T12:00:00.000Z)
                        dt = datetime.fromisoformat(self.creation_time.replace('Z', '+00:00'))
                        # 時間部分だけを取得してタイムコードに変換
                        hours = dt.hour
                        minutes = dt.minute
                        seconds = dt.second
                        frames = int(dt.microsecond / 1000000 * 30)  # 30fpsとして計算
                        
                        self.timecode_offset = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
                        print(f"計算されたタイムコードオフセット: {self.timecode_offset}")
                    except Exception as e:
                        print(f"タイムコードオフセットの計算中にエラーが発生しました: {e}")
            
            # タイムコードトラックを探す
            for stream in metadata.get("streams", []):
                if stream.get("codec_type") == "video" and "tags" in stream:
                    tags = stream["tags"]
                    if "timecode" in tags:
                        # タイムコードトラックが存在する場合はそれを優先
                        self.timecode_offset = tags["timecode"]
                        print(f"ビデオストリームからタイムコードを検出: {self.timecode_offset}")
                        break
            
            if not self.timecode_offset:
                print(f"警告: タイムコードが検出されませんでした。デフォルトの00:00:00:00を使用します。")
                self.timecode_offset = "00:00:00:00"
                
        except subprocess.CalledProcessError as e:
            print(f"FFprobeエラー: {e.stderr}")
            print(f"警告: メタデータの抽出に失敗しました。デフォルトのタイムコードを使用します。")
            self.timecode_offset = "00:00:00:00"
        except Exception as e:
            print(f"メタデータ抽出中にエラーが発生しました: {str(e)}")
            print(f"警告: デフォルトのタイムコードを使用します。")
            self.timecode_offset = "00:00:00:00"

    def apply_timecode_offset(self, timecode_ms: int) -> int:
        """タイムコード (ミリ秒) にオフセットを適用します。"""
        if not self.timecode_offset or self.timecode_offset == "00:00:00:00":
            return timecode_ms

        offset_ms = self.converter.hhmmssff_to_ms(self.timecode_offset)
        return timecode_ms + offset_ms

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
            logger.error(f"FFmpegエラー: {e.stderr}")
            raise Exception(f"FFmpeg error: {e.stderr}")

    def transcribe(
        self,
        model_name: str = "small",
        language: str = "ja",
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        beam_size: int = 5,
        condition_on_previous: bool = True,
        device: str = "auto",
        compute_type: str = "default", # for faster-whisper
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Extracts audio, performs transcription, and evaluates segments using batch API call."""
        logger.info(f">>> Entering transcribe method for {os.path.basename(self.filepath)}")
        config = ConfigManager().load_config()
        enable_preprocessing = config.get("audio", {}).get("preprocessing", False)

        if not self.audio_filepath or not os.path.exists(self.audio_filepath):
            logger.info("オーディオファイルパスが無効です。オーディオ抽出を試みます...")
            self.extract_audio()

            if not self.audio_filepath or not os.path.exists(self.audio_filepath):
                 logger.error("オーディオ抽出に失敗、またはファイルが見つかりません。文字起こしをスキップします。")
                 self.transcription_result = {"text": "ERROR: Audio Extraction Failed", "segments": [], "language": "error"}
                 self.segments = []
                 # Restore original exit log placement
                 logger.info("文字起こしリソースを解放しました。") 
                 logger.info(f"<<< Exiting transcribe method for {os.path.basename(self.filepath)}") 
                 return

        logger.info(f"'{self.filepath}' の文字起こしを開始中...")
        logger.info(f"モデル: {model_name}, デバイス: {device}, ComputeType: {compute_type}")
        # logger.debug("--- Past initial info logs, before device/compute type checks ---") # Remove this debug log

        # --- Restore the original logic block --- 
        # Use device setting from whisper config if auto
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"自動デバイス検出: {device}")
        
        # Select compute type for faster-whisper based on device
        if FASTER_WHISPER_AVAILABLE and compute_type == "default":
            if device == "cuda":
                # Check GPU capability for FP16/INT8
                # Defaulting to int8 for broader compatibility, even on CUDA
                compute_type = "int8" 
                logger.info(f"Faster-Whisper: GPU detected, using compute_type='{compute_type}' for better compatibility")
            else:
                compute_type = "int8"
                logger.info(f"Faster-Whisper: CPU detected, using compute_type='{compute_type}'")
        elif not FASTER_WHISPER_AVAILABLE:
             compute_type = "default" # Ensure compute_type is not used for standard whisper
        
        # Determine which audio file to use based on preprocessing setting
        audio_file_to_use = self.audio_filepath
        preprocessed_audio_path = ""
        if enable_preprocessing:
            logger.info("音声前処理が有効です。試行します...")
            try:
                preprocessed_audio_path = self._preprocess_audio(self.audio_filepath)
                if preprocessed_audio_path and os.path.exists(preprocessed_audio_path):
                    audio_file_to_use = preprocessed_audio_path
                    logger.info(f"前処理済み音声を使用: {audio_file_to_use}")
                else:
                    logger.warning("音声前処理に失敗しました。元のファイルを使用します。")
            except Exception as preprocess_err:
                logger.error(f"音声前処理中に予期せぬエラー: {preprocess_err}")
                logger.warning("元の音声ファイルを使用します。")
        else:
            logger.info("音声前処理は無効です。")
        
        # Load Model function (definition only)
        model = None
        def load_model_safely(m_name, dev):
            nonlocal model # Allow modification of the outer scope variable
            try:
                logger.info(f"Whisperモデル '{m_name}' ({dev}) をロード中...")
                # Disable FP16 on CPU explicitly for standard whisper
                use_fp16 = False if dev == 'cpu' else None 
                model = whisper.load_model(m_name, device=dev)
                logger.info(f"モデル '{m_name}' ({dev}) のロード成功.")
                return model
            except Exception as e:
                logger.error(f"モデル '{m_name}' ({dev}) のロード中にエラー: {e}")
                if dev == "cuda": torch.cuda.empty_cache() # Free memory on CUDA failure
                return None
        # --- End of restored block ---

        # Remove the hardcoded values
        # if device == "auto": device = "cpu"
        # if compute_type == "default": compute_type = "int8" if FASTER_WHISPER_AVAILABLE else "default"
        # audio_file_to_use = self.audio_filepath
        # preprocessed_audio_path = ""
        # model = None
        # def load_model_safely(m_name, dev):
        #     nonlocal model
        #     try: logger.info(f"(Simplified) Loading model {m_name} on {dev}..."); model = whisper.load_model(m_name, device=dev); logger.info("Load successful."); return model
        #     except Exception as e: logger.error(f"Load failed: {e}"); return None

        # Perform transcription (try block starts here)
        # logger.debug("Preparing for transcription try block...") # Remove this debug log
        try:
            raw_segments_for_json = [] # To store raw data for transcription_whisper_result
            self.segments = [] # Reset internal segments list
            
            # --- Actual Transcription Call (Faster-Whisper or Standard Whisper) ---
            if FASTER_WHISPER_AVAILABLE and compute_type != "default":
                logger.info("Attempting transcription using faster-whisper...")
                try:
                    logger.debug(f"Loading faster-whisper model: {model_name}, device={device}, compute_type={compute_type}")
                    fw_model = WhisperModel(model_name, device=device, compute_type=compute_type)
                    logger.debug("Faster-whisper model loaded successfully.")
                    
                    # Determine language parameter for faster-whisper
                    effective_language = None # Default to None for auto-detect
                    # Get language from config (passed via transcribe arguments)
                    config_lang = config.get('whisper', {}).get('language', 'auto') 
                    if config_lang and config_lang.lower() != "auto":
                        effective_language = config_lang # Use specified language if not 'auto'
                    logger.info(f"Using language '{effective_language if effective_language else 'auto'}' for faster-whisper.")

                    segments_iterator, info = fw_model.transcribe(
                        audio_file_to_use,
                        language=effective_language,
                        initial_prompt=initial_prompt,
                        temperature=temperature,
                        beam_size=beam_size,
                        condition_on_previous_text=condition_on_previous,
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    detected_lang = info.language
                    logger.info(f"Faster-Whisper 検出言語: {detected_lang}, 確率: {info.language_probability:.2f}")
                    total_duration_ms = (self.duration or 0) * 1000

                    logger.info("Faster-Whisper セグメントを処理中 (評価は後で一括実行)...")
                    segment_count = 0
                    for segment in segments_iterator:
                        start_ms = int(segment.start * 1000)
                        end_ms = int(segment.end * 1000)
                        text = segment.text.strip()
                        start_timecode = self.converter.ms_to_hhmmssff(start_ms)
                        end_timecode = self.converter.ms_to_hhmmssff(end_ms)
                        
                        app_segment = Segment(
                            start_timecode=start_timecode,
                            end_timecode=end_timecode,
                            transcription=text,
                            transcription_good_reason=None,
                            transcription_bad_reason=None
                        )
                        self.segments.append(app_segment)
                        raw_segments_for_json.append({"start": segment.start, "end": segment.end, "text": text})
                        segment_count += 1
                        
                        # Report progress
                        if progress_callback and total_duration_ms > 0:
                             progress = min(segment.end * 1000 / total_duration_ms, 1.0)
                             progress_callback(int(progress*100), 100)

                    self.transcription_result = {
                        "text": " ".join([s["text"] for s in raw_segments_for_json]),
                        "segments": raw_segments_for_json,
                        "language": detected_lang
                    }
                    logger.info(f"faster-whisper 文字起こし完了. {len(self.segments)} 個の内部セグメントを作成 (評価前).")
                except Exception as fw_err:
                     logger.error(f"faster-whisper実行中にエラー: {fw_err}", exc_info=True)
                     logger.warning("標準のwhisperにフォールバックします...")
                     # Ensure faster-whisper specific flags are reset if falling back
                     compute_type = "default"
                     # FASTER_WHISPER_AVAILABLE = False # Don't disable permanently, just for this run
                     # Fall through to standard whisper block by ensuring the condition below is met
                     model = None # Ensure standard whisper model is loaded if fallback happens

            # Standard Whisper execution block (runs if faster-whisper failed or is disabled, or fallback needed)
            # Use 'model is None' to check if standard model needs loading (either initially or due to fallback)
            # logger.debug(f"Checking if standard whisper should be used (model is None: {model is None}, FASTER_WHISPER_AVAILABLE: {FASTER_WHISPER_AVAILABLE}, compute_type: {compute_type})...") # Remove this debug log
            if model is None and (not FASTER_WHISPER_AVAILABLE or compute_type == "default"):
                logger.info("Attempting transcription using standard whisper...")
                # model = load_model_safely(model_name, device) <-- Moved call below
                
                # Load the model here
                model = load_model_safely(model_name, device)
                
                if not model:
                     # If standard model also fails, record error and exit transcribe method
                     logger.critical(f"標準whisperモデル '{model_name}' ({device}) のロードにも失敗しました。文字起こしを中止します。")
                     # ... (rest of the error handling for model load failure) ...
                     # Ensure audio file cleanup happens before returning
                     if preprocessed_audio_path and os.path.exists(preprocessed_audio_path):
                          try: os.remove(preprocessed_audio_path); logger.info(f"一時前処理ファイルを削除: {preprocessed_audio_path}")
                          except OSError as e: logger.warning(f"一時ファイル削除エラー: {e}")
                     logger.info("文字起こしリソースを解放しました。") 
                     logger.info(f"<<< Exiting transcribe method for {os.path.basename(self.filepath)}") 
                     return # Exit transcribe

                # Standard whisper transcription with word timestamps
                # Note: Standard whisper's progress isn't easily usable here.
                whisper_options = dict(
                    language=language if language and language != "auto" else None, 
                    initial_prompt=initial_prompt, 
                    temperature=temperature, 
                    beam_size=beam_size, 
                    condition_on_previous_text=condition_on_previous,
                    word_timestamps=True,
                    fp16 = False if device == 'cpu' else None # Disable FP16 on CPU
                )
                logger.info(f"標準Whisper オプション: {whisper_options}")
                logger.debug("Calling standard whisper model.transcribe()...")
                self.transcription_result = model.transcribe(
                    self.audio_filepath, # Use potentially preprocessed audio
                    **whisper_options
                )
                if progress_callback:
                    progress_callback(100, 100) # Mark as complete after standard whisper finishes
                logger.info(f"標準whisper 文字起こし完了.")
            # --- End Transcription Call ---

            # --- Post-Transcription Processing (Common logic if transcription_result has segments) ---
            # This block now only populates self.segments if standard whisper ran and self.segments is empty
            if not self.segments and isinstance(self.transcription_result, dict) and self.transcription_result.get("segments"):
                whisper_segments = self.transcription_result.get("segments", [])
                logger.info(f"標準Whisperセグメントを処理中 ({len(whisper_segments)}個) (評価は後で一括実行)...")
                raw_segments_for_json = [] # Reset for standard whisper case
                for idx, segment_data in enumerate(whisper_segments):
                    start_ms = int(segment_data["start"] * 1000)
                    end_ms = int(segment_data["end"] * 1000)
                    text = segment_data.get("text", "").strip()
                    start_timecode = self.converter.ms_to_hhmmssff(start_ms)
                    end_timecode = self.converter.ms_to_hhmmssff(end_ms)

                    segment = Segment(
                        start_timecode=start_timecode,
                        end_timecode=end_timecode,
                        transcription=text,
                        transcription_good_reason=None,
                        transcription_bad_reason=None
                    )
                    self.segments.append(segment)
                    raw_segments_for_json.append({"start": segment_data["start"], "end": segment_data["end"], "text": text}) # Add raw data
                # Update transcription_result text if standard whisper was used
                if not self.transcription_result.get("text"): # Avoid overwriting if faster-whisper populated it
                     self.transcription_result["text"] = " ".join([s.get("text", "") for s in whisper_segments])
                self.transcription_result["segments"] = raw_segments_for_json # Ensure raw segments are stored
                     
                logger.info(f"{len(self.segments)}個の内部セグメントオブジェクトを作成しました (評価前)。")
            # --- End Post-Transcription Segment Creation ---
            
            # --- Batch Evaluate Transcriptions --- 
            if self.gemini_client and self.segments:
                logger.info(f"一括文字起こし評価を開始 (対象セグメント数: {len(self.segments)})...")
                transcriptions_to_evaluate = [seg.transcription for seg in self.segments]
                try:
                    evaluation_results = self.gemini_client.evaluate_transcription_batch(transcriptions_to_evaluate)
                    
                    if len(evaluation_results) == len(self.segments):
                        logger.info("バッチ評価結果をセグメントに割り当て中...")
                        for segment, (good, bad) in zip(self.segments, evaluation_results):
                            segment.transcription_good_reason = good
                            segment.transcription_bad_reason = bad
                        logger.info("バッチ評価結果の割り当て完了。")
                    else:
                        logger.warning(f"バッチ文字起こし評価が不正な数の結果を返しました ({len(evaluation_results)} vs {len(self.segments)})。評価タグは割り当てられません。")
                except Exception as eval_err:
                    logger.error(f"バッチ文字起こし評価中にエラーが発生しました: {eval_err}", exc_info=True)
            elif not self.gemini_client:
                logger.warning("GeminiClientが利用できないため、文字起こし評価をスキップします。")
            # --- End Batch Evaluation ---
            
            # --- English Re-transcription block REMOVED --- 

        except Exception as e:
            logger.error(f"文字起こし中にエラーが発生しました: {str(e)}")
            if 'model' in locals() and model is not None:
                del model
            if 'fw_model' in locals() and 'fw_model' in locals() and fw_model is not None:
                 del fw_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPUキャッシュをクリアしました。")
            logger.info("文字起こしリソースを解放しました。")
            # Restore original exit log position
            logger.info(f"<<< Exiting transcribe method for {os.path.basename(self.filepath)}") 
            # raise Exception(f"Whisper transcription error: {e}") # Re-raise if needed, or handle

        finally:
            # ... (Resource cleanup) ...
            logger.info("文字起こしリソースを解放しました。") 
            logger.info(f"<<< Exiting transcribe method for {os.path.basename(self.filepath)}")

    def _preprocess_audio(self, audio_path: str) -> Optional[str]:
        """音声ファイルの前処理（ノイズ除去と音量調整）を行います。"""
        try:
            print(f"音声ファイルを前処理中: {audio_path}")
            
            # 出力ファイルパスを生成
            base_dir = os.path.dirname(audio_path)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            processed_path = os.path.join(base_dir, f"{base_name}_processed.wav")
            
            # pydubを使用して音声を読み込み
            audio = AudioSegment.from_file(audio_path)
            
            # 音量の正規化（headroomを0.5に調整してより大きな音量に）
            print(f"音量を正規化中...")
            normalized_audio = audio.normalize(headroom=0.5)
            
            # ノイズ除去（FFmpegを使用）
            print(f"ノイズ除去フィルターを適用中...")
            normalized_audio.export("temp_normalized.wav", format="wav")
            
            # FFmpegのノイズ除去フィルターを適用（周波数帯域を拡大し、ノイズ除去強度を調整）
            command = [
                "ffmpeg",
                "-i", "temp_normalized.wav",
                "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-20",  # 元のフィルター設定
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

    def generate_edl_data(self, use_timecode_offset: bool = True) -> Tuple[EDLData, str]:
        """
        Generates EDL data from the segments.
        
        Args:
            use_timecode_offset: Whether to apply the timecode offset.
            
        Returns:
            A tuple containing the EDLData object and the record start timecode.
        """
        edl = EDLData(title=f"{os.path.basename(self.filepath)}_Transcription", fcm="NON-DROP FRAME")
        reel_name = f"TAPE{self.file_index:02d}"
        
        record_start_ms = 0
        if use_timecode_offset:
             record_start_ms = self.converter.hhmmssff_to_ms(self.timecode_offset)
        
        record_start_tc = self.converter.ms_to_hhmmssff(record_start_ms)
        print(f"Generating EDL with Record Start: {record_start_tc} (Offset Applied: {use_timecode_offset})")

        current_record_ms = record_start_ms

        for i, segment in enumerate(self.segments):
            source_start_ms = self.converter.hhmmssff_to_ms(segment.start_timecode)
            source_end_ms = self.converter.hhmmssff_to_ms(segment.end_timecode)
            duration_ms = source_end_ms - source_start_ms
            
            record_end_ms = current_record_ms + duration_ms
            
            event = {
                "event_num": f"{i+1:03d}",
                "reel_name": reel_name,
                "track_type": "A", # Audio only for transcription
                "transition": "C", # Cut
                "source_in": segment.start_timecode,
                "source_out": segment.end_timecode,
                "record_in": self.converter.ms_to_hhmmssff(current_record_ms),
                "record_out": self.converter.ms_to_hhmmssff(record_end_ms),
                "clip_name": f"Segment {i+1}",
                "comment": segment.transcription, # Use transcription as comment
                "scene_description": segment.scene_description # Pass scene description
            }
            edl.add_event(event)
            
            current_record_ms = record_end_ms
        
        self.edl_data = edl # Store generated EDL data
        return edl, record_start_tc

    def generate_srt_data(self) -> SRTData:
        """
        Generates SRT data from the segments.
        The SRTData object already holds the segments and the converter.
        """
        # Clear existing segments in srt_data and add current ones
        self.srt_data.segments = [] 
        for segment in self.segments:
             self.srt_data.add_segment(segment)
        # Segments in self.srt_data now have scene info assigned (if analysis ran)
        # SRTData's write_to_file method will handle the rest using the converter
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

    def _frames_to_timecode(self, total_frames: int) -> str:
        """フレーム数からHH:MM:SS:FF形式のタイムコードに変換します（30 fps）。"""
        hours = total_frames // (3600 * 30)
        minutes = (total_frames % (3600 * 30)) // (60 * 30)
        secs = (total_frames % (60 * 30)) // 30
        frames = total_frames % 30
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

    def run_scene_analysis(self, frame_analysis_rate: int, capture_output_dir: Optional[str] = None):
        """Runs scene analysis if the analyzer is available."""
        if not self.scene_analyzer:
            print("Scene analyzer not provided. Skipping scene analysis.")
            return

        print("Starting scene analysis...")
        # The progress callback is now handled during SceneAnalyzer initialization.
        # Remove the nested wrapper and the progress_callback argument from analyze_video call.

        try:
             # Call analyze_video without the progress_callback argument, but with capture_output_dir
             self.detected_scenes = self.scene_analyzer.analyze_video(
                 self.filepath,
                 frame_analysis_rate,
                 capture_output_dir=capture_output_dir # Pass the directory
             )
             print(f"Scene analysis complete. Found {len(self.detected_scenes)} scenes.")
             self.assign_scenes_to_segments() # Assign scenes after analysis
        except Exception as e:
             print(f"Error during scene analysis: {e}")
             self.detected_scenes = [] # Ensure list is empty on error
        finally:
            # Final progress update might need adjustment if handled differently now
            # Let's rely on the analyzer's final _report_progress(1.0)
            pass
            # if progress_callback:
            #     progress_callback(0.9) # REMOVED - Handled by analyzer's callback

    def assign_scenes_to_segments(self):
         """Assigns detected scene information to transcription segments based on time overlap."""
         if not self.detected_scenes or not self.segments:
             print("No scenes detected or no segments to assign to. Skipping scene assignment.")
             return

         print(f"Assigning {len(self.detected_scenes)} scenes to {len(self.segments)} segments...")
         assigned_count = 0
         for segment in self.segments:
             segment_start_ms = self.converter.hhmmssff_to_ms(segment.start_timecode)
             segment_end_ms = self.converter.hhmmssff_to_ms(segment.end_timecode)
             segment_mid_ms = segment_start_ms + (segment_end_ms - segment_start_ms) / 2

             best_match_scene = None
             for scene in self.detected_scenes:
                 if scene.start_ms <= segment_mid_ms < scene.end_ms:
                     best_match_scene = scene
                     break # Found the containing scene

             if best_match_scene:
                 segment.scene_id = best_match_scene.scene_id
                 segment.scene_description = best_match_scene.description
                 assigned_count += 1
             # else: # Optional: Log segments that don't fall into any scene
             #    print(f"  Warning: Segment {segment.start_timecode}-{segment.end_timecode} did not match any scene.")

         print(f"Assigned scene information to {assigned_count} segments.")

    def get_intermediate_data_as_dict(self) -> Dict[str, Any]:
        """Gathers all processed data into a dictionary for JSON serialization."""
        logger.info("中間データ辞書を作成中...") # Changed to logger
        
        # Ensure transcription_result exists and has expected keys, provide defaults otherwise
        whisper_result = self.transcription_result if isinstance(self.transcription_result, dict) else {}

        data = {
            "source_filepath": self.filepath,
            "file_index": self.file_index,
            "extracted_audio_filepath": self.audio_filepath if self.audio_filepath and os.path.exists(self.audio_filepath) else None,
            "metadata": {
                "duration_seconds": self.duration,
                "creation_time_utc": self.creation_time,
                "timecode_offset": self.timecode_offset,
            },
            # Ensure transcription_result format is consistent even on errors
            "transcription_whisper_result": {
                "text": whisper_result.get("text", ""),
                "segments": whisper_result.get("segments", []),
                "language": whisper_result.get("language", "unknown")
            },
            "detected_scenes": [scene.to_dict(self.converter) for scene in self.detected_scenes], # Use Scene.to_dict
            "final_segments": [segment.to_dict() for segment in self.segments] # Use Segment.to_dict
        }
        logger.info("中間データ辞書作成完了。") # Changed to logger
        return data

