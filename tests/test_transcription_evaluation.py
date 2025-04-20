import os
import sys
import logging
import json

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from mp4_to_edl_srt.mp4_file import MP4File, ConfigManager
    from mp4_to_edl_srt.timecode_utils import TimecodeConverter
    from mp4_to_edl_srt.scene_analysis import SceneAnalyzer
    from mp4_to_edl_srt.api_client import GeminiClient # For type hint checking if needed
except ImportError as e:
    logger.error(f"必要なモジュールのインポートに失敗しました: {e}")
    logger.error("プロジェクトルートから実行しているか、PYTHONPATHを確認してください。")
    sys.exit(1)

# --- Test Configuration ---
TEST_VIDEO_PATH = r"G:\ts\GH012880.MP4" # The file with repetition issue
FILE_INDEX = 1 # Arbitrary index for testing

# --- Execution ---
if __name__ == "__main__":
    logger.info("--- Transcription Evaluation Test Start ---")

    if not os.path.exists(TEST_VIDEO_PATH):
        logger.error(f"テスト動画ファイルが見つかりません: {TEST_VIDEO_PATH}")
        sys.exit(1)

    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.config
        whisper_config = config.get('whisper', {})
        logger.info(f"Loaded Whisper Config: {whisper_config}")

        # Initialize components
        converter = TimecodeConverter()
        # Initialize SceneAnalyzer to make GeminiClient available for evaluation
        scene_analyzer = SceneAnalyzer() 
        
        # Initialize MP4File
        logger.info(f"Initializing MP4File for: {TEST_VIDEO_PATH}")
        mp4_file = MP4File(
            filepath=TEST_VIDEO_PATH, 
            file_index=FILE_INDEX, 
            converter=converter, 
            scene_analyzer=scene_analyzer # Pass analyzer to enable AI eval
        )

        # Ensure audio is extracted (optional, transcription might do it)
        if not mp4_file.audio_filepath or not os.path.exists(mp4_file.audio_filepath):
             logger.info("Extracting audio...")
             mp4_file.extract_audio()

        # Run transcription and evaluation
        logger.info("Running transcribe method (including AI evaluation)...")
        mp4_file.transcribe(
            model_name=whisper_config.get('model', 'small'),
            #language=whisper_config.get('language', 'ja'),
            initial_prompt=whisper_config.get('initial_prompt'),
            temperature=float(whisper_config.get('temperature', 0.0)),
            beam_size=int(whisper_config.get('beam_size', 5)),
            condition_on_previous=whisper_config.get('condition_on_previous', True), # Uses value from config
            # device and compute_type will be determined inside transcribe
        )
        logger.info("Transcription and evaluation finished.")

        # Output the evaluated segments as JSON
        if mp4_file.segments:
            logger.info("--- Evaluated Segments (JSON Output) ---")
            segments_list_of_dicts = [seg.to_dict() for seg in mp4_file.segments]
            json_output = json.dumps(segments_list_of_dicts, indent=4, ensure_ascii=False)
            print(json_output)
            logger.info("--- End JSON Output ---")
        else:
            logger.info("No segments were generated.")

    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生しました: {e}", exc_info=True)

    logger.info("--- Transcription Evaluation Test End ---") 