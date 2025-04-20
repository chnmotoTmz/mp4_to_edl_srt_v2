import unittest
import os
import sys
import logging

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from mp4_to_edl_srt.mp4_file import MP4File
    from mp4_to_edl_srt.timecode_utils import TimecodeConverter
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is set correctly.")
    sys.exit(1)

# Configure logging for the test
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Test Configuration ---
# Use a real MP4 file known to have audio. Using the one from the last JSON example.
# Ensure this path is accessible when running the test.
TEST_MP4_PATH = r"G:\ts\GH012664.MP4" # Make sure this file exists and has audio
TEST_FILE_INDEX = 99 # Arbitrary index for testing

# Use a smaller model for faster testing? Or the configured one? Let's use 'tiny' for speed.
TEST_WHISPER_MODEL = "tiny" 
# TEST_WHISPER_MODEL = "large-v3" # ConfigManager().config.get('whisper', {}).get('model', 'large-v3') # Use configured model


class TestMP4FileTranscribe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up resources needed for all tests in this class."""
        logger.info("Setting up TestMP4FileTranscribe class...")
        if not os.path.exists(TEST_MP4_PATH):
            raise FileNotFoundError(f"Test MP4 file not found: {TEST_MP4_PATH}")
            
        cls.converter = TimecodeConverter()
        # Initialize MP4File without a scene analyzer for this test
        cls.mp4_file_instance = MP4File(TEST_MP4_PATH, TEST_FILE_INDEX, cls.converter, scene_analyzer=None)
        
        # Extract audio once for all tests in this class
        try:
            logger.info(f"Extracting audio for {TEST_MP4_PATH}...")
            cls.mp4_file_instance.extract_audio()
            cls.audio_extracted = True
            logger.info(f"Audio extracted to: {cls.mp4_file_instance.audio_filepath}")
        except Exception as e:
            logger.error(f"Failed to extract audio during setUpClass: {e}", exc_info=True)
            cls.audio_extracted = False
            
    @classmethod
    def tearDownClass(cls):
         """Clean up resources after all tests in this class are run."""
         logger.info("Tearing down TestMP4FileTranscribe class...")
         if hasattr(cls, 'mp4_file_instance') and cls.mp4_file_instance and cls.mp4_file_instance.audio_filepath:
             if os.path.exists(cls.mp4_file_instance.audio_filepath):
                 try:
                     os.remove(cls.mp4_file_instance.audio_filepath)
                     logger.info(f"Cleaned up audio file: {cls.mp4_file_instance.audio_filepath}")
                 except OSError as e:
                     logger.warning(f"Could not remove audio file during teardown: {e}")

    def test_transcribe_populates_results(self):
        """Test that transcribe method populates transcription_result and segments."""
        logger.info("Running test_transcribe_populates_results...")
        self.assertTrue(self.audio_extracted, "Audio extraction failed in setUpClass, cannot run test.")
        
        # Ensure results are initially empty or None before calling transcribe
        self.mp4_file_instance.transcription_result = {}
        self.mp4_file_instance.segments = []
        
        logger.info(f"Calling transcribe with model: {TEST_WHISPER_MODEL}")
        try:
            self.mp4_file_instance.transcribe(
                model_name=TEST_WHISPER_MODEL, 
                language='ja', # Assuming Japanese for the test file
                # Force standard whisper for this test by setting compute_type="default"
                compute_type="default" 
                # Add other necessary parameters if they affect execution significantly
                # temperature=0.0, 
                # beam_size=5, 
                # condition_on_previous=True
            )
            transcribe_successful = True
        except Exception as e:
            logger.error(f"transcribe method raised an exception: {e}", exc_info=True)
            transcribe_successful = False
            
        self.assertTrue(transcribe_successful, "transcribe method raised an exception.")

        # --- Assertions ---
        logger.info("Checking transcription results...")
        
        # Check if transcription_result dict is populated and has expected keys
        self.assertIsNotNone(self.mp4_file_instance.transcription_result, "transcription_result should not be None")
        self.assertIsInstance(self.mp4_file_instance.transcription_result, dict, "transcription_result should be a dict")
        self.assertIn("text", self.mp4_file_instance.transcription_result, "'text' key missing in transcription_result")
        self.assertIn("segments", self.mp4_file_instance.transcription_result, "'segments' key missing in transcription_result")
        self.assertIn("language", self.mp4_file_instance.transcription_result, "'language' key missing in transcription_result")
        
        # Check if the main segments list (list of Segment objects) is populated
        self.assertIsNotNone(self.mp4_file_instance.segments, "segments list should not be None")
        self.assertIsInstance(self.mp4_file_instance.segments, list, "segments should be a list")
        
        # Check if results are actually non-empty (assuming the test audio has speech)
        # This might fail if the audio is truly silent or whisper fails badly
        self.assertTrue(len(self.mp4_file_instance.transcription_result.get("segments", [])) > 0, 
                        "Whisper segments in transcription_result should not be empty")
        self.assertTrue(len(self.mp4_file_instance.segments) > 0, 
                        "MP4File segments list should not be empty")
        self.assertTrue(len(self.mp4_file_instance.transcription_result.get("text", "").strip()) > 0,
                         "Transcription text should not be empty")
                         
        # Check detected language (optional but good)
        # Note: Language detection might not always be perfect
        detected_lang = self.mp4_file_instance.transcription_result.get("language")
        self.assertIsNotNone(detected_lang, "Detected language should not be None")
        self.assertEqual(detected_lang, 'ja', f"Expected language 'ja', but got '{detected_lang}'")
        
        logger.info("test_transcribe_populates_results finished successfully.")

if __name__ == '__main__':
    logger.info("Starting MP4File Transcribe Unit Test Runner...")
    unittest.main() 