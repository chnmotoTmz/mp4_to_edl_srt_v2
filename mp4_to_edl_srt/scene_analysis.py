import cv2
import time # For simulating processing time
import random
import os # Import os module
from typing import List, Callable, Optional
# from .scene import Scene # Relative import
from mp4_to_edl_srt.scene import Scene # Changed to absolute import
from .api_client import GeminiClient # Import GeminiClient
import logging # Use logging for better output control

logger = logging.getLogger(__name__) # Use logger instead of print

class SceneAnalyzer:
    """Analyzes video frames to detect scene changes. (Placeholder implementation)"""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu', progress_callback: Optional[Callable[[float], None]] = None):
        """
        Initializes the SceneAnalyzer.

        Args:
            model_path: Path to the scene analysis model (currently unused placeholder).
            device: Device to run the model on ('cpu' or 'cuda', currently unused).
            progress_callback: Optional function to report progress (0.0 to 1.0).
        """
        logger.info(f"Initializing SceneAnalyzer (Placeholder). Model path: {model_path}, Device: {device}") # Changed to logger
        self.model_path = model_path
        self.device = device
        self._load_model() # Placeholder for model loading
        self.progress_callback = progress_callback
        # Initialize Gemini Client
        try:
            self.gemini_client = GeminiClient() # Assumes API key is handled within GeminiClient (e.g., via env vars)
            logger.info("GeminiClient initialized successfully.") # Changed to logger
        except Exception as e:
            logger.warning(f"Failed to initialize GeminiClient: {e}. Scene descriptions will not be generated.") # Changed to logger
            self.gemini_client = None

    def _load_model(self):
        """Placeholder for loading the actual scene analysis model."""
        logger.info("Loading scene analysis model (Placeholder)...") # Changed to logger
        # In a real implementation, load the PyTorch/ONNX/etc. model here
        time.sleep(0.5) # Simulate loading time
        logger.info("Model loaded (Placeholder).") # Changed to logger

    def _report_progress(self, value: float):
        """Reports progress if a callback is provided."""
        if self.progress_callback:
            self.progress_callback(min(max(value, 0.0), 1.0))

    def analyze_video(self, video_path: str, frame_analysis_rate: int = 30, capture_output_dir: Optional[str] = None) -> List[Scene]:
        """
        Analyzes the video to detect scenes and optionally captures thumbnails.

        Args:
            video_path: Path to the video file.
            frame_analysis_rate: Analyze every Nth frame.
            capture_output_dir: Optional directory path to save scene thumbnails.

        Returns:
            A list of detected Scene objects.
        """
        logger.info(f"Starting scene analysis for: {video_path} (Rate: {frame_analysis_rate})") # Changed to logger
        if capture_output_dir:
            logger.info(f"Scene thumbnails will be saved to: {capture_output_dir}") # Changed to logger
            os.makedirs(capture_output_dir, exist_ok=True) # Ensure directory exists
            
        self._report_progress(0.0)

        # --- Open video once to get metadata --- 
        cap_meta = cv2.VideoCapture(video_path)
        if not cap_meta.isOpened():
            logger.error(f"Error: Could not open video file {video_path} for metadata.") # Changed to logger
            return []
        fps = cap_meta.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_meta.release()
        # --- Metadata check --- 
        if fps <= 0 or total_frames <= 0:
             logger.error(f"Error: Could not read video metadata (FPS: {fps}, Frames: {total_frames})") # Changed to logger
             return []

        logger.info(f"Video Info - FPS: {fps:.2f}, Total Frames: {total_frames}") # Changed to logger

        scenes: List[Scene] = []
        scene_id_counter = 1
        current_scene_start_ms = 0

        # Define evaluation tag categories
        GOOD_REASONS = {"Scenic", "Landmark", "Informative", "Action"}
        BAD_REASONS = {"Privacy", "PoorQuality", "Irrelevant"}
        # "Generic" is treated as neither good nor bad by default

        # --- Placeholder Scene Detection Logic ---
        # Use total_frames if rate is 0 or less, avoid division by zero
        num_simulated_steps = total_frames // frame_analysis_rate if frame_analysis_rate > 0 else total_frames
        last_progress_report_step = 0

        for i in range(num_simulated_steps):
            current_frame_index = i * frame_analysis_rate
            current_time_ms = int((current_frame_index / fps) * 1000)

            # --- Simulate scene change --- 
            # Using slightly different logic for simulation to ensure at least one scene change usually
            is_scene_change = (i > 0 and random.random() < 0.05 and current_time_ms > current_scene_start_ms) or (i == num_simulated_steps // 2) # Force one change mid-way for testing
            # Ensure the last segment gets created if no change detected at the very end
            is_last_iteration = (i == num_simulated_steps - 1) 

            # Process scene change or if it's the last iteration to capture the final scene segment
            if (is_scene_change and current_time_ms > current_scene_start_ms):
                scene_end_ms = current_time_ms
                thumbnail_path = None
                # Initialize description and tags before API call
                scene_desc = f"Scene {scene_id_counter}: Analysis pending..."
                scene_evaluation_tag = None # Will be filled by API response

                # --- Capture Thumbnail & Generate Description/Tag via API ---
                if capture_output_dir:
                    try:
                        # Calculate middle frame index for thumbnail
                        middle_frame_index = int(((current_scene_start_ms + scene_end_ms) / 2 / 1000) * fps)
                        middle_frame_index = min(max(middle_frame_index, 0), total_frames - 1) # Clamp index
                        cap_thumb = cv2.VideoCapture(video_path)
                        if cap_thumb.isOpened():
                            cap_thumb.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
                            ret, frame = cap_thumb.read()
                            if ret:
                                image_filename = f"scene_{scene_id_counter:04d}.jpg"
                                thumbnail_path = os.path.join(capture_output_dir, image_filename)
                                cv2.imwrite(thumbnail_path, frame)
                                logger.info(f"    Thumbnail saved: {thumbnail_path}")
                                
                                if self.gemini_client and thumbnail_path:
                                    try:
                                        # Call updated analyze_scene which returns desc and tag
                                        analysis_result = self.gemini_client.analyze_scene(thumbnail_path)
                                        scene_desc = analysis_result.get('description', f"Scene {scene_id_counter}: No description from API.")
                                        scene_evaluation_tag = analysis_result.get('evaluation_tag') # Get the tag
        
                                        if analysis_result.get('error'):
                                            logger.warning(f"    Gemini API returned an error for scene {scene_id_counter}: {scene_desc}")
                                            # Keep default scene_desc and potentially null scene_evaluation_tag
                                        else:
                                            # Description is already cleaned/limited in api_client now
                                            logger.info(f"    API Analysis Result: Desc='{scene_desc}', Tag='{scene_evaluation_tag}'")
        
                                    except Exception as api_err:
                                        logger.error(f"    Error calling/processing Gemini API for scene {scene_id_counter}: {api_err}")
                                        scene_desc = f"Scene {scene_id_counter}: API call failed."
                                        scene_evaluation_tag = None # Ensure tag is None on error
                                elif not self.gemini_client:
                                     logger.warning(f"    Gemini client not available for scene {scene_id_counter}.")
                                     scene_desc = f"Scene {scene_id_counter}: Gemini client not available."
                                     scene_evaluation_tag = None # Ensure tag is None 
                            else:
                                logger.warning(f"    Failed to read frame {middle_frame_index} for scene {scene_id_counter}")
                                scene_desc = f"Scene {scene_id_counter}: Thumbnail capture failed."
                                scene_evaluation_tag = None # Ensure tag is None 
                            cap_thumb.release()
                        else:
                           logger.warning(f"    Could not re-open video to capture thumbnail for scene {scene_id_counter}") 
                           scene_desc = f"Scene {scene_id_counter}: Video open failed for thumbnail."
                           scene_evaluation_tag = None # Ensure tag is None 
                    except Exception as e:
                         logger.error(f"    Error capturing thumbnail or analyzing scene {scene_id_counter}: {e}")
                         scene_desc = f"Scene {scene_id_counter}: Thumbnail/Analysis error."
                         scene_evaluation_tag = None
                else:
                     logger.info(f"Scene {scene_id_counter}: Thumbnail capture disabled.")
                     scene_desc = f"Scene {scene_id_counter}: Thumbnail capture disabled."
                     scene_evaluation_tag = None # No analysis if no thumbnail
                # --- End Capture & Analysis --- 

                # --- Determine good/bad reason from API tag --- 
                scene_good_reason = None
                scene_bad_reason = None
                if scene_evaluation_tag:
                    if scene_evaluation_tag in GOOD_REASONS:
                        scene_good_reason = scene_evaluation_tag
                    elif scene_evaluation_tag in BAD_REASONS:
                        scene_bad_reason = scene_evaluation_tag
                    # If tag is "Generic" or unknown, both reasons remain None (default)
                # --- End Reason Determination --- 

                new_scene = Scene(
                    start_ms=current_scene_start_ms,
                    end_ms=scene_end_ms,
                    scene_id=scene_id_counter,
                    description=scene_desc, # Use description from API or fallback
                    thumbnail_path=thumbnail_path,
                    scene_good_reason=scene_good_reason, # Pass determined reason
                    scene_bad_reason=scene_bad_reason,   # Pass determined reason
                    scene_evaluation_tag=scene_evaluation_tag # Pass the raw tag from API
                )
                scenes.append(new_scene)
                logger.info(f"  -> Detected Scene Change: {new_scene}") # __str__ includes reasons
                current_scene_start_ms = scene_end_ms
                scene_id_counter += 1

            if i % 100 == 0: time.sleep(0.01)

            if i - last_progress_report_step >= 50: 
                 progress = (i + 1) / num_simulated_steps if num_simulated_steps > 0 else 1.0
                 self._report_progress(progress * 0.95) 
                 last_progress_report_step = i
        # --- End Loop --- 

        # Add the final scene
        final_scene_end_ms = int((total_frames / fps) * 1000)
        if current_scene_start_ms < final_scene_end_ms:
            thumbnail_path = None
            scene_desc = f"Scene {scene_id_counter}: Analysis pending..."
            scene_evaluation_tag = None
            # --- Capture Thumbnail & Generate Description/Tag for Final Scene --- 
            if capture_output_dir:
                 try:
                     # Calculate middle frame index for thumbnail
                     middle_frame_index = int(((current_scene_start_ms + final_scene_end_ms) / 2 / 1000) * fps)
                     middle_frame_index = min(max(middle_frame_index, 0), total_frames - 1)
                     cap_thumb = cv2.VideoCapture(video_path)
                     if cap_thumb.isOpened():
                          cap_thumb.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
                          ret, frame = cap_thumb.read()
                          if ret:
                               image_filename = f"scene_{scene_id_counter:04d}.jpg"
                               thumbnail_path = os.path.join(capture_output_dir, image_filename)
                               cv2.imwrite(thumbnail_path, frame)
                               logger.info(f"    Thumbnail saved: {thumbnail_path}")
                               
                               if self.gemini_client and thumbnail_path:
                                   try:
                                       analysis_result = self.gemini_client.analyze_scene(thumbnail_path)
                                       scene_desc = analysis_result.get('description', f"Scene {scene_id_counter}: No description from API.")
                                       scene_evaluation_tag = analysis_result.get('evaluation_tag')
      
                                       if analysis_result.get('error'):
                                            logger.warning(f"    Gemini API returned an error for final scene {scene_id_counter}: {scene_desc}")
                                       else:
                                           logger.info(f"    API Analysis Result (Final): Desc='{scene_desc}', Tag='{scene_evaluation_tag}'")
                                   except Exception as api_err:
                                       logger.error(f"    Error calling/processing Gemini API for final scene {scene_id_counter}: {api_err}")
                                       scene_desc = f"Scene {scene_id_counter}: API call failed."
                                       scene_evaluation_tag = None
                               elif not self.gemini_client:
                                    logger.warning(f"    Gemini client not available for final scene {scene_id_counter}.")
                                    scene_desc = f"Scene {scene_id_counter}: Gemini client not available."
                                    scene_evaluation_tag = None
                          else:
                               logger.warning(f"    Failed to read frame {middle_frame_index} for final scene {scene_id_counter}")
                               scene_desc = f"Scene {scene_id_counter}: Thumbnail capture failed."
                               scene_evaluation_tag = None
                          cap_thumb.release()
                     else:
                          logger.warning(f"    Could not re-open video to capture thumbnail for final scene {scene_id_counter}")
                          scene_desc = f"Scene {scene_id_counter}: Video open failed for thumbnail."
                          scene_evaluation_tag = None
                 except Exception as e:
                      logger.error(f"    Error capturing thumbnail or analyzing final scene {scene_id_counter}: {e}")
                      scene_desc = f"Scene {scene_id_counter}: Thumbnail/Analysis error."
                      scene_evaluation_tag = None
            else:
                 logger.info(f"Final scene {scene_id_counter}: Thumbnail capture disabled.")
                 scene_desc = f"Final scene {scene_id_counter}: Thumbnail capture disabled."
                 scene_evaluation_tag = None # No analysis if no thumbnail
            # --- End Final Capture & Analysis --- 

            # --- Determine good/bad reason from API tag --- 
            scene_good_reason = None
            scene_bad_reason = None
            if scene_evaluation_tag:
                if scene_evaluation_tag in GOOD_REASONS:
                    scene_good_reason = scene_evaluation_tag
                elif scene_evaluation_tag in BAD_REASONS:
                    scene_bad_reason = scene_evaluation_tag
            # --- End Reason Determination --- 

            final_scene = Scene(
                start_ms=current_scene_start_ms,
                end_ms=final_scene_end_ms,
                scene_id=scene_id_counter,
                description=scene_desc,
                thumbnail_path=thumbnail_path,
                scene_good_reason=scene_good_reason, # Pass determined reason
                scene_bad_reason=scene_bad_reason,   # Pass determined reason
                scene_evaluation_tag=scene_evaluation_tag # Pass the raw tag from API
            )
            scenes.append(final_scene)
            logger.info(f"  -> Added Final Scene: {final_scene}")

        self._report_progress(1.0)
        logger.info(f"Scene analysis finished. Detected {len(scenes)} scenes.")
        return scenes 