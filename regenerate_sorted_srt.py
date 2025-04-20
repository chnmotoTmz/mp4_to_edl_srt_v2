import os
import glob
import json
import argparse
import logging
import sys
from typing import List

# Add project root to path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from mp4_to_edl_srt.segment import Segment
    from mp4_to_edl_srt.srt_data import SRTData
    from mp4_to_edl_srt.timecode_utils import TimecodeConverter
except ImportError as e:
    print(f"Error importing modules: {e}. Make sure script is run from project root or PYTHONPATH is set.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def regenerate_sorted_srt(output_folder: str, output_filename: str = "combined_output_sorted.srt"):
    """
    Reads intermediate JSON data files from an output folder, combines segments,
    sorts them by absolute timecode, and writes a new combined SRT file.

    Args:
        output_folder: The folder containing the *_data.json files (can be nested).
        output_filename: The name for the regenerated sorted SRT file.
    """
    output_folder = os.path.normpath(output_folder)
    if not os.path.isdir(output_folder):
        logger.error(f"Output folder not found: {output_folder}")
        return

    # Recursively find all _data.json files
    # Use os.path.join for robust path construction
    json_pattern = os.path.join(output_folder, '**', '*_data.json')
    json_files = glob.glob(json_pattern, recursive=True)

    if not json_files:
        logger.warning(f"No '*_data.json' files found in {output_folder} or its subdirectories.")
        return

    logger.info(f"Found {len(json_files)} JSON data files. Processing...")

    all_segments: List[Segment] = []
    converter = TimecodeConverter() # Needed for SRTData and offset conversion

    for json_path in json_files:
        logger.debug(f"Reading {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract offset and segments
            metadata = data.get("metadata", {})
            # Handle potential None value from JSON before passing to Segment
            timecode_offset = metadata.get("timecode_offset")
            final_segments_data = data.get("final_segments", [])

            # Use a default offset if missing in JSON, log warning
            effective_offset = timecode_offset if timecode_offset is not None else '00:00:00:00'
            if timecode_offset is None:
                logger.warning(f"Timecode offset missing in {json_path}. Using default '00:00:00:00'.")

            if not final_segments_data:
                logger.warning(f"No final segments found in {json_path}. Skipping file.")
                continue

            # Create Segment objects with the offset
            for seg_data in final_segments_data:
                try:
                    # Ensure required fields exist, provide defaults or skip
                    start_tc = seg_data.get("start_timecode")
                    end_tc = seg_data.get("end_timecode")
                    transcription = seg_data.get("transcription", "")

                    if start_tc is None or end_tc is None:
                         logger.warning(f"Skipping segment due to missing timecode in {json_path}: {seg_data}")
                         continue

                    segment = Segment(
                        start_timecode=start_tc,
                        end_timecode=end_tc,
                        transcription=transcription,
                        scene_id=seg_data.get("scene_id"),
                        scene_description=seg_data.get("scene_description"),
                        transcription_good_reason=seg_data.get("transcription_good_reason"),
                        transcription_bad_reason=seg_data.get("transcription_bad_reason"),
                        # Pass the offset extracted (or defaulted) from the JSON's metadata
                        source_timecode_offset=effective_offset
                    )
                    all_segments.append(segment)
                except Exception as e_seg:
                     # Log specific error during segment creation
                     logger.error(f"Error creating segment from data in {json_path}. Error: {e_seg}. Data: {seg_data}")

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {json_path}. Skipping file.")
        except Exception as e_file:
            # Catch other potential errors during file processing
            logger.error(f"Error processing file {json_path}: {e_file}", exc_info=True)

    if not all_segments:
        logger.error("No valid segments could be extracted from any JSON file. Cannot generate SRT.")
        return

    logger.info(f"Total segments extracted: {len(all_segments)}. Generating sorted SRT...")

    # Create SRTData and write sorted file
    srt_data = SRTData(converter=converter)
    srt_data.segments = all_segments # Directly assign the list of segments with offsets

    output_srt_path = os.path.join(output_folder, output_filename)
    try:
        srt_data.write_to_file(output_srt_path) # This method already handles sorting by absolute time
        logger.info(f"Successfully generated sorted SRT file: {output_srt_path}")
    except Exception as e_write:
        logger.error(f"Error writing sorted SRT file: {e_write}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate a combined SRT file sorted by absolute timecode from intermediate JSON data.")
    parser.add_argument("output_folder", help="Path to the output folder containing the '*_data.json' files (searches recursively).")
    parser.add_argument("-o", "--output_filename", default="combined_output_sorted.srt", help="Filename for the regenerated sorted SRT file (default: combined_output_sorted.srt).")

    args = parser.parse_args()

    regenerate_sorted_srt(args.output_folder, args.output_filename) 