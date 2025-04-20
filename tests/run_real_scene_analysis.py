# tests/run_real_scene_analysis.py
import os
import sys
import logging
import json # Add json import

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure logging to see details from api_client etc.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from mp4_to_edl_srt.scene_analysis import SceneAnalyzer
    from mp4_to_edl_srt.api_client import GeminiClient # Needed to check availability if desired
    from mp4_to_edl_srt.timecode_utils import TimecodeConverter # Import TimecodeConverter
except ImportError as e:
    logger.error(f"必要なモジュールのインポートに失敗しました: {e}")
    logger.error("プロジェクトルートから実行しているか、PYTHONPATHを確認してください。")
    sys.exit(1)

# --- テスト設定 ---
# ↓↓↓↓↓↓ ★★★ 実際のテスト用動画ファイルへのパスに変更してください ★★★ ↓↓↓↓↓↓
# 例: TEST_VIDEO_PATH = "G:/ts/GH012777.MP4"
# 例: TEST_VIDEO_PATH = os.path.join(project_root, "sample_video.mp4") # プロジェクト内にサンプルを置く場合
TEST_VIDEO_PATH = r"G:\ts\GH012664.MP4" # Use the video file from the last JSON example
# ↑↑↑↑↑↑ ★★★ 実際のテスト用動画ファイルへのパスに変更してください ★★★ ↑↑↑↑↑↑

# サムネイル画像の出力先ディレクトリ (tests/outputs/ など)
CAPTURE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_scene_outputs")
FRAME_ANALYSIS_RATE = 30 # 何フレームごとにシーンチェンジを判定するか（仮）

# --- 実行 ---
if __name__ == "__main__":
    logger.info("--- Scene Analyzer 実動作テスト開始 ---")

    if not os.path.exists(TEST_VIDEO_PATH):
        logger.error(f"テスト動画ファイルが見つかりません: {TEST_VIDEO_PATH}")
        logger.error("スクリプト内の TEST_VIDEO_PATH を正しいパスに修正してください。")
        sys.exit(1)

    logger.info(f"テスト動画: {TEST_VIDEO_PATH}")
    logger.info(f"キャプチャ出力先: {CAPTURE_OUTPUT_DIR}")
    logger.info(f"フレーム解析レート: {FRAME_ANALYSIS_RATE}")

    try:
        # SceneAnalyzer を初期化 (GeminiClientも内部で初期化される)
        logger.info("SceneAnalyzer を初期化中...")
        analyzer = SceneAnalyzer()
        logger.info("SceneAnalyzer 初期化完了.")

        if not analyzer.gemini_client:
             logger.warning("GeminiClientが初期化されませんでした。APIキーを確認してください。説明生成はスキップされます。")

        # シーン分析を実行
        logger.info("analyze_video を実行中...")
        detected_scenes = analyzer.analyze_video(
            video_path=TEST_VIDEO_PATH,
            frame_analysis_rate=FRAME_ANALYSIS_RATE,
            capture_output_dir=CAPTURE_OUTPUT_DIR
        )
        logger.info("analyze_video 実行完了.")

        # 結果を出力
        logger.info(f"検出されたシーン数: {len(detected_scenes)}")
        if detected_scenes:
            logger.info("--- 検出されたシーン詳細 ---")
            for i, scene in enumerate(detected_scenes):
                logger.info(f"  シーン {i+1}:")
                logger.info(f"    Scene ID: {scene.scene_id}")
                # 時間情報を TimecodeConverter を使って変換する例 (もし必要なら)
                # try:
                #     from mp4_to_edl_srt.timecode_utils import TimecodeConverter
                #     converter = TimecodeConverter()
                #     start_tc = converter.ms_to_hhmmssff(scene.start_ms)
                #     end_tc = converter.ms_to_hhmmssff(scene.end_ms)
                #     logger.info(f"    期間 (ms): {scene.start_ms} - {scene.end_ms}")
                #     logger.info(f"    期間 (TC): {start_tc} - {end_tc}")
                # except ImportError:
                logger.info(f"    開始 (ms): {scene.start_ms}")
                logger.info(f"    終了 (ms): {scene.end_ms}")

                logger.info(f"    説明: {scene.description}")
                logger.info(f"    サムネイルパス: {scene.thumbnail_path}")
                # --- Add logging for the new evaluation tag --- 
                logger.info(f"    評価タグ (API Raw): {scene.scene_evaluation_tag}") 
                logger.info(f"    評価タグ (Good Reason): {scene.scene_good_reason}") 
                logger.info(f"    評価タグ (Bad Reason): {scene.scene_bad_reason}") 
                # --- End logging addition --- 
                # サムネイルファイルの存在確認
                if scene.thumbnail_path and os.path.exists(scene.thumbnail_path):
                    logger.info("      -> サムネイルファイル存在確認: OK")
                elif scene.thumbnail_path:
                    logger.warning("      -> サムネイルファイル存在確認: NG (ファイルが見つかりません)")

            logger.info("--- テスト結果出力完了 ---")
        else:
            logger.info("シーンは検出されませんでした。")

    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生しました: {e}", exc_info=True)

    logger.info("--- Scene Analyzer 実動作テスト終了 ---")

    # --- Add JSON output ---
    if detected_scenes:
        try:
            # Instantiate TimecodeConverter
            converter = TimecodeConverter()
            # Convert Scene objects to dictionaries, passing the converter
            scenes_list_of_dicts = [scene.to_dict(converter) for scene in detected_scenes]
            # Convert the list of dictionaries to a JSON string
            json_output = json.dumps(scenes_list_of_dicts, indent=4, ensure_ascii=False)
            print("\n--- JSON Output ---")
            print(json_output)
            print("--- End JSON Output ---")
        except Exception as e:
            logger.error(f"JSON出力中にエラーが発生しました: {e}", exc_info=True)
    # --- End JSON output --- 