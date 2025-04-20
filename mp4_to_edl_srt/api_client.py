import os
import json
import base64
import requests
import logging
import subprocess
import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Callable, TypeVar, Union
from dotenv import load_dotenv
from pathlib import Path
import tempfile
from PIL import Image
from faster_whisper import WhisperModel
import torch
import gc
import traceback
from datetime import datetime
from dataclasses import dataclass
import re # Import regex module

logger = logging.getLogger(__name__)

# 環境変数のロード
load_dotenv()

# faster-whisperのサポートを追加
try:
    FASTER_WHISPER_AVAILABLE = True
    logger.info("faster-whisperライブラリが利用可能です。高速モードが使用できます。")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.info("faster-whisperライブラリが見つかりません。標準モードで実行します。")
    logger.info("高速モードを使用するには次のコマンドを実行してください: pip install faster-whisper")

# AttrDictの実装
class AttrDict(dict):
    """辞書をオブジェクト風に使えるようにするクラス"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

# APIクライアントの可用性フラグ
API_CLIENT_AVAILABLE = True

# 型定義
T = TypeVar('T')


@dataclass
class GeminiResult:
    """Gemini APIの結果を格納するデータクラス"""
    scene_type: str
    location: str
    objects: List[str]
    keywords: List[str]



class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Gemini APIクライアントを初期化し、複数のキーをロード"""
        # If a specific key is provided, use only that one
        if api_key:
            self.api_keys = [api_key]
            logger.info("指定されたAPIキーを直接使用します。")
        else:
            # Load keys from environment variables in order of preference
            keys_to_try = ["GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3", "GEMINI_API_KEY4", "GEMINI_API_KEY5", "GEMINI_API_KEY6", "GEMINI_API_KEY7", "GEMINI_API_KEY"] # Add original as fallback
            self.api_keys = []
            for key_name in keys_to_try:
                key_value = os.getenv(key_name)
                if key_value:
                    self.api_keys.append(key_value)
                    logger.info(f"環境変数 {key_name} からAPIキーをロードしました")
            
            if not self.api_keys:
                 logger.error("有効なGemini APIキーが環境変数（GEMINI_API_KEY, GEMINI_API_KEY1-3）に見つかりません。")
                 raise ValueError("有効なGemini APIキーが設定されていません")

        self.current_key_index = 0
        self.api_key = self.api_keys[self.current_key_index] # Start with the first key
        logger.info(f"{len(self.api_keys)}個のAPIキーで初期化しました。Index {self.current_key_index} から開始します。")

        # APIの最新エンドポイントとモデル名 (Consider making these configurable)
        # Using v1beta as it often has newer models like 1.5 flash
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models" 
        self.model = "gemini-2.0-flash" # Use standard stable model
        self.vision_model = "gemini-2.0-flash" # Use the same for consistency (or gemini-pro-vision if needed)
        
        # リトライ設定
        self.max_retries = 3
        self.base_delay = 2  # 基本遅延（秒）
        self.max_delay = 10  # 最大遅延（秒）
        
        # API接続テスト (Uses the initial self.api_key)
        if not self._test_api_connection():
             # If the first key fails, maybe try the next one immediately?
             # Or just rely on rotation during actual calls. Let's rely on rotation for now.
             logger.warning("初期APIキーでの接続テストに失敗しました。実際のAPI呼び出し時にローテーションを試みます。")
             # We might not want to raise an error here, let the actual calls handle rotation.
             # raise ConnectionError("初期APIキーでのGemini APIへの接続に失敗しました。")
    
    def _test_api_connection(self) -> bool:
        """現在のAPIキーが有効かテストする"""
        if not self.api_key:
            logger.error("Gemini APIキーが設定されていません。")
            return False
        
        try:
            logger.info(f"Gemini API接続テスト実行中... (Key Index: {self.current_key_index})")
            # ... (rest of the test logic remains the same, using self.api_key) ...
            # Simple prompt test
            response = requests.post(
                f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                 json={"contents": [{"parts": [{"text": "Test"}]}]},
                 timeout=10 # Add timeout
            )
            
            if response and response.status_code == 200:
                logger.info(f"Gemini API接続テスト成功 (Key Index: {self.current_key_index})")
                return True
            else:
                # ... (existing error logging for test failure) ...
                 status = response.status_code if response else "不明"
                 error_text = response.text if response else "応答なし"
                 logger.error(f"Gemini API接続テスト失敗 (Key Index: {self.current_key_index}): ステータス {status}, エラー: {error_text[:200]}")
                 return False
                
        except requests.exceptions.Timeout:
            logger.error(f"Gemini API接続テスト中にタイムアウト発生 (Key Index: {self.current_key_index})")
            return False
        except Exception as e:
            logger.error(f"Gemini API接続テスト中に例外発生 (Key Index: {self.current_key_index}): {str(e)}")
            # Avoid logging full traceback for simple test? Maybe okay.
            # logger.error(traceback.format_exc()) 
            return False

    def _rotate_api_key(self) -> bool:
        """次の利用可能なAPIキーにローテーションする。もうキーがなければFalseを返す"""
        if len(self.api_keys) <= 1:
            logger.warning("キーローテーション試行: APIキーが1つしかないためローテーションできません。")
            return False # Cannot rotate if only one key
            
        prev_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_index]
        # Avoid logging the actual key value
        logger.warning(f"APIキーをローテーションしました。旧Index: {prev_index} -> 新Index: {self.current_key_index}")
        # Optional: Test the new key? Better not to avoid loops.
        return True
        
    def _retry_operation(self, operation: Callable, max_retries=3, is_vision=False):
        """操作の再試行を処理し、キーエラー時にローテーションを試みる"""
        retries = 0
        last_status_code = None
        # Track keys tried *within this specific operation attempt* to prevent infinite loops if all keys fail similarly
        keys_already_tried_indices = set() 
        
        while retries < max_retries:
            response = None # Initialize response to None at the start of each retry iteration
            # Ensure we start with the current key for this attempt
            current_active_key_index = self.current_key_index
            if current_active_key_index in keys_already_tried_indices:
                 # We have rotated back to a key already tried in this sequence, means all keys likely failed.
                 logger.error(f"全APIキー ({len(self.api_keys)}個) を試行しましたが、操作に失敗しました。試行を中止します。")
                 # Return a dummy error response or the last response? Let's return None.
                 return None 
                 
            keys_already_tried_indices.add(current_active_key_index)

            try:
                model_info = "Vision API" if is_vision else "Text API"
                logger.info(f"Gemini {model_info} リクエスト実行 (試行 {retries+1}/{max_retries}, KeyIndex: {current_active_key_index})")
                
                # The operation lambda will use the current self.api_key
                response = operation() 
                last_status_code = response.status_code if response else None
                
                if response and response.status_code == 200:
                    logger.info(f"Gemini API レスポンス成功: ステータスコード=200 (KeyIndex: {current_active_key_index})")
                    return response # Success!

                # --- Handle Errors ---
                elif response:
                    logger.error(f"Gemini API エラー: ステータスコード={response.status_code} (KeyIndex: {current_active_key_index})")
                    try:
                        error_detail = response.json() # Try to get JSON error detail
                        logger.error(f"エラー詳細: {error_detail}")
                    except json.JSONDecodeError:
                        error_detail = response.text[:200] # Get text snippet if not JSON
                        logger.error(f"エラー応答 (非JSON): {error_detail}...")
                        
                    # --- Key Rotation Logic ---
                    # Rotate on specific errors indicative of key/quota issues
                    if response.status_code in [401, 403, 429]: 
                        logger.warning(f"キー関連エラー ({response.status_code}) またはレート制限のためキーローテーションを試みます。")
                        rotated = self._rotate_api_key() # Switch self.api_key for the *next* attempt
                        if not rotated:
                             logger.error("ローテーション可能なキーが他にありません。")
                             # If rotation isn't possible, let the normal retry logic continue/fail
                        # Apply appropriate delay, especially for rate limits
                        wait_time = self.base_delay * (2 ** retries)
                        if response.status_code == 429:
                            # Use potentially longer delay for rate limiting, respecting max_delay
                            # The API response *might* include a 'Retry-After' header, but parsing that adds complexity.
                            # Let's use exponential backoff with a higher base or multiplier for 429.
                            wait_time = min(self.base_delay * (2**(retries + 1)), self.max_delay) # Increase delay more for 429
                            logger.warning(f"レート制限エラー(429)。ローテーション後/再試行前 に {wait_time:.1f} 秒待機します...")
                        else:
                            wait_time = min(wait_time, self.max_delay)
                            logger.info(f"エラー ({response.status_code})。再試行前に {wait_time:.1f} 秒待機します...")
                        time.sleep(wait_time)
                    else:
                        # For other errors, just wait and retry
                        wait_time = min(self.base_delay * (2 ** retries), self.max_delay)
                        logger.info(f"エラー ({response.status_code})。再試行前に {wait_time:.1f} 秒待機します...")
                        time.sleep(wait_time)
                else:
                    # Don't Rotate Key, just wait
                    wait_time = min(self.base_delay * (2 ** retries), self.max_delay)
                    logger.info(f"エラー ({response.status_code})。再試行前に {wait_time:.1f} 秒待機します...")
                    time.sleep(wait_time)
            
            except requests.exceptions.Timeout:
                 logger.error(f"接続タイムアウト (試行 {retries+1}/{max_retries})")
                 last_status_code = "Timeout" # Indicate timeout
                 # Try rotating key even on timeout, as it might be specific to a region/endpoint associated with the key
                 logger.warning(f"タイムアウトのためキーローテーションを試みます。")
                 rotated = self._rotate_api_key()
                 if not rotated:
                      logger.error("ローテーション可能なキーが他にありません。")
                 wait_time = min(self.base_delay * (2 ** retries), self.max_delay)
                 time.sleep(wait_time)
            except requests.exceptions.ConnectionError as e:
                logger.error(f"接続エラー (試行 {retries+1}/{max_retries}): {str(e)}")
                last_status_code = "ConnectionError"
                # Try rotating key on connection error too
                logger.warning(f"接続エラーのためキーローテーションを試みます。")
                rotated = self._rotate_api_key()
                if not rotated:
                     logger.error("ローテーション可能なキーが他にありません。")
                wait_time = min(self.base_delay * (2 ** retries), self.max_delay)
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"API呼び出し中に予期せぬ例外 (試行 {retries+1}/{max_retries}): {str(e)}")
                # Consider logging traceback only once or at higher severity
                # logger.error(traceback.format_exc()) 
                last_status_code = "Exception"
                # Don't rotate on general exceptions, just retry after delay
                wait_time = min(self.base_delay * (2 ** retries), self.max_delay)
                logger.info(f"予期せぬ例外発生。再試行前に {wait_time:.1f} 秒待機します...") # Added log for clarity
                time.sleep(wait_time)
            
            # If response is None after the try-except block (meaning operation() returned None or an exception occurred)
            if response is None:
                 logger.error(f"API呼び出し応答なし、または例外発生 (試行 {retries+1}/{max_retries})。キーローテーションを試みます。")
                 last_status_code = last_status_code or "NoResponse" # Set status if not already set by exception
                 rotated = self._rotate_api_key()
                 if not rotated:
                     logger.error("ローテーション可能なキーが他にありません。")
                 # Wait before the next retry regardless of rotation success
                 wait_time = min(self.base_delay * (2 ** retries), self.max_delay)
                 logger.info(f"再試行前に {wait_time:.1f} 秒待機します...")
                 time.sleep(wait_time)
            
            retries += 1
        
        # All retries failed
        logger.error(f"Gemini API呼び出しが{max_retries}回の再試行の末に失敗しました。最終ステータスコード: {last_status_code}")
        return None # Indicate failure

    def generate_content(self, prompt, temperature=0.7, retry_on_empty=True):
        """
        Gemini APIを使用してテキストを生成する
        """
        if not self.api_key:
            logger.warning("API_KEY未設定のためデモモードでレスポンスを返します")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            demo_result = {
                "choices": [{
                    "message": {
                        "content": f"デモレスポンス（{current_time}）: {prompt[:30]}...",
                    }
                }],
                "created": "1970-01-01T00:00:00.000000Z"  # デモモード識別用のタイムスタンプ
            }
            logger.info(f"デモモードレスポンス生成: timestamp=1970-01-01T00:00:00.000000Z")
            return AttrDict(demo_result)
        
        try:
            logger.info(f"Gemini APIコンテンツ生成: プロンプト長={len(prompt)}")
            
            response = self._retry_operation(
                lambda: requests.post(
                    f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
            )
            
            if response and response.status_code == 200:
                return response.json()
            else:
                logger.error("Gemini APIからのレスポンスが無効です")
                logger.error(f"エラー詳細: {response.status_code if response else 'なし'} - {response.text if response else 'レスポンスなし'}")
                return {"api_error": True, "error": "APIレスポンスエラー"}
                
        except Exception as e:
            logger.error(f"コンテンツ生成中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e)}
    
    def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Gemini Vision APIを使用して画像を分析する"""
        try:
            logger.info(f"Gemini Vision API画像分析: {image_path}")
            
            # 画像をbase64エンコード
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            response = self._retry_operation(
                lambda: requests.post(
                    f"{self.base_url}/{self.vision_model}:generateContent?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                            ]
                        }]
                    }
                ),
                is_vision=True
            )
            
            if response and response.status_code == 200:
                return response.json()
            else:
                logger.error("Gemini Vision APIからのレスポンスが無効です")
                logger.error(f"エラー詳細: {response.status_code if response else 'なし'} - {response.text if response else 'レスポンスなし'}")
                return {"api_error": True, "error": "画像分析APIエラー"}
                
        except Exception as e:
            logger.error(f"画像分析中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e)}
    
    def analyze_scene(self, image_path: str) -> Dict[str, Any]:
        """
        画像に基づいてシーンの説明と評価タグを生成します。

        評価タグの基準:
        - Scenic: 景色が特に素晴らしい
        - Landmark: 重要なランドマーク
        - Informative: 有用な情報（看板、地図）
        - Action: 臨場感のある活動
        - Privacy: 個人情報懸念 (顔がはっきり写っている等)
        - PoorQuality: 低品質（ブレ、ボケ、暗すぎる等）
        - Irrelevant: 無関係・撮影ミス・意図不明
        - Generic: 上記に当てはまらない一般的な光景
        """
        try:
            # Updated prompt to request both description and evaluation tag
            evaluation_tags = ["Scenic", "Landmark", "Informative", "Action", "Privacy", "PoorQuality", "Irrelevant", "Generic"]
            prompt = f"""この画像の内容を、主要な被写体や状況に焦点を当てて、日本語の1文（約100文字）で簡潔に説明してください。
さらに、以下の評価基準とタグリストに基づいて、このシーンに最も適した評価タグを1つだけ選択し、次の形式で応答してください:
説明: [生成された説明文]
評価タグ: [選択されたタグ]

評価基準:
- Scenic: 景色が特に素晴らしい
- Landmark: 重要なランドマーク
- Informative: 有用な情報（看板、地図）
- Action: 臨場感のある活動
- Privacy: 個人情報懸念 (顔がはっきり写っている等)
- PoorQuality: 低品質（ブレ、ボケ、暗すぎる等）
- Irrelevant: 無関係・撮影ミス・意図不明
- Generic: 上記に当てはまらない一般的な光景

評価タグリスト: {', '.join(evaluation_tags)}
"""
            # Call analyze_image which handles the actual API call and base64 encoding
            # analyze_image should return the raw text response from the API
            response_dict = self.analyze_image(image_path, prompt)

            # Check if analyze_image returned an error dictionary
            if "api_error" in response_dict and response_dict["api_error"]:
                logger.error(f"画像分析API呼び出し中にエラー発生: {response_dict.get('error')}")
                # Return a dictionary indicating the error, preserving the error flag
                return {"description": f"API Error: {response_dict.get('error')}", "evaluation_tag": None, "error": True}

            # If successful, extract the generated text directly
            try:
                # Assuming analyze_image response structure is consistent: candidates -> content -> parts -> text
                generated_text = response_dict.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

                if not generated_text:
                     logger.warning(f"Gemini APIから空のテキストが返されました。 Image: {image_path}")
                     # Return generic error or specific tags? Let's return generic for now.
                     return {"description": "(説明文生成スキップ)", "evaluation_tag": "Generic", "error": False}

                # --- Parse the response text to extract description and tag ---
                description = "(説明文抽出失敗)"
                evaluation_tag = "Generic" # Default tag if parsing fails
                # Corrected regex to capture description stopping before the tag or end of string
                desc_match = re.search(r"説明:\s*(.*?)(?:\n*評価タグ:|\Z)", generated_text, re.IGNORECASE | re.DOTALL)
                tag_match = re.search(r"評価タグ:\s*(\w+)", generated_text, re.IGNORECASE)

                if desc_match:
                    description = desc_match.group(1).strip().replace('\n', ' ')[:150] # Clean and limit
                else:
                     logger.warning(f"応答テキストから「説明:」を抽出できませんでした。 Text: {generated_text[:100]}...")
                     # Use the whole text as description as a fallback, maybe?
                     description = generated_text.strip().replace('\n', ' ')[:150]

                if tag_match:
                    parsed_tag = tag_match.group(1).strip()
                    # Validate if the parsed tag is in our expected list
                    if parsed_tag in evaluation_tags:
                        evaluation_tag = parsed_tag
                    else:
                        logger.warning(f"APIが予期しない評価タグを返しました: '{parsed_tag}'. 'Generic'を使用します。")
                        evaluation_tag = "Generic" # Fallback to Generic
                else:
                     logger.warning(f"応答テキストから「評価タグ:」を抽出できませんでした。 Text: {generated_text[:100]}... 'Generic'を使用します。")
                     evaluation_tag = "Generic" # Fallback to Generic

                logger.info(f"シーン分析成功: Desc='{description[:50]}...', Tag='{evaluation_tag}'")
                return {"description": description, "evaluation_tag": evaluation_tag, "error": False}
                # --- End Parsing ---

            except (IndexError, KeyError, AttributeError) as e:
                # This catches errors in accessing parts of the 'response_dict' from analyze_image
                logger.error(f"シーン分析レスポンスの解析に失敗: {str(e)}")
                logger.error(f"応答辞書構造: {response_dict}") # Log the structure for debugging
                logger.error(traceback.format_exc())
                return {"description": "Response parsing error.", "evaluation_tag": None, "error": True}

        except Exception as e:
            logger.error(f"シーン分析中に予期せぬ例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"description": f"Unexpected error: {str(e)}", "evaluation_tag": None, "error": True}
    
    def analyze_scene_context(self, images: List[str], transcript: str = "") -> Dict[str, Any]:
        """複数の画像とトランスクリプトから文脈を分析する"""
        try:
            # 最大3枚の画像を選択
            selected_images = images[:3]
            logger.info(f"シーン文脈分析: {len(selected_images)}枚の画像, トランスクリプト長={len(transcript)}")
            
            # 最初の画像を分析
            if selected_images:
                result = self.analyze_scene(selected_images[0])
                
                # エラーチェック
                if "api_error" in result and result.get("api_error") == True:
                    logger.error("シーン文脈分析のための最初の画像分析に失敗")
                    return {"api_error": True, "error": result.get("error", "シーン分析エラー")}
                
                # 追加のコンテキスト分析（複数画像がある場合）
                if len(selected_images) > 1 and transcript:
                    # 複数の画像を使って文脈を強化...（実装はAPIの機能によって異なる）
                    pass
                
                return result
            else:
                logger.error("シーン文脈分析に画像が提供されていません")
                return {"api_error": True, "error": "画像なしエラー", "scene_type": "不明", "keywords": ["エラー"]}
                
        except Exception as e:
            logger.error(f"シーン文脈分析中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e), "scene_type": "不明", "keywords": ["エラー"]}
    
    def generate_editing_suggestions(self, scenes_data: List[Dict], full_transcript: str) -> Dict[str, Any]:
        """シーンデータとトランスクリプトからビデオ編集の提案を生成する"""
        try:
            scenes_str = json.dumps(scenes_data, ensure_ascii=False)
            # トランスクリプトが長すぎる場合は切り詰める
            max_transcript_len = 4000
            if len(full_transcript) > max_transcript_len:
                logger.warning(f"トランスクリプトが長すぎるため切り詰めます ({len(full_transcript)} -> {max_transcript_len}文字)")
                transcript_summary = full_transcript[:max_transcript_len] + "..."
            else:
                transcript_summary = full_transcript
            
            prompt = f"""
            以下のビデオシーンデータとトランスクリプトに基づいて、ビデオの編集方法の提案を行ってください。
            結果はJSON形式で提供してください:
            
            シーンデータ: {scenes_str}
            
            トランスクリプト: {transcript_summary}
            
            以下のJSON形式で返してください:
            {{
              "title": "ビデオのタイトル案",
              "overview": "ビデオの概要（200文字以内）",
              "highlights": ["重要なハイライトシーン（最大5つ）"],
              "suggested_clips": [
                {{
                  "start_time": "開始時間（秒）",
                  "end_time": "終了時間（秒）",
                  "description": "クリップの説明"
                }}
              ],
              "editing_notes": ["編集上の注意点"]
            }}
            """
            
            response = self.generate_content(prompt)
            
            if "api_error" in response:
                logger.error("編集提案の生成に失敗")
                return {"api_error": True, "error": response.get("error", "生成エラー"), "title": "API接続エラー", "overview": "Gemini APIとの接続に問題が発生しました。APIキーとネットワーク接続を確認してください。"}
            
            # レスポンスからJSONを抽出
            try:
                text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                # テキストからJSONブロックを抽出
                json_str = text
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_str)
                # タイムスタンプを追加
                result["filming_date"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                return result
                
            except Exception as e:
                logger.error(f"編集提案のJSON解析に失敗: {str(e)}")
                logger.error(f"原文: {text if 'text' in locals() else 'テキストなし'}")
                logger.error(traceback.format_exc())
                return {"api_error": True, "error": "JSON解析エラー", "title": "解析エラー", "overview": "レスポンスの解析中にエラーが発生しました。", "filming_date": "1970-01-01T00:00:00.000000Z"}
                
        except Exception as e:
            logger.error(f"編集提案生成中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            # デモモードで使用するフォールバック
            return {
                "api_error": True, 
                "error": str(e), 
                "title": "処理エラー", 
                "overview": f"ビデオ分析中にエラーが発生しました: {str(e)}", 
                "filming_date": "1970-01-01T00:00:00.000000Z"
            }
    
    def generate_video_summary(self, scenes: List[Dict], transcript: str) -> Dict[str, Any]:
        """動画全体のサマリーを生成"""
        try:
            # シーンとトランスクリプトをJSON文字列に変換
            scenes_str = json.dumps(scenes, ensure_ascii=False)
            
            # トランスクリプトが長すぎる場合は切り詰める
            max_transcript_len = 4000
            if len(transcript) > max_transcript_len:
                logger.warning(f"トランスクリプトが長すぎるため切り詰めます ({len(transcript)} -> {max_transcript_len}文字)")
                transcript_summary = transcript[:max_transcript_len] + "..."
            else:
                transcript_summary = transcript
            
            prompt = f"""
            以下のビデオシーンデータとトランスクリプトに基づいて、ビデオの要約を行ってください。
            
            シーンデータ: {scenes_str}
            
            トランスクリプト: {transcript_summary}
            
            以下のJSON形式で返してください:
            {{
              "title": "ビデオのタイトル",
              "overview": "ビデオの概要（200文字以内）",
              "concepts": ["ビデオのコンセプト/テーマ（最大5つ）"],
              "key_moments": [
                {{
                  "time": "時間（秒）",
                  "description": "重要な瞬間の説明"
                }}
              ],
              "filming_date": "撮影日時の推測（ISO 8601形式）",
              "filming_location": "撮影場所の推測"
            }}
            """
            
            response = self._retry_operation(
                lambda: requests.post(
                    f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
            )
            
            if not response or response.status_code != 200:
                logger.error("ビデオ要約の生成に失敗")
                logger.error(f"エラー詳細: {response.status_code if response else 'なし'} - {response.text if response else 'レスポンスなし'}")
                return {"api_error": True, "title": "API接続エラー", "overview": "Gemini APIとの接続に問題が発生しました。APIキーとネットワーク接続を確認してください。", "filming_date": "1970-01-01T00:00:00.000000Z"}
            
            # レスポンスからJSONを抽出
            try:
                data = response.json()
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                
                # テキストからJSONブロックを抽出
                json_str = text
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_str)
                logger.info(f"ビデオ要約生成成功: {result.get('title', 'タイトルなし')}")
                return result
                
            except Exception as e:
                logger.error(f"ビデオ要約のJSON解析に失敗: {str(e)}")
                logger.error(f"原文: {text if 'text' in locals() else 'テキストなし'}")
                logger.error(traceback.format_exc())
                return {"api_error": True, "error": "JSON解析エラー", "title": "解析エラー", "overview": "レスポンスの解析中にエラーが発生しました。", "filming_date": "1970-01-01T00:00:00.000000Z"}
                
        except Exception as e:
            logger.error(f"ビデオ要約生成中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e), "title": "エラー", "overview": "処理中にエラーが発生しました。", "filming_date": "1970-01-01T00:00:00.000000Z"}

    # --- Add New Method for Transcription Quality Evaluation --- 
    def evaluate_transcription_quality(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """文字起こしテキストの品質をGemini APIを使用して評価する"""
        if not text:
            logger.debug("空のテキストが渡されたため、文字起こし評価をスキップします。")
            return None, "Inaccurate" # Return None, Inaccurate for empty text directly

        prompt = f"""
Evaluate the quality of the following Japanese transcription segment: "{text}"

Prioritize identifying potential issues like misrecognition or typos.

Consider aspects like:
- Is it descriptive or emotional? (good_reason: "Descriptive" or "Emotional")
- Does it contain filler words or hesitations? (bad_reason: "Filler")
- Does it seem repetitive? (bad_reason: "Repetitive")
- Does it seem inaccurate (e.g., noise markers like [音楽], ellipses ..., empty)? (bad_reason: "Inaccurate" or "Noise")
- Is there potential misrecognition or typo (e.g., contextually strange words, unnatural phrasing)? (bad_reason: "Misrecognition" or "Typo")
- Is it conversationally incoherent or completely out of context? (bad_reason: "Incoherent")
- Is it just a simple, short, neutral phrase? (reasons are null)

Respond ONLY with a JSON object containing two keys: "good_reason" and "bad_reason". The values should be the appropriate tag (string) or null if no reason applies. Return ONLY the JSON object.

Example 1:
Input: "うーん、まあ、そうですね..."
Output: {{"good_reason": null, "bad_reason": "Filler"}}

Example 2:
Input: "わー、すごく綺麗！感動しました！"
Output: {{"good_reason": "Emotional", "bad_reason": null}}

Example 3:
Input: "えーと、えーと、えーと、"
Output: {{"good_reason": null, "bad_reason": "Repetitive"}}

Example 4:
Input: "確認します。"
Output: {{"good_reason": null, "bad_reason": null}}

Example 5:
Input: "[音楽]"
Output: {{"good_reason": null, "bad_reason": "Noise"}}

Example 6:
Input: "これはペンですん。"
Output: {{"good_reason": null, "bad_reason": "Typo"}}

Example 7:
Input: "空を飛ぶ象を見た。"
Output: {{"good_reason": null, "bad_reason": "Misrecognition"}}

Example 8:
Input: "緑色のアイデアが眠る。"
Output: {{"good_reason": null, "bad_reason": "Incoherent"}}
"""
        # Use lambda for the API call, similar to other methods
        response = self._retry_operation(
            lambda: requests.post(
                f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=15 # Slightly longer timeout for potentially complex evaluation
            ),
            is_vision=False
        )

        content = "" # Initialize content to avoid potential UnboundLocalError in except block
        if response:
            try:
                result_json = response.json()
                # Extract text from the typical Gemini response structure
                content = result_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                
                # Clean the response text - remove potential markdown/json block quotes
                cleaned_content = re.sub(r'```json\n?|```', '', content).strip()
                
                logger.debug(f"Gemini Transcription Eval Raw Response: {cleaned_content}")
                
                # Parse the JSON string from the response
                eval_data = json.loads(cleaned_content)
                
                good_reason = eval_data.get("good_reason")
                bad_reason = eval_data.get("bad_reason")
                
                # Ensure returned values are strings or None
                good_reason = str(good_reason) if good_reason is not None else None
                bad_reason = str(bad_reason) if bad_reason is not None else None
                
                logger.info(f"Gemini Transcription Eval Result: Good='{good_reason}', Bad='{bad_reason}' for text: '{text[:30]}...'")
                return good_reason, bad_reason
                
            except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
                logger.error(f"Gemini文字起こし評価応答の解析に失敗: {e}. Raw response: {content[:200]}")
                return None, None # Failed to parse, return no reasons
        else:
            logger.error("Gemini文字起こし評価API呼び出しに失敗しました。リトライ上限超過またはその他のエラー。")
            return None, None # API call failed after retries

    # --- End New Method ---

    # --- Add New Method for Batch Transcription Quality Evaluation --- 
    def evaluate_transcription_batch(self, texts: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
        """文字起こしテキストのリストの品質をGemini APIを使用して一括評価する"""
        if not texts:
            logger.debug("空のテキストリストが渡されたため、バッチ文字起こし評価をスキップします。")
            return []

        # Format the list of texts for the prompt (e.g., numbered list)
        formatted_texts = "\n".join([f"{i+1}. \"{text}\"" for i, text in enumerate(texts)])
        
        prompt = f"""
Evaluate the quality of the following list of Japanese transcription segments. For each segment, determine a 'good_reason' and a 'bad_reason' based on the criteria below. Respond ONLY with a single JSON array (list) where each element is an object {{"good_reason": TAG_OR_NULL, "bad_reason": TAG_OR_NULL}}. The order of elements in the response array MUST match the order of the input segments.

Input Segments:
{formatted_texts}

Evaluation Criteria & Tags:
- Descriptive or Emotional? (good_reason: "Descriptive" or "Emotional")
- Contains filler words or hesitations? (bad_reason: "Filler")
- Seems repetitive? (bad_reason: "Repetitive")
- Seems inaccurate (e.g., noise markers like [音楽], ellipses ..., empty)? (bad_reason: "Inaccurate" or "Noise")
- Potential misrecognition or typo (e.g., contextually strange words, unnatural phrasing)? (bad_reason: "Misrecognition" or "Typo")
- Conversationally incoherent or completely out of context? (bad_reason: "Incoherent")
- Simple, short, neutral phrase? (good_reason: null, bad_reason: null)

Output Format Example (for 3 input segments):
[
  {{"good_reason": null, "bad_reason": "Filler"}},
  {{"good_reason": "Emotional", "bad_reason": null}},
  {{"good_reason": null, "bad_reason": null}}
]

Respond ONLY with the JSON array.
"""
        # Use lambda for the API call
        response = self._retry_operation(
            lambda: requests.post(
                f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30 # Increase timeout slightly for potentially longer processing
            ),
            is_vision=False
        )

        results: List[Tuple[Optional[str], Optional[str]]] = [(None, None)] * len(texts) # Initialize with defaults
        content = "" # Initialize content

        if response:
            try:
                result_json = response.json()
                # Extract text from the typical Gemini response structure
                content = result_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                
                # Clean the response text - remove potential markdown/json block quotes
                # And find the JSON array within the text
                match = re.search(r'\[\s*(.*?)\s*\]', content, re.DOTALL) # Find content within [] brackets
                if match:
                    json_str = match.group(0) # Get the full matched string including brackets
                    logger.debug(f"Gemini Batch Transcription Eval Raw JSON: {json_str}")
                    eval_data_list = json.loads(json_str)
                    
                    if isinstance(eval_data_list, list) and len(eval_data_list) == len(texts):
                        parsed_results = []
                        for i, eval_data in enumerate(eval_data_list):
                            if isinstance(eval_data, dict):
                                good_reason = eval_data.get("good_reason")
                                bad_reason = eval_data.get("bad_reason")
                                good_reason = str(good_reason) if good_reason is not None else None
                                bad_reason = str(bad_reason) if bad_reason is not None else None
                                parsed_results.append((good_reason, bad_reason))
                            else:
                                logger.warning(f"バッチ評価結果の要素 {i} が辞書形式ではありません: {eval_data}")
                                parsed_results.append((None, None)) # Append default on format error
                        results = parsed_results # Assign successfully parsed results
                        logger.info(f"Gemini Batch Transcription Eval successful for {len(results)} segments.")
                    else:
                         logger.error(f"Geminiバッチ評価応答のリスト長が入力と一致しません ({len(eval_data_list)} vs {len(texts)}) またはリスト形式ではありません。")
                else:
                    logger.error(f"Geminiバッチ評価応答からJSON配列が見つかりませんでした。 Raw response: {content[:500]}")

            except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
                logger.error(f"Geminiバッチ文字起こし評価応答の解析に失敗: {e}. Raw response: {content[:500]}")
                # Results remain as default (None, None)
            except Exception as e:
                logger.error(f"Geminiバッチ評価処理中に予期せぬエラー: {e}. Raw response: {content[:500]}")
                # Results remain as default (None, None)
        else:
            logger.error("Geminiバッチ文字起こし評価API呼び出しに失敗しました。リトライ上限超過またはその他のエラー。")
            # Results remain as default (None, None)

        return results
    # --- End Batch Method ---
