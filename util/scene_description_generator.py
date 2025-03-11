import argparse
import re
import time
import os
from typing import List, Dict, Tuple
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
import datetime
import requests
import json

# 設定定数
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCztyJmLgpfj9ko5G5Y48AxnzP2Nj78xYM")  # 環境変数からAPIキーを取得
MODEL = "gemini-1.5-flash-8b"
SCENE_DURATION = 10  # シーンの長さ（秒）- 10秒に変更
MAX_RETRIES = 3  # API呼び出しの最大リトライ回数
RETRY_DELAY = 2  # リトライ間の遅延（秒）
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "0c939952353f8ab495951a1420a13b1f5b08001d")  # 環境変数からAPIキーを取得
DISABLE_WEB_SEARCH = os.environ.get("DISABLE_WEB_SEARCH", "0") == "1"  # ウェブ検索を無効化するかどうか

# 一時的なシナリオファイル名（古い関数との互換性用）
TEMP_SCENARIO_FILE = "temp_scenario.md"

class SRTBlock:
    """SRTファイル内の1つの字幕ブロックを表します。"""
    
    def __init__(self, index: int, timestamp: str, text: str):
        """
        SRTブロックを初期化します。
        
        Args:
            index: 字幕のインデックス
            timestamp: 字幕のタイムスタンプ（"00:00:00,000 --> 00:00:05,000"の形式）
            text: 字幕のテキスト
        """
        self.index = index
        self.timestamp = timestamp
        self.text = text
        
        # タイムスタンプから開始時間と終了時間を秒単位で抽出
        timestamp_parts = timestamp.split(" --> ")
        self.start_time = self._parse_timestamp(timestamp_parts[0])
        self.end_time = self._parse_timestamp(timestamp_parts[1])
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """タイムスタンプを秒単位の浮動小数点数に変換します。"""
        h, m, s = timestamp_str.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def get_formatted_timestamp(self, seconds: float) -> str:
        """秒数をSRT形式のタイムスタンプに変換します。"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    
    def __str__(self) -> str:
        """SRTブロックの文字列表現を返します。"""
        return f"{self.index}\n{self.timestamp}\n{self.text}"

def setup_gemini_api():
    """Gemini APIの設定を行います。"""
    genai.configure(api_key=API_KEY)

def parse_srt_file(file_path: str) -> List[SRTBlock]:
    """
    SRTファイルを解析してSRTBlockのリストを返します。
    
    Args:
        file_path: SRTファイルのパス
        
    Returns:
        SRTBlockのリスト
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # SRTブロックを分割（空行で区切られている）
    blocks = content.strip().split("\n\n")
    srt_blocks = []
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        
        try:
            index = int(lines[0])
            timestamp = lines[1]
            text = '\n'.join(lines[2:])
            srt_blocks.append(SRTBlock(index, timestamp, text))
        except ValueError:
            # インデックスが整数でない場合はスキップ
            continue
    
    return srt_blocks

def create_scene_blocks(duration: int, total_duration: int) -> List[Tuple[float, float]]:
    """
    指定した時間間隔でシーンを分割します。
    
    Args:
        duration: 各シーンの長さ（秒）
        total_duration: 全体の長さ（秒）
        
    Returns:
        (開始時間, 終了時間)のタプルのリスト
    """
    scenes = []
    start_time = 0.0
    
    while start_time < total_duration:
        end_time = min(start_time + duration, total_duration)
        scenes.append((start_time, end_time))
        start_time = end_time
    
    return scenes

def extract_keywords_from_scene(srt_blocks: List[SRTBlock], start_time: float, end_time: float) -> str:
    """
    指定した時間範囲内の字幕から重要なキーワードを抽出します。
    
    Args:
        srt_blocks: SRTBlockのリスト
        start_time: シーンの開始時間（秒）
        end_time: シーンの終了時間（秒）
        
    Returns:
        抽出されたキーワード
    """
    # 指定した時間範囲内の字幕を収集
    scene_blocks = [block for block in srt_blocks if 
                   (block.start_time <= end_time and block.end_time >= start_time)]
    
    # すべてのテキストを結合
    combined_text = " ".join([block.text for block in scene_blocks])
    
    # 重要なキーワードを抽出
    # 観光地特有の用語リスト
    tourism_terms = [
        "温泉", "外湯", "旅館", "観光", "名所", "グルメ", "料理", "名物",
        "カニ", "海鮮", "和食", "会席", "バイキング",
        "ロープウェイ", "寺院", "神社", "お土産", "特産品"
    ]
    
    # 交通関連の用語リスト
    transport_terms = [
        "特急", "列車", "電車", "鉄道", "駅", "乗換", "プラットフォーム",
        "時刻表", "路線", "切符", "指定席", "自由席"
    ]
    
    # 一般的な助詞や接続詞などのストップワード
    stop_words = [
        "は", "が", "の", "に", "を", "で", "と", "も", "や", "から", "まで", "より", 
        "ね", "よ", "た", "だ", "です", "ます", "ました", "ません", "ない", "する", 
        "した", "して", "しない", "ある", "あり", "ない", "なく", "なり", "なる"
    ]
    
    # 特徴的なパターンを検出
    special_patterns = [
        (r'[^\s]{1,10}温泉', 5),        # 〜温泉
        (r'[^\s]{1,10}駅', 5),          # 〜駅
        (r'[^\s]{1,10}寺', 4),          # 〜寺
        (r'[^\s]{1,10}神社', 4),        # 〜神社
        (r'[^\s]{1,10}公園', 4),        # 〜公園
        (r'[^\s]{1,10}橋', 3),          # 〜橋
        (r'[^\s]{1,10}通り', 3),        # 〜通り
        (r'[^\s]{1,10}街', 3),          # 〜街
        (r'特急[^\s]{1,10}', 5),        # 特急〜
        (r'[^\s]{1,10}号', 5),          # 〜号（列車名）
    ]
    
    # キーワードとウェイトを収集
    keywords_weight = {}
    
    # 特徴的なパターンからキーワードを抽出
    for pattern, weight in special_patterns:
        matches = re.findall(pattern, combined_text)
        for match in matches:
            keywords_weight[match] = weight
    
    # 観光地特有の用語を検出
    for term in tourism_terms:
        if term in combined_text:
            keywords_weight[term] = 4
    
    # 交通関連の用語を検出
    for term in transport_terms:
        if term in combined_text:
            keywords_weight[term] = 4
    
    # 固有名詞や一般的な名詞を抽出
    words = re.findall(r'[一-龠々〆ヵヶ]{2,}', combined_text)
    for word in words:
        if word not in stop_words and len(word) >= 2:
            if word not in keywords_weight:
                keywords_weight[word] = 1
    
    # ウェイトでソートして上位を選択
    sorted_keywords = sorted(keywords_weight.items(), key=lambda x: (-x[1], x[0]))
    top_keywords = [keyword for keyword, _ in sorted_keywords[:5]]  # 上位5つを選択
    
    print(f"時間帯 {format_time(start_time)} - {format_time(end_time)} の抽出キーワード: {top_keywords}")
    return " ".join(top_keywords)

def format_time(seconds: float) -> str:
    """秒数を読みやすい形式に変換します。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def search_web(query: str, num_results: int = 3) -> str:
    """
    Serper APIを使用してWeb検索を行い、結果を返します。
    
    Args:
        query: 検索クエリ
        num_results: 取得する検索結果の数
        
    Returns:
        検索結果のテキスト
    """
    # ウェブ検索が無効化されている場合は空の文字列を返す
    if DISABLE_WEB_SEARCH:
        print(f"ウェブ検索は無効化されています。検索クエリ '{query}' はスキップされました。")
        return "ウェブ検索は無効化されています。"
    
    print(f"\n=== Serper API リクエスト ===")
    print(f"検索クエリ: {query}")
    print(f"結果数: {num_results}")
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "gl": "jp",
        "hl": "ja",
        "num": num_results
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # APIキーの一部を表示（セキュリティのため完全なキーは表示しない）
    display_key = SERPER_API_KEY[:4] + "..." + SERPER_API_KEY[-4:] if len(SERPER_API_KEY) > 8 else "設定されていません"
    print(f"API Key: {display_key}")
    print(f"リクエスト URL: {url}")
    print(f"リクエスト ペイロード: {payload}")
    
    try:
        print(f"\nリクエスト送信中...")
        start_time = time.time()
        response = requests.request("POST", url, headers=headers, data=payload)
        end_time = time.time()
        
        print(f"レスポンス受信: {response.status_code} ({(end_time - start_time):.2f}秒)")
        data = response.json()
        
        if response.status_code != 200:
            print(f"検索APIエラー: ステータスコード {response.status_code}")
            print(f"エラーレスポンス: {data}")
            return f"検索結果を取得できませんでした。エラー: {response.status_code}"
        
        print(f"\n=== Serper API レスポンス ===")
        # レスポンスの概要を表示
        result_count = len(data.get("organic", [])) 
        print(f"検索結果数: {result_count}")
        
        if "organic" not in data or len(data["organic"]) == 0:
            print("検索結果がありません")
            return "関連する検索結果がありませんでした。"
        
        results = []
        print(f"\n検索結果概要:")
        for i, item in enumerate(data["organic"][:num_results]):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            print(f"  結果 {i+1}: {title[:40]}{'...' if len(title) > 40 else ''}")
            results.append(f"タイトル: {title}\n概要: {snippet}")
        
        print(f"\n=== 検索処理完了 ===")
        return "\n\n".join(results)
    except Exception as e:
        print(f"検索中にエラーが発生しました: {e}")
        import traceback
        print(traceback.format_exc())
        return f"検索処理中にエラーが発生しました: {e}"

def match_scene_with_scenario(keywords: str, scenario: str) -> str:
    """
    キーワードからシナリオ内の最も関連性の高いシーンを見つけます。
    
    Args:
        keywords: シーンから抽出されたキーワード
        scenario: 全体のシナリオのテキスト
        
    Returns:
        関連するシナリオのセクション
    """
    # シナリオをセクションに分割（## で始まる行で区切る）
    sections = re.split(r'\n##\s+', scenario)
    
    # 最初のセクションが ##なしで始まる場合の処理
    if not sections[0].startswith('##'):
        sections[0] = sections[0].lstrip()
    else:
        sections[0] = '## ' + sections[0]
    
    best_section = ""
    best_score = 0
    
    # キーワードをスペースで分割
    keyword_list = keywords.split()
    
    for section in sections:
        score = 0
        for keyword in keyword_list:
            if keyword in section:
                score += 1
        
        # スコアが高い（キーワードとの一致度が高い）セクションを選択
        if score > best_score:
            best_score = score
            best_section = section
    
    # マッチするセクションがなければ空文字を返す
    if best_score == 0:
        return ""
    
    # セクション名（見出し）を抽出
    section_title = re.search(r'##\s+(.*?)$', best_section, re.MULTILINE)
    if section_title:
        section_name = section_title.group(1)
    else:
        section_name = "不明なセクション"
    
    print(f"マッチしたシナリオセクション: {section_name} (一致スコア: {best_score})")
    return best_section

def generate_caption_prompt(keywords: str, scenario_section: str, search_results: str) -> str:
    """
    テロップ生成のためのプロンプトを作成します。
    
    Args:
        keywords: シーンから抽出されたキーワード
        scenario_section: 関連するシナリオのセクション
        search_results: Web検索の結果
        
    Returns:
        生成されたプロンプト
    """
    prompt = """
あなたは旅行ドキュメンタリーの字幕・テロップライターです。以下の情報に基づいて、魅力的な字幕テキストを1つだけ生成してください：

【キーワード】
{}

【シナリオセクション】
{}

【Web検索結果】
{}

以下の点に注意して、最大60文字程度の簡潔で魅力的な字幕テキストを1つだけ作成してください：
1. 視聴者に見どころや旅の魅力を伝える内容にする
2. シナリオセクションの内容と関連するWeb検索情報を活用する
3. ナレーターが説明している感じの文体にする
4. テロップの前に「テロップ例：」などの表現を入れないこと
5. マークダウン記法（**など）を使用しないこと
6. 改行を入れないこと
7. タイムスタンプを入れないこと
8. 複数の選択肢を提示しないこと
9. 短くて簡潔なテロップにすること

【生成テロップ】
""".format(keywords, scenario_section, search_results)
    
    return prompt

def call_gemini_api(prompt: str) -> str:
    """
    Gemini APIを呼び出して、生成テキストを取得します。
    
    Args:
        prompt: APIに送信するプロンプト
        
    Returns:
        生成されたテキスト
    """
    try:
        # Gemini APIの設定
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL)
        
        # プロンプトを送信
        response = model.generate_content(prompt)
        
        # レスポンスからテキストを抽出
        text = response.text
        
        # APIのレートリミットに達した場合のリトライ
        retries = 0
        while not text and retries < MAX_RETRIES:
            print(f"APIレスポンスが空でした。{RETRY_DELAY}秒後にリトライします...")
            time.sleep(RETRY_DELAY)
            response = model.generate_content(prompt)
            text = response.text
            retries += 1
        
        if not text:
            return "APIからのレスポンスが取得できませんでした。"
        
        return text
    
    except ResourceExhausted:
        print("APIの利用制限に達しました。しばらく待ってから再試行してください。")
        return "APIの利用制限に達しました。"
    
    except ServiceUnavailable:
        print("サービスが一時的に利用できません。しばらく待ってから再試行してください。")
        return "サービスが一時的に利用できません。"
    
    except Exception as e:
        print(f"API呼び出し中にエラーが発生しました: {e}")
        return f"エラーが発生しました: {str(e)}"

def create_new_caption_block(index: int, start_time: float, end_time: float, text: str) -> SRTBlock:
    """
    新しいテロップブロックを作成します。
    
    Args:
        index: ブロックのインデックス
        start_time: 開始時間（秒）
        end_time: 終了時間（秒）
        text: テロップのテキスト
        
    Returns:
        新しいSRTBlock
    """
    # タイムスタンプをフォーマット
    start_formatted = format_srt_timestamp(start_time)
    end_formatted = format_srt_timestamp(end_time)
    timestamp = f"{start_formatted} --> {end_formatted}"
    
    return SRTBlock(index, timestamp, text)

def format_srt_timestamp(seconds: float) -> str:
    """秒数をSRT形式のタイムスタンプに変換します。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def generate_opening_prompt(scenario_text: str, search_results: str) -> str:
    """
    オープニングテロップを生成するためのプロンプトを作成します。
    
    Args:
        scenario_text: シナリオのテキスト
        search_results: Web検索の結果
        
    Returns:
        生成されたプロンプト
    """
    prompt = """
あなたはテレビ番組や映画のオープニングテロップを作成するプロフェッショナルです。
以下の情報に基づいて、魅力的なオープニングテロップを生成してください：

【シナリオ概要】
{}

【Web検索結果】
{}

以下の点に注意して、番組/映画のオープニングとして適切な字幕テキストを作成してください：
1. テレビ番組や映画のプロローグのような簡潔で印象的な内容にする
2. 旅の魅力や見どころを視聴者に伝える内容を含める
3. 全体で100〜150文字程度にまとめる
4. 改行は必要最小限にする（2〜3行程度に収める）
5. 「ナレーション：」などの表記は入れない
6. 旅番組の導入部らしい語り口調にする

【オープニングテロップ】
""".format(scenario_text[:500], search_results)  # シナリオは長すぎる場合があるため先頭500文字だけ使用
    
    return prompt

def generate_ending_prompt(scenario_text: str, search_results: str) -> str:
    """
    エンディングテロップ（エンドロール）を生成するためのプロンプトを作成します。
    
    Args:
        scenario_text: シナリオのテキスト
        search_results: Web検索の結果
        
    Returns:
        生成されたプロンプト
    """
    prompt = """
あなはテレビ番組や映画のエンディング（エンドロール）を作成するプロフェッショナルです。
以下の情報に基づいて、映画のようなエンドロールテキストを生成してください：

【シナリオ概要】
{}

【Web検索結果】
{}

以下の点に注意して、番組/映画のエンドロールとして適切な字幕テキストを作成してください：
1. 番組制作情報（制作年、制作協力など）を含める
2. 撮影地や協力施設への謝辞を含める
3. 全体で150〜200文字程度にまとめる
4. 適切な場所で改行を入れて、エンドロールらしい体裁にする
5. 実際のテレビ番組や映画のエンドロールを参考にする
6. 旅行関連の情報（観光協会、交通機関など）への謝辞も含める
7. 完全にフィクショナルな内容ではなく、シナリオと検索結果に基づく現実的な内容にする

【エンドロールテキスト】
""".format(scenario_text[:500], search_results)
    
    return prompt

def generate_caption_with_gemini(text_in_scene: str, scene_start: float, scene_end: float, scenario_text: str = "", search_result: str = "") -> str:
    """
    Gemini AIを使用してシーン説明を生成します。
    
    Args:
        text_in_scene: シーン内の字幕テキスト
        scene_start: シーンの開始時間（秒）
        scene_end: シーンの終了時間（秒）
        scenario_text: シナリオのテキスト（あれば）
        search_result: Web検索の結果（あれば）
        
    Returns:
        生成されたシーン説明
    """
    setup_gemini_api()
    
    start_time_formatted = format_time(scene_start)
    end_time_formatted = format_time(scene_end)
    
    # 入力情報をログとして表示
    print(f"\n時間帯 {start_time_formatted} - {end_time_formatted} の字幕:")
    print(text_in_scene)
    
    # シナリオがある場合
    if scenario_text:
        print("\nシナリオ情報:")
        print(scenario_text[:300] + "..." if len(scenario_text) > 300 else scenario_text)
    
    # 検索結果がある場合かつウェブ検索が有効な場合に表示
    if search_result and not DISABLE_WEB_SEARCH:
        print("\n関連する検索結果:")
        print(search_result[:300] + "..." if len(search_result) > 300 else search_result)
    
    # プロンプトの作成
    prompt = f"""
時間帯 {start_time_formatted} から {end_time_formatted} までのシーンの説明を作成してください。
以下の情報に基づいてください:

### 字幕テキスト
{text_in_scene}

"""

    # シナリオがある場合はプロンプトに追加
    if scenario_text:
        prompt += f"""
### シナリオ情報
{scenario_text}

"""

    # ウェブ検索が有効で検索結果がある場合はプロンプトに追加
    if search_result and not DISABLE_WEB_SEARCH:
        prompt += f"""
### 関連する情報
{search_result}

"""

    # 指示と出力フォーマットについての説明
    prompt += """
上記の情報に基づいて、このシーンで何が起きているかを簡潔に説明してください。
映像には何が写っているかを想像し、それをシーン説明として表現してください。

注意事項:
- 説明は50文字以内で簡潔にまとめてください
- 先頭に【シーン】と付けてください
- 字幕の内容をそのまま繰り返さないでください
- 検索情報をそのまま引用しないでください
- 時間情報を出力に含めないでください
- 客観的な説明を心がけ、「～と思われる」などの主観的な表現は避けてください

出力例:
【シーン】バスは雪道をゆっくり進んでいる
"""

    print(f"\n=== Gemini API リクエスト ===")
    print(f"モデル: {MODEL}")
    # APIキーの一部を表示（セキュリティのため完全なキーは表示しない）
    display_key = API_KEY[:4] + "..." + API_KEY[-4:] if len(API_KEY) > 8 else "設定されていません"
    print(f"API Key: {display_key}")
    print(f"プロンプト長: {len(prompt)} 文字")
    print(f"プロンプト内容の概要: 時間帯 {start_time_formatted} - {end_time_formatted} のシーン説明を生成")

    # Gemini API呼び出し関数のリトライ機能
    def call_api_with_retry(prompt: str) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"\nGemini APIにリクエスト送信中... (試行 {attempt+1}/{MAX_RETRIES})")
                start_time = time.time()
                model = genai.GenerativeModel(MODEL)
                response = model.generate_content(prompt)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                print(f"\n=== Gemini API レスポンス ===")
                print(f"処理時間: {elapsed_time:.2f}秒")
                print(f"レスポンス受信: 成功")
                
                if not response.text:
                    print("警告: レスポンスにテキストが含まれていません")
                    
                print(f"レスポンス長: {len(response.text)} 文字")
                print(f"レスポンス内容: {response.text[:100]}{'...' if len(response.text) > 100 else ''}")
                
                return response.text
            except (ResourceExhausted, ServiceUnavailable) as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"API呼び出しエラー: {e}. {RETRY_DELAY}秒後にリトライします...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"リトライ回数を超えました: {e}")
                    return "【シーン】（情報なし）"
            except Exception as e:
                print(f"API呼び出し中に予期しないエラーが発生しました: {e}")
                import traceback
                print(traceback.format_exc())
                return "【シーン】（情報なし）"
    
    # APIを呼び出してシーン説明を生成
    caption = call_api_with_retry(prompt)
    
    # 先頭に【シーン】が付いていない場合は追加
    if not caption.strip().startswith("【シーン】"):
        print("警告: レスポンスに【シーン】が含まれていないため追加します")
        caption = "【シーン】" + caption.strip()
    
    # 説明が長すぎる場合は短く切り詰める
    if len(caption) > 50:
        print(f"説明が長すぎます（{len(caption)}文字）。50文字に切り詰めます。")
        caption = caption[:47] + "..."
    
    print(f"\n生成されたシーン説明: {caption}")
    print(f"=== シーン説明生成完了 ===")
    return caption

def generate_captions_from_scenario(input_srt_file: str, output_srt_file: str, scenario_file: str, scene_duration: int = SCENE_DURATION, gemini_api_key: str = None, serper_api_key: str = None, disable_web_search: bool = None) -> None:
    """
    SRTファイルとシナリオファイルからシーン説明を生成し、新しいSRTファイルを作成します。
    
    Args:
        input_srt_file: 入力SRTファイルのパス
        output_srt_file: 出力SRTファイルのパス
        scenario_file: シナリオファイルのパス
        scene_duration: 各シーンの長さ（秒）
        gemini_api_key: Gemini API Key（あれば環境変数を上書き）
        serper_api_key: Serper API Key（あれば環境変数を上書き）
        disable_web_search: ウェブ検索を無効化するかどうか
    """
    global API_KEY, SERPER_API_KEY, DISABLE_WEB_SEARCH
    
    # APIキーを設定
    if gemini_api_key:
        API_KEY = gemini_api_key
    if serper_api_key:
        SERPER_API_KEY = serper_api_key
    if disable_web_search is not None:
        DISABLE_WEB_SEARCH = disable_web_search
    
    print(f"処理を開始します: {input_srt_file} + {scenario_file} -> {output_srt_file}")
    print(f"ウェブ検索モード: {'無効' if DISABLE_WEB_SEARCH else '有効'}")
    
    # シナリオファイルの読み込み
    try:
        with open(scenario_file, 'r', encoding='utf-8') as file:
            scenario_text = file.read()
    except Exception as e:
        print(f"シナリオファイルの読み込み中にエラーが発生しました: {e}")
        return
    
    # SRTファイルの解析
    srt_blocks = parse_srt_file(input_srt_file)
    if not srt_blocks:
        print(f"エラー: {input_srt_file} から字幕ブロックを解析できませんでした。")
        return
    
    # 最後の字幕ブロックから動画の最大時間を取得
    max_time = max(block.end_time for block in srt_blocks)
    
    # シーンブロックの作成
    scene_blocks = create_scene_blocks(scene_duration, max_time)
    
    # シーン説明の生成と挿入
    output_blocks = []
    scene_descriptions = []
    
    # API呼び出しの間隔を調整するための変数（秒）
    api_call_interval = 3  # 3秒間隔
    
    for scene_index, (scene_start, scene_end) in enumerate(scene_blocks):
        print(f"\n*** シーン {scene_index+1}/{len(scene_blocks)}: {format_time(scene_start)} - {format_time(scene_end)} ***")
        
        # シーン内の字幕を収集
        texts_in_scene = []
        for block in srt_blocks:
            if (block.start_time <= scene_end and block.end_time >= scene_start):
                texts_in_scene.append(block.text)
        
        # シーン内に字幕がなければスキップ
        if not texts_in_scene:
            print(f"シーン {scene_index+1} にはテキストがありません。スキップします。")
            continue
        
        # シーン内のテキストを結合
        text_in_scene = "\n".join(texts_in_scene)
        
        # キーワードを抽出してWeb検索
        search_result = ""
        if not DISABLE_WEB_SEARCH:  # ウェブ検索が有効な場合のみ実行
            keywords = extract_keywords_from_scene(srt_blocks, scene_start, scene_end)
            if keywords:
                search_query = f"{keywords} 特徴 説明"
                search_result = search_web(search_query)
                
                # Web検索の後に少し待機してAPIレート制限を回避
                if scene_index < len(scene_blocks) - 1:  # 最後のシーン以外で待機
                    print(f"\n検索API呼び出し後、{api_call_interval}秒間待機します...")
                    time.sleep(api_call_interval)
        
        # Gemini APIでシーン説明を生成（シナリオ付き）
        scene_description = generate_caption_with_gemini(text_in_scene, scene_start, scene_end, scenario_text, search_result)
        scene_descriptions.append((scene_start, scene_end, scene_description))
        
        # 次のシーンの処理前に待機してAPIレート制限を回避
        if scene_index < len(scene_blocks) - 1:  # 最後のシーン以外で待機
            progress = f"進捗: {scene_index+1}/{len(scene_blocks)} シーン処理完了 ({(scene_index+1)*100/len(scene_blocks):.1f}%)"
            print(f"\n{progress}")
            print(f"Gemini API呼び出し後、{api_call_interval}秒間待機します...")
            time.sleep(api_call_interval)
    
    # シーン説明ブロックを作成して元の字幕ブロックと結合
    all_blocks = []
    
    # まず元の字幕ブロックをコピー
    for block in srt_blocks:
        all_blocks.append(block)
    
    # シーン説明ブロックを作成して追加
    for scene_index, (scene_start, scene_end, scene_description) in enumerate(scene_descriptions):
        # シーン説明用の新しいインデックスを作成（既存のブロック数 + シーンインデックス）
        new_index = len(srt_blocks) + scene_index + 1
        
        # SRTフォーマットのタイムスタンプを生成
        start_timestamp = SRTBlock(0, "", "").get_formatted_timestamp(scene_start)
        end_timestamp = SRTBlock(0, "", "").get_formatted_timestamp(scene_start + 5)  # 5秒間表示
        timestamp = f"{start_timestamp} --> {end_timestamp}"
        
        # シーン説明ブロックを作成
        scene_block = SRTBlock(new_index, timestamp, scene_description)
        all_blocks.append(scene_block)
    
    # インデックス順にソート
    all_blocks.sort(key=lambda block: (block.start_time, block.index))
    
    # インデックスを振り直し
    for i, block in enumerate(all_blocks):
        block.index = i + 1
    
    # 出力ファイルに書き込み
    output_dir = os.path.dirname(output_srt_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_srt_file, 'w', encoding='utf-8') as file:
        for i, block in enumerate(all_blocks):
            file.write(str(block))
            if i < len(all_blocks) - 1:
                file.write("\n\n")
    
    print(f"\n処理完了: {len(scene_descriptions)} 個のシーン説明を生成し、{output_srt_file} に保存しました。")

def add_scene_descriptions_to_srt(input_srt_file: str, output_srt_file: str, scene_duration: int = SCENE_DURATION, gemini_api_key: str = None, serper_api_key: str = None, disable_web_search: bool = None) -> None:
    """
    SRTファイルの字幕からシーン説明を生成し、新しいSRTファイルを作成します。
    
    Args:
        input_srt_file: 入力SRTファイルのパス
        output_srt_file: 出力SRTファイルのパス
        scene_duration: 各シーンの長さ（秒）
        gemini_api_key: Gemini API Key（あれば環境変数を上書き）
        serper_api_key: Serper API Key（あれば環境変数を上書き）
        disable_web_search: ウェブ検索を無効化するかどうか
    """
    global API_KEY, SERPER_API_KEY, DISABLE_WEB_SEARCH
    
    # APIキーを設定
    if gemini_api_key:
        API_KEY = gemini_api_key
    if serper_api_key:
        SERPER_API_KEY = serper_api_key
    if disable_web_search is not None:
        DISABLE_WEB_SEARCH = disable_web_search
        
    print(f"処理を開始します: {input_srt_file} -> {output_srt_file}")
    print(f"ウェブ検索モード: {'無効' if DISABLE_WEB_SEARCH else '有効'}")
    
    # SRTファイルの解析
    srt_blocks = parse_srt_file(input_srt_file)
    if not srt_blocks:
        print(f"エラー: {input_srt_file} から字幕ブロックを解析できませんでした。")
        return
    
    # 最後の字幕ブロックから動画の最大時間を取得
    max_time = max(block.end_time for block in srt_blocks)
    
    # シーンブロックの作成
    scene_blocks = create_scene_blocks(scene_duration, max_time)
    
    # シーン説明の生成と挿入
    output_blocks = []
    scene_descriptions = []
    
    # API呼び出しの間隔を調整するための変数（秒）
    api_call_interval = 3  # 3秒間隔
    
    for scene_index, (scene_start, scene_end) in enumerate(scene_blocks):
        print(f"\n*** シーン {scene_index+1}/{len(scene_blocks)}: {format_time(scene_start)} - {format_time(scene_end)} ***")
        
        # シーン内の字幕を収集
        texts_in_scene = []
        for block in srt_blocks:
            if (block.start_time <= scene_end and block.end_time >= scene_start):
                texts_in_scene.append(block.text)
        
        # シーン内に字幕がなければスキップ
        if not texts_in_scene:
            print(f"シーン {scene_index+1} にはテキストがありません。スキップします。")
            continue
        
        # シーン内のテキストを結合
        text_in_scene = "\n".join(texts_in_scene)
        
        # キーワードを抽出してWeb検索
        search_result = ""
        if not DISABLE_WEB_SEARCH:  # ウェブ検索が有効な場合のみ実行
            keywords = extract_keywords_from_scene(srt_blocks, scene_start, scene_end)
            if keywords:
                search_query = f"{keywords} 特徴 説明"
                search_result = search_web(search_query)
                
                # Web検索の後に少し待機してAPIレート制限を回避
                if scene_index < len(scene_blocks) - 1:  # 最後のシーン以外で待機
                    print(f"\n検索API呼び出し後、{api_call_interval}秒間待機します...")
                    time.sleep(api_call_interval)
        
        # Gemini APIでシーン説明を生成
        scene_description = generate_caption_with_gemini(text_in_scene, scene_start, scene_end, search_result=search_result)
        scene_descriptions.append((scene_start, scene_end, scene_description))
        
        # 次のシーンの処理前に待機してAPIレート制限を回避
        if scene_index < len(scene_blocks) - 1:  # 最後のシーン以外で待機
            progress = f"進捗: {scene_index+1}/{len(scene_blocks)} シーン処理完了 ({(scene_index+1)*100/len(scene_blocks):.1f}%)"
            print(f"\n{progress}")
            print(f"Gemini API呼び出し後、{api_call_interval}秒間待機します...")
            time.sleep(api_call_interval)
    
    # シーン説明ブロックを作成して元の字幕ブロックと結合
    all_blocks = []
    
    # まず元の字幕ブロックをコピー
    for block in srt_blocks:
        all_blocks.append(block)
    
    # シーン説明ブロックを作成して追加
    for scene_index, (scene_start, scene_end, scene_description) in enumerate(scene_descriptions):
        # シーン説明用の新しいインデックスを作成（既存のブロック数 + シーンインデックス）
        new_index = len(srt_blocks) + scene_index + 1
        
        # SRTフォーマットのタイムスタンプを生成
        start_timestamp = SRTBlock(0, "", "").get_formatted_timestamp(scene_start)
        end_timestamp = SRTBlock(0, "", "").get_formatted_timestamp(scene_start + 5)  # 5秒間表示
        timestamp = f"{start_timestamp} --> {end_timestamp}"
        
        # シーン説明ブロックを作成
        scene_block = SRTBlock(new_index, timestamp, scene_description)
        all_blocks.append(scene_block)
    
    # インデックス順にソート
    all_blocks.sort(key=lambda block: (block.start_time, block.index))
    
    # インデックスを振り直し
    for i, block in enumerate(all_blocks):
        block.index = i + 1
    
    # 出力ファイルに書き込み
    output_dir = os.path.dirname(output_srt_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_srt_file, 'w', encoding='utf-8') as file:
        for i, block in enumerate(all_blocks):
            file.write(str(block))
            if i < len(all_blocks) - 1:
                file.write("\n\n")
    
    print(f"\n処理完了: {len(scene_descriptions)} 個のシーン説明を生成し、{output_srt_file} に保存しました。")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='シナリオとSRTファイルからテロップを生成します。')
    parser.add_argument('input_file', help='入力SRTファイルのパス')
    parser.add_argument('output_file', help='出力SRTファイルのパス')
    parser.add_argument('scenario_file', help='シナリオファイルのパス')
    parser.add_argument('--duration', type=int, default=SCENE_DURATION, help='シーンの長さ（秒）')
    parser.add_argument('--gemini_api_key', help='GeminiのAPIキー')
    parser.add_argument('--serper_api_key', help='SerperのAPIキー')
    
    args = parser.parse_args()
    
    generate_captions_from_scenario(
        args.input_file,
        args.output_file,
        args.scenario_file,
        args.duration,
        args.gemini_api_key,
        args.serper_api_key
    )

if __name__ == "__main__":
    main() 