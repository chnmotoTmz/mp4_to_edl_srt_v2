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
API_KEY = "AIzaSyCztyJmLgpfj9ko5G5Y48AxnzP2Nj78xYM"  # 実際のAPIキーに置き換えてください
MODEL = "gemini-1.5-flash-8b"
SCENE_DURATION = 10  # シーンの長さ（秒）- 10秒に変更
MAX_RETRIES = 3  # API呼び出しの最大リトライ回数
RETRY_DELAY = 2  # リトライ間の遅延（秒）
SERPER_API_KEY = "0c939952353f8ab495951a1420a13b1f5b08001d"  # SerperのAPIキー

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
    print(f"検索クエリ: {query}")
    
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
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        data = response.json()
        
        if response.status_code != 200:
            print(f"検索APIエラー: ステータスコード {response.status_code}")
            return f"検索結果を取得できませんでした。エラー: {response.status_code}"
        
        if "organic" not in data or len(data["organic"]) == 0:
            print("検索結果がありません")
            return "関連する検索結果がありませんでした。"
        
        results = []
        for item in data["organic"][:num_results]:
            title = item.get("title", "タイトルなし")
            snippet = item.get("snippet", "スニペットなし")
            link = item.get("link", "#")
            results.append(f"【{title}】\n{snippet}\n{link}\n")
        
        return "\n".join(results)
    
    except Exception as e:
        print(f"検索API呼び出し中にエラーが発生しました: {e}")
        return f"検索中にエラーが発生しました: {str(e)}"

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

def generate_captions_from_scenario(input_file: str, output_file: str, scenario_file: str, scene_duration: int = SCENE_DURATION, gemini_api_key: str = None, serper_api_key: str = None):
    """
    SRTファイルとシナリオからテロップを生成し、新しいSRTファイルを作成します。
    
    Args:
        input_file: 入力SRTファイルのパス
        output_file: 出力SRTファイルのパス
        scenario_file: シナリオファイルのパス
        scene_duration: シーンの長さ（秒）
        gemini_api_key: GeminiのAPIキー（オプション）
        serper_api_key: SerperのAPIキー（オプション）
    """
    # APIキーを設定
    global API_KEY, SERPER_API_KEY
    if gemini_api_key:
        API_KEY = gemini_api_key
    if serper_api_key:
        SERPER_API_KEY = serper_api_key
    
    # GeminiのAPIを設定
    setup_gemini_api()
    
    # SRTファイルを解析
    print(f"SRTファイル {input_file} を解析中...")
    srt_blocks = parse_srt_file(input_file)
    if not srt_blocks:
        print("SRTファイルが空か、解析できませんでした。")
        return
    
    # 最終の終了時間を取得
    total_duration = max(block.end_time for block in srt_blocks)
    print(f"動画の総時間: {format_time(total_duration)}")
    
    # シナリオファイルを読み込む
    try:
        with open(scenario_file, 'r', encoding='utf-8') as file:
            scenario_text = file.read()
        print(f"シナリオファイル {scenario_file} を読み込みました。")
    except Exception as e:
        print(f"シナリオファイルの読み込み中にエラーが発生しました: {e}")
        return
    
    # シーンブロックを作成
    scenes = create_scene_blocks(scene_duration, total_duration)
    print(f"{len(scenes)}個のシーンに分割しました。")
    
    # 新しいSRTブロックを格納するリスト
    new_srt_blocks = []
    
    # オープニングテロップのためのWeb検索
    opening_search_query = "旅行番組 オープニング テロップ 構成"
    opening_search_results = search_web(opening_search_query)
    
    # オープニングテロップを生成
    print("\nオープニングテロップを生成中...")
    opening_prompt = generate_opening_prompt(scenario_text, opening_search_results)
    opening_caption = call_gemini_api(opening_prompt).strip()
    
    # オープニングテロップの整形
    opening_caption = opening_caption.replace("**", "").replace("*", "")
    for prefix in ["オープニングテロップ：", "ナレーション："]:
        if opening_caption.startswith(prefix):
            opening_caption = opening_caption[len(prefix):].strip()
    
    print(f"オープニングテロップ:\n{opening_caption}")
    
    # オープニングテロップを追加（10秒間表示）
    opening_block = create_new_caption_block(1, 0, 10, opening_caption)
    new_srt_blocks.append(opening_block)
    
    # インデックスオフセット（オープニングブロックを追加したため）
    index_offset = 1
    
    # 各シーンに対してテロップを生成
    for i, (start_time, end_time) in enumerate(scenes):
        # オープニングの時間は飛ばす
        if start_time < 10:
            continue
            
        print(f"\nシーン {i+1}/{len(scenes)} を処理中 ({format_time(start_time)} - {format_time(end_time)})...")
        
        # シーンからキーワードを抽出
        keywords = extract_keywords_from_scene(srt_blocks, start_time, end_time)
        if not keywords:
            print("  キーワードが抽出できませんでした。次のシーンに進みます。")
            continue
        
        # シナリオとマッチするセクションを検索
        scenario_section = match_scene_with_scenario(keywords, scenario_text)
        
        # 検索クエリを作成し、Web検索を実行
        search_query = f"{keywords} 観光 情報"
        search_results = search_web(search_query)
        
        # テロップ生成のプロンプトを作成
        prompt = generate_caption_prompt(keywords, scenario_section, search_results)
        
        # Gemini APIを呼び出してテロップを生成
        generated_caption = call_gemini_api(prompt).strip()
        
        # 生成されたテロップを整形（不要な改行やマークダウン記法を除去）
        generated_caption = generated_caption.replace("\n", " ").replace("**", "").replace("*", "")
        
        # テロップ例：などの接頭辞がある場合は削除
        prefixes_to_remove = ["テロップ例：", "テロップ例1：", "テロップ例2：", "テロップ案：", "テロップ："]
        for prefix in prefixes_to_remove:
            if generated_caption.startswith(prefix):
                generated_caption = generated_caption[len(prefix):].strip()
        
        print(f"  生成されたテロップ: {generated_caption}")
        
        # テロップが長すぎる場合は制限
        if len(generated_caption) > 60:
            generated_caption = generated_caption[:57] + "..."
        
        # 新しいSRTブロックを作成
        caption_block = create_new_caption_block(i + index_offset, start_time, end_time, generated_caption)
        new_srt_blocks.append(caption_block)
        
        # 進捗状況を表示（10シーンごと）
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scenes)} シーンの処理完了")
    
    # エンドロールのためのWeb検索
    ending_search_query = "旅行番組 エンディング エンドロール 制作協力 謝辞"
    ending_search_results = search_web(ending_search_query)
    
    # エンドロールテロップを生成
    print("\nエンドロールテロップを生成中...")
    ending_prompt = generate_ending_prompt(scenario_text, ending_search_results)
    ending_caption = call_gemini_api(ending_prompt).strip()
    
    # エンドロールの開始時間を設定（最後のテロップの終了時間）
    if new_srt_blocks:
        ending_start_time = new_srt_blocks[-1].end_time
    else:
        ending_start_time = total_duration - 20  # エンドロールの長さを20秒と仮定
    
    # エンドロールテロップは20秒間表示
    ending_end_time = min(ending_start_time + 20, total_duration)
    
    # エンドロールブロックを作成
    ending_block = create_new_caption_block(len(new_srt_blocks) + 1, ending_start_time, ending_end_time, ending_caption)
    new_srt_blocks.append(ending_block)
    
    print(f"エンドロールテロップ:\n{ending_caption}")
    
    # 新しいSRTファイルを書き出し
    if new_srt_blocks:
        write_srt_file(new_srt_blocks, output_file)
        print(f"\n新しいテロップを {output_file} に保存しました。")
    else:
        print("\nテロップが生成されませんでした。")

def write_srt_file(srt_blocks: List[SRTBlock], file_path: str):
    """SRTブロックのリストをSRTファイルに書き出します。"""
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, block in enumerate(srt_blocks):
            # インデックスを1から順番に振り直す
            file.write(f"{i+1}\n{block.timestamp}\n{block.text}\n\n")

def add_scene_descriptions_to_srt(input_file: str, output_file: str, scene_duration: int = SCENE_DURATION, overall_scenario: str = "", gemini_api_key: str = None, serper_api_key: str = None):
    """
    SRTファイルにシーン説明を追加します。
    この関数は古いバージョンとの互換性のために残されています。
    新しいgenerate_captions_from_scenarioを呼び出します。
    
    Args:
        input_file: 入力SRTファイルのパス
        output_file: 修正されたSRTファイルを保存するパス
        scene_duration: シーンの長さ（秒）
        overall_scenario: 全体のシナリオの説明（オプション）
        gemini_api_key: Gemini APIキー（オプション）
        serper_api_key: Serper APIキー（オプション）
    """
    # シナリオが提供された場合、一時ファイルに保存
    if overall_scenario:
        with open(TEMP_SCENARIO_FILE, 'w', encoding='utf-8') as f:
            f.write(overall_scenario)
        
        try:
            # 新しい関数を呼び出し
            generate_captions_from_scenario(
                input_file,
                output_file,
                TEMP_SCENARIO_FILE,
                scene_duration,
                gemini_api_key,
                serper_api_key
            )
        finally:
            # 一時ファイルを削除
            if os.path.exists(TEMP_SCENARIO_FILE):
                try:
                    os.remove(TEMP_SCENARIO_FILE)
                except:
                    pass
    else:
        # シナリオがない場合は空の一時ファイルを作成
        with open(TEMP_SCENARIO_FILE, 'w', encoding='utf-8') as f:
            f.write("# 自動生成シナリオ\n\n## シーン1\n動画シーン\n")
        
        try:
            # 新しい関数を呼び出し
            generate_captions_from_scenario(
                input_file,
                output_file,
                TEMP_SCENARIO_FILE,
                scene_duration,
                gemini_api_key,
                serper_api_key
            )
        finally:
            # 一時ファイルを削除
            if os.path.exists(TEMP_SCENARIO_FILE):
                try:
                    os.remove(TEMP_SCENARIO_FILE)
                except:
                    pass

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