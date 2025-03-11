import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading
import os
import sys
import json
from scene_description_generator import add_scene_descriptions_to_srt, SCENE_DURATION, TEMP_SCENARIO_FILE

# 設定ファイルのパス
CONFIG_FILE = "config.json"

# APIキーをハードコード
GEMINI_API_KEY = "AIzaSyCztyJmLgpfj9ko5G5Y48AxnzP2Nj78xYM"
SERPER_API_KEY = "0c939952353f8ab495951a1420a13b1f5b08001d"

class RedirectText:
    """コンソール出力をGUIのテキストウィジェットにリダイレクトするクラス"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
        
    def write(self, string):
        self.buffer += string
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")
        
    def flush(self):
        pass

class SRTSceneDescriptionGeneratorGUI:
    """SRTシーン説明ジェネレーターのGUIクラス"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SRTシーン説明ジェネレーター")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # 設定を読み込み
        self.config = self.load_config()
        
        # メインフレーム
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 入力ファイル選択
        input_frame = ttk.LabelFrame(main_frame, text="入力SRTファイル", padding="5")
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_path = tk.StringVar(value=self.config.get("last_input_path", ""))
        ttk.Entry(input_frame, textvariable=self.input_path, width=70).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="参照...", command=self.browse_input_file).pack(side=tk.RIGHT, padx=5)
        
        # 出力ファイル選択
        output_frame = ttk.LabelFrame(main_frame, text="出力SRTファイル", padding="5")
        output_frame.pack(fill=tk.X, pady=5)
        
        self.output_path = tk.StringVar(value=self.config.get("last_output_path", ""))
        ttk.Entry(output_frame, textvariable=self.output_path, width=70).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="参照...", command=self.browse_output_file).pack(side=tk.RIGHT, padx=5)
        
        # 設定フレーム
        settings_frame = ttk.LabelFrame(main_frame, text="設定", padding="5")
        settings_frame.pack(fill=tk.X, pady=5)
        
        # シーン時間
        ttk.Label(settings_frame, text="シーン時間（秒）:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.scene_duration = tk.IntVar(value=self.config.get("scene_duration", SCENE_DURATION))
        ttk.Spinbox(settings_frame, from_=5, to=60, increment=5, textvariable=self.scene_duration, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # シナリオ説明
        ttk.Label(settings_frame, text="シナリオ説明:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.scenario_text = tk.Text(settings_frame, wrap=tk.WORD, width=60, height=3)
        self.scenario_text.grid(row=1, column=1, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=2)
        # 保存されたシナリオがあれば挿入
        if "last_scenario" in self.config:
            self.scenario_text.insert("1.0", self.config["last_scenario"])
        # スクロールバーを追加
        scenario_scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=self.scenario_text.yview)
        self.scenario_text.configure(yscrollcommand=scenario_scrollbar.set)
        scenario_scrollbar.grid(row=1, column=4, sticky='ns', pady=2)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # ログエリア
        log_frame = ttk.LabelFrame(main_frame, text="ログ", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.configure(state="disabled")
        
        # リダイレクト標準出力
        self.stdout_redirector = RedirectText(self.log_text)
        sys.stdout = self.stdout_redirector
        
        # ボタンエリア
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.progress = ttk.Progressbar(button_frame, orient=tk.HORIZONTAL, length=200, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, padx=5)
        
        # ボタンを右側に配置し、サイズを大きくする
        self.generate_button = ttk.Button(button_frame, text="シーン説明を生成", command=self.generate_descriptions, width=20)
        self.generate_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 処理中フラグ
        self.processing = False
        
        # 終了時に設定を保存するためのイベントハンドラを設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_config(self):
        """設定ファイルを読み込む"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"設定ファイルの読み込みに失敗しました: {e}")
        return {}
    
    def save_config(self):
        """設定ファイルに保存する"""
        config = {
            "last_input_path": self.input_path.get(),
            "last_output_path": self.output_path.get(),
            "scene_duration": self.scene_duration.get(),
            "last_scenario": self.scenario_text.get("1.0", tk.END).strip()
        }
        
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"設定ファイルの保存に失敗しました: {e}")
    
    def on_closing(self):
        """ウィンドウが閉じられるときの処理"""
        self.save_config()
        # 標準出力を元に戻す
        sys.stdout = sys.__stdout__
        self.root.destroy()
    
    def browse_input_file(self):
        """入力SRTファイルを選択するダイアログを表示"""
        file_path = filedialog.askopenfilename(
            title="入力SRTファイルを選択",
            filetypes=[("SRTファイル", "*.srt"), ("すべてのファイル", "*.*")]
        )
        if file_path:
            self.input_path.set(file_path)
            # 出力パスが空の場合、自動的に設定
            if not self.output_path.get():
                base, ext = os.path.splitext(file_path)
                self.output_path.set(f"{base}_with_descriptions{ext}")
    
    def browse_output_file(self):
        """出力SRTファイルを選択するダイアログを表示"""
        file_path = filedialog.asksaveasfilename(
            title="出力SRTファイルを選択",
            filetypes=[("SRTファイル", "*.srt"), ("すべてのファイル", "*.*")],
            defaultextension=".srt"
        )
        if file_path:
            self.output_path.set(file_path)
    
    def generate_descriptions(self):
        """シーン説明の生成処理を開始"""
        if self.processing:
            return
        
        # 入力ファイルの確認
        input_file = self.input_path.get()
        if not input_file or not os.path.isfile(input_file):
            print("エラー: 有効な入力SRTファイルを選択してください。")
            return
        
        # 出力ファイルの確認
        output_file = self.output_path.get()
        if not output_file:
            print("エラー: 出力SRTファイルを指定してください。")
            return
        
        # 処理開始
        self.processing = True
        self.generate_button.configure(state="disabled")
        self.progress.start()
        
        # 設定を保存
        self.save_config()
        
        # パラメータの取得
        scene_duration = self.scene_duration.get()
        # テキストエリアからシナリオを取得
        scenario = self.scenario_text.get("1.0", tk.END).strip()
        
        # 別スレッドで処理を実行
        thread = threading.Thread(
            target=self.process_file,
            args=(input_file, output_file, scene_duration, scenario)
        )
        thread.daemon = True
        thread.start()
    
    def process_file(self, input_file, output_file, scene_duration, scenario):
        """別スレッドで実行される処理"""
        try:
            print(f"処理を開始します: {input_file}")
            add_scene_descriptions_to_srt(
                input_file, 
                output_file, 
                scene_duration, 
                scenario, 
                GEMINI_API_KEY, 
                SERPER_API_KEY
            )
            print(f"処理が完了しました。出力ファイル: {output_file}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
        finally:
            # GUI更新はメインスレッドで行う
            self.root.after(0, self.processing_complete)
    
    def processing_complete(self):
        """処理完了時の後処理"""
        self.progress.stop()
        self.generate_button.configure(state="normal")
        self.processing = False

def main():
    root = tk.Tk()
    app = SRTSceneDescriptionGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 