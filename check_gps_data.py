import json
import os
import sys
import subprocess
import argparse

def extract_gps_data_with_exiftool(video_filepath):
    """
    exiftoolを使ってビデオファイルからGPSデータを抽出する
    
    Args:
        video_filepath (str): 動画ファイルの絶対パス
        
    Returns:
        list: GPSデータのリスト（各項目は辞書形式）
        errors: エラーメッセージのリスト
    """
    gps_data = []
    errors = []
    
    # exiftoolの実行ファイルのパスを取得
    exiftool_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exiftool-13.28_64')
    exiftool_path = os.path.join(exiftool_dir, 'exiftool.exe')
    
    if not os.path.exists(exiftool_path):
        errors.append(f"exiftoolが見つかりません: {exiftool_path}")
        return gps_data, errors
        
    try:
        # exiftoolを実行してGPSデータをJSON形式で抽出
        cmd = [exiftool_path, "-G", "-n", "-j", "-ee", video_filepath]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.returncode != 0:
            errors.append(f"exiftoolの実行中にエラーが発生しました: {result.stderr}")
            return gps_data, errors
        
        # JSON形式の出力を解析
        try:
            data = json.loads(result.stdout)
            if not data:
                errors.append(f"exiftoolがJSON形式のデータを返しませんでした: {video_filepath}")
                return gps_data, errors
            
            # 詳細なデバッグ情報
            print("抽出されたメタデータキー:")
            for key in data[0].keys():
                if "GPS" in key:
                    print(f"  {key}: {data[0][key]}")
            
            # GPSトラックデータを処理
            track_data = data[0].get("Composite:GPSTrack", [])
            if isinstance(track_data, list) and track_data:
                print("GPSトラックデータが見つかりました")
                for point in track_data:
                    gps_point = {
                        "timestamp": point.get("GPSDateTime", ""),
                        "latitude": point.get("GPSLatitude"),
                        "longitude": point.get("GPSLongitude"),
                        "altitude": point.get("GPSAltitude"),
                        "speed": point.get("GPSSpeed"),
                        "track": point.get("GPSTrack")
                    }
                    gps_data.append(gps_point)
            else:
                print("GPSトラックデータが見つかりません、単一のGPSポイントを確認します")
                # 単一のGPSポイントの場合
                if "GPS:GPSLatitude" in data[0] and "GPS:GPSLongitude" in data[0]:
                    print("標準GPSデータが見つかりました")
                    gps_point = {
                        "timestamp": data[0].get("GPS:GPSDateTime", ""),
                        "latitude": data[0].get("GPS:GPSLatitude"),
                        "longitude": data[0].get("GPS:GPSLongitude"),
                        "altitude": data[0].get("GPS:GPSAltitude"),
                        "speed": data[0].get("GPS:GPSSpeed"),
                        "track": data[0].get("GPS:GPSTrack")
                    }
                    gps_data.append(gps_point)
                else:
                    # GoPro特有のメタデータ構造に対応
                    if "GoPro:GPSLatitude" in data[0] and "GoPro:GPSLongitude" in data[0]:
                        print("GoPro固有のGPSデータが見つかりました")
                        gps_point = {
                            "timestamp": data[0].get("GoPro:GPSDateTime", data[0].get("ExifIFD:CreateDate", "")),
                            "latitude": data[0].get("GoPro:GPSLatitude"),
                            "longitude": data[0].get("GoPro:GPSLongitude"),
                            "altitude": data[0].get("GoPro:GPSAltitude"),
                            "speed": data[0].get("GoPro:GPSSpeed"),
                            "track": data[0].get("GoPro:GPSTrack")
                        }
                        gps_data.append(gps_point)
                    else:
                        print("標準的なGPSデータが見つかりません、テレメトリーデータを確認します")
                        # テレメトリーデータのための別のコマンドを試す
                        temp_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_telemetry.json')
                        cmd = [exiftool_path, "-ee", "-j", video_filepath]
                        telemetry_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        
                        with open(temp_json, 'w', encoding='utf-8') as f:
                            f.write(telemetry_result.stdout)
                            
                        if os.path.exists(temp_json) and os.path.getsize(temp_json) > 0:
                            with open(temp_json, 'r', encoding='utf-8') as f:
                                try:
                                    telemetry_data = json.load(f)
                                    # GoPro固有のテレメトリーデータをチェック
                                    if "QuickTime:GPSCoordinates" in telemetry_data[0]:
                                        print("QuickTime GPSデータが見つかりました")
                                        coords = telemetry_data[0].get("QuickTime:GPSCoordinates", "")
                                        if coords and "+" in coords:
                                            lat_lon = coords.split("+")
                                            if len(lat_lon) >= 2:
                                                try:
                                                    lat = float(lat_lon[0])
                                                    lon = float(lat_lon[1])
                                                    gps_point = {
                                                        "timestamp": telemetry_data[0].get("QuickTime:CreateDate", ""),
                                                        "latitude": lat,
                                                        "longitude": lon,
                                                        "altitude": 0,  # 高度情報はこの形式では利用できない場合が多い
                                                        "speed": 0,     # 速度情報もこの形式では利用できない場合が多い
                                                        "track": 0      # 方位情報もこの形式では利用できない場合が多い
                                                    }
                                                    gps_data.append(gps_point)
                                                except (ValueError, IndexError):
                                                    pass
                                except json.JSONDecodeError:
                                    errors.append(f"テレメトリーJSONの解析に失敗しました: {temp_json}")
        
        except json.JSONDecodeError as e:
            errors.append(f"JSONの解析に失敗しました: {e}")
            return gps_data, errors
            
    except subprocess.CalledProcessError as e:
        errors.append(f"exiftoolの実行に失敗しました: {e}")
    except Exception as e:
        errors.append(f"GPSデータの抽出中にエラーが発生しました: {e}")
        
    return gps_data, errors

def save_gps_json(gps_data, output_file="gps_data.json"):
    """GPSデータをJSONファイルとして保存"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(gps_data, f, ensure_ascii=False, indent=2)
        print(f"GPSデータを保存しました: {output_file}")
        return True
    except Exception as e:
        print(f"GPSデータの保存に失敗しました: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='GoProのMP4ファイルからGPSデータを抽出して表示する')
    parser.add_argument('video_file', help='処理するMP4ファイルのパス')
    parser.add_argument('--save', help='GPSデータを保存するファイル名', default='gps_data.json')
    parser.add_argument('--txt', help='テキストファイルとして保存するファイル名')
    
    if len(sys.argv) == 1:
        # 引数がない場合はデフォルトでgps_data.jsonを読む
        try:
            with open('gps_data.json', 'r') as f:
                data = json.load(f)
                
            print(f"総データ数: {len(data)}")
            print("\n最初の10件のGPSデータ:")
            for i, point in enumerate(data[:10]):
                print(f"\nポイント {i+1}:")
                print(f"  緯度: {point['latitude']}")
                print(f"  経度: {point['longitude']}")
                print(f"  高度: {point.get('altitude', 'N/A')}m")
                print(f"  速度: {point.get('speed', 'N/A')}m/s")
                print(f"  時刻: {point.get('timestamp', 'N/A')}")

        except FileNotFoundError:
            print("GPSデータファイルが見つかりません")
        except json.JSONDecodeError:
            print("JSONファイルの解析に失敗しました")
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
    else:
        args = parser.parse_args()
        
        if not os.path.exists(args.video_file):
            print(f"ファイルが見つかりません: {args.video_file}")
            return
            
        print(f"ファイル {args.video_file} からGPSデータを抽出しています...")
        
        gps_data, errors = extract_gps_data_with_exiftool(args.video_file)
        
        if errors:
            print("\nエラー:")
            for error in errors:
                print(f"- {error}")
                
        if not gps_data:
            print("GPSデータが見つかりませんでした")
            return
            
        print(f"\n総データ数: {len(gps_data)}")
        print("\n最初の10件のGPSデータ:")
        for i, point in enumerate(gps_data[:10]):
            print(f"\nポイント {i+1}:")
            print(f"  緯度: {point.get('latitude', 'N/A')}")
            print(f"  経度: {point.get('longitude', 'N/A')}")
            print(f"  高度: {point.get('altitude', 'N/A')}m")
            print(f"  速度: {point.get('speed', 'N/A')}m/s")
            print(f"  時刻: {point.get('timestamp', 'N/A')}")
            
        # GPSデータをJSONとして保存
        save_gps_json(gps_data, args.save)
        
        # テキストファイルとしても保存する場合
        if args.txt:
            try:
                with open(args.txt, 'w', encoding='utf-8') as f:
                    f.write(f"総データ数: {len(gps_data)}\n\n")
                    for i, point in enumerate(gps_data):
                        f.write(f"ポイント {i+1}:\n")
                        f.write(f"  緯度: {point.get('latitude', 'N/A')}\n")
                        f.write(f"  経度: {point.get('longitude', 'N/A')}\n")
                        f.write(f"  高度: {point.get('altitude', 'N/A')}m\n")
                        f.write(f"  速度: {point.get('speed', 'N/A')}m/s\n")
                        f.write(f"  時刻: {point.get('timestamp', 'N/A')}\n\n")
                print(f"テキストデータを保存しました: {args.txt}")
            except Exception as e:
                print(f"テキストファイルの保存に失敗しました: {e}")

if __name__ == "__main__":
    main()