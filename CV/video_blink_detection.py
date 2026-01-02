"""
Video Blink Detection - 動画から瞬き検出プログラム

このプログラムは動画ファイルから人物の瞬きを自動検出し、詳細なログとレポートを生成します。
MediaPipe Face Landmarkerを使用して顔のランドマークを検出し、
目のアスペクト比（EAR）を計算することで瞬きを判定します。

Author: OchiLab
Version: 1.0.0
License: MIT License
copyright (c) 2024 OchiLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from datetime import datetime, timedelta
import csv
import sys
import os

# 目のランドマークのインデックス
LEFT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]

def calculate_eye_aspect_ratio(eye_landmarks):
    """目のアスペクト比（EAR）を計算"""
    # 垂直方向の距離
    vertical_1 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - 
                                 np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    vertical_2 = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - 
                                 np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    
    # 水平方向の距離
    horizontal = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - 
                                np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    
    # EAR計算
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def format_timestamp(milliseconds):
    """ミリ秒をHH:MM:SS.mmm形式に変換"""
    td = timedelta(milliseconds=milliseconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    ms = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def main():
    # コマンドライン引数から動画ファイルパスを取得
    if len(sys.argv) < 2:
        print("使い方: python video_blink_detection.py <動画ファイルパス>")
        print("例: python video_blink_detection.py sample_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # 動画ファイルの存在確認
    if not os.path.exists(video_path):
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        sys.exit(1)
    
    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けません: {video_path}")
        sys.exit(1)
    
    # 動画の情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n=== 動画情報 ===")
    print(f"ファイル: {video_path}")
    print(f"解像度: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"総フレーム数: {total_frames}")
    print(f"再生時間: {format_timestamp(duration * 1000)}")
    print(f"\n処理を開始します...\n")
    
    # CSVファイルの準備
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_filename = f"blink_log_{video_name}_{timestamp}.csv"
    
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['瞬き番号', 'タイムスタンプ', 'フレーム番号', '左目EAR', '右目EAR', '平均EAR'])
    
    # Face Landmarkerモデルの初期化
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # 瞬き検出の設定
    EAR_THRESHOLD = 0.2  # この値以下で目が閉じていると判定
    CONSECUTIVE_FRAMES = 2  # この回数連続で閉じていたら瞬きと判定
    
    blink_records = []  # 瞬きの記録
    total_blinks = 0
    frame_counter = 0
    ear_values = []
    current_frame = 0
    
    # 瞬き検出中の一時データ
    blink_start_frame = None
    blink_ear_values = []
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            current_frame += 1
            
            # 進捗表示
            if current_frame % 30 == 0:
                progress = (current_frame / total_frames) * 100
                print(f"処理中... {progress:.1f}% ({current_frame}/{total_frames} フレーム) - 瞬き検出: {total_blinks}回", end='\r')
            
            # BGRからRGBに変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # タイムスタンプを取得（ミリ秒）
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # 顔検出を実行
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # 顔が検出された場合
            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                
                # 左目と右目のランドマークを取得
                left_eye = [landmarks[i] for i in LEFT_EYE_INDEXES]
                right_eye = [landmarks[i] for i in RIGHT_EYE_INDEXES]
                
                # 両目のEARを計算
                left_ear = calculate_eye_aspect_ratio(left_eye)
                right_ear = calculate_eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                ear_values.append(ear)
                
                # 瞬き検出
                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if blink_start_frame is None:
                        blink_start_frame = current_frame
                    blink_ear_values.append((left_ear, right_ear, ear))
                else:
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        total_blinks += 1
                        
                        # 瞬き中の平均EARを計算
                        avg_left_ear = np.mean([v[0] for v in blink_ear_values])
                        avg_right_ear = np.mean([v[1] for v in blink_ear_values])
                        avg_ear = np.mean([v[2] for v in blink_ear_values])
                        
                        # 瞬きを記録
                        blink_record = {
                            'blink_number': total_blinks,
                            'timestamp': format_timestamp(timestamp_ms),
                            'frame': blink_start_frame,
                            'left_ear': avg_left_ear,
                            'right_ear': avg_right_ear,
                            'avg_ear': avg_ear,
                            'duration_frames': frame_counter
                        }
                        blink_records.append(blink_record)
                        
                        # CSVに書き込み
                        csv_writer.writerow([
                            total_blinks,
                            format_timestamp(timestamp_ms),
                            blink_start_frame,
                            f"{avg_left_ear:.4f}",
                            f"{avg_right_ear:.4f}",
                            f"{avg_ear:.4f}"
                        ])
                    
                    # リセット
                    frame_counter = 0
                    blink_start_frame = None
                    blink_ear_values = []
    
    # リソースの解放
    cap.release()
    csv_file.close()
    
    # 結果の表示
    print(f"\n\n=== 処理完了 ===")
    print(f"総フレーム数: {current_frame}")
    print(f"総瞬き回数: {total_blinks}")
    
    if ear_values:
        print(f"\n=== EAR統計情報 ===")
        print(f"平均EAR: {np.mean(ear_values):.4f}")
        print(f"最小EAR: {np.min(ear_values):.4f}")
        print(f"最大EAR: {np.max(ear_values):.4f}")
        print(f"標準偏差: {np.std(ear_values):.4f}")
    
    if total_blinks > 0:
        print(f"\n=== 瞬き詳細 ===")
        print(f"瞬き頻度: {total_blinks / (duration / 60):.2f} 回/分")
        print(f"\n最初の5回の瞬き:")
        for i, record in enumerate(blink_records[:5]):
            print(f"  {i+1}. {record['timestamp']} (フレーム {record['frame']}) - EAR: {record['avg_ear']:.4f}")
    
    print(f"\n結果をCSVファイルに保存しました: {csv_filename}")
    
    # 詳細なレポートファイルを作成
    report_filename = f"blink_report_{video_name}_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("瞬き検出レポート\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"動画ファイル: {video_path}\n")
        f.write(f"解析日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        
        f.write("--- 動画情報 ---\n")
        f.write(f"解像度: {width}x{height}\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"総フレーム数: {total_frames}\n")
        f.write(f"再生時間: {format_timestamp(duration * 1000)}\n\n")
        
        f.write("--- 検出結果 ---\n")
        f.write(f"総瞬き回数: {total_blinks}\n")
        if total_blinks > 0:
            f.write(f"瞬き頻度: {total_blinks / (duration / 60):.2f} 回/分\n")
        f.write(f"\n")
        
        if ear_values:
            f.write("--- EAR統計 ---\n")
            f.write(f"平均EAR: {np.mean(ear_values):.4f}\n")
            f.write(f"最小EAR: {np.min(ear_values):.4f}\n")
            f.write(f"最大EAR: {np.max(ear_values):.4f}\n")
            f.write(f"標準偏差: {np.std(ear_values):.4f}\n\n")
        
        f.write("--- 全瞬き記録 ---\n")
        f.write(f"{'No.':<5} {'タイムスタンプ':<15} {'フレーム':<10} {'左目EAR':<12} {'右目EAR':<12} {'平均EAR':<12}\n")
        f.write("-" * 80 + "\n")
        for record in blink_records:
            f.write(f"{record['blink_number']:<5} {record['timestamp']:<15} {record['frame']:<10} "
                   f"{record['left_ear']:<12.4f} {record['right_ear']:<12.4f} {record['avg_ear']:<12.4f}\n")
    
    print(f"詳細レポートを保存しました: {report_filename}")

if __name__ == "__main__":
    main()
