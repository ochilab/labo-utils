"""
Camera Recorder - ウェブカメラ録画プログラム

このプログラムはウェブカメラからの映像をリアルタイムで録画します。
録画された動画には、フレーム番号とタイムスタンプが焼き込まれ、
後から正確な時間の確認が可能です。

Features:
- リアルタイムプレビュー表示
- フレーム番号とタイムスタンプの焼き込み
- スペースキーで録画開始/停止
- 実測FPSによる正確な録画

Author: OchiLab
Version: 1.0.0
License: MIT License

Copyright (c) 2024 OchiLab

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
from datetime import datetime
import os

def main():
    # ウェブカメラの初期化
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("エラー: カメラを開けません")
        return

    # カメラの解像度を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # カメラのFPSは信頼できないことが多いので、実測する
    print("FPSを測定中...")
    frame_count = 0
    start_time = datetime.now()
    
    while frame_count < 30:  # 30フレーム測定
        ret, _ = cap.read()
        if ret:
            frame_count += 1
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    measured_fps = frame_count / elapsed if elapsed > 0 else 30
    
    # 測定したFPSを使用（通常30fpsに近い値になるはず）
    fps = int(round(measured_fps))
    
    print("=" * 60)
    print("ウェブカメラ録画プログラム")
    print("=" * 60)
    print(f"解像度: {width}x{height}")
    print(f"カメラ報告FPS: {reported_fps}")
    print(f"実測FPS: {measured_fps:.2f}")
    print(f"使用FPS: {fps}")
    print("\n操作方法:")
    print("  SPACE: 録画開始/停止")
    print("  q: 終了")
    print("=" * 60 + "\n")
    
    # 録画用の変数
    is_recording = False
    video_writer = None
    recording_start_time = None
    output_filename = None
    recording_frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを取得できません")
            break
        
        # 録画中のみフレーム番号をカウント
        if is_recording:
            recording_frame_number += 1
        
        # 現在時刻を取得
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # フレームを反転（鏡のように）
        frame = cv2.flip(frame, 1)
        
        # フォント設定
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        
        # フレーム番号のテキスト
        frame_text = f"Frame: {recording_frame_number}"
        (frame_text_width, frame_text_height), _ = cv2.getTextSize(frame_text, font, font_scale, thickness)
        
        # タイムスタンプのテキスト
        timestamp_text = timestamp_str
        (time_text_width, time_text_height), _ = cv2.getTextSize(timestamp_text, font, font_scale, thickness)
        
        # 最も幅の広いテキストを基準に背景の幅を決定
        max_text_width = max(frame_text_width, time_text_width)
        
        # 左下の位置を計算
        y_bottom = height - 5
        
        # 背景の矩形を描画（フレーム番号とタイムスタンプの両方を含む）
        total_height = frame_text_height + time_text_height + padding * 3
        cv2.rectangle(frame, 
                     (5, y_bottom - total_height), 
                     (5 + max_text_width + padding * 2, y_bottom), 
                     (0, 0, 0), 
                     -1)
        
        # フレーム番号を描画（上側）
        cv2.putText(frame, 
                   frame_text, 
                   (5 + padding, y_bottom - time_text_height - padding * 2), 
                   font, 
                   font_scale, 
                   (255, 255, 0),  # 黄色
                   thickness)
        
        # タイムスタンプを描画（下側）
        cv2.putText(frame, 
                   timestamp_text, 
                   (5 + padding, y_bottom - padding), 
                   font, 
                   font_scale, 
                   (0, 255, 0),  # 緑
                   thickness)
        
        # 録画中の場合
        if is_recording:
            # 録画時間を計算
            elapsed_time = (current_time - recording_start_time).total_seconds()
            elapsed_str = f"REC: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
            
            # 録画インジケーターを描画（右上）
            rec_color = (0, 0, 255) if int(elapsed_time * 2) % 2 == 0 else (0, 100, 255)  # 点滅効果
            cv2.circle(frame, (width - 30, 30), 10, rec_color, -1)
            cv2.putText(frame, 
                       elapsed_str, 
                       (width - 150, 40), 
                       font, 
                       0.7, 
                       (0, 0, 255), 
                       2)
            
            # フレームを書き込み
            video_writer.write(frame)
        
        # プレビュー表示
        display_frame = frame.copy()
        
        # 操作説明を表示（下部）
        instruction = "SPACE: Start/Stop Recording | Q: Quit"
        cv2.putText(display_frame, 
                   instruction, 
                   (10, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, 
                   (255, 255, 255), 
                   2)
        
        cv2.imshow('Camera Recording', display_frame)
        
        # キー入力処理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # スペースキー
            if not is_recording:
                # 録画開始
                recording_frame_number = 0  # フレーム番号をリセット
                recording_start_time = datetime.now()
                timestamp_for_filename = recording_start_time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"recording_{timestamp_for_filename}.mp4"
                
                # VideoWriterの初期化
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                
                is_recording = True
                print(f"[{timestamp_str}] 録画開始: {output_filename}")
            else:
                # 録画停止
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                
                # ファイルサイズを取得
                file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
                elapsed_time = (datetime.now() - recording_start_time).total_seconds()
                
                print(f"[{timestamp_str}] 録画停止: {output_filename}")
                print(f"  録画時間: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}")
                print(f"  ファイルサイズ: {file_size:.2f} MB")
    
    # リソースの解放
    if video_writer is not None:
        video_writer.release()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n終了しました")

if __name__ == "__main__":
    main()
