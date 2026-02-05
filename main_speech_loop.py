import cv2
import sys
import os
import tempfile
import asyncio
import uuid  
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl

# --- Google Generative AI SDK ---
import google.generativeai as genai

import edge_tts
from AI_agent_frame import Ui_MainWindow

# --- 設定區 ---
API_KEY = "AIzaSyBdfjLFS2Pd8eiDphgZm2DJZa_QwRjyfoU" 
MODEL_NAME = "gemini-flash-latest"
TTS_VOICE = "zh-TW-HsiaoChenNeural"

# --- 影片播放執行緒 (邏輯不變，它會傻傻地照著列表播) ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    video_switched = pyqtSignal(str) 

    def __init__(self, idle_paths, talk_path, width=400, height=600):
        super().__init__()
        self.idle_paths = idle_paths 
        self.talk_path = talk_path
        
        self.current_path = self.idle_paths[0]
        self.is_talking_mode = False 
        self.idle_index = 0 
        
        self._run_flag = True
        self.resize_w = width
        self.resize_h = height

    def set_talking_mode(self, talking: bool):
        self.is_talking_mode = talking

    def run(self):
        cap = cv2.VideoCapture(self.current_path)

        while self._run_flag:
            ret, frame = cap.read()
            
            if ret:
                try:
                    frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                
                self.change_pixmap_signal.emit(qt_image)
                self.msleep(33) 
            
            else:
                # --- 影片結束 (Loop End) ---
                next_path = ""

                if self.is_talking_mode:
                    next_path = self.talk_path
                else:
                    # [這裡的邏輯會自動處理輪播]
                    if self.current_path == self.talk_path:
                        self.idle_index = 0
                    else:
                        # 依序切換到列表中的下一個
                        self.idle_index = (self.idle_index + 1) % len(self.idle_paths)
                    
                    next_path = self.idle_paths[self.idle_index]

                if next_path != self.current_path:
                    self.current_path = next_path
                    cap.release()
                    
                    if os.path.exists(self.current_path):
                        cap = cv2.VideoCapture(self.current_path)
                        self.video_switched.emit(self.current_path)
                    else:
                        print(f"找不到影片: {self.current_path}")
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# --- LLMWorker (不變) ---
class LLMWorker(QThread):
    response_received = pyqtSignal(str) 

    def __init__(self, history):
        super().__init__()
        self.history = history 
        genai.configure(api_key=API_KEY.strip())

    def run(self):
        try:
            gemini_history = []
            system_instruction = "你是一個有用的助手。"
            last_user_message = ""
            for msg in self.history:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    if msg == self.history[-1]:
                        last_user_message = content
                    else:
                        gemini_history.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    gemini_history.append({"role": "model", "parts": [content]})

            model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction)
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(last_user_message)
            self.response_received.emit(response.text)
        except Exception as e:
            error_msg = str(e)
            print(f"Gemini Error: {error_msg}")
            if "429" in error_msg:
                self.response_received.emit("錯誤：請求太頻繁或額度不足，請稍後再試。")
            elif "404" in error_msg:
                self.response_received.emit(f"錯誤：找不到模型 {MODEL_NAME}。")
            else:
                self.response_received.emit(f"連線錯誤: {error_msg}")

# --- TTSWorker (不變) ---
class TTSWorker(QThread):
    audio_ready = pyqtSignal(str)
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.voice = TTS_VOICE
    async def _generate_audio_async(self, file_path):
        communicate = edge_tts.Communicate(self.text, self.voice)
        await communicate.save(file_path)
    def run(self):
        try:
            temp_dir = tempfile.gettempdir()
            unique_filename = f"speech_{uuid.uuid4()}.mp3"
            speech_file_path = str(Path(temp_dir) / unique_filename)
            asyncio.run(self._generate_audio_async(speech_file_path))
            self.audio_ready.emit(speech_file_path)
        except Exception as e:
            print(f"TTS Error: {e}")

# --- 主程式 ---
class MyMainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainApp, self).__init__(parent)
        
        self.setupUi(self)
        self.setup_chat_ui()
        
        # --- [修改重點] 設定播放清單順序 ---
        # 想要 2 次 idle1, 1 次 idle2 -> 就把 idle1 加兩次進去
        # 順序變成：[idle1 -> idle1 -> idle2 -> idle1 -> idle1 -> idle2 ...]
        self.IDLE_PATHS = [
            "./vedio/avatar_idle2.mp4", 
            "./vedio/avatar_idle2.mp4", 
            "./vedio/avatar_idle2.mp4", 
            "./vedio/avatar_idle1.mp4"
        ]
        self.TALK_PATH = "./vedio/avatar_talk.mp4"
        
        self.setup_agent_ui() 
        self.setup_signal_connections()

        self.player = QtMultimedia.QMediaPlayer()
        self.player.setVolume(80)
        self.player.stateChanged.connect(self.handle_player_state_changed)

        self.current_audio_path = None
        self.pending_audio_path = None
        
        self.conversation_history = [{"role": "system", "content": "你是一個專業的 AAEON 智慧助理，說話簡潔有力，並使用繁體中文回答。"}]

        QtCore.QTimer.singleShot(500, lambda: self.add_chat_bubble("你好！我是 AAEON 智慧助理 (Gemini 1.5)，請下達指令。", is_user=False))
        
        shadow = QGraphicsDropShadowEffect(self.frame_3)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 8))
        self.frame_3.setGraphicsEffect(shadow)

    def setup_agent_ui(self):
        layout = QtWidgets.QVBoxLayout(self.agent)
        layout.setContentsMargins(0, 50, 0, 0)
        layout.addStretch(1)
        
        self.avatar_label = QtWidgets.QLabel(self.agent)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setScaledContents(True) 
        layout.addWidget(self.avatar_label)

        self.video_thread = VideoThread(self.IDLE_PATHS, self.TALK_PATH, width=400, height=600)
        self.video_thread.change_pixmap_signal.connect(self.update_avatar_frame)
        self.video_thread.video_switched.connect(self.on_video_switched)
        self.video_thread.start()

    def update_avatar_frame(self, q_img):
        self.avatar_label.setPixmap(QPixmap.fromImage(q_img))

    def play_audio(self, new_file_path):
        self.player.stop()
        self.player.setMedia(QtMultimedia.QMediaContent())
        
        self.video_thread.set_talking_mode(True)
        
        current_video = self.video_thread.current_path
        
        if current_video == self.TALK_PATH:
            self.start_player_immediately(new_file_path)
        else:
            self.pending_audio_path = new_file_path
            print("等待 Idle 動作結束後切換為說話...")

    def start_player_immediately(self, file_path):
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            try:
                os.remove(self.current_audio_path)
            except Exception:
                pass
        
        self.current_audio_path = file_path
        try:
            url = QUrl.fromLocalFile(file_path)
            content = QtMultimedia.QMediaContent(url)
            self.player.setMedia(content)
            self.player.play()
        except Exception as e:
            print(f"播放失敗: {e}")

    def on_video_switched(self, new_path):
        if new_path == self.TALK_PATH and self.pending_audio_path:
            print("影片動作已就緒，開始播放音訊。")
            self.start_player_immediately(self.pending_audio_path)
            self.pending_audio_path = None 

    def handle_player_state_changed(self, state):
        if state == QtMultimedia.QMediaPlayer.StoppedState:
            if self.pending_audio_path is None:
                self.video_thread.set_talking_mode(False)

    def setup_chat_ui(self):
        self.chat_main_layout = QtWidgets.QVBoxLayout(self.frame)
        self.chat_main_layout.setContentsMargins(10, 0, 10, 10)
        self.chat_main_layout.setSpacing(10)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea {border: none; background: transparent;} QScrollBar:vertical {background: #E0E0E0; width: 8px; border-radius: 4px;} QScrollBar::handle:vertical {background: #A0A0A0; border-radius: 4px;}")
        self.scroll_content_widget = QtWidgets.QWidget()
        self.scroll_content_widget.setStyleSheet("background-color: transparent;")
        self.scroll_area.setWidget(self.scroll_content_widget)
        self.messages_layout = QtWidgets.QVBoxLayout(self.scroll_content_widget)
        self.messages_layout.setContentsMargins(5, 5, 5, 5)
        self.messages_layout.setSpacing(15)
        self.messages_layout.addStretch()
        self.chat_main_layout.addWidget(self.scroll_area)
        self.input_container = QtWidgets.QHBoxLayout()
        self.input_box = QtWidgets.QLineEdit()
        self.input_box.setPlaceholderText("請輸入訊息...")
        self.input_box.setMinimumHeight(40)
        self.input_box.setStyleSheet("QLineEdit {border: 2px solid #ccc; border-radius: 20px; padding: 5px 15px; font-size: 16px; background-color: #FFFFFF;} QLineEdit:focus {border: 2px solid #009FDC;}")
        self.send_btn = QtWidgets.QPushButton("發送")
        self.send_btn.setMinimumHeight(40)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setStyleSheet("QPushButton {background-color: #009FDC; color: white; border-radius: 20px; padding: 5px 20px; font-weight: bold; font-size: 16px;} QPushButton:hover {background-color: #008CC2;} QPushButton:pressed {background-color: #0079A8;}")
        self.input_container.addWidget(self.input_box)
        self.input_container.addWidget(self.send_btn)
        self.chat_main_layout.addLayout(self.input_container)

    def setup_signal_connections(self):
        self.send_btn.clicked.connect(self.handle_send)
        self.input_box.returnPressed.connect(self.handle_send)

    def handle_send(self):
        text = self.input_box.text().strip()
        if not text: return
        
        if self.player.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.player.stop()
        
        self.pending_audio_path = None
        self.video_thread.set_talking_mode(False) 

        self.add_chat_bubble(text, is_user=True)
        self.input_box.clear()
        self.conversation_history.append({"role": "user", "content": text})
        self.input_box.setPlaceholderText("Gemini 思考中...")
        self.input_box.setEnabled(False)
        self.llm_worker = LLMWorker(self.conversation_history)
        self.llm_worker.response_received.connect(self.handle_ai_response)
        self.llm_worker.start()

    def handle_ai_response(self, response_text):
        self.conversation_history.append({"role": "assistant", "content": response_text})
        self.add_chat_bubble(response_text, is_user=False)
        self.input_box.setEnabled(True)
        self.input_box.setPlaceholderText("請輸入訊息...")
        self.input_box.setFocus()
        self.tts_worker = TTSWorker(response_text)
        self.tts_worker.audio_ready.connect(self.play_audio)
        self.tts_worker.start()

    def closeEvent(self, event):
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
        
        if self.current_audio_path:
            self.player.stop()
            self.player.setMedia(QtMultimedia.QMediaContent())
            try:
                if os.path.exists(self.current_audio_path):
                    os.remove(self.current_audio_path)
            except Exception:
                pass
        event.accept()

    def add_chat_bubble(self, text, is_user):
        bubble_widget = QtWidgets.QWidget()
        bubble_layout = QtWidgets.QHBoxLayout(bubble_widget)
        bubble_layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setMaximumWidth(600)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = QtGui.QFont()
        font.setPointSize(11)
        label.setFont(font)
        if is_user:
            label.setStyleSheet("QLabel {background-color: #0080FF; color: white; border-radius: 10px; padding: 10px;}")
            bubble_layout.addStretch()
            bubble_layout.addWidget(label)
        else:
            label.setStyleSheet("QLabel {background-color: #FFFFFF; color: black; border: 1px solid #E0E0E0; border-radius: 10px; padding: 10px;}")
            bubble_layout.addWidget(label)
            bubble_layout.addStretch()
        self.messages_layout.addWidget(bubble_widget)
        QtCore.QTimer.singleShot(100, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont("Microsoft JhengHei", 10)
    app.setFont(font)
    window = MyMainApp()
    window.show()
    sys.exit(app.exec_())
