import os
import sys

# --- 路徑與環境設定 (必須放在最前面) ---

# 1. 取得絕對路徑
ai_agent_dir = os.path.dirname(os.path.abspath(__file__)) # .../ai_agent
project_root = os.path.dirname(ai_agent_dir) # .../MuseTalk
musetalk_root = os.path.join(project_root, "MuseTalk") # .../MuseTalk/MuseTalk

# 2. 強制將工作目錄切換到 MuseTalk 根目錄 (為了解決 preprocessing.py 的路徑問題)
print(f"正在切換工作目錄至: {musetalk_root}")
if os.path.exists(musetalk_root):
    os.chdir(musetalk_root)
else:
    print(f"嚴重錯誤: 找不到目錄 {musetalk_root}")

# 3. 更新 Python 搜尋路徑
if ai_agent_dir not in sys.path:
    sys.path.insert(0, ai_agent_dir)
if musetalk_root not in sys.path:
    sys.path.insert(0, musetalk_root)

# --- 標準 Imports ---
import cv2
import tempfile
import asyncio
import uuid  
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl

from musetalk_worker import MuseTalkWorker
import google.generativeai as genai
import edge_tts
from AI_agent_frame import Ui_MainWindow

# --- 設定區 ---
API_KEY = "AIzaSyBPWC3wGHW5NFFp1gFRvUK170v3VnTHCF8" 
MODEL_NAME = "gemini-flash-latest"
TTS_VOICE = "zh-TW-HsiaoChenNeural"

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, idle_paths, width=400, height=600):
        super().__init__()
        self.idle_paths = idle_paths 
        self.current_path = self.idle_paths[0]
        self.idle_index = 0 
        self._run_flag = True
        self.resize_w = width
        self.resize_h = height
        self.mode = "IDLE" 
        self.generated_frames = [] 
        self.gen_index = 0

    def start_talking(self, frames):
        self.generated_frames = frames
        self.gen_index = 0
        self.mode = "TALK_GEN"

    def stop_talking(self):
        self.mode = "IDLE"

    def run(self):
        cap = cv2.VideoCapture(self.current_path)
        while self._run_flag:
            if self.mode == "TALK_GEN":
                if self.gen_index < len(self.generated_frames):
                    qt_image = self.generated_frames[self.gen_index]
                    if qt_image.width() != self.resize_w:
                        qt_image = qt_image.scaled(self.resize_w, self.resize_h, Qt.KeepAspectRatio)
                    self.change_pixmap_signal.emit(qt_image)
                    self.gen_index += 1
                    self.msleep(40) 
                else:
                    self.mode = "IDLE"
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(self.current_path)
            else:
                ret, frame = cap.read()
                if ret:
                    try:
                        frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()
                        self.change_pixmap_signal.emit(qt_image)
                    except: pass
                    self.msleep(33)
                else:
                    self.idle_index = (self.idle_index + 1) % len(self.idle_paths)
                    self.current_path = self.idle_paths[self.idle_index]
                    cap.release()
                    cap = cv2.VideoCapture(self.current_path)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class LLMWorker(QThread):
    response_received = pyqtSignal(str) 
    def __init__(self, history):
        super().__init__()
        self.history = history 
        genai.configure(api_key=API_KEY.strip())
    def run(self):
        try:
            gemini_history = []
            system_instruction = "你現在是研揚科技AAEON的專業助理請以繁體中文回答問題"
            last_user_message = ""
            for msg in self.history:
                if msg["role"] == "system": system_instruction = msg["content"]
                elif msg["role"] == "user":
                    if msg == self.history[-1]: last_user_message = msg["content"]
                    else: gemini_history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_history.append({"role": "model", "parts": [msg["content"]]})
            model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction)
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(last_user_message)
            self.response_received.emit(response.text)
        except Exception as e:
            error_msg = str(e)
            print(f"Gemini Error: {error_msg}")
            if "429" in error_msg: self.response_received.emit("錯誤：請求太頻繁。但我這邊作一個長度與邊點符號的測試。")
            else: self.response_received.emit(f"連線錯誤: {error_msg}")

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

class MyMainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainApp, self).__init__(parent)
        self.setupUi(self)
        self.setup_chat_ui()
        
        # 絕對路徑設定 Idle 影片
        self.IDLE_PATHS = [
            os.path.join(ai_agent_dir, "vedio/avatar_idle2.mp4"),
            os.path.join(ai_agent_dir, "vedio/avatar_idle2.mp4"),
            os.path.join(ai_agent_dir, "vedio/avatar_idle2.mp4"),
            os.path.join(ai_agent_dir, "vedio/avatar_idle1.mp4")
        ]
        
        self.setup_agent_ui() 
        self.setup_signal_connections()

        self.player = QtMultimedia.QMediaPlayer()
        self.player.setVolume(80)
        
        self.current_audio_path = None
        self.pending_audio_path = None
        self.conversation_history = [{"role": "system", "content": "你現在是研揚科技AAEON的專業助理請以繁體中文回答問題回答內容請盡量控制在30字以內且禁止輸出任何標點符號若語句需停頓請務必使用空格取代以利語音朗讀"}]
        
        # [新增] 暫存 AI 回應文字
        self.pending_ai_text = None

        QtCore.QTimer.singleShot(500, lambda: self.add_chat_bubble("你好！我是 AAEON 智慧助理，有什麼需要協助的嗎？", is_user=False))
        
        shadow = QGraphicsDropShadowEffect(self.frame_3)
        shadow.setBlurRadius(10); shadow.setColor(QColor(0, 0, 0, 8))
        self.frame_3.setGraphicsEffect(shadow)

    def setup_agent_ui(self):
        layout = QtWidgets.QVBoxLayout(self.agent)
        layout.setContentsMargins(0, 50, 0, 0)
        layout.addStretch(1)
        self.avatar_label = QtWidgets.QLabel(self.agent)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setScaledContents(True) 
        layout.addWidget(self.avatar_label)

        # 初始化 MuseTalk (確保路徑正確)
        avatar_talk_path = os.path.join(ai_agent_dir, "vedio/avatar_small_talk.mp4")
        self.musetalk_worker = MuseTalkWorker(
            avatar_video_path=avatar_talk_path,
            model_config_path="./models/musetalkV15/musetalk.json",
            model_path="./models/musetalkV15/unet.pth",
            whisper_path="./models/whisper"
        )
        self.musetalk_worker.frames_ready.connect(self.on_musetalk_ready)
        self.musetalk_worker.start()

        self.video_thread = VideoThread(self.IDLE_PATHS, width=400, height=600)
        self.video_thread.change_pixmap_signal.connect(self.update_avatar_frame)
        self.video_thread.start()

    def update_avatar_frame(self, q_img):
        self.avatar_label.setPixmap(QPixmap.fromImage(q_img))

    def start_player_immediately(self, file_path):
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            try: os.remove(self.current_audio_path)
            except: pass
        self.current_audio_path = file_path
        url = QUrl.fromLocalFile(file_path)
        self.player.setMedia(QtMultimedia.QMediaContent(url))
        self.player.play()

    def setup_chat_ui(self):
        self.chat_main_layout = QtWidgets.QVBoxLayout(self.frame)
        self.chat_main_layout.setContentsMargins(10, 0, 10, 10)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea {border: none; background: transparent;}")
        self.scroll_content_widget = QtWidgets.QWidget()
        self.scroll_content_widget.setStyleSheet("background: transparent;")
        self.scroll_area.setWidget(self.scroll_content_widget)
        self.messages_layout = QtWidgets.QVBoxLayout(self.scroll_content_widget)
        self.messages_layout.addStretch()
        self.chat_main_layout.addWidget(self.scroll_area)
        self.input_container = QtWidgets.QHBoxLayout()
        self.input_box = QtWidgets.QLineEdit()
        self.input_box.setPlaceholderText("請輸入訊息...")
        self.input_box.setMinimumHeight(40)
        self.input_box.setStyleSheet("QLineEdit {border: 2px solid #ccc; border-radius: 20px; padding: 5px 15px; background: white;}")
        self.send_btn = QtWidgets.QPushButton("發送")
        self.send_btn.setMinimumHeight(40)
        self.send_btn.setStyleSheet("QPushButton {background-color: #009FDC; color: white; border-radius: 20px; padding: 5px 20px;}")
        self.input_container.addWidget(self.input_box)
        self.input_container.addWidget(self.send_btn)
        self.chat_main_layout.addLayout(self.input_container)

    def setup_signal_connections(self):
        self.send_btn.clicked.connect(self.handle_send)
        self.input_box.returnPressed.connect(self.handle_send)

    def handle_send(self):
        text = self.input_box.text().strip()
        if not text: return
        self.player.stop()
        self.video_thread.stop_talking()
        self.add_chat_bubble(text, is_user=True)
        self.input_box.clear()
        self.conversation_history.append({"role": "user", "content": text})
        
        # 使用者送出後，進入等待狀態
        self.input_box.setPlaceholderText("思考中...")
        self.input_box.setEnabled(False)
        
        self.llm_worker = LLMWorker(self.conversation_history)
        self.llm_worker.response_received.connect(self.handle_ai_response)
        self.llm_worker.start()

    def handle_ai_response(self, response_text):
        # 1. 存入歷史
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # 2. [修改] 暫存文字，不馬上顯示氣泡，也不解鎖輸入框
        self.pending_ai_text = response_text
        self.input_box.setPlaceholderText("思考中...") # 更新狀態提示
        
        # 3. 開始 TTS -> MuseTalk
        self.tts_worker = TTSWorker(response_text)
        self.tts_worker.audio_ready.connect(lambda p: self.musetalk_worker.run_inference(p))
        self.tts_worker.start()

    def on_musetalk_ready(self, frames):
        # 1. 播放動作
        self.video_thread.start_talking(frames)
        
        # 2. 播放聲音 (透過 musetalk_worker 保存的路徑)
        if self.musetalk_worker.audio_path:
            self.start_player_immediately(self.musetalk_worker.audio_path)
            
        # 3. [新增] 影片準備好後，才顯示文字氣泡
        if self.pending_ai_text:
            self.add_chat_bubble(self.pending_ai_text, is_user=False)
            self.pending_ai_text = None
        
        # 4. [新增] 恢復輸入框
        self.input_box.setEnabled(True)
        self.input_box.setPlaceholderText("請輸入訊息...")
        self.input_box.setFocus()

    def add_chat_bubble(self, text, is_user):
        bubble = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(bubble)
        layout.setContentsMargins(0,0,0,0)
        lbl = QtWidgets.QLabel(text)
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(600)
        lbl.setFont(QtGui.QFont("Microsoft JhengHei", 11))
        lbl.setStyleSheet(f"background-color: {'#0080FF' if is_user else 'white'}; color: {'white' if is_user else 'black'}; border-radius: 10px; padding: 10px; border: {'none' if is_user else '1px solid #ddd'};")
        if is_user: layout.addStretch(); layout.addWidget(lbl)
        else: layout.addWidget(lbl); layout.addStretch()
        self.messages_layout.addWidget(bubble)
        QtCore.QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum()))

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Microsoft JhengHei", 10))
    window = MyMainApp()
    window.show()
    sys.exit(app.exec_())