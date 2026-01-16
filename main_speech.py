import sys
import os
import tempfile
import asyncio
import uuid 
import struct 
import random # [新增] 需要用到隨機數來決定眨眼間隔
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt5.QtMultimedia import QAudioProbe

import edge_tts
from openai import OpenAI
from AI_agent_frame import Ui_MainWindow

# --- 設定區 ---
CHAT_API_KEY = "ollama"
CHAT_BASE_URL = "http://localhost:11434/v1"
CHAT_MODEL_NAME = "llama3" 
TTS_VOICE = "zh-TW-HsiaoChenNeural"

# ... (LLMWorker 和 TTSWorker 類別保持不變，不用修改) ...
class LLMWorker(QThread):
    response_received = pyqtSignal(str) 
    def __init__(self, history):
        super().__init__()
        self.history = history
        self.client = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_BASE_URL)
    def run(self):
        try:
            response = self.client.chat.completions.create(
                model=CHAT_MODEL_NAME, messages=self.history, temperature=0.7
            )
            self.response_received.emit(response.choices[0].message.content)
        except Exception as e:
            self.response_received.emit(f"文字連線錯誤: {str(e)}")

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

# --- 主程式修改區 ---
class MyMainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainApp, self).__init__(parent)
        
        self.setupUi(self)
        self.setup_chat_ui()
        self.setup_agent_ui() # 這裡會載入新的眨眼圖片
        self.setup_signal_connections()

        self.player = QtMultimedia.QMediaPlayer()
        self.player.setVolume(80)

        self.probe = QAudioProbe()
        self.probe.setSource(self.player)
        self.probe.audioBufferProbed.connect(self.process_audio_buffer)

        # --- 說話動畫計時器 ---
        self.talk_anim_timer = QtCore.QTimer(self)
        self.talk_anim_timer.setInterval(150)
        self.talk_anim_timer.timeout.connect(self.update_talk_animation)

        # --- 說話延遲閉嘴計時器 ---
        self.stop_talking_timer = QtCore.QTimer(self)
        self.stop_talking_timer.setInterval(300)
        self.stop_talking_timer.setSingleShot(True)
        self.stop_talking_timer.timeout.connect(self.stop_talking_immediate)

        # --- [新增] 眨眼相關計時器 ---
        # 1. 觸發器：決定多久眨一次眼 (隨機長間隔)
        self.blink_trigger_timer = QtCore.QTimer(self)
        self.blink_trigger_timer.setSingleShot(True) # 每次只觸發一次，然後重新設定隨機時間
        self.blink_trigger_timer.timeout.connect(self.try_start_blink)
        
        # 2. 動畫器：執行快速的眨眼動作 (短固定間隔)
        self.blink_anim_timer = QtCore.QTimer(self)
        self.blink_anim_timer.setInterval(30) # 50ms 切換一張圖，動作比較快
        self.blink_anim_timer.timeout.connect(self.update_blink_animation)
        
        # 眨眼序列索引 (0:睜眼, 1:半閉, 2:全閉) -> 順序: 睜->半->全->半->睜
        self.blink_sequence = [0, 1, 2, 1, 0] 
        self.blink_step = 0

        self.is_talking = False
        self.is_blinking = False # [新增] 追蹤是否正在眨眼中
        self.current_audio_path = None
        self.conversation_history = [{"role": "system", "content": "你是一個專業的 AAEON 智慧助理，說話簡潔有力，並使用繁體中文回答。"}]

        # 啟動第一次的眨眼排程
        self.schedule_next_blink()

        QtCore.QTimer.singleShot(500, lambda: self.add_chat_bubble("你好！我是 AAEON 智慧助理，請下達指令。", is_user=False))

        shadow = QGraphicsDropShadowEffect(self.frame_3)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 8))
        self.frame_3.setGraphicsEffect(shadow)

    def setup_agent_ui(self):
        layout = QtWidgets.QVBoxLayout(self.agent)
        layout.setContentsMargins(50, 100, 50, 0) 
        self.avatar_label = QtWidgets.QLabel(self.agent)
        self.avatar_label.setScaledContents(True)
        self.avatar_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        
        # --- [修改] 載入兩組圖片：說話用 & 眨眼用 ---
        # 1. 說話嘴型圖 (睜眼狀態)
        talk_paths = [
            "./pic/avatar_idle.png",   # [0] 靜止閉嘴 (基底圖)
            "./pic/avatar_talk1.png",  # [1] 張嘴小
            "./pic/avatar_talk2.png"   # [2] 張嘴大
        ]
        # 2. 眨眼圖 (閉嘴狀態)
        blink_paths = [
            "./pic/avatar_idle.png",         # [0] 睜眼 (同基底圖)
            "./pic/avatar_blink_half.png",   # [1] 半閉眼
            "./pic/avatar_blink_closed.png"  # [2] 全閉眼
        ]

        self.talk_pixmaps = self._load_pixmaps(talk_paths)
        self.blink_pixmaps = self._load_pixmaps(blink_paths)

        # 設定初始圖片
        self.set_idle_image()
        layout.addWidget(self.avatar_label)

    def _load_pixmaps(self, paths):
        """ 輔助函式：載入圖片列表 """
        pixmaps = []
        for path in paths:
            pix = QtGui.QPixmap(path)
            if pix.isNull():
                # print(f"Missing image: {path}") # 除錯用
                pix = QtGui.QPixmap(500, 500)
                pix.fill(Qt.transparent)
            pixmaps.append(pix)
        return pixmaps

    # --- [新增] 眨眼控制邏輯 ---
    def schedule_next_blink(self):
        """ 排程下一次的眨眼 (隨機 2~6 秒後觸發) """
        # 只有在「沒說話」且「沒在眨眼」時才排程，避免衝突
        if not self.is_talking and not self.is_blinking:
            next_interval = random.randint(2000, 7000) # 隨機毫秒數
            self.blink_trigger_timer.start(next_interval)

    def try_start_blink(self):
        """ 嘗試開始眨眼動作 """
        # 再次檢查：如果正在說話，就取消這次眨眼，重新排程
        if self.is_talking:
            self.schedule_next_blink()
            return
            
        # 開始眨眼動畫
        self.is_blinking = True
        self.blink_step = 0
        self.blink_anim_timer.start()

    def update_blink_animation(self):
        """ 執行眨眼動畫序列 (睜->半->全->半->睜) """
        if not self.blink_pixmaps: return
        
        # 取得當前步驟對應的圖片索引
        pix_index = self.blink_sequence[self.blink_step]
        self.avatar_label.setPixmap(self.blink_pixmaps[pix_index])
        
        self.blink_step += 1
        # 如果序列跑完了
        if self.blink_step >= len(self.blink_sequence):
            self.blink_anim_timer.stop()
            self.is_blinking = False
            self.set_idle_image() # 確保回到靜止狀態
            self.schedule_next_blink() # 排程下一次眨眼

    # --- 音訊處理邏輯 (有修改以配合眨眼) ---
    def process_audio_buffer(self, buffer):
        fmt = buffer.format()
        if fmt.sampleType() != QtMultimedia.QAudioFormat.SignedInt: return
        
        try:
            ptr = buffer.constData()
            ptr.setsize(buffer.byteCount())
            raw_data = bytes(ptr)
            count = len(raw_data) // 2
            shorts = struct.unpack(f"{count}h", raw_data)
            
            max_amplitude = 0
            for i in range(0, len(shorts), 50):
                val = abs(shorts[i])
                if val > max_amplitude: max_amplitude = val
            
            THRESHOLD = 500
            
            if max_amplitude > THRESHOLD:
                # [有聲音 -> 開始說話]
                if self.stop_talking_timer.isActive():
                    self.stop_talking_timer.stop()

                if not self.is_talking:
                    self.is_talking = True
                    # [重要] 開始說話時，強制停止任何正在進行的眨眼或眨眼排程
                    self.blink_trigger_timer.stop()
                    self.blink_anim_timer.stop()
                    self.is_blinking = False 
                    
                    self.talk_anim_timer.start()
            else:
                # [沒聲音 -> 準備停止]
                if self.is_talking and not self.stop_talking_timer.isActive():
                    self.stop_talking_timer.start()

        except Exception:
            pass

    def stop_talking_immediate(self):
        """ 真的閉嘴 """
        self.is_talking = False
        self.talk_anim_timer.stop()
        self.set_idle_image()
        # [重要] 停止說話後，重新開始排程眨眼
        self.schedule_next_blink()

    def update_talk_animation(self):
        """ 說話嘴型切換 (只用 talk_pixmaps) """
        if not self.talk_pixmaps: return
        # 在索引 1(小張) 和 2(大張) 之間隨機切換，偶爾回 0(閉嘴)
        next_index = random.choice([1, 2, 1, 2, 0]) 
        self.avatar_label.setPixmap(self.talk_pixmaps[next_index])

    def set_idle_image(self):
        """ 回到靜止狀態 (睜眼閉嘴) """
        if self.talk_pixmaps:
            # 使用 talk_pixmaps[0] 也就是 avatar_idle.png
            self.avatar_label.setPixmap(self.talk_pixmaps[0])

    # --- UI & Handle Functions (保持不變) ---
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
            self.stop_talking_immediate()

        self.add_chat_bubble(text, is_user=True)
        self.input_box.clear()
        self.conversation_history.append({"role": "user", "content": text})
        self.input_box.setPlaceholderText("思考中...")
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

    def play_audio(self, new_file_path):
        self.player.stop()
        self.stop_talking_immediate()
        self.player.setMedia(QtMultimedia.QMediaContent()) 
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            try:
                os.remove(self.current_audio_path)
            except Exception:
                pass
        self.current_audio_path = new_file_path
        try:
            url = QUrl.fromLocalFile(new_file_path)
            content = QtMultimedia.QMediaContent(url)
            self.player.setMedia(content)
            self.player.play()
        except Exception as e:
            print(f"播放失敗: {e}")

    def closeEvent(self, event):
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