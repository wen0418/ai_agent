import sys
import cv2  # 必須安裝: pip install opencv-python
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 引入 Google 最新版 SDK
from google import genai
from google.genai import types

# 匯入你原本的介面檔案
from AI_agent_frame import Ui_MainWindow

# --- LLM 設定區 ---
API_KEY = "AIzaSyAcfVQMO0agc2StxF1lAY8MWsFwnwIHyFY" 
MODEL_NAME = "gemini-2.0-flash"
# -------------------------------------

class VideoThread(QThread):
    """
    專門負責讀取影片並發送畫面的執行緒。
    加上 .copy() 是為了解決移動視窗時記憶體衝突導致的閃退問題。
    """
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._run_flag = True

    def run(self):
        # 檢查檔案是否存在
        if not os.path.exists(self.video_path):
            print(f"找不到影片檔案: {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        
        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # 1. 轉換顏色空間 (BGR -> RGB)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. 取得圖片資訊
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                # [核心修正]：加上 .copy()。
                # 這會複製一份影像數據，讓 GUI 執行緒擁有獨立的緩衝區。
                # 這樣移動視窗導致 GUI 暫停時，就不會因為讀取到背景執行緒已釋放的記憶體而崩潰。
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                
                # 3. 發送訊號給主介面更新
                self.change_pixmap_signal.emit(convert_to_Qt_format)
                
                # 4. 控制播放速度 (約 30 FPS)
                self.msleep(30)
            else:
                # 影片結束，重新開始 (循環播放)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap.release()

    def stop(self):
        """停止影片播放"""
        self._run_flag = False
        self.wait()

class LLMWorker(QThread):
    response_received = pyqtSignal(str) 

    def __init__(self, history):
        super().__init__()
        self.history = history 
        self.client = genai.Client(api_key=API_KEY.strip())

    def run(self):
        try:
            formatted_history = []
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
                        formatted_history.append(
                            types.Content(role="user", parts=[types.Part.from_text(text=content)])
                        )
                elif role == "assistant":
                    formatted_history.append(
                        types.Content(role="model", parts=[types.Part.from_text(text=content)])
                    )

            chat = self.client.chats.create(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7, 
                ),
                history=formatted_history
            )

            response = chat.send_message(last_user_message)
            self.response_received.emit(response.text)

        except Exception as e:
            self.response_received.emit(f"連線錯誤: {str(e)}")


class MyMainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainApp, self).__init__(parent)
        
        self.setupUi(self)
        self.setup_chat_ui()
        self.setup_agent_ui() 
        self.setup_signal_connections()

        # 陰影效果
        shadow = QGraphicsDropShadowEffect(self.frame_3)
        shadow.setBlurRadius(10) 
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 8)) 
        self.frame_3.setGraphicsEffect(shadow)

        self.conversation_history = [
            {"role": "system", "content": "你是一個專業的 AAEON 智慧助理，說話簡潔有力，並使用繁體中文回答。"}
        ]

        QtCore.QTimer.singleShot(500, lambda: self.add_chat_bubble("你好！我是 AAEON 智慧助理，很高興為您服務。", is_user=False))

    def setup_chat_ui(self):
        self.chat_main_layout = QtWidgets.QVBoxLayout(self.frame)
        self.chat_main_layout.setContentsMargins(10, 0, 10, 10)
        self.chat_main_layout.setSpacing(10)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea {border: none; background: transparent;}")
        
        self.scroll_content_widget = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.scroll_content_widget)
        
        self.messages_layout = QtWidgets.QVBoxLayout(self.scroll_content_widget)
        self.messages_layout.addStretch()

        self.chat_main_layout.addWidget(self.scroll_area)

        self.input_container = QtWidgets.QHBoxLayout()
        self.input_box = QtWidgets.QLineEdit()
        self.input_box.setPlaceholderText("請輸入訊息...")
        self.input_box.setMinimumHeight(40)

        self.send_btn = QtWidgets.QPushButton("發送")
        self.send_btn.setMinimumHeight(40)

        self.input_container.addWidget(self.input_box)
        self.input_container.addWidget(self.send_btn)
        self.chat_main_layout.addLayout(self.input_container)

    def setup_agent_ui(self):
        """改用 OpenCV 播放影片的 UI 設定"""
        layout = QtWidgets.QVBoxLayout(self.agent)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.avatar_display = QtWidgets.QLabel(self.agent)
        self.avatar_display.setAlignment(Qt.AlignCenter)
        self.avatar_display.setFixedSize(400, 600) # 強制設定標籤為 300x300
        self.avatar_display.setScaledContents(True) 
        layout.addWidget(self.avatar_display)

        # layout.addStretch(1)

        # 啟動影片播放執行緒
        # 提示：請確保路徑 ./vedio/ 確實存在
        video_path = "./vedio/avatar_talk.mp4" 
        self.video_thread = VideoThread(video_path)
        self.video_thread.change_pixmap_signal.connect(self.update_avatar_frame)
        self.video_thread.start()

    def update_avatar_frame(self, q_img):
        """接收訊號並更新畫面"""
        pixmap = QPixmap.fromImage(q_img)
        self.avatar_display.setPixmap(pixmap)

    def setup_signal_connections(self):
        self.send_btn.clicked.connect(self.handle_send)
        self.input_box.returnPressed.connect(self.handle_send)

    def handle_send(self):
        text = self.input_box.text().strip()
        if not text: return

        self.add_chat_bubble(text, is_user=True)
        self.input_box.clear()
        self.conversation_history.append({"role": "user", "content": text})
        self.input_box.setEnabled(False)

        self.worker = LLMWorker(self.conversation_history)
        self.worker.response_received.connect(self.handle_ai_response)
        self.worker.start()

    def handle_ai_response(self, response_text):
        self.conversation_history.append({"role": "assistant", "content": response_text})
        self.add_chat_bubble(response_text, is_user=False)
        self.input_box.setEnabled(True)
        self.input_box.setFocus()

    def add_chat_bubble(self, text, is_user):
        bubble_widget = QtWidgets.QWidget()
        bubble_layout = QtWidgets.QHBoxLayout(bubble_widget)
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setMaximumWidth(600)
        
        if is_user:
            label.setStyleSheet("background-color: #0080FF; color: white; border-radius: 10px; padding: 10px;")
            bubble_layout.addStretch()
            bubble_layout.addWidget(label)
        else:
            label.setStyleSheet("background-color: #FFFFFF; color: black; border: 1px solid #E0E0E0; border-radius: 10px; padding: 10px;")
            bubble_layout.addWidget(label)
            bubble_layout.addStretch()

        self.messages_layout.addWidget(bubble_widget)
        QtCore.QTimer.singleShot(100, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """當視窗關閉時，必須確保影片執行緒也安全關閉，避免報錯"""
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainApp()
    window.show()
    sys.exit(app.exec_())