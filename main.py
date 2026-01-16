import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 引入 Google 最新版 SDK
from google import genai
from google.genai import types

# 匯入你原本的介面檔案
from AI_agent_frame import Ui_MainWindow

# --- LLM 設定區 ---
# [重要] 請填入您的新 API Key (建議去刪除舊的並重新申請)
API_KEY = "AIzaSyAcfVQMO0agc2StxF1lAY8MWsFwnwIHyFY" 

# [修改] 改用 Gemini 2.0 Flash (目前預覽版名稱通常為 -exp)
# 如果未來正式版推出，名稱可能會變成 "gemini-2.0-flash" 或 "gemini-2.0-flash-001"
MODEL_NAME = "gemini-2.0-flash"
# -------------------------------------

class LLMWorker(QThread):
    response_received = pyqtSignal(str) 

    def __init__(self, history):
        super().__init__()
        self.history = history 
        # 初始化新的 Client
        self.client = genai.Client(api_key=API_KEY.strip())

    def run(self):
        try:
            # 1. 整理歷史訊息格式
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

            # 2. 設定與呼叫模型
            # 注意：Gemini 2.0 可能會有更快的響應速度
            chat = self.client.chats.create(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7, 
                ),
                history=formatted_history
            )

            # 3. 發送訊息
            response = chat.send_message(last_user_message)

            # 4. 取得回應文字
            ai_text = response.text
            self.response_received.emit(ai_text)

        except Exception as e:
            error_msg = str(e)
            print(f"Error detail: {error_msg}")
            
            if "404" in error_msg:
                self.response_received.emit(f"錯誤：找不到模型 {MODEL_NAME}。請確認您的 API Key 是否有權限存取 2.0 預覽版，或嘗試改回 gemini-1.5-flash。")
            elif "403" in error_msg or "Illegal header" in error_msg:
                self.response_received.emit("錯誤：API Key 無效或格式錯誤。")
            elif "429" in error_msg:
                self.response_received.emit("錯誤：請求太頻繁，請稍後再試。")
            else:
                self.response_received.emit(f"連線錯誤: {error_msg}")


class MyMainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainApp, self).__init__(parent)
        
        self.setupUi(self)
        self.setup_chat_ui()
        self.setup_agent_ui()
        self.setup_signal_connections()

        shadow = QGraphicsDropShadowEffect(self.frame_3)
        shadow.setBlurRadius(10) 
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 8)) 
        self.frame_3.setGraphicsEffect(shadow)

        self.conversation_history = [
            {"role": "system", "content": "你是一個專業的 AAEON 智慧助理，說話簡潔有力，並使用繁體中文回答。"}
        ]

        # 更新歡迎訊息，讓使用者知道現在是用 2.0
        QtCore.QTimer.singleShot(500, lambda: self.add_chat_bubble("你好！我是 AAEON 智慧助理 (Gemini 2.0 Flash版)，請下達指令。", is_user=False))

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
        if not text:
            return

        self.add_chat_bubble(text, is_user=True)
        self.input_box.clear()
        
        self.conversation_history.append({"role": "user", "content": text})

        self.input_box.setPlaceholderText("Gemini 2.0 思考中...")
        self.input_box.setEnabled(False)

        self.worker = LLMWorker(self.conversation_history)
        self.worker.response_received.connect(self.handle_ai_response)
        self.worker.start()

    def handle_ai_response(self, response_text):
        self.conversation_history.append({"role": "assistant", "content": response_text})
        self.add_chat_bubble(response_text, is_user=False)
        self.input_box.setEnabled(True)
        self.input_box.setPlaceholderText("請輸入訊息...")
        self.input_box.setFocus()

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

    def setup_agent_ui(self):
        layout = QtWidgets.QVBoxLayout(self.agent)
        layout.setContentsMargins(50, 100, 50, 0) 
        
        self.avatar_label = QtWidgets.QLabel(self.agent)
        
        pixmap = QtGui.QPixmap("./pic/avatar.png")
        
        if pixmap.isNull():
            print("錯誤: 找不到 './pic/avatar.png'，請確認檔案位置。")
            self.avatar_label.setText("") 
        else:
            self.avatar_label.setPixmap(pixmap)
            self.avatar_label.setScaledContents(True)
            self.avatar_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        layout.addWidget(self.avatar_label)

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont("Microsoft JhengHei", 10)
    app.setFont(font)

    window = MyMainApp()
    window.show()
    sys.exit(app.exec_())