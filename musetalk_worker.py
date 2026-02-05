import sys
import os
import cv2
import torch
import numpy as np
import copy
import traceback
import shutil 
import gc
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

# --- 路徑修復 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.utils import load_all_model, datagen
    from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
    from musetalk.utils.blending import get_image
except ImportError as e:
    print(f"MuseTalk Import Error: {e}")

class MuseTalkWorker(QThread):
    frames_ready = pyqtSignal(list)
    status_update = pyqtSignal(str)

    def __init__(self, 
                 avatar_video_path,
                 gpu_id=0,
                 model_config_path="./models/musetalkV15/musetalk.json",
                 model_path="./models/musetalkV15/unet.pth",
                 whisper_path="./models/whisper"):
        super().__init__()
        
        self.avatar_path = avatar_video_path
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        self.unet_config = model_config_path
        self.unet_model_path = model_path
        self.whisper_path = whisper_path
        
        self.is_initialized = False
        self.is_initializing = False
        self.audio_path = None
        self.input_latent_list = [] 
        self.batch_size = 16

    def initialize_models(self):
        if self.is_initialized or self.is_initializing:
            print("Model is already initialized or initializing. Skipping duplicate call.")
            return

        self.is_initializing = True

        try:
            self.status_update.emit("正在載入 MuseTalk 模型 (請耐心等待)...")
            print(f"Loading MuseTalk models...")
            
            with torch.no_grad():
                self.vae, self.unet, self.pe = load_all_model(
                    unet_model_path=self.unet_model_path,
                    vae_type="sd-vae",
                    unet_config=self.unet_config,
                    device=self.device
                )
            self.timesteps = torch.tensor([0], device=self.device)
            
            self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_path)
            from transformers import WhisperModel
            self.whisper = WhisperModel.from_pretrained(self.whisper_path).to(self.device).eval()
            self.whisper.requires_grad_(False)
            
            self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

            self.status_update.emit("正在預處理 Avatar (這一步會比較久)...")
            self._preprocess_avatar()
            
            self.is_initialized = True
            self.status_update.emit("MuseTalk 準備就緒！")
            print("=== MuseTalk Initialization Complete ===")
            
        except Exception as e:
            print(f"Model Init Error: {e}")
            traceback.print_exc()
            self.status_update.emit(f"模型載入失敗: {e}")
        finally:
            self.is_initializing = False
            torch.cuda.empty_cache()
            gc.collect()

    def _preprocess_avatar(self):
        if not os.path.exists(self.avatar_path):
            raise FileNotFoundError(f"找不到 Avatar 影片: {self.avatar_path}")

        temp_dir = os.path.join(os.path.dirname(self.avatar_path), "temp_preprocess_frames")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        print(f"正在提取 Frame 到暫存區: {temp_dir}")
        
        cap = cv2.VideoCapture(self.avatar_path)
        self.avatar_frames = []
        img_paths = []
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            self.avatar_frames.append(frame)
            img_path = os.path.join(temp_dir, f"{idx:08d}.png")
            cv2.imwrite(img_path, frame)
            img_paths.append(img_path)
            idx += 1
            
        cap.release()
        print(f"共提取 {len(self.avatar_frames)} 幀，開始計算 Landmark...")

        self.coord_list, _ = get_landmark_and_bbox(img_paths, 0)

        try: shutil.rmtree(temp_dir)
        except: pass

        print("開始 VAE Encode (將圖片轉為 Latents)...")
        self.input_latent_list = []
        
        with torch.no_grad():
            for i, (bbox, frame) in enumerate(zip(self.coord_list, self.avatar_frames)):
                if bbox == coord_placeholder:
                    self.input_latent_list.append(None)
                    continue
                
                x1, y1, x2, y2 = bbox
                y2 = min(y2 + 10, frame.shape[0]) 
                
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                
                latents = self.vae.get_latents_for_unet(crop_frame)
                self.input_latent_list.append(latents)
                
                if i % 20 == 0:
                    gc.collect()

    def run_inference(self, audio_path):
        if self.is_initializing:
            print("Model is still initializing, please wait...")
            self.status_update.emit("模型初始化中，請稍候...")
            return

        if not self.is_initialized:
            print("Model not initialized, initializing now...")
            self.audio_path = audio_path
            self.start()
        else:
            self.audio_path = audio_path
            self.start()

    def run(self):
        if not self.is_initialized:
            self.initialize_models()
            if not self.audio_path: return

        if not self.input_latent_list:
            return

        try:
            self.status_update.emit("正在生成嘴型...")
            
            with torch.no_grad():
                # 1. 音訊處理
                whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(self.audio_path)
                whisper_chunks = self.audio_processor.get_whisper_chunk(
                    whisper_input_features, 
                    self.device, 
                    self.unet.model.dtype,
                    self.whisper, 
                    librosa_length,
                    fps=25,
                    audio_padding_length_left=2,
                    audio_padding_length_right=2
                )
                
                audio_frame_num = len(whisper_chunks)
                
                # 2. 準備 Batch [關鍵修改：精準對齊，不再循環]
                run_latent_list = []
                run_coord_list = []
                run_ori_frame_list = []
                total_avatar_frames = len(self.input_latent_list)
                
                for i in range(audio_frame_num):
                    # 如果還在 Avatar 影片長度內，就照順序用
                    if i < total_avatar_frames:
                        idx = i
                    else:
                        # 如果超過長度，固定使用第 0 幀 (Idle Pic)
                        # 這樣頭部會固定住，但嘴巴會繼續跟著聲音動
                        idx = 0
                    
                    # 容錯：萬一選到的幀沒有臉 (None)，則嘗試找第 0 幀或任何有效幀
                    if self.input_latent_list[idx] is None:
                        idx = 0 
                        if self.input_latent_list[idx] is None:
                             for k in range(total_avatar_frames):
                                 if self.input_latent_list[k] is not None:
                                     idx = k
                                     break
                    
                    run_latent_list.append(self.input_latent_list[idx])
                    run_coord_list.append(self.coord_list[idx])
                    run_ori_frame_list.append(self.avatar_frames[idx])

                # 3. 推論
                gen = datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=run_latent_list,
                    batch_size=self.batch_size,
                    delay_frame=0,
                    device=self.device
                )
                
                res_frame_list = []
                for i, (whisper_batch, latent_batch) in enumerate(gen):
                    audio_feature_batch = self.pe(whisper_batch)
                    latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                    
                    pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = self.vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        res_frame_list.append(res_frame)

                # 4. 合成
                final_qt_images = []
                for i, res_frame in enumerate(res_frame_list):
                    try:
                        bbox = run_coord_list[i]
                        ori_frame = copy.deepcopy(run_ori_frame_list[i])
                        x1, y1, x2, y2 = bbox
                        y2 = min(y2 + 10, ori_frame.shape[0])

                        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode='jaw', fp=self.fp)
                        
                        rgb_frame = cv2.cvtColor(combine_frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888).copy()
                        final_qt_images.append(qt_img)
                    except: continue

                # [已刪除] 原本的 Step 5. 補尾幀代碼已刪除
                # 現在輸出的影片長度會嚴格等於 Audio 長度，最後一幀剛好是 Idle Frame

            self.frames_ready.emit(final_qt_images)
            self.status_update.emit("生成完畢")

        except Exception as e:
            print(f"Inference Error: {e}")
            traceback.print_exc()
            self.status_update.emit("生成發生錯誤")
        finally:
            torch.cuda.empty_cache()