import google.generativeai as genai
import os

# 請替換成你的真實 API Key
MY_API_KEY = "AIzaSyAcfVQMO0agc2StxF1lAY8MWsFwnwIHyFY"

try:
    genai.configure(api_key=MY_API_KEY)
    
    print(f"正在查詢 API Key 可用的生成模型...\n")
    
    available_models = []
    for m in genai.list_models():
        # 我們只關心支援 'generateContent' (對話/文字生成) 的模型
        if 'generateContent' in m.supported_generation_methods:
            print(f"名稱: {m.name}")
            print(f"描述: {m.description}")
            print("-" * 30)
            available_models.append(m.name)
            
    print(f"\n查詢完成！共找到 {len(available_models)} 個可用模型。")
    print("建議在 main_speech_loop.py 中使用以下名稱 (去掉 'models/' 前綴):")
    for model in available_models:
        clean_name = model.replace("models/", "")
        print(f"'{clean_name}'")

except Exception as e:
    print(f"查詢失敗: {e}")
