import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
from datasets import Dataset, Audio
import os

# 1. CẤU HÌNH ĐƯỜNG DẪN (Trỏ đúng vào folder vừa train xong)
PEFT_MODEL_ID = "./phowhisper-finetuned-local/checkpoint-500" # Lấy checkpoint cuối cùng
BASE_MODEL_ID = "vinai/PhoWhisper-small"
DATA_PATH = r"C:\Users\phamm\Downloads\Compressed\archive\vivos" # Đường dẫn dataset của bạn

# 2. LOAD MODEL & ADAPTER (Lắp não mới vào)
print("Đang load model...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language="vietnamese", task="transcribe")
# Load model gốc
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, device_map="auto")
# Load phần LoRA vừa train
model = PeftModel.from_pretrained(model, PEFT_MODEL_ID)
model.eval() # Chuyển sang chế độ kiểm tra

# 3. HÀM LOAD 1 FILE AUDIO ĐỂ TEST
def predict_audio(audio_path):
    # Đọc file audio
    import librosa
    audio, rate = librosa.load(audio_path, sr=16000)
    
    # Xử lý đầu vào
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CHÌA KHÓA ĐỂ SỬA LỖI LẶP TỪ (313% -> 15%) ---
    # no_repeat_ngram_size=3: Không cho phép lặp lại cụm 3 từ giống hệt nhau
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=100,
            no_repeat_ngram_size=3,  # <--- CÁI PHANH Ở ĐÂY
            repetition_penalty=1.2,   # <--- PHẠT NẶNG NẾU LẶP
            language="vietnamese"
        )
    
    # Giải mã ra chữ
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# 4. CHẠY TEST TRÊN 5 MẪU NGẪU NHIÊN ĐỂ LÀM BÁO CÁO
print("\n" + "="*80)
print(f"{'THỰC TẾ (Reference)':<40} | {'DỰ ĐOÁN (Hypothesis - Đã fix lỗi)':<40}")
print("="*80)

# Lấy thủ công vài file trong folder test để demo
test_path = os.path.join(DATA_PATH, "test", "prompts.txt")
waves_path = os.path.join(DATA_PATH, "test", "waves")

count = 0
with open(test_path, encoding="utf-8") as f:
    for line in f:
        if count >= 10: break # Test 10 câu thôi
        parts = line.strip().split(" ", 1)
        if len(parts) != 2: continue
        
        file_id, text_ref = parts
        speaker_id = file_id.split("_")[0]
        audio_file = os.path.join(waves_path, speaker_id, f"{file_id}.wav")
        
        if os.path.exists(audio_file):
            # Dự đoán
            text_pred = predict_audio(audio_file)
            
            # In ra màn hình
            print(f"{text_ref:<40} | {text_pred:<40}")
            count += 1

print("="*80)
print("Xong! Hãy copy bảng trên vào báo cáo phần 'Kết quả thực nghiệm'.")