import torch
import numpy as np
import sounddevice as sd
import queue
import time
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor
)

# ==========================================
# 1. CẤU HÌNH 
# ==========================================
# Đường dẫn model (SỬA LẠI NẾU CẦN)
MODEL_PATH = r"wav2vec2_vivos_best_checkpoint\checkpoint-3645"
SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.05  # Độ nhạy mic
PAUSE_LIMIT = 0.8         # Thời gian chờ ngắt câu (giây)

# ==========================================
# 2. LOAD MODEL
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️  Thiết bị: {device}")
print("⏳ Đang load model...")

try:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(device)
    print("✅ Model sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi: {e}")
    print("👉 Kiểm tra lại đường dẫn MODEL_PATH")
    exit()

audio_queue = queue.Queue()

# ==========================================
# 3. HÀM GHI ÂM (CALLBACK)
# ==========================================
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# ==========================================
# 4. HÀM DỊCH
# ==========================================
def transcribe(audio_buffer):
    if len(audio_buffer) == 0: return ""
    audio_input = np.concatenate(audio_buffer).flatten()
    input_values = processor(audio_input, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return text.lower()

# ==========================================
# 5. VÒNG LẶP CHÍNH
# ==========================================
def main():
    print("="*50)
    print("🎙️  CHẾ ĐỘ RẢNH TAY (HANDS-FREE)")
    print("👉 Bạn cứ nói, khi ngưng khoảng 1 giây máy sẽ tự dịch.")
    print("👉 Nhấn Ctrl + C để dừng chương trình.")
    print("="*50)

    buffer = []
    silence_start_time = None
    is_speaking = False
    
    # --- ĐÃ SỬA: THÊM TRY Ở ĐÂY ---
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            while True:
                if not audio_queue.empty():
                    indata = audio_queue.get()
                    energy = np.sqrt(np.mean(indata**2))
                    
                    if energy > ENERGY_THRESHOLD:
                        is_speaking = True
                        silence_start_time = None
                        buffer.append(indata)
                        print("🔴 Đang nghe...   ", end="\r")
                    
                    else:
                        if is_speaking:
                            buffer.append(indata)
                            if silence_start_time is None:
                                silence_start_time = time.time()
                            
                            if time.time() - silence_start_time > PAUSE_LIMIT:
                                print("🟡 Đang dịch...   ", end="\r")
                                text = transcribe(buffer)
                                print(f"🗣️  : {text}                                ")
                                buffer = []
                                is_speaking = False
                                silence_start_time = None
                                print("⚪ Chờ câu mới... ", end="\r")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Đã dừng chương trình.")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")

if __name__ == "__main__":
    main()