import torch
import librosa
import pandas as pd
import os
import evaluate
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor
)
from tqdm import tqdm
import unicodedata

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (GIỮ NGUYÊN NHƯ CŨ)
# ==========================================
model_path = r"checkpoint-3645"
vivos_test_path = r"C:\Users\phamm\Downloads\Compressed\archive\vivos\test"
report_path = r"Ket_Qua_Danh_Gia.csv"

# ==========================================
# 2. HÀM CHUẨN BỊ DỮ LIỆU
# ==========================================
def load_vivos_test_data(root_path):
    prompts_path = os.path.join(root_path, "prompts.txt")
    waves_dir = os.path.join(root_path, "waves")
    
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {prompts_path}")

    with open(prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    data = []
    print("⏳ Đang quét file audio...")
    for line in lines:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            file_id, text = parts
            speaker_id = file_id.split("_")[0]
            # Tạo đường dẫn đầy đủ đến file wav
            full_path = os.path.join(waves_dir, speaker_id, f"{file_id}.wav")
            
            if os.path.exists(full_path):
                data.append({"path": full_path, "text": text})
            else:
                pass # Bỏ qua cảnh báo cho gọn màn hình
    
    return data

# ==========================================
# 3. LOAD MODEL & METRIC (ĐÃ SỬA LỖI)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️  Đang chạy trên thiết bị: {device}")

print("⏳ Đang load model...")
try:
    # --- SỬA LỖI 1: Load Tokenizer từ Local (để lấy Vocab của bạn) ---
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        model_path, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        word_delimiter_token="|"
    )

    # --- SỬA LỖI 2: Load Feature Extractor từ Online (Fix lỗi thiếu file config) ---
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

    # Gộp lại thành Processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load Model Weights
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    print("✅ Load model thành công!")

except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    print("👉 Hãy chắc chắn trong folder model có file 'vocab.json', 'config.json', 'model.safetensors' (hoặc pytorch_model.bin)")
    exit()

wer_metric = evaluate.load("wer")

# ==========================================
# 4. BẮT ĐẦU ĐÁNH GIÁ
# ==========================================
dataset = load_vivos_test_data(vivos_test_path)
print(f"✅ Tìm thấy {len(dataset)} mẫu kiểm thử.")

references = []
predictions = []

print("🚀 Bắt đầu chạy test (Việc này sẽ mất vài phút)...")

for item in tqdm(dataset):
    # 1. Load Audio
    speech, sr = librosa.load(item["path"], sr=16000) 
    
    # 2. Xử lý input
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
    
    # 3. Dự đoán
    with torch.no_grad():
        logits = model(input_values).logits
    
    # 4. Decode ra chữ
    pred_ids = torch.argmax(logits, dim=-1)
    
    # --- SỬA LỖI 3: Thêm skip_special_tokens=True để xóa [PAD] ---
    transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    
    # 5. Lưu lại
    ref_norm = item["text"].lower().strip()
    pred_norm = transcription.lower().strip()
    
    predictions.append(pred_norm)
    references.append(ref_norm)

# ==========================================
# 5. TÍNH ĐIỂM VÀ LƯU BÁO CÁO
# ==========================================
print("\n📊 Đang tính toán WER...")
wer_score = wer_metric.compute(predictions=predictions, references=references)

print("="*40)
print(f"🏆 KẾT QUẢ CUỐI CÙNG:")
print(f"👉 WER (Tỷ lệ lỗi): {wer_score * 100:.2f}%")
print(f"👉 Độ chính xác (Accuracy): {(1 - wer_score) * 100:.2f}%")
print("="*40)

# Lưu file Excel
df = pd.DataFrame({
    "Audio Path": [d['path'] for d in dataset],
    "Gốc (Reference)": references,
    "Máy đoán (Prediction)": predictions
})

df.to_csv(report_path, index=False, encoding='utf-8-sig')
print(f"✅ Đã lưu báo cáo chi tiết tại: {report_path}")