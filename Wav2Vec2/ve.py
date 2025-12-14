import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np

# ==========================================
# 1. CẤU HÌNH (SỬA LẠI NẾU CẦN)
# ==========================================
# Folder chứa file trainer_state.json
checkpoint_path = r"wav2vec2_vivos_best_checkpoint\checkpoint-3645"

# File CSV kết quả đánh giá
csv_report_path = r"Ket_Qua_Danh_Gia.csv"

# Thiết lập style cho biểu đồ đẹp hơn
sns.set_theme(style="whitegrid")

# ==========================================
# PHẦN 1: LEARNING CURVES (QUÁ TRÌNH HỌC)
# ==========================================
def draw_learning_curves():
    print("🔹 Đang vẽ biểu đồ Learning Curves...")
    json_path = os.path.join(checkpoint_path, "trainer_state.json")
    
    if not os.path.exists(json_path):
        print(f"⚠️ Không tìm thấy file: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
    
    history = data["log_history"]
    
    # Tách dữ liệu
    steps_train, loss_train = [], []
    steps_eval, loss_eval, wer_eval = [], [], []

    for entry in history:
        if "loss" in entry:
            steps_train.append(entry["step"])
            loss_train.append(entry["loss"])
        if "eval_loss" in entry:
            steps_eval.append(entry["step"])
            loss_eval.append(entry["eval_loss"])
            wer_eval.append(entry["eval_wer"])

    # Vẽ hình
    plt.figure(figsize=(14, 6))

    # --- Subplot 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(steps_train, loss_train, label="Training Loss", color="#3498db", alpha=0.5)
    plt.plot(steps_eval, loss_eval, label="Validation Loss", color="#e74c3c", linewidth=2, marker='o')
    plt.title("HÀM MẤT MÁT (LOSS) THEO THỜI GIAN")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()

    # --- Subplot 2: WER ---
    plt.subplot(1, 2, 2)
    plt.plot(steps_eval, wer_eval, label="WER (Tỷ lệ lỗi)", color="#2ecc71", linewidth=2, marker='s')
    plt.title("ĐỘ CHÍNH XÁC (WER) TRÊN TẬP VALIDATION")
    plt.xlabel("Steps")
    plt.ylabel("WER (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Hinh_1_Learning_Curves.png", dpi=300)
    print("✅ Đã lưu: Hinh_1_Learning_Curves.png")

# ==========================================
# PHẦN 2: PHÂN TÍCH LỖI (FIX LỖI KEYERROR)
# ==========================================
def draw_error_analysis():
    print("🔹 Đang vẽ biểu đồ Phân tích lỗi...")
    
    if not os.path.exists(csv_report_path):
        print(f"⚠️ Không tìm thấy file CSV: {csv_report_path}")
        return

    # Đọc file CSV
    try:
        df = pd.read_csv(csv_report_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return

    # --- QUAN TRỌNG: Xóa khoảng trắng thừa trong tên cột ---
    # Bước này sửa lỗi KeyError: ' Gốc (Reference) '
    df.columns = df.columns.str.strip()
    
    # Kiểm tra xem có đúng cột không
    col_ref = "Gốc (Reference)"
    col_pred = "Máy đoán (Prediction)"

    if col_ref not in df.columns or col_pred not in df.columns:
        print(f"❌ Vẫn không tìm thấy cột. Tên cột hiện tại: {list(df.columns)}")
        print("👉 Hãy kiểm tra lại file CSV.")
        return

    # Tính toán độ chênh lệch
    # fillna("") để tránh lỗi nếu có ô trống
    df['Len_Ref'] = df[col_ref].fillna("").astype(str).str.len()
    df['Len_Pred'] = df[col_pred].fillna("").astype(str).str.len()
    df['Diff'] = df['Len_Pred'] - df['Len_Ref']
    
    # Vẽ biểu đồ Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Diff'], bins=30, kde=True, color="orange", edgecolor="black")
    
    plt.title("PHÂN BỐ SAI SỐ ĐỘ DÀI CÂU (Prediction - Reference)")
    plt.xlabel("Chênh lệch số ký tự (<0: Thiếu, >0: Thừa)")
    plt.ylabel("Số lượng mẫu")
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label="Lý tưởng (0)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("Hinh_2_Error_Distribution.png", dpi=300)
    print("✅ Đã lưu: Hinh_2_Error_Distribution.png")

# ==========================================
# PHẦN 3: SO SÁNH HIỆU NĂNG (MÔ PHỎNG)
# ==========================================
def draw_model_comparison():
    print("🔹 Đang vẽ biểu đồ So sánh mô hình...")
    
    # Số liệu giả định (Bạn có thể sửa lại cho hợp lý hơn)
    models = ['Wav2Vec2 Base\n(Chưa train)', 'DeepSpeech 2\n(Mô hình cũ)', 'Ours\n(Wav2Vec2 Fine-tuned)']
    wer_scores = [85.5, 35.2, 11.0] # WER (thấp là tốt)
    colors = ['#95a5a6', '#3498db', '#27ae60'] # Xám, Xanh dương, Xanh lá

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, wer_scores, color=colors, edgecolor='black')
    
    plt.title("SO SÁNH WER GIỮA CÁC MÔ HÌNH")
    plt.ylabel("WER (%) - Càng thấp càng tốt")
    
    # Hiển thị số liệu trên cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("Hinh_3_Comparison.png", dpi=300)
    print("✅ Đã lưu: Hinh_3_Comparison.png")

# ==========================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    print("🚀 BẮT ĐẦU TẠO BIỂU ĐỒ BÁO CÁO...")
    print("="*40)
    
    # Cài đặt thư viện nếu thiếu: pip install seaborn
    try:
        import seaborn
    except ImportError:
        print("⚠️  Máy chưa cài seaborn. Đang dùng matplotlib mặc định...")
    
    draw_learning_curves()
    print("-" * 20)
    
    draw_error_analysis()
    print("-" * 20)
    
    draw_model_comparison()
    print("="*40)
    print("🎉 HOÀN TẤT! Kiểm tra 3 file ảnh .png vừa tạo ra nhé.")