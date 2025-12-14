"""
Script đánh giá model PhoWhisper - Chỉ tạo 2 hình:
1. Learning Curves (Loss, WER)
2. Error Distribution
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import librosa
import os
from tqdm import tqdm
import evaluate
from pathlib import Path

# ==========================================
# 1. LOAD TRAINING HISTORY
# ==========================================
def load_training_history(checkpoint_path):
    """Đọc trainer_state.json để lấy lịch sử training"""
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    
    with open(trainer_state_path, 'r', encoding='utf-8') as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state['log_history']
    
    # Tách ra train loss và eval metrics
    train_logs = []
    eval_logs = []
    
    for log in log_history:
        if 'loss' in log and 'eval_loss' not in log:
            train_logs.append({
                'epoch': log['epoch'],
                'loss': log['loss'],
                'step': log['step']
            })
        elif 'eval_loss' in log:
            eval_logs.append({
                'epoch': log['epoch'],
                'eval_loss': log['eval_loss'],
                'eval_wer': log.get('eval_wer', None),
                'step': log['step']
            })
    
    return train_logs, eval_logs

# ==========================================
# 2. VẼ LEARNING CURVES
# ==========================================
def plot_learning_curves(train_logs, eval_logs, save_path='phowhisper_learning_curves.png'):
    """Vẽ biểu đồ Loss và WER theo epoch"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot 1: Training Loss vs Validation Loss
    train_epochs = [log['epoch'] for log in train_logs]
    train_losses = [log['loss'] for log in train_logs]
    
    eval_epochs = [log['epoch'] for log in eval_logs]
    eval_losses = [log['eval_loss'] for log in eval_logs]
    
    axes[0].plot(train_epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
    axes[0].plot(eval_epochs, eval_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Validation WER
    eval_wers = [log['eval_wer'] for log in eval_logs if log['eval_wer'] is not None]
    
    axes[1].plot(eval_epochs[:len(eval_wers)], eval_wers, 'g-o', label='Validation WER', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('WER (%)', fontsize=12)
    axes[1].set_title('Word Error Rate on Validation Set', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu learning curves vào: {save_path}")
    
    # Nhận xét
    print("\n" + "="*80)
    print("NHẬN XÉT VỀ BIỂU ĐỒ LEARNING CURVES:")
    print("="*80)
    
    # Kiểm tra hội tụ
    if len(train_losses) > 10:
        recent_train_loss = np.mean(train_losses[-5:])
        initial_train_loss = np.mean(train_losses[:5])
        improvement = ((initial_train_loss - recent_train_loss) / initial_train_loss) * 100
        
        print(f"📊 Training Loss giảm từ {initial_train_loss:.4f} → {recent_train_loss:.4f} ({improvement:.1f}% cải thiện)")
    
    if len(eval_losses) > 2:
        recent_eval_loss = np.mean(eval_losses[-2:])
        initial_eval_loss = eval_losses[0]
        print(f"📊 Validation Loss giảm từ {initial_eval_loss:.4f} → {recent_eval_loss:.4f}")
    
    # Kiểm tra dao động
    if len(train_losses) > 10:
        train_std = np.std(train_losses[-10:])
        print(f"📊 Độ dao động Training Loss (10 epochs cuối): {train_std:.4f}")
        if train_std > 0.5:
            print("⚠️  Mô hình có dao động mạnh")
        else:
            print("✅ Mô hình hội tụ ổn định")
    
    # Kiểm tra overfitting
    if len(eval_losses) > 2 and len(train_losses) > 2:
        final_gap = eval_losses[-1] - train_losses[-1]
        print(f"📊 Khoảng cách Train-Validation Loss cuối: {final_gap:.4f}")
        if final_gap > 0.5:
            print("⚠️  Có dấu hiệu Overfitting")
        else:
            print("✅ Không có dấu hiệu overfitting nghiêm trọng")
    
    print("="*80 + "\n")

# ==========================================
# 3. ĐÁNH GIÁ MODEL TRÊN TEST SET
# ==========================================
def evaluate_model_on_test_set(model, processor, test_data_path, max_samples=200):
    """Đánh giá model trên test set của VIVOS"""
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Load test data
    prompts_path = os.path.join(test_data_path, "prompts.txt")
    waves_dir = os.path.join(test_data_path, "waves")
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    references = []
    predictions = []
    individual_wers = []
    
    print(f"\n🔍 Đang đánh giá trên {min(len(lines), max_samples)} mẫu test...")
    
    model.eval()
    
    for i, line in enumerate(tqdm(lines[:max_samples])):
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
            
        file_id, reference_text = parts
        speaker_id = file_id.split("_")[0]
        audio_path = os.path.join(waves_dir, speaker_id, f"{file_id}.wav")
        
        if not os.path.exists(audio_path):
            continue
        
        try:
            # Load audio
            audio, rate = librosa.load(audio_path, sr=16000)
            
            # Predict
            input_features = processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(model.device)
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    max_new_tokens=100,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    language="vietnamese"
                )
            
            prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            ref_text = reference_text.lower()
            pred_text = prediction.lower()
            
            references.append(ref_text)
            predictions.append(pred_text)
            
            # Tính WER cho từng mẫu
            try:
                wer_single = wer_metric.compute(predictions=[pred_text], references=[ref_text])
                individual_wers.append(wer_single * 100)
            except:
                individual_wers.append(0)
            
        except Exception as e:
            continue
    
    # Tính metrics tổng thể
    wer_score = wer_metric.compute(predictions=predictions, references=references)
    cer_score = cer_metric.compute(predictions=predictions, references=references)
    
    print("\n" + "="*80)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET:")
    print("="*80)
    print(f"📊 WER (Word Error Rate): {wer_score*100:.2f}%")
    print(f"📊 CER (Character Error Rate): {cer_score*100:.2f}%")
    print(f"📊 Số mẫu đánh giá: {len(predictions)}")
    print("="*80 + "\n")
    
    # In một vài ví dụ
    print("\n🔍 MỘT SỐ VÍ DỤ DỰ ĐOÁN:")
    print("-"*80)
    for i in range(min(5, len(predictions))):
        print(f"\nMẫu {i+1}:")
        print(f"  Reference:  {references[i]}")
        print(f"  Prediction: {predictions[i]}")
    print("-"*80 + "\n")
    
    return {
        'wer': wer_score * 100,
        'cer': cer_score * 100,
        'num_samples': len(predictions),
        'individual_wers': individual_wers
    }

# ==========================================
# 4. VẼ ERROR DISTRIBUTION
# ==========================================
def plot_error_distribution(individual_wers, overall_wer, save_path='phowhisper_error_distribution.png'):
    """Vẽ phân phối lỗi WER trên các mẫu test"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot 1: Histogram của WER
    axes[0].hist(individual_wers, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(overall_wer, color='red', linestyle='--', linewidth=2, label=f'Mean WER: {overall_wer:.2f}%')
    axes[0].set_xlabel('WER (%)', fontsize=12)
    axes[0].set_ylabel('Số lượng mẫu', fontsize=12)
    axes[0].set_title('Phân Phối Word Error Rate', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Phân loại mẫu theo độ chính xác
    wer_categories = {
        'Xuất sắc\n(WER < 20%)': sum(1 for w in individual_wers if w < 20),
        'Tốt\n(20% ≤ WER < 40%)': sum(1 for w in individual_wers if 20 <= w < 40),
        'Trung bình\n(40% ≤ WER < 60%)': sum(1 for w in individual_wers if 40 <= w < 60),
        'Kém\n(WER ≥ 60%)': sum(1 for w in individual_wers if w >= 60)
    }
    
    categories = list(wer_categories.keys())
    counts = list(wer_categories.values())
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    
    bars = axes[1].bar(range(len(categories)), counts, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(categories)))
    axes[1].set_xticklabels(categories, fontsize=10)
    axes[1].set_ylabel('Số lượng mẫu', fontsize=12)
    axes[1].set_title('Phân Loại Mẫu Theo Độ Chính Xác', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Thêm số lượng lên từng cột
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(individual_wers)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu error distribution vào: {save_path}")
    
    # Phân tích chi tiết
    print("\n" + "="*80)
    print("PHÂN TÍCH ERROR DISTRIBUTION:")
    print("="*80)
    print(f"📊 WER trung bình: {overall_wer:.2f}%")
    print(f"📊 WER thấp nhất: {min(individual_wers):.2f}%")
    print(f"📊 WER cao nhất: {max(individual_wers):.2f}%")
    print(f"📊 Độ lệch chuẩn: {np.std(individual_wers):.2f}%")
    print(f"📊 Trung vị (Median): {np.median(individual_wers):.2f}%")
    print("\n📊 Phân loại mẫu:")
    for category, count in wer_categories.items():
        percentage = (count / len(individual_wers)) * 100
        print(f"   {category.replace(chr(10), ' ')}: {count} mẫu ({percentage:.1f}%)")
    print("="*80 + "\n")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*80)
    print("ĐÁNH GIÁ MODEL PHOWHISPER - TẠO 2 HÌNH")
    print("="*80 + "\n")
    
    # Cấu hình
    CHECKPOINT_PATH = "./phowhisper-finetuned-local/checkpoint-500"
    BASE_MODEL_ID = "vinai/PhoWhisper-small"
    TEST_DATA_PATH = r"C:\Users\phamm\Downloads\Compressed\archive\vivos\test"
    
    # 1. Vẽ Learning Curves
    print("📊 Bước 1/4: Vẽ Learning Curves...")
    train_logs, eval_logs = load_training_history(CHECKPOINT_PATH)
    plot_learning_curves(train_logs, eval_logs)
    
    # 2. Load model
    print("\n📦 Bước 2/4: Load model...")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL_ID, 
        language="vietnamese", 
        task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
    
    # 3. Đánh giá trên test set
    print("\n📊 Bước 3/4: Đánh giá trên test set...")
    if os.path.exists(TEST_DATA_PATH):
        results = evaluate_model_on_test_set(model, processor, TEST_DATA_PATH, max_samples=200)
        
        # 4. Vẽ Error Distribution
        print("\n📊 Bước 4/4: Vẽ Error Distribution...")
        plot_error_distribution(results['individual_wers'], results['wer'])
        
        # Lưu kết quả
        with open('phowhisper_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'wer': results['wer'],
                'cer': results['cer'],
                'num_samples': results['num_samples']
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu kết quả vào: phowhisper_evaluation_results.json")
        
        print("\n" + "="*80)
        print("✅ KẾT QUẢ CUỐI CÙNG:")
        print("="*80)
        print(f"📊 Đã tạo 2 hình:")
        print(f"   1. phowhisper_learning_curves.png")
        print(f"   2. phowhisper_error_distribution.png")
        print(f"\n📊 File kết quả: phowhisper_evaluation_results.json")
        print("="*80)
    else:
        print(f"⚠️  Không tìm thấy test data tại: {TEST_DATA_PATH}")
    
    print("\n✅ HOÀN THÀNH!")
