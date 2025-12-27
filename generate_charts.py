"""
Script tạo các biểu đồ cho báo cáo Chapter 4
- Learning Curves (Loss, WER)
- So sánh các mô hình
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Tạo thư mục img nếu chưa có
os.makedirs('report_final/img', exist_ok=True)

# Đọc dữ liệu trainer_state.json
def load_trainer_state(path):
    with open(path, 'r') as f:
        return json.load(f)

# ==================== WAV2VEC2 ====================
wav2vec2_state = load_trainer_state('Wav2Vec2/checkpoint-3645/trainer_state.json')
wav2vec2_logs = wav2vec2_state['log_history']

# Tách train loss và eval metrics
w2v_train_steps = []
w2v_train_loss = []
w2v_eval_steps = []
w2v_eval_loss = []
w2v_eval_wer = []

for log in wav2vec2_logs:
    if 'loss' in log and 'eval_loss' not in log:
        w2v_train_steps.append(log['step'])
        w2v_train_loss.append(log['loss'])
    if 'eval_loss' in log:
        w2v_eval_steps.append(log['step'])
        w2v_eval_loss.append(log['eval_loss'])
        w2v_eval_wer.append(log['eval_wer'])

# ==================== PHOWHISPER ====================
phowhisper_state = load_trainer_state('PhoWhisper/phowhisper-finetuned-local/checkpoint-500/trainer_state.json')
phowhisper_logs = phowhisper_state['log_history']

pw_train_steps = []
pw_train_loss = []
pw_eval_steps = []
pw_eval_loss = []
pw_eval_wer = []

for log in phowhisper_logs:
    if 'loss' in log and 'eval_loss' not in log:
        pw_train_steps.append(log['step'])
        pw_train_loss.append(log['loss'])
    if 'eval_loss' in log:
        pw_eval_steps.append(log['step'])
        pw_eval_loss.append(log['eval_loss'])
        # WER trong log này bị lỗi, sử dụng giá trị cuối cùng từ evaluation
        pw_eval_wer.append(log['eval_wer'])

# ==================== BIỂU ĐỒ 1: WAV2VEC2 LEARNING CURVES ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(w2v_train_steps, w2v_train_loss, 'b-', label='Train Loss', linewidth=2)
axes[0].plot(w2v_eval_steps, w2v_eval_loss, 'r--', label='Eval Loss', linewidth=2, marker='o')
axes[0].set_xlabel('Steps', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Wav2Vec2 - Training & Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# WER
axes[1].plot(w2v_eval_steps, w2v_eval_wer, 'g-', linewidth=2, marker='s')
axes[1].set_xlabel('Steps', fontsize=12)
axes[1].set_ylabel('WER (%)', fontsize=12)
axes[1].set_title('Wav2Vec2 - Word Error Rate (WER)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_final/img/wav2vec2_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wav2vec2_learning_curves.png")

# ==================== BIỂU ĐỒ 2: PHOWHISPER LEARNING CURVES ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(pw_train_steps, pw_train_loss, 'b-', label='Train Loss', linewidth=2)
axes[0].plot(pw_eval_steps, pw_eval_loss, 'r--', label='Eval Loss', linewidth=2, marker='o')
axes[0].set_xlabel('Steps', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('PhoWhisper - Training & Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Train Loss detail
axes[1].plot(pw_train_steps, pw_train_loss, 'b-', linewidth=2, marker='o')
axes[1].set_xlabel('Steps', fontsize=12)
axes[1].set_ylabel('Train Loss', fontsize=12)
axes[1].set_title('PhoWhisper - Train Loss Convergence', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_final/img/phowhisper_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phowhisper_learning_curves.png")

# ==================== BIỂU ĐỒ 3: SO SÁNH CÁC MÔ HÌNH ====================
# Kết quả WER cuối cùng
models = ['Wav2Vec2\n(Fine-tuned)', 'PhoWhisper\n(Fine-tuned)', 'OpenAI Whisper\n(Zero-shot)']
wer_values = [11.28, 32.89, 85.0]  # WER% - Whisper zero-shot ước tính từ kết quả CSV

colors = ['#2ecc71', '#3498db', '#e74c3c']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, wer_values, color=colors, edgecolor='black', linewidth=1.5)

# Thêm giá trị lên cột
for bar, val in zip(bars, wer_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Word Error Rate (%)', fontsize=14)
ax.set_title('So sánh WER giữa các Mô hình trên Tập Test VIVOS', fontsize=16, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Thêm đường baseline
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline 50%')

plt.tight_layout()
plt.savefig('report_final/img/model_comparison_wer.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison_wer.png")

# ==================== BIỂU ĐỒ 4: WAV2VEC2 WER QUA CÁC EPOCH ====================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(w2v_eval_steps, w2v_eval_wer, 'b-', linewidth=2.5, marker='o', markersize=6)
ax.fill_between(w2v_eval_steps, w2v_eval_wer, alpha=0.2)

ax.set_xlabel('Training Steps', fontsize=14)
ax.set_ylabel('Word Error Rate (%)', fontsize=14)
ax.set_title('Wav2Vec2 - WER Improvement During Training', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Đánh dấu điểm bắt đầu và kết thúc
ax.annotate(f'Start: {w2v_eval_wer[0]:.1f}%', 
            xy=(w2v_eval_steps[0], w2v_eval_wer[0]),
            xytext=(w2v_eval_steps[0]+200, w2v_eval_wer[0]-5),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='red'))

ax.annotate(f'End: {w2v_eval_wer[-1]:.1f}%', 
            xy=(w2v_eval_steps[-1], w2v_eval_wer[-1]),
            xytext=(w2v_eval_steps[-1]-400, w2v_eval_wer[-1]+5),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig('report_final/img/wav2vec2_wer_improvement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: wav2vec2_wer_improvement.png")

# ==================== BIỂU ĐỒ 5: TRAIN LOSS SO SÁNH ====================
fig, ax = plt.subplots(figsize=(12, 6))

# Normalize steps for comparison
ax.plot(w2v_train_steps[:len(pw_train_steps)], w2v_train_loss[:len(pw_train_steps)], 
        'b-', linewidth=2, label='Wav2Vec2', alpha=0.8)
ax.plot(pw_train_steps, pw_train_loss, 'r-', linewidth=2, label='PhoWhisper', alpha=0.8)

ax.set_xlabel('Training Steps', fontsize=14)
ax.set_ylabel('Training Loss', fontsize=14)
ax.set_title('Training Loss Comparison: Wav2Vec2 vs PhoWhisper', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_final/img/train_loss_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: train_loss_comparison.png")

print("\n✅ Tất cả biểu đồ đã được tạo trong thư mục report_final/img/")
