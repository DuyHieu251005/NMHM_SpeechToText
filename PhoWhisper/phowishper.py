import os
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
import evaluate
import matplotlib.pyplot as plt

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Lưu ý: Windows cần chữ r đằng trước string
DATA_PATH = r"C:\Users\phamm\Downloads\Compressed\archive\vivos"
MODEL_ID = "vinai/PhoWhisper-small"
OUTPUT_DIR = "./phowhisper-finetuned-local"

# 1. HÀM LOAD DỮ LIỆU TỪ Ổ CỨNG
def load_vivos_local(root_path):
    print(f"Đang đọc dữ liệu từ: {root_path}")
    def load_split(split_name):
        prompts_path = os.path.join(root_path, split_name, "prompts.txt")
        waves_path = os.path.join(root_path, split_name, "waves")
        data = []
        if not os.path.exists(prompts_path):
            print(f"Không tìm thấy file: {prompts_path}")
            return Dataset.from_list([])
            
        with open(prompts_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2: continue
                file_id, text = parts
                speaker_id = file_id.split("_")[0]
                audio_file = os.path.join(waves_path, speaker_id, f"{file_id}.wav")
                if os.path.exists(audio_file):
                    data.append({"audio": audio_file, "sentence": text})
        
        return Dataset.from_list(data).cast_column("audio", Audio(sampling_rate=16000))

    return DatasetDict({
        "train": load_split("train"),
        "test": load_split("test")
    })

# 2. DATA COLLATOR (BẮT BUỘC CHO WHISPER)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Thay padding token bằng -100 để không tính loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

# 3. HÀM MAIN (QUAN TRỌNG TRÊN WINDOWS)
if __name__ == "__main__":
    # --- A. CHUẨN BỊ DỮ LIỆU ---
    dataset = load_vivos_local(DATA_PATH)
    if len(dataset['train']) == 0:
        print("Lỗi: Không đọc được dữ liệu. Kiểm tra lại đường dẫn!")
        exit()
        
    print(f"Số lượng mẫu Train: {len(dataset['train'])}")
    print(f"Số lượng mẫu Test: {len(dataset['test'])}")

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="vietnamese", task="transcribe")

    def prepare_dataset(batch):
        audio = batch["audio"]
        # Chuyển audio thành spectrogram
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # Tokenize text
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    # Map dữ liệu (Giảm num_proc xuống 1 để tránh lỗi Windows)
    print("Đang xử lý dữ liệu (Feature Extraction)...")
    # Lấy mẫu nhỏ chạy thử cho nhanh (Bỏ .select(...) nếu muốn chạy full)
    # train_dataset = dataset["train"].select(range(500)).map(prepare_dataset) 
    train_dataset = dataset["train"].map(prepare_dataset) # Chạy full
    
    # Test lấy 200 mẫu thôi cho nhanh
    test_dataset = dataset["test"].select(range(200)).map(prepare_dataset)

    # --- B. LOAD MODEL & LORA (KHÔNG DÙNG 8-BIT ĐỂ NÉ LỖI WINDOWS) ---
    print("Đang load model...")
    # Lưu ý: Không dùng load_in_8bit=True vì lỗi bitsandbytes trên Windows
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")
    
    # Cấu hình LoRA
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # --- C. TRAIN CONFIG ---
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # Giảm xuống 4 nếu VRAM < 8GB
        gradient_accumulation_steps=4, # Tăng lên để bù cho batch size nhỏ
        learning_rate=1e-3,
        max_steps=5000, # Chạy 500 steps demo
        fp16=True, # Bắt buộc True để nhẹ VRAM
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        predict_with_generate=True,
        report_to=["none"], # Không cần wandb
        dataloader_num_workers=0 # Bắt buộc = 0 trên Windows để không lỗi
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
    )

    # --- D. START TRAIN ---
    print("Bắt đầu training...")
    trainer.train()

    # --- E. VẼ BIỂU ĐỒ & XUẤT KẾT QUẢ ---
    print("Đang vẽ biểu đồ...")
    history = trainer.state.log_history
    train_loss = [x['loss'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_wer = [x['eval_wer'] for x in history if 'eval_wer' in x]
    steps = [x['step'] for x in history if 'loss' in x]
    eval_steps = [x['step'] for x in history if 'eval_loss' in x]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_loss, label='Training Loss')
    # Xử lý độ dài mảng để vẽ (đơn giản hóa)
    if len(eval_loss) > 0:
        plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    if len(eval_wer) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps, eval_wer, label='Validation WER', color='orange', marker='x')
        plt.title('Word Error Rate (WER)')
        plt.legend()
        plt.grid(True)
    
    plt.savefig("training_result.png")
    print("Đã lưu biểu đồ vào file training_result.png")
    plt.show()