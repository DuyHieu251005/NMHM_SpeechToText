"""
Training OpenAI Whisper cho Vietnamese Speech Recognition
Đồ án Nhập môn Học máy - HCMUS

Thông số lấy từ report và kết quả đánh giá:
- Model: openai/whisper-small
- Dataset: VIVOS (15 giờ, 12,420 mẫu)
- Target WER: < 15%

Siêu tham số từ báo cáo (Chapter 3):
- per_device_train_batch_size: 16
- learning_rate: 1e-5
- warmup_steps: 100
- num_train_epochs: 15
- fp16: True
- generation_max_length: 225
"""

import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, DatasetDict, Audio, load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate
import numpy as np
from pathlib import Path


# ==================== CẤU HÌNH ====================

@dataclass
class TrainingConfig:
    """Cấu hình training từ báo cáo"""
    
    # Model
    model_name: str = "openai/whisper-small"  # 244M params
    
    # Dataset
    data_path: str = r"D:\Data\vivos"  # Đường dẫn VIVOS
    processed_data_path: str = "./processed_data/vivos_processed"  # Dữ liệu đã xử lý
    
    # Training hyperparameters (từ report Chapter 3)
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    num_train_epochs: int = 15
    
    # Mixed precision
    fp16: bool = True  # Bật để tiết kiệm VRAM
    
    # Generation
    generation_max_length: int = 225
    
    # Evaluation & Saving
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 3
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # Output
    output_dir: str = "./whisper-finetuned-vivos"
    
    # System
    dataloader_num_workers: int = 0  # 0 cho Windows
    seed: int = 42


# ==================== DATA COLLATOR ====================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator cho Whisper Seq2Seq training
    - Pad input features về cùng độ dài
    - Pad labels và mask với -100 để không tính loss
    """
    processor: Any
    decoder_start_token_id: int = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding token id với -100 để không tính loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Loại bỏ BOS token nếu có
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch


# ==================== LOAD DATASET ====================

def load_vivos_from_disk(data_path: str) -> DatasetDict:
    """Load VIVOS dataset từ đường dẫn gốc"""
    print(f"📂 Đang load dữ liệu từ: {data_path}")
    
    def load_split(split_name: str) -> Dataset:
        prompts_path = os.path.join(data_path, split_name, "prompts.txt")
        waves_path = os.path.join(data_path, split_name, "waves")
        
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"Không tìm thấy: {prompts_path}")
        
        data = []
        with open(prompts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                
                file_id, text = parts
                speaker_id = file_id.split("_")[0]
                audio_file = os.path.join(waves_path, speaker_id, f"{file_id}.wav")
                
                if os.path.exists(audio_file):
                    data.append({
                        "audio": audio_file,
                        "sentence": text.lower().strip()  # Normalize text
                    })
        
        print(f"  ✅ {split_name}: {len(data)} samples")
        dataset = Dataset.from_list(data)
        return dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return DatasetDict({
        "train": load_split("train"),
        "test": load_split("test")
    })


def prepare_dataset(
    batch: Dict,
    processor: WhisperProcessor
) -> Dict:
    """Prepare features cho một batch"""
    audio = batch["audio"]
    
    # Extract log-mel spectrogram
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Tokenize transcription
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    
    return batch


# ==================== METRICS ====================

def compute_wer_metrics(pred, processor, metric):
    """
    Tính Word Error Rate (WER)
    
    WER = (S + D + I) / N × 100%
    - S: Substitution (thay thế sai)
    - D: Deletion (bỏ sót)
    - I: Insertion (thêm thừa)
    - N: Tổng số từ gốc
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 với pad token để decode
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions và labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Tính WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}


# ==================== TRAINING ====================

def train_whisper(config: TrainingConfig):
    """Main training function"""
    
    print("=" * 60)
    print("🎤 TRAINING OPENAI WHISPER CHO VIETNAMESE ASR")
    print("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ===== LOAD PROCESSOR =====
    print("\n📦 Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        config.model_name,
        language="vietnamese",
        task="transcribe"
    )
    
    # ===== LOAD DATASET =====
    print("\n📊 Loading dataset...")
    
    # Thử load dữ liệu đã xử lý trước
    if os.path.exists(config.processed_data_path):
        print(f"  📁 Tìm thấy dữ liệu đã xử lý: {config.processed_data_path}")
        dataset = load_from_disk(config.processed_data_path)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        print("  ⚠️ Chưa có dữ liệu đã xử lý, đang load từ VIVOS...")
        dataset = load_vivos_from_disk(config.data_path)
        
        # Prepare features
        print("\n⚙️ Preparing features...")
        prepare_fn = lambda batch: prepare_dataset(batch, processor)
        
        train_dataset = dataset["train"].map(
            prepare_fn,
            remove_columns=dataset["train"].column_names,
            desc="Processing train"
        )
        eval_dataset = dataset["test"].map(
            prepare_fn,
            remove_columns=dataset["test"].column_names,
            desc="Processing test"
        )
    
    print(f"\n📈 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Eval:  {len(eval_dataset)} samples")
    
    # ===== LOAD MODEL =====
    print("\n🤖 Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    
    # Cấu hình generation
    model.generation_config.language = "vietnamese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    # Đếm parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable:    {trainable_params:,}")
    
    # ===== DATA COLLATOR =====
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    
    # ===== METRICS =====
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        return compute_wer_metrics(pred, processor, wer_metric)
    
    # ===== TRAINING ARGUMENTS =====
    print("\n⚙️ Cấu hình training (từ báo cáo):")
    print(f"   Batch size: {config.per_device_train_batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Warmup steps: {config.warmup_steps}")
    print(f"   FP16: {config.fp16}")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        
        # Batch size
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Learning rate
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        
        # Training duration
        num_train_epochs=config.num_train_epochs,
        
        # Mixed precision
        fp16=config.fp16 and device == "cuda",
        
        # Evaluation
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        
        # Saving
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        
        # Logging
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        
        # Best model
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # System
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        
        # Disable wandb
        push_to_hub=False,
    )
    
    # ===== TRAINER =====
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        ]
    )
    
    # ===== START TRAINING =====
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU TRAINING...")
    print("=" * 60)
    
    train_result = trainer.train()
    
    # ===== SAVE MODEL =====
    print("\n💾 Saving model...")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # ===== FINAL EVALUATION =====
    print("\n📊 Final Evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING HOÀN THÀNH!")
    print("=" * 60)
    print(f"\n📁 Model saved to: {config.output_dir}")
    print(f"📈 Final WER: {eval_metrics.get('eval_wer', 'N/A'):.2f}%")
    
    return trainer, eval_metrics


# ==================== INFERENCE ====================

def transcribe_audio(
    audio_path: str,
    model_path: str = "./whisper-finetuned-vivos",
    device: str = None
) -> str:
    """
    Transcribe một file audio sử dụng model đã fine-tune
    
    Args:
        audio_path: Đường dẫn file audio
        model_path: Đường dẫn model đã train
        device: cuda hoặc cpu
    
    Returns:
        Văn bản transcription
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model và processor
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    
    # Load audio
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Prepare input
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            max_length=225,
            language="vi",
            task="transcribe"
        )
    
    # Decode
    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]
    
    return transcription


# ==================== MAIN ====================

def main():
    """Main entry point"""
    
    # Cấu hình - Thay đổi theo môi trường của bạn
    config = TrainingConfig(
        # ⚠️ Thay đổi đường dẫn này
        data_path=r"D:\Data\vivos",
        
        # Model size - chọn theo VRAM
        # "openai/whisper-tiny"   - 39M params, ~2GB VRAM
        # "openai/whisper-base"   - 74M params, ~3GB VRAM  
        # "openai/whisper-small"  - 244M params, ~5GB VRAM (Khuyến nghị)
        # "openai/whisper-medium" - 769M params, ~10GB VRAM
        model_name="openai/whisper-small",
        
        # Điều chỉnh batch size theo VRAM
        # - GPU 8GB: batch_size=8
        # - GPU 12GB: batch_size=16
        # - GPU 24GB: batch_size=32
        per_device_train_batch_size=16,
        
        # Epochs
        num_train_epochs=15,
        
        # Output
        output_dir="./whisper-finetuned-vivos"
    )
    
    # Kiểm tra đường dẫn
    if not os.path.exists(config.data_path):
        print(f"\n❌ Không tìm thấy thư mục dữ liệu: {config.data_path}")
        print("Vui lòng:")
        print("1. Tải VIVOS từ: https://ailab.hcmus.edu.vn/vivos")
        print("2. Giải nén vào thư mục data_path")
        print("3. Cập nhật đường dẫn trong TrainingConfig")
        return
    
    # Train
    trainer, metrics = train_whisper(config)
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
