import os
import shutil
import re
import json
import pandas as pd
import numpy as np
import torch
import gc
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Union

# ==========================================
# CAU HINH DUONG DAN
# ==========================================
DATA_PATH = "./vivos"
OUTPUT_DIR = "./wav2vec2-vivos-final"

# ==========================================
# HAM LOAD DU LIEU
# ==========================================
def load_vivos_from_local(root_path):
    datasets = {}
    for split in ["train", "test"]:
        prompts_path = os.path.join(root_path, split, "prompts.txt")
        waves_dir = os.path.join(root_path, split, "waves")
        
        if not os.path.exists(prompts_path):
            print(f"Loi: Khong tim thay file {prompts_path}")
            continue

        with open(prompts_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        audio_paths, sentences = [], []
        for line in lines:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                file_id, text = parts
                full_path = os.path.join(waves_dir, file_id.split("_")[0], f"{file_id}.wav")
                if os.path.exists(full_path):
                    audio_paths.append(full_path)
                    sentences.append(text)
        
        ds = Dataset.from_pandas(pd.DataFrame({"audio": audio_paths, "sentence": sentences}))
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        datasets[split] = ds
    
    if not datasets:
        raise ValueError("Khong load duoc du lieu. Hay kiem tra lai duong dan DATA_PATH.")
        
    return DatasetDict(datasets)

# ==========================================
# TIEN XU LY TEXT (PREPROCESSING)
# ==========================================
chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\‘\”\']"

def remove_special_characters(batch):
    # Loai bo ky tu dac biet va chuyen ve chu thuong
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower() + " "
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

# ==========================================
# DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
            
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# ==========================================
# CHUONG TRINH CHINH
# ==========================================
def main():
    # 1. Don dep thu muc cu
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load du lieu
    print(f"Dang load du lieu tu: {os.path.abspath(DATA_PATH)}")
    vivos = load_vivos_from_local(DATA_PATH)
    print(f"   - Train: {len(vivos['train'])} mau")
    print(f"   - Test: {len(vivos['test'])} mau")

    # 3. Xu ly Text (Clean Text)
    vivos = vivos.map(remove_special_characters)

    # 4. Tao Vocab (Tu dien)
    print("Dang tao bo tu dien (Vocab)...")
    vocab_train = vivos["train"].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=vivos["train"].column_names)
    vocab_test = vivos["test"].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=vivos["test"].column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    vocab_path = os.path.join(OUTPUT_DIR, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # Giai phong RAM
    del vocab_train, vocab_test, vocab_list
    gc.collect()

    # 5. Khoi tao Processor
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 6. Chuan bi Audio (Encoding)
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    print("Dang xu ly Audio (Encoding)...")
    # num_proc=1 de tranh loi tren Windows
    vivos_encoded = vivos.map(
        prepare_dataset, 
        remove_columns=vivos["train"].column_names, 
        num_proc=1, 
        writer_batch_size=200 
    )

    del vivos
    gc.collect()

    # 7. Cau hinh Model
    print("Dang tai Model Pre-trained...")
    model_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    
    config = Wav2Vec2Config.from_pretrained(
        model_id, 
        vocab_size=len(processor.tokenizer), 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        attention_dropout=0.1, hidden_dropout=0.1, feat_proj_dropout=0.0,
        mask_time_prob=0.05, layerdrop=0.0,
    )

    model = Wav2Vec2ForCTC.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
    model.freeze_feature_encoder()

    # 8. Metric WER
    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 9. Training Arguments
    use_fp16 = torch.cuda.is_available()
    print(f"Trang thai GPU: {'Co (Bat FP16)' if use_fp16 else 'Khong (Chay CPU/FP32)'}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=4,   # Giam xuong neu bi loi bo nho
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=200, save_steps=400, logging_steps=50,
        num_train_epochs=5,
        gradient_checkpointing=True,
        fp16=use_fp16,
        learning_rate=1e-4,
        warmup_steps=300,
        save_total_limit=2,
        dataloader_num_workers=0, # Bat buoc = 0 tren Windows
        report_to=["tensorboard"]
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vivos_encoded["train"],
        eval_dataset=vivos_encoded["test"],
        tokenizer=processor.feature_extractor,
    )

    print("Bat dau qua trinh huan luyen tren Local...")
    trainer.train()

    # 10. Luu model
    print("Dang luu model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Hoan tat! Model da luu tai: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()