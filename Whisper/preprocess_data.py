"""
Tiền xử lý dữ liệu VIVOS cho Vietnamese Speech Recognition
Đồ án Nhập môn Học máy - HCMUS

File này thực hiện:
1. Load dữ liệu VIVOS từ ổ cứng
2. Tiền xử lý audio (resampling 16kHz)
3. Chuẩn hóa text (lowercase)
4. Trích xuất đặc trưng (Log-Mel Spectrogram cho Whisper)
5. Tokenization
6. Lưu dữ liệu đã xử lý để sử dụng cho training
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Audio processing
import librosa
import soundfile as sf

# Dataset handling
from datasets import Dataset, DatasetDict, Audio

# Transformers
from transformers import WhisperProcessor, WhisperFeatureExtractor


# ==================== CẤU HÌNH ====================
@dataclass
class DataConfig:
    """Cấu hình cho tiền xử lý dữ liệu"""
    # Đường dẫn đến thư mục VIVOS
    data_path: str = r"D:\Data\vivos"  # Thay đổi theo đường dẫn của bạn
    
    # Thông số audio
    sample_rate: int = 16000  # Tần số lấy mẫu chuẩn cho ASR
    max_duration: float = 30.0  # Độ dài tối đa audio (giây)
    min_duration: float = 0.5  # Độ dài tối thiểu audio (giây)
    
    # Thông số xử lý
    normalize_text: bool = True  # Chuẩn hóa text về lowercase
    remove_punctuation: bool = False  # Giữ lại dấu câu
    
    # Mô hình processor
    whisper_model: str = "openai/whisper-small"
    
    # Thư mục output
    output_dir: str = "./processed_data"
    
    # Cache
    use_cache: bool = True


# ==================== HÀM TIỀN XỬ LÝ ====================

def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt
    - Chuyển về chữ thường
    - Loại bỏ khoảng trắng thừa
    """
    # Chuyển về lowercase
    text = text.lower().strip()
    
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text


def load_audio(audio_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """
    Load và resample audio về tần số mục tiêu
    
    Args:
        audio_path: Đường dẫn file audio
        target_sr: Tần số lấy mẫu mục tiêu (mặc định 16kHz)
    
    Returns:
        Numpy array chứa audio waveform hoặc None nếu lỗi
    """
    try:
        # Load audio với librosa
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"Lỗi load audio {audio_path}: {e}")
        return None


def get_audio_duration(audio: np.ndarray, sr: int = 16000) -> float:
    """Tính độ dài audio (giây)"""
    return len(audio) / sr


def validate_audio(audio: np.ndarray, config: DataConfig) -> bool:
    """
    Kiểm tra audio có hợp lệ không
    - Không quá ngắn hoặc quá dài
    - Không phải audio im lặng
    """
    duration = get_audio_duration(audio, config.sample_rate)
    
    # Kiểm tra độ dài
    if duration < config.min_duration or duration > config.max_duration:
        return False
    
    # Kiểm tra có âm thanh không (RMS > threshold)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.001:  # Ngưỡng cho audio im lặng
        return False
    
    return True


def load_vivos_dataset(data_path: str) -> DatasetDict:
    """
    Load bộ dữ liệu VIVOS từ ổ cứng
    
    Cấu trúc thư mục VIVOS:
    vivos/
    ├── train/
    │   ├── prompts.txt
    │   └── waves/
    │       └── VIVOSSPK01/
    │           └── VIVOSSPK01_001.wav
    └── test/
        ├── prompts.txt
        └── waves/
    """
    print(f"📂 Đang đọc dữ liệu từ: {data_path}")
    
    def load_split(split_name: str) -> Dataset:
        """Load một split (train hoặc test)"""
        prompts_path = os.path.join(data_path, split_name, "prompts.txt")
        waves_path = os.path.join(data_path, split_name, "waves")
        
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"Không tìm thấy file: {prompts_path}")
        
        data = []
        skipped = 0
        
        with open(prompts_path, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {split_name}"):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    skipped += 1
                    continue
                
                file_id, text = parts
                speaker_id = file_id.split("_")[0]
                audio_file = os.path.join(waves_path, speaker_id, f"{file_id}.wav")
                
                if os.path.exists(audio_file):
                    data.append({
                        "file_id": file_id,
                        "speaker_id": speaker_id,
                        "audio": audio_file,
                        "sentence": text
                    })
                else:
                    skipped += 1
        
        print(f"  ✅ Loaded {len(data)} samples, skipped {skipped}")
        
        # Tạo Dataset và cast audio column
        dataset = Dataset.from_list(data)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset
    
    return DatasetDict({
        "train": load_split("train"),
        "test": load_split("test")
    })


def prepare_features_whisper(
    batch: Dict,
    processor: WhisperProcessor,
    config: DataConfig
) -> Dict:
    """
    Chuẩn bị features cho Whisper model
    
    Thực hiện:
    1. Trích xuất Log-Mel Spectrogram từ audio
    2. Tokenize text thành labels
    """
    audio = batch["audio"]
    
    # Trích xuất Log-Mel Spectrogram
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Chuẩn hóa text
    text = batch["sentence"]
    if config.normalize_text:
        text = normalize_vietnamese_text(text)
    
    # Tokenize text
    labels = processor.tokenizer(text).input_ids
    
    return {
        "input_features": input_features,
        "labels": labels,
        "text": text
    }


def preprocess_dataset(
    dataset: DatasetDict,
    config: DataConfig,
    save: bool = True
) -> DatasetDict:
    """
    Pipeline tiền xử lý hoàn chỉnh
    """
    print("\n🔄 Đang khởi tạo Processor...")
    processor = WhisperProcessor.from_pretrained(
        config.whisper_model,
        language="vietnamese",
        task="transcribe"
    )
    
    print("\n🔄 Đang xử lý dữ liệu...")
    
    def prepare_fn(batch):
        return prepare_features_whisper(batch, processor, config)
    
    processed_dataset = DatasetDict()
    
    for split in ["train", "test"]:
        print(f"\n  📊 Xử lý {split} set ({len(dataset[split])} samples)...")
        processed = dataset[split].map(
            prepare_fn,
            remove_columns=dataset[split].column_names,
            desc=f"Processing {split}"
        )
        processed_dataset[split] = processed
    
    # Lưu dữ liệu đã xử lý
    if save:
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Đang lưu dữ liệu vào {config.output_dir}...")
        processed_dataset.save_to_disk(str(output_path / "vivos_processed"))
        
        # Lưu config
        config_dict = {
            "data_path": config.data_path,
            "sample_rate": config.sample_rate,
            "whisper_model": config.whisper_model,
            "train_samples": len(processed_dataset["train"]),
            "test_samples": len(processed_dataset["test"])
        }
        with open(output_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print("  ✅ Đã lưu thành công!")
    
    return processed_dataset


def get_dataset_statistics(dataset: DatasetDict) -> Dict:
    """
    Tính toán các thống kê về dataset
    """
    stats = {}
    
    for split in ["train", "test"]:
        data = dataset[split]
        
        # Thống kê text
        texts = [item["sentence"] for item in data]
        text_lengths = [len(t.split()) for t in texts]
        
        stats[split] = {
            "num_samples": len(data),
            "avg_text_length": np.mean(text_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
        }
    
    return stats


# ==================== MAIN ====================

def main():
    """Hàm chính chạy tiền xử lý"""
    print("=" * 60)
    print("🎤 TIỀN XỬ LÝ DỮ LIỆU VIVOS CHO WHISPER")
    print("=" * 60)
    
    # Cấu hình - Thay đổi đường dẫn phù hợp
    config = DataConfig(
        data_path=r"D:\Data\vivos",  # ⚠️ Thay đổi đường dẫn này
        whisper_model="openai/whisper-small",
        output_dir="./processed_data"
    )
    
    # Kiểm tra đường dẫn
    if not os.path.exists(config.data_path):
        print(f"\n❌ Không tìm thấy thư mục dữ liệu: {config.data_path}")
        print("Vui lòng thay đổi đường dẫn trong DataConfig!")
        return
    
    # Load dataset
    print("\n" + "=" * 40)
    print("📥 BƯỚC 1: LOAD DỮ LIỆU")
    print("=" * 40)
    
    dataset = load_vivos_dataset(config.data_path)
    
    # Thống kê
    print("\n" + "=" * 40)
    print("📊 BƯỚC 2: THỐNG KÊ DỮ LIỆU")
    print("=" * 40)
    
    stats = get_dataset_statistics(dataset)
    for split, split_stats in stats.items():
        print(f"\n{split.upper()}:")
        for key, value in split_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Tiền xử lý
    print("\n" + "=" * 40)
    print("⚙️ BƯỚC 3: TIỀN XỬ LÝ")
    print("=" * 40)
    
    processed = preprocess_dataset(dataset, config, save=True)
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH!")
    print(f"📁 Dữ liệu đã được lưu tại: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
