#  Vietnamese Speech-to-Text Web Application

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Đồ án Nhập môn Học máy** - Trường ĐH Khoa học Tự nhiên, ĐHQG TP.HCM

Ứng dụng web chuyển đổi giọng nói tiếng Việt thành văn bản sử dụng 3 mô hình AI:

| Mô hình | WER | Mô tả |
|---------|-----|-------|
| **Wav2Vec2** | 11.28%  | Mô hình nhẹ, nhanh - Fine-tuned trên VIVOS |
| **PhoWhisper** | 32.89% | Tối ưu cho tiếng Việt bởi VinAI |
| **OpenAI Whisper** | ~85% | Mô hình đa ngôn ngữ (zero-shot) |

##  Tính năng

-  Upload file audio (WAV, MP3, M4A, FLAC, WebM, OGG)
-  Ghi âm trực tiếp từ microphone
-  Chọn giữa 3 mô hình AI
-  Giao diện đẹp, thân thiện (Bootstrap 5)
-  Hỗ trợ drag & drop
-  Responsive trên mọi thiết bị  

##  Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/DuyHieu251005/NMHM_SpeechToText.git
cd NMHM_SpeechToText/app
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

> **Lưu ý**: Package `imageio-ffmpeg` sẽ tự động cài đặt FFmpeg bundled, không cần cài FFmpeg thủ công!

### 4. Chạy ứng dụng
```bash
python app.py
```

### 5. Truy cập ứng dụng
Mở trình duyệt và truy cập: **http://localhost:5000**

##  Yêu cầu hệ thống

| Yêu cầu | Tối thiểu | Khuyến nghị |
|---------|-----------|-------------|
| Python | 3.8+ | 3.10 hoặc 3.11 |
| RAM | 8GB | 16GB |
| GPU | Không bắt buộc | NVIDIA với CUDA |
| Dung lượng | ~5GB | ~10GB |

##  Cài đặt FFmpeg (TÙY CHỌN)

>  **Không bắt buộc!** Ứng dụng đã sử dụng `imageio-ffmpeg` để xử lý audio tự động.

Nếu muốn cài đặt FFmpeg hệ thống để hỗ trợ thêm:

### Windows (sử dụng winget):
```powershell
winget install --id Gyan.FFmpeg -e --source winget
```

### Linux (Ubuntu/Debian):
```bash
sudo apt update && sudo apt install ffmpeg
```

### macOS:
```bash
brew install ffmpeg
```

##  Cài đặt CUDA (Tùy chọn - cho GPU NVIDIA)

Nếu bạn có GPU NVIDIA và muốn tăng tốc inference:

1. Cài đặt [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads)
2. Cài đặt PyTorch với CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

##  Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `CUDA out of memory` | GPU không đủ VRAM | Set `device = "cpu"` trong app.py |
| `No module named 'xxx'` | Thiếu thư viện | `pip install -r requirements.txt` |
| Microphone không hoạt động | Trình duyệt chưa cấp quyền | Nhấn 🔒 trên thanh địa chỉ → Cho phép Microphone |
| `Model loading error` | Thiếu bộ nhớ | Đóng các ứng dụng khác, tăng RAM |

##  Cấu trúc dự án

```
NMHM_SpeechToText/
├── app/
│   ├── app.py              # Flask backend
│   ├── requirements.txt    # Dependencies
│   └── templates/
│       └── index.html      # Giao diện web
├── Wav2Vec2/
│   ├── checkpoint-3645/    # Model fine-tuned (WER 11.28%)
│   └── *.py                # Scripts training/evaluation
├── PhoWhisper/
│   ├── phowhisper-finetuned-local/  # LoRA adapters
│   └── *.py                # Scripts training/evaluation
├── Whisper/
│   └── *.csv               # Kết quả đánh giá
├── report_final/           # Báo cáo LaTeX
└── README.md
```

##  Sử dụng

1. **Chọn mô hình**: Dropdown menu để chọn Wav2Vec2, PhoWhisper, hoặc Whisper
2. **Upload file**: Kéo thả hoặc click để chọn file audio
3. **Ghi âm**: Chuyển sang tab "Ghi Âm" và nhấn nút 🎙️
4. **Chuyển đổi**: Nhấn nút "Chuyển đổi thành văn bản"
5. **Kết quả**: Văn bản sẽ hiển thị bên dưới, có thể sao chép

##  API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Trang chủ |
| POST | `/transcribe` | Chuyển đổi audio → văn bản |
| GET | `/health` | Kiểm tra trạng thái server |

## Nhóm thực hiện

| Họ tên | MSSV |
|--------|------|
| Đặng Anh Kiệt | 23127077 |
| Phạm Minh Triết | 23127132 |
| Trần Quang Phúc | 23127302 |
| Kiều Duy Hiếu | 23127365 |

**GVHD**: Bùi Tiến Lên, Lê Nhựt Nam, Võ Nhật Tân

## License

MIT License
