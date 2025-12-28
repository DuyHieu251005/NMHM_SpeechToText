# Vietnamese Speech-to-Text Web Application

Ứng dụng web chuyển đổi giọng nói tiếng Việt thành văn bản sử dụng 3 mô hình AI:
- **Wav2Vec2**: Mô hình nhẹ, nhanh
- **PhoWhisper**: Tối ưu cho tiếng Việt bởi VinAI
- **OpenAI Whisper**: Mô hình đa ngôn ngữ mạnh mẽ

## Tính năng

Upload file audio (WAV, MP3, M4A, FLAC)  
Ghi âm trực tiếp từ microphone  
Chọn giữa 3 mô hình AI  
Giao diện đẹp, thân thiện  
Hỗ trợ drag & drop  

## Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/DuyHieu251005/NMHM_SpeechToText.git
cd NMHM_SpeechToText/app
```

### 2. Tạo môi trường ảo
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

### 4. Chạy ứng dụng
```bash
python app.py
```

### 5. Truy cập ứng dụng
Mở trình duyệt và truy cập: **http://localhost:5000**

## Yêu cầu hệ thống

- Python 3.8+ (khuyến nghị 3.10 hoặc 3.11)
- RAM: tối thiểu 8GB (khuyến nghị 16GB)
- GPU: NVIDIA GPU với CUDA (khuyến nghị, không bắt buộc)
- Dung lượng: ~5GB cho các mô hình
- **FFmpeg**: Bắt buộc để xử lý audio từ microphone

## Cài đặt FFmpeg (BẮT BUỘC)

FFmpeg là thư viện xử lý audio/video, cần thiết để decode các định dạng webm/ogg từ trình duyệt.

### Windows (sử dụng winget):
```powershell
winget install --id Gyan.FFmpeg -e --source winget
```
*Sau khi cài, restart terminal hoặc VS Code để nhận PATH mới.*

### Windows (thủ công):
1. Tải từ https://www.gyan.dev/ffmpeg/builds/
2. Giải nén vào `C:\ffmpeg`
3. Thêm `C:\ffmpeg\bin` vào biến môi trường PATH

### Linux (Ubuntu/Debian):
```bash
sudo apt update && sudo apt install ffmpeg
```

### macOS:
```bash
brew install ffmpeg
```

### Kiểm tra cài đặt:
```bash
ffmpeg -version
```

## Cài đặt CUDA (Tùy chọn - cho GPU NVIDIA)

Nếu bạn có GPU NVIDIA và muốn tăng tốc inference:

1. Cài đặt [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads)
2. Cài đặt PyTorch với CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Xử lý lỗi thường gặp

### Lỗi `audioread.exceptions.NoBackendError`
**Nguyên nhân**: Thiếu FFmpeg  
**Giải pháp**: Cài đặt FFmpeg theo hướng dẫn ở trên, sau đó restart terminal/VS Code

### Lỗi `CUDA out of memory`
**Nguyên nhân**: GPU không đủ VRAM  
**Giải pháp**: Sử dụng CPU bằng cách set `device = "cpu"` trong app.py

### Lỗi `No module named 'xxx'`
**Nguyên nhân**: Thiếu thư viện  
**Giải pháp**: `pip install -r requirements.txt`

### Lỗi microphone không hoạt động
**Nguyên nhân**: Trình duyệt chưa được cấp quyền  
**Giải pháp**: Nhấn biểu tượng khóa trên thanh địa chỉ → Cho phép Microphone → Refresh trang

## Cấu trúc thư mục

```
app/
├── app.py              # Flask backend
├── requirements.txt    # Dependencies
├── README.md          # Hướng dẫn
└── templates/
    └── index.html     # Giao diện web
```

## Sử dụng

1. **Chọn mô hình**: Dropdown menu để chọn Wav2Vec2, PhoWhisper, hoặc OpenAI Whisper
2. **Upload file**: Kéo thả hoặc click để chọn file audio
3. **Ghi âm**: Chuyển sang tab "Ghi Âm" và nhấn nút microphone
4. **Chuyển đổi**: Nhấn nút "Chuyển đổi thành văn bản"
5. **Kết quả**: Văn bản sẽ hiển thị bên dưới, có thể sao chép

## API Endpoints

- `GET /`: Trang chủ
- `POST /transcribe`: Chuyển đổi audio thành văn bản
- `GET /health`: Kiểm tra trạng thái server

## License

MIT License
