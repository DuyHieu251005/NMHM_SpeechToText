# Vietnamese Speech-to-Text Web Application

Ứng dụng web chuyển đổi giọng nói tiếng Việt thành văn bản sử dụng 3 mô hình AI:
- **Wav2Vec2**: Mô hình nhẹ, nhanh
- **PhoWhisper**: Tối ưu cho tiếng Việt bởi VinAI
- **OpenAI Whisper**: Mô hình đa ngôn ngữ mạnh mẽ

## Tính năng

✅ Upload file audio (WAV, MP3, M4A, FLAC)  
✅ Ghi âm trực tiếp từ microphone  
✅ Chọn giữa 3 mô hình AI  
✅ Giao diện đẹp, thân thiện  
✅ Hỗ trợ drag & drop  

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

- Python 3.8+
- RAM: tối thiểu 8GB (khuyến nghị 16GB)
- GPU: NVIDIA GPU với CUDA (khuyến nghị, không bắt buộc)
- Dung lượng: ~5GB cho các mô hình

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
