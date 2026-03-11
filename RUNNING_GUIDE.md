# KinetoCheck - Running Guide

Complete guide to run the KinetoCheck application with trained model.

## Prerequisites

- **Backend**: Python 3.11+, PyTorch (CUDA optional), trained model at `Backend/weights/stgat_best.pt`
- **Frontend**: Flutter SDK 3.9+

---

## 1. Start Backend API Server

```powershell
# Navigate to backend
cd Backend

# Install dependencies (if not done)
pip install -r requirements.txt

# Start FastAPI server
python run.py
```

The server will start at: **http://localhost:8000**

You can test it by visiting: http://localhost:8000/docs (Swagger UI)

---

## 2. Configure Frontend API Endpoint

Edit `frontend/lib/api/api_service.dart`:

```dart
static const String baseUrl = 'http://localhost:8000';  // For local development
```

### Platform-specific URLs:
- **Windows/Mac/Linux**: `http://localhost:8000`
- **Android Emulator**: `http://10.0.2.2:8000`
- **iOS Simulator**: `http://localhost:8000`
- **Physical Device**: Use your computer's IP address (e.g., `http://192.168.1.100:8000`)

---

## 3. Run Flutter App

```powershell
# Navigate to frontend
cd frontend

# Get dependencies
flutter pub get

# Run on connected device/emulator
flutter run
```

Or run from VS Code:
1. Open `frontend/lib/main.dart`
2. Press `F5` or click "Run" → "Start Debugging"

---

## 4. Using the App

1. **Select Video**: Click "Select Video" and choose a movement video
2. **Preview**: The video will appear in the preview box (click play icon to watch)
3. **Analyze**: Click "Analyze Movement" to upload and process
4. **Results**: See if the movement is correctly executed with confidence scores

### Result Interpretation:
- **Green (Correctly Executed)**: Movement performed correctly
- **Orange (Needs Improvement)**: Movement has execution issues
- Confidence percentage shows model certainty

---

## Troubleshooting

### Backend Issues

**"Model not found"**
- Ensure `Backend/weights/stgat_best.pt` exists
- If training on Colab, download the model and place it in `Backend/weights/`

**"CUDA not available" (if you have GPU)**
- Install CUDA PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

### Frontend Issues

**"Connection refused" / "Network error"**
- Check backend is running at http://localhost:8000
- Verify the `baseUrl` in `api_service.dart` matches your setup
- For Android emulator, use `http://10.0.2.2:8000`
- For physical device, use your computer's IP address

**"Cannot pick video file"**
- Ensure you granted file/storage permissions
- On Android: Check app permissions in Settings

---

## API Endpoints

- `POST /api/v1/predict/video` - Upload video for analysis
- `POST /api/v1/predict/keypoints` - Send pre-extracted keypoints
- `GET /api/v1/models` - List available models
- `GET /api/v1/health` - Check server status
- `GET /docs` - Interactive API documentation (Swagger UI)

---

## Development Tips

### Hot Reload (Flutter)
- Save files in your editor for instant UI updates
- Press `r` in terminal for manual hot reload
- Press `R` for full app restart

### Backend Changes
- Restart `run.py` after code changes
- Model changes require full restart

### GPU Training vs CPU Inference
- Training: Use GPU (10-50x faster) via Colab or local CUDA
- Inference: CPU is fast enough for single videos (~2-5 seconds)

---

## File Structure

```
Backend/
  weights/stgat_best.pt    ← Trained model
  run.py                   ← Start server here
  app/
    api/routes.py          ← API endpoints
    services/inference_service.py  ← Model inference logic

frontend/
  lib/
    main.dart              ← App entry point
    api/api_service.dart   ← Backend API client
    screens/video_upload_screen.dart  ← Main UI
```

---

## Next Steps

- Add real-time camera analysis
- Implement exercise tracking history
- Add multiple movement types
- Export analysis reports
- User authentication
