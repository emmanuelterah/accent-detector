# 🎙️ Accent Analysis Tool

A powerful web application that analyzes accents in videos using advanced machine learning models. This tool can detect various English accents (British, American, Australian, Indian, and Other) from video content, providing both transcription and accent classification with confidence scores.

## ✨ Features

- **Video Processing**
  - Supports YouTube and Google Drive video URLs
  - Automatic video downloading and audio extraction
  - Smart format selection for optimal quality

- **Accent Analysis**
  - Detects 5 different English accents:
    - British
    - American
    - Australian
    - Indian
    - Other
  - Provides confidence scores for predictions
  - Includes full speech transcription

- **Modern UI**
  - Beautiful glassmorphism design
  - Responsive layout
  - Real-time processing feedback
  - Clean and intuitive interface

## 🛠️ Technical Stack

- **Backend**
  - Python 3.x
  - Streamlit for web interface
  - PyTorch for deep learning
  - Transformers library for accent classification
  - Faster-Whisper for speech recognition
  - yt-dlp for video downloading
  - FFmpeg for audio processing

- **Models**
  - Whisper (base) for speech transcription
  - Custom accent classification model (dima806/english_accents_classification)

## 📋 Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- CUDA-capable GPU (optional, but recommended for faster processing)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/accent-detector.git
   cd accent-detector
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install FFmpeg:
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - **Linux**: `sudo apt-get install ffmpeg`
   - **macOS**: `brew install ffmpeg`

## 🎮 Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter a video URL (YouTube or Google Drive) in the input field

4. Click "Analyze Accent" and wait for the results

5. View the results:
   - Detected accent
   - Confidence score
   - Full transcription
   - Detailed explanation

## 🔧 Configuration

The application uses several configuration parameters that can be modified in the code:

- `MODEL_CACHE_DIR`: Directory for caching ML models
- `TEMP_DIR`: Directory for temporary files
- Audio processing duration (default: 60 seconds)
- Model parameters and settings

## 📝 Notes

- The application processes the first 60 seconds of audio for analysis
- Temporary files are automatically cleaned up after processing
- For best results, use videos with clear speech and minimal background noise
- Processing time may vary depending on video length and system specifications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading
- [Streamlit](https://streamlit.io/) for the web interface
- [Hugging Face](https://huggingface.co/) for the accent classification model 