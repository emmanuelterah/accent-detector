# Accent Analysis Tool

This tool analyzes English accents from video URLs, providing classification and confidence scores for different English accents (British, American, Australian, etc.).

## Features

- Accepts video URLs (Google Drive or direct MP4 links)
- Extracts audio from videos
- Analyzes speaker's accent
- Provides accent classification and confidence scores
- Simple web interface using Streamlit

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the web interface (default: http://localhost:8501)
2. Enter a video URL
3. Click "Analyze" to get results
4. View the accent classification and confidence score

## Technical Details

The tool uses:
- Whisper for speech recognition
- Resemblyzer for voice embeddings
- XGBoost for accent classification
- Streamlit for the web interface

## Note

This is a proof-of-concept tool for evaluating English accents in hiring scenarios. The accuracy may vary based on audio quality and speaking conditions. 