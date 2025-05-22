# Accent Analysis Tool

A tool that analyzes accents in speech using AI models. The tool can process video URLs and provide accent classification along with transcription.

## Features

- Accent classification (British, American, Australian, Other)
- Speech transcription
- Support for Google Drive and direct MP4 video links
- Modern, responsive UI
- Real-time analysis
- Memory-optimized for cloud deployment

## Resource Optimization

The app is optimized for Streamlit Cloud's free tier:
- Uses Whisper 'tiny' model for faster processing
- Implements half-precision (FP16) for memory efficiency
- CPU-only PyTorch installation
- Automatic memory cleanup
- Model caching for faster subsequent runs

## Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)

3. Sign in with your GitHub account

4. Click "New app"

5. Select your forked repository

6. Set the following:
   - Main file path: `streamlit_app.py`
   - Python version: 3.9
   - Requirements file: `requirements-streamlit.txt`

7. Click "Deploy"

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements-streamlit.txt
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

## Usage

1. Enter a video URL (Google Drive or direct MP4 link)
2. Click "Analyze Accent"
3. Wait for the analysis to complete
4. View the results:
   - Detected accent
   - Confidence score
   - Speech transcription
   - Detailed explanation

## Notes

- First-time initialization may take a few minutes as models are downloaded
- Analysis time depends on video length and server load
- For best results, use clear audio with minimal background noise
- Keep videos under 2 minutes for optimal performance
- App may take longer to start after 24 hours of inactivity (Streamlit Cloud limitation)

## Technical Details

- Uses Whisper (tiny) for speech transcription
- Uses WavLM for accent classification
- Implements model caching for faster subsequent runs
- Handles various video formats and sources
- Provides detailed error handling and logging 