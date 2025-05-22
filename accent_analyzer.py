import os
import tempfile
import requests
import numpy as np
from faster_whisper import WhisperModel
from pathlib import Path
import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
from pydub import AudioSegment
import io
import re
import subprocess
import shutil
import gdown
import gc
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class AccentAnalyzer:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AccentAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                # Set device
                self.device = torch.device("cpu")
                logger.info(f"Using device: {self.device}")
                
                # Initialize models
                self._initialize_models()
                
                # Set accent labels
                self.accent_labels = ['British', 'American', 'Australian', 'Other']
                
                self._initialized = True
                logger.info("AccentAnalyzer initialized successfully")
            except Exception as e:
                logger.error(f"Error during initialization: {str(e)}")
                self._cleanup()
                raise

    def _initialize_models(self):
        """Initialize models with proper error handling and cleanup."""
        try:
            # Initialize Whisper model
            logger.info("Initializing Whisper model...")
            self.whisper_model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8",
                download_root=MODEL_CACHE_DIR
            )
            
            # Initialize WavLM model and feature extractor
            logger.info("Initializing WavLM model...")
            self.model_name = "microsoft/wavlm-base"
            
            # Load feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                cache_dir=MODEL_CACHE_DIR
            )
            
            # Load model
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                num_labels=4,
                ignore_mismatched_sizes=True,
                cache_dir=MODEL_CACHE_DIR
            )
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            self._cleanup()
            raise

    def _cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'whisper_model'):
                del self.whisper_model
            if hasattr(self, 'feature_extractor'):
                del self.feature_extractor
            
            gc.collect()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self._cleanup()

    def is_google_drive_url(self, url):
        """Check if the URL is a Google Drive URL."""
        drive_regex = r'https?://drive\.google\.com/(?:file/d/|open\?id=)([a-zA-Z0-9_-]+)'
        return bool(re.match(drive_regex, url))

    def get_google_drive_file_id(self, url):
        """Extract file ID from Google Drive URL."""
        if '/file/d/' in url:
            return re.search(r'/file/d/([a-zA-Z0-9_-]+)', url).group(1)
        elif 'open?id=' in url:
            return re.search(r'open\?id=([a-zA-Z0-9_-]+)', url).group(1)
        return None

    def download_google_drive_video(self, url):
        """Download video from Google Drive URL using gdown."""
        temp_path = None
        try:
            # Extract file ID
            file_id = self.get_google_drive_file_id(url)
            if not file_id:
                raise Exception("Invalid Google Drive URL format")

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_path = temp_file.name
            temp_file.close()

            # Download using gdown
            output = gdown.download(id=file_id, output=temp_path, quiet=True)
            
            if not output or not os.path.exists(temp_path):
                raise Exception("Failed to download from Google Drive")

            # Verify the file exists and has content
            if os.path.getsize(temp_path) == 0:
                raise Exception("Downloaded file is empty")

            return temp_path

        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise Exception(f"Error downloading from Google Drive: {str(e)}")

    def download_video(self, url):
        """Download video from URL to temporary file."""
        temp_path = None
        try:
            if self.is_google_drive_url(url):
                return self.download_google_drive_video(url)
            
            # First, check if the URL is accessible
            head_response = requests.head(url, allow_redirects=True)
            if head_response.status_code != 200:
                raise Exception(f"URL is not accessible: {head_response.status_code}")

            # Get content type
            content_type = head_response.headers.get('content-type', '')
            if not any(video_type in content_type.lower() for video_type in ['video', 'mp4', 'quicktime']):
                raise Exception(f"URL does not point to a video file. Content-Type: {content_type}")

            # Download the file
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise Exception(f"Failed to download video: {response.status_code}")

            # Create a temporary file with .mp4 extension
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_path = temp_file.name
            temp_file.close()

            # Download the file in chunks
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify the file exists and has content
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise Exception("Downloaded file is empty or does not exist")

            return temp_path

        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise Exception(f"Error downloading video: {str(e)}")

    def extract_audio(self, video_path):
        """Extract audio from video file using pydub."""
        audio_path = None
        try:
            # Verify the video file exists and has content
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise Exception("Video file is empty or does not exist")

            # Create audio path
            audio_path = video_path.replace('.mp4', '.wav')

            try:
                # Load video and extract audio
                video = AudioSegment.from_file(video_path)
                
                # Convert to mono and set sample rate
                audio = video.set_channels(1).set_frame_rate(16000)
                
                # Export as WAV
                audio.export(audio_path, format="wav")
                
            except Exception as e:
                raise Exception(f"Error extracting audio: {str(e)}")

            # Verify the audio file was created
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise Exception("Failed to create audio file")

            return audio_path

        except Exception as e:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            raise Exception(f"Error processing video: {str(e)}")

    def get_transcription(self, audio_path):
        """Get transcription using Whisper."""
        try:
            segments, _ = self.whisper_model.transcribe(audio_path)
            return " ".join([segment.text for segment in segments])
        except Exception as e:
            raise Exception(f"Error getting transcription: {str(e)}")

    def preprocess_audio(self, audio_path):
        """Preprocess audio for the model using librosa."""
        try:
            # Load audio file using librosa
            waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            # Extract features
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            return inputs.to(self.device)

        except Exception as e:
            raise Exception(f"Error preprocessing audio: {str(e)}")

    def predict_accent(self, audio_path):
        """Predict accent using the model."""
        try:
            inputs = self.preprocess_audio(audio_path)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get prediction and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = float(probabilities[0][predicted_class].item())
            
            return predicted_class, confidence

        except Exception as e:
            raise Exception(f"Error predicting accent: {str(e)}")

    def analyze_accent(self, url):
        """Main function to analyze accent from video URL."""
        video_path = None
        audio_path = None
        try:
            logger.info(f"Starting accent analysis for URL: {url}")
            
            # Download and process video
            video_path = self.download_video(url)
            logger.info("Video downloaded successfully")
            
            audio_path = self.extract_audio(video_path)
            logger.info("Audio extracted successfully")

            # Get transcription
            transcription = self.get_transcription(audio_path)
            logger.info("Transcription completed")
            
            # Get accent prediction
            accent_idx, confidence = self.predict_accent(audio_path)
            logger.info(f"Accent prediction completed: {self.accent_labels[accent_idx]}")

            # Cleanup temporary files
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

            return {
                'accent': self.accent_labels[accent_idx],
                'confidence': float(confidence),
                'transcription': transcription,
                'explanation': f"Based on the analysis of speech patterns and pronunciation, the speaker appears to have a {self.accent_labels[accent_idx].lower()} accent with {confidence:.1%} confidence."
            }

        except Exception as e:
            logger.error(f"Error during accent analysis: {str(e)}")
            # Cleanup on error
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            return {
                'error': str(e),
                'accent': None,
                'confidence': 0.0,
                'transcription': None,
                'explanation': None
            } 