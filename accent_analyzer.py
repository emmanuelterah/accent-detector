import os
import tempfile
import numpy as np
import torch
import yt_dlp
import subprocess
import logging
import shutil
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
import glob
import re
import gdown
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def find_ffmpeg():
    """Find FFmpeg executable in system PATH or common installation locations."""
    # Check if ffmpeg is in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
        
    # Check common installation locations
    common_paths = [
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
            
    return None

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
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using device: {self.device}")
                
                # Find FFmpeg
                self.ffmpeg_path = find_ffmpeg()
                if not self.ffmpeg_path:
                    raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in your system PATH.")
                logger.info(f"Found FFmpeg at: {self.ffmpeg_path}")
                
                # Initialize models
                self._initialize_models()
                
                # Set accent labels
                self.accent_labels = ['British', 'American', 'Australian', 'Indian', 'Other']
                
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
            
            # Initialize accent classification model
            logger.info("Initializing accent classification model...")
            self.model_name = "dima806/english_accents_classification"
            
            # Load feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                cache_dir=MODEL_CACHE_DIR
            )
            
            # Load model with correct number of labels (5 instead of 4)
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                num_labels=5,  # Model has 5 classes
                ignore_mismatched_sizes=True,
                cache_dir=MODEL_CACHE_DIR
            )
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Update accent labels to match model's classes
            self.accent_labels = ['British', 'American', 'Australian', 'Indian', 'Other']
            
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
            
            torch.cuda.empty_cache()
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

    def list_available_formats(self, url):
        """List available formats for a video URL."""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise Exception("Could not extract video information")
                
                formats = info.get('formats', [])
                logger.info("\nAvailable formats:")
                for f in formats:
                    logger.info(f"Format: {f.get('format_id')} - {f.get('ext')} - {f.get('format_note')} - {f.get('resolution')}")
                return formats
        except Exception as e:
            logger.error(f"Error listing formats: {str(e)}")
            return []

    def download_video(self, url: str) -> str:
        """Download video from URL (YouTube or Google Drive) and return local path."""
        try:
            # Create temp directory if it doesn't exist
            os.makedirs(TEMP_DIR, exist_ok=True)
            
            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(TEMP_DIR, f"video_{timestamp}.mp4")
            
            # Check if it's a Google Drive URL
            if self.is_google_drive_url(url):
                file_id = self.get_google_drive_file_id(url)
                if not file_id:
                    raise ValueError("Invalid Google Drive URL")
                
                # Download using gdown
                output = gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    output_path,
                    quiet=False
                )
                
                if output is None:
                    raise Exception("Failed to download from Google Drive")
                
                return output_path
            
            # For YouTube URLs, use yt-dlp with specific format
            ydl_opts = {
                'format': 'best[ext=mp4]/best',  # Prefer MP4 format
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
                'ffmpeg_location': self.ffmpeg_path,  # Use found ffmpeg path
                'merge_output_format': 'mp4',
                'verbose': True  # Enable verbose output for debugging
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # First try to get video info
                    info = ydl.extract_info(url, download=False)
                    logger.info(f"Available formats: {[f['format_id'] for f in info['formats']]}")
                    
                    # Download the video
                    ydl.download([url])
                    
                    # Verify the file was downloaded
                    if os.path.exists(output_path):
                        logger.info(f"Successfully downloaded video to {output_path}")
                        return output_path
                    else:
                        raise Exception("Video file not found after download")
                        
            except Exception as e:
                logger.error(f"Error downloading video: {str(e)}")
                # Try alternative format if first attempt fails
                ydl_opts['format'] = 'best'
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    if os.path.exists(output_path):
                        return output_path
                    raise Exception(f"Failed to download video after retry: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in download_video: {str(e)}")
            raise Exception(f"Error downloading video: {str(e)}")

    def extract_audio(self, video_path):
        """Extract first 60 seconds of audio using ffmpeg."""
        duration_sec = 60
        audio_path = video_path.rsplit('.', 1)[0] + ".wav"
        command = [
            self.ffmpeg_path,
            '-y',
            '-i', video_path,
            '-t', str(duration_sec),  # only first N seconds
            '-ac', '1',
            '-ar', '16000',
            audio_path
        ]
        try:
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise Exception(f"Error extracting audio: {str(e)}")

    def get_transcription(self, audio_path):
        """Get transcription using Whisper."""
        try:
            segments, _ = self.whisper_model.transcribe(audio_path)
            return " ".join([segment.text for segment in segments])
        except Exception as e:
            raise Exception(f"Error getting transcription: {str(e)}")

    def preprocess_audio(self, audio_path):
        """Preprocess audio for the model."""
        try:
            # Load audio file using librosa
            waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            # Extract features
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=16000 * 60,  # Process up to 60 seconds
                truncation=True
            )
            
            # Convert inputs to the same dtype as the model
            inputs = {k: v.to(dtype=torch.float32, device=self.device) for k, v in inputs.items()}
            
            return inputs

        except Exception as e:
            raise Exception(f"Error preprocessing audio: {str(e)}")

    def predict_accent(self, audio_path):
        """Predict accent using the model."""
        try:
            inputs = self.preprocess_audio(audio_path)
            
            # Run inference
            with torch.no_grad():
                self.model.eval()  # Ensure model is in eval mode
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