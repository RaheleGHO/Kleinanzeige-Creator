import os
import logging
from typing import List, Dict, Tuple, Optional
import streamlit as st
from openai import OpenAI
from extract_frames import extract_frames
from extract_audio import extract_audio
from transcribe_audio import transcribe
from langdetect import detect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """A class to handle video processing including frame extraction, audio transcription,
    and description generation."""
    
    def __init__(self, video_path: str):
        """Initialize the VideoProcessor with video path and output directories."""
        self.video_path = video_path
        self.frames_dir = "outputs/frames/"
        self.audio_path = "outputs/audio/audio.mp3"
        self.transcripts_path = "outputs/transcripts/transcript.txt"
        self.description_path = "outputs/description/description.txt"
        
        # Create output directories if they don't exist
        for path in [self.frames_dir, os.path.dirname(self.audio_path),
                    os.path.dirname(self.transcripts_path),
                    os.path.dirname(self.description_path)]:
            os.makedirs(path, exist_ok=True)

    def generate_description(self, transcript: str, target_lang: str) -> str:
        """Generate a product description using GPT-4 based on video transcript.
        
        Args:
            transcript: The full text transcript of the video
            target_lang: Target language code (e.g. 'en', 'de')
            
        Returns:
            str: Generated product description in target language
        """
        try:
            # First generate description in English
            prompt = """
            Create a concise product description in English based on this video transcript.
            Format the description as line-by-line bullet points:
            - Product name and type
            - Key features (one per line)
            - Condition
            - Age suitability (if mentioned)
            - Price (if mentioned)
            
            Keep it brief and factual. Use simple language. Only include information from the transcript.
            Start each line with a hyphen and space.
            
            Transcript:
            {transcript}
            """
            
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.format(transcript=transcript)}],
                temperature=0.7
            )
            
            description = response.choices[0].message.content
            
            # Translate to target language if needed
            if target_lang != "en":
                description = self.translate_text(description, target_lang)
            
            return description
        except Exception as e:
            logger.error(f"Failed to generate description: {str(e)}")
            raise

    def process_video(self, target_lang: str) -> Tuple[List[Dict], str]:
        """Process the video through the full pipeline.
        
        Args:
            target_lang: Selected target language for transcription
            
        Returns:
            tuple: (transcripts, description) where:
                - transcripts: List of transcription segments
                - description: Generated product description
        """
        try:
            # Extract frames
            with st.spinner("Extracting frames..."):
                extract_frames(self.video_path, self.frames_dir, frame_interval=60)
                logger.info(f"Extracted frames to {self.frames_dir}")

            # Extract audio
            with st.spinner("Extracting audio..."):
                extract_audio(self.video_path, self.audio_path)
                logger.info(f"Extracted audio to {self.audio_path}")

            # Transcribe audio
            with st.spinner("Transcribing audio..."):
                transcript = transcribe(self.audio_path)
                
                # Detect and translate if needed
                detected_lang = self.detect_language(transcript)
                if detected_lang and detected_lang != target_lang:
                    with st.spinner("Translating transcription..."):
                        transcript = self.translate_text(
                            transcript,
                            target_lang
                        )
                
                self._save_transcripts(transcript)
                logger.info(f"Saved transcripts to {self.transcripts_path}")

            # Generate description
            with st.spinner("Generating description..."):
                description = self.generate_description(transcript, target_lang)
                self._save_description(description)
                logger.info(f"Saved description to {self.description_path}")
            
            return transcript, description
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise

    def _save_transcripts(self, transcript: str) -> None:
        """Save transcription text to file.
        
        Args:
            transcript: The full transcription text
        """
        with open(self.transcripts_path, 'w', encoding="utf-8") as f:
            f.write(transcript)

    def _save_description(self, description: str) -> None:
        """Save generated description to file.
        
        Args:
            description: The generated description text
        """
        with open(self.description_path, 'w', encoding="utf-8") as f:
            f.write(description)

    def detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the given text.
        
        Args:
            text: Input text to detect language from
            
        Returns:
            str: Detected language code or None if detection fails
        """
        try:
            return detect(text)
        except:
            return None

    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text to target language using GPT-4.
        
        Args:
            text: Text to translate
            target_lang: Target language code (e.g. 'en', 'de')
            
        Returns:
            str: Translated text
        """
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": f"Translate the following text to {target_lang}. Keep the meaning accurate."
                }, {
                    "role": "user",
                    "content": text
                }],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text

class VideoApp:
    """Streamlit application for video processing."""
    
    def __init__(self):
        """Initialize the Streamlit app with supported languages and processing settings."""
        self.video_path = None
        self.processor = None
        self.target_languages = {
            "English": "en",
            "German": "de",
            "French": "fr",
            "Spanish": "es",
            "Italian": "it"
        }
        
    def _upload_video(self) -> Optional[str]:
        """Handle video file upload and return temporary file path.
        
        Returns:
            str: Path to the uploaded video file or None if no file uploaded
        """
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv"],
            help="Supported formats: MP4, MOV, AVI, MKV"
        )
        
        if uploaded_file is not None:
            # Create temp directory if it doesn't exist
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return file_path
        return None
        
    def run(self):
        """Run the Streamlit application."""
        st.title("Video Transcription & Description Generator")
        
        # Upload video
        self.video_path = self._upload_video()
        
        if self.video_path is None:
            st.warning("Please upload a video file to begin")
            return
            
        # Display video
        st.video(self.video_path)
        
        # Language selection
        target_lang = st.selectbox(
            "Select target language for transcription:",
            options=list(self.target_languages.keys())
        )
        
        # Initialize processor
        self.processor = VideoProcessor(self.video_path)
        
        if st.button("Process Video"):
            try:
                with st.spinner("Processing video..."):
                    transcripts, description = self.processor.process_video(target_lang)
                
                # Display results
                st.success("Processing complete!")
                self._display_transcription(transcripts)
                self._display_description(description)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Application error: {str(e)}")

    def _display_transcription(self, transcript: str) -> None:
        """Display transcription text.
        
        Args:
            transcript: The full transcription text
        """
        st.subheader("Transcription")
        st.write(transcript)

    def _display_description(self, description: str) -> None:
        """Display generated description.
        
        Args:
            description: The generated description text
        """
        st.subheader("Generated Description")
        st.write(description)


if __name__ == "__main__":
    app = VideoApp()
    app.run()
