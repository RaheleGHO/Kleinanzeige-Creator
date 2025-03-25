# Video Transcription & Description Generator

A Streamlit application that processes video files to:
1. Extract key frames
2. Extract and transcribe audio
3. Generate product descriptions from the transcription
4. Support multiple languages (English, German, French, Spanish, Italian)

## Features
- Video frame extraction at configurable intervals
- Audio extraction to MP3 format
- Audio transcription using OpenAI Whisper
- Product description generation using GPT-4
- Multi-language support with auto-detection and translation

## Installation

1. Clone this repository:
```bash
git clone [repository_url]
cd EbayPro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. In the web interface:
   - Upload a video file
   - Select target language
   - Click "Process Video"

3. The app will:
   - Extract frames and save to `outputs/frames/`
   - Extract audio and save to `outputs/audio/audio.mp3`
   - Generate transcription and save to `outputs/transcripts/transcript.txt`
   - Generate product description and save to `outputs/description/description.txt`

## Configuration

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key for Whisper and GPT-4 access

## Dependencies

Listed in requirements.txt:
- streamlit
- openai
- moviepy
- opencv-python
- langdetect

Note: Python standard library modules (os, logging, etc.) are not listed as they come with Python.

## Example Output

After processing a video, you'll get:
- Key frames as JPG images
- Audio file in MP3 format
- Text transcription
- Formatted product description

## Notes

- Video processing may take several minutes depending on video length
- Larger videos will consume more OpenAI API credits
- Supported video formats: MP4, MOV, AVI, MKV
