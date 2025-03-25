from moviepy import VideoFileClip 
import os

def extract_audio(video_path, output_audio_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    # Load video and extract audio
    video = VideoFileClip(video_path)
    audio = video.audio

    if audio:
        audio.write_audiofile(output_audio_path, codec="mp3")
        print(f"Audio extracted and saved to {output_audio_path}")
    else:
        print("No audio track found in the video.")

    # Close the video to free resources
    video.close()

