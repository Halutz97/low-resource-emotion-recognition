from moviepy.editor import VideoFileClip

# Replace 'your_video_file.mp4' with the path to your MP4 file
video_file_path = 'your_video_file.mp4'
audio_file_path = 'extracted_audio.wav'  # Output audio file in WAV format

# Load the video file
video = VideoFileClip(video_file_path)

# Extract the audio
audio = video.audio

# Write the audio to a file (in WAV format)
audio.write_audiofile(audio_file_path)

# Close the audio and video files to free resources
audio.close()
video.close()
