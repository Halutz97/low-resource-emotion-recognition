import os
from pydub import AudioSegment


def convert_stereo_to_mono(input_path, output_path):
    # Load the stereo audio file
    stereo_audio = AudioSegment.from_wav(input_path)
    
    # Convert to mono by averaging the channels
    mono_audio = stereo_audio.set_channels(1)
    
    # Export the mono audio file
    mono_audio.export(output_path, format="wav")


input_folder = r'C:\Users\DANIEL\Desktop\thesis\AudioWAV_resampled'
output_folder = r'C:\Users\DANIEL\Desktop\thesis\AudioWAV_resampled_mono'

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        input_wav = os.path.join(input_folder, file)
        output_wav = os.path.join(output_folder, file)
        convert_stereo_to_mono(input_wav, output_wav)


print("DONE!")