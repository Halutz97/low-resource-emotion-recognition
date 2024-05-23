# This script converts video files from .flv to .mp4
import os
from moviepy.editor import VideoFileClip

def convert_flv_to_mp4(input_file, output_file):
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping conversion.")
        return
    video = VideoFileClip(input_file)
    video.write_videofile(output_file, codec='libx264')  # 'libx264' is typically used for MP4 files

# Example usage

def main():
    # Convert all files in a directory
    input_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST"
    output_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST_MP4"
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in os.listdir(input_dir):
        if file.endswith(".flv"):
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, file.replace(".flv", ".mp4"))
            convert_flv_to_mp4(input_file, output_file)

if __name__ == "__main__":
    main()




