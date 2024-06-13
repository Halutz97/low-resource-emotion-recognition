import os
import subprocess

def convert_flv_to_mp4(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".flv"):
            flv_file = os.path.join(folder_path, filename)
            mp4_file = os.path.join(folder_path, filename.replace(".flv", ".mp4"))
            subprocess.run(["ffmpeg", "-i", flv_file, "-c:v", "libx264", "-c:a", "aac", mp4_file])
            print(f"Converted {flv_file} to {mp4_file}")

folder_path = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\VideoFlash"  # Replace with your folder path
convert_flv_to_mp4(folder_path)