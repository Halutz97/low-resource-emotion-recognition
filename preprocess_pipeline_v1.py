from util import extract_audio_from_video
from util import detect_broken_audio_files
from util import renaming_MELD_files
from util import match_labels
import os
import shutil
import librosa
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import Wav2Vec2ForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

def handle_MELD(directory):
    renaming_MELD_files.rename_audio_files(directory)

def default_case():
    print("No dataset match found.")

dataset_switch = {
    "MELD": handle_MELD,
    "CREMA-D": handle_MELD
}

def switch_case(dataset, *args, **kwargs):
    if dataset in dataset_switch:
        dataset_switch[dataset](*args, **kwargs)
    else:
        default_case()

def main():
    toggle_controls = [True, True, True, True]
    dataset = "MELD"
    video_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\videos"
    labels_path = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\dev_sent_emo.csv"

    extract_audio_files_from_video = toggle_controls[0]
    rename_files = toggle_controls[1]
    detect_broken_files = toggle_controls[2]
    extract_corrected_labels = toggle_controls[3]
    
    audio_directory = os.path.join(os.path.dirname(video_directory), os.path.basename(video_directory) + "_audio")
    corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")



    if extract_audio_files_from_video:
        if not os.path.exists(audio_directory):
            os.makedirs(audio_directory)
        if not os.listdir(audio_directory):
            extract_audio_from_video.extract_files(video_directory, audio_directory, num_files_to_process=50)
        else:
            print("Audio file folder not empty.")

    if rename_files:
        switch_case(dataset, audio_directory)
    
    if detect_broken_files:
        detect_broken_audio_files.process_files(audio_directory)
    
    if extract_corrected_labels:
        match_labels.match_emotion_labels(labels_path,  corrected_labels_path, audio_directory)

if __name__ == "__main__":
    main()