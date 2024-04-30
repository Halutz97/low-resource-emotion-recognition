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


def handle_MELD(directory):
    renaming_MELD_files.rename_audio_files(directory)

def handle_CREMA_D(directory):
    # Extracting the labels from the file names
    index = []
    filename = []
    speaker = []
    line = []
    emotions = []
    intensity = []

    for i, file in enumerate(os.listdir(directory)):
        index.append(i)
        filename.append(file)
        speaker.append(file.split("_")[0])
        line.append(file.split("_")[1])
        emotions.append(file.split("_")[2])
        intensity.append(file.split("_")[3][:2])

    # Creating a csv file with the extracted labels
    labels = pd.DataFrame(list(zip(index, filename, speaker, line, emotions, intensity)), columns = ["Index", "filename", "Speaker", "Line", "Emotion", "Intensity"])
    labels.to_csv(os.path.join(os.path.dirname(directory), "labels.csv"), index=False)

    return


def handle_IEMOCAP(directory):
    pass


def default_case():
    print("No dataset match found.")

dataset_switch = {
    "MELD": handle_MELD,
    "CREMA-D": handle_CREMA_D,
    "IEMOCAP": handle_IEMOCAP
}

def switch_case(dataset, *args, **kwargs):
    if dataset in dataset_switch:
        dataset_switch[dataset](*args, **kwargs)
    else:
        default_case()

def main():
    toggle_controls = [True, True, True, True]
    dataset = "CREMA-D"
    audio_directory = ""
    corrected_labels_path = ""
    labels_path = ""
    video_directory = ""


    match dataset:
        case "MELD":

            video_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\videos"
            labels_path = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\dev_sent_emo.csv"

            toggle_controls = [True, True, True, True]

            audio_directory = os.path.join(os.path.dirname(video_directory), os.path.basename(video_directory) + "_audio")
            corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")

        case "CREMA-D":
            old_audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\AudioWAV"
            audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio"
            labels_path = os.path.join(os.path.dirname(audio_directory), "labels.csv")

            # Create the destination directory if it doesn't exist
            if not os.path.exists(audio_directory):
                os.makedirs(audio_directory, exist_ok=True)

                # Copy all files from the source to the destination directory
                for filename in os.listdir(old_audio_directory):
                    source_path = os.path.join(old_audio_directory, filename)
                    destination_path = os.path.join(audio_directory, filename)
                    shutil.copy2(source_path, destination_path)

            if not os.path.exists(labels_path):
                handle_CREMA_D(audio_directory)

            toggle_controls = [False, False, True, True]
            corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")


        case "IEMOCAP":
            pass



    extract_audio_files_from_video = toggle_controls[0]
    rename_files = toggle_controls[1]
    detect_broken_files = toggle_controls[2]
    extract_corrected_labels = toggle_controls[3]
    



    if extract_audio_files_from_video:
        print("Extracting audio files from video files...")
        if not os.path.exists(audio_directory):
            os.makedirs(audio_directory)
        if not os.listdir(audio_directory):
            extract_audio_from_video.extract_files(video_directory, audio_directory, num_files_to_process=50)
        else:
            print("Audio file folder not empty.")

        print("Done.")

    if rename_files:
        print("Renaming files...")
        switch_case(dataset, audio_directory)
        print("Done.")
    
    if detect_broken_files:
        print("Detecting broken audio files...")
        detect_broken_audio_files.process_files(audio_directory)
        print("Done.")
    
    if extract_corrected_labels:
        print("Extracting corrected labels...")
        match_labels.match_emotion_labels(labels_path,  corrected_labels_path, audio_directory, dataset)
        print("Done.")

if __name__ == "__main__":
    main()