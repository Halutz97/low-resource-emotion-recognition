import numpy as np
import pandas as pd
import os
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
from audio_model import AudioModel
from visual_model import VisualModel
# from visual_model import classify_visual_file


def get_dataset(dataset, directory):
    # Define the dataset and the directory
    if dataset == "CREMA-D":
        data = pd.read_csv(os.path.join(directory,"CREMA-D\labels_testing.csv"))
        directory = os.path.join(directory,"CREMA-D\audio_testing")
        my_encoding_dict_dataset = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3}

    elif dataset == "CREMA-D-voted":
        data = pd.read_csv(os.path.join(directory,"CREMA-D\labels_v_testing.csv"))
        directory = os.path.join(directory,"CREMA-D\audio_v_testing")
        my_encoding_dict_dataset = {'N': 0, 'A': 1, 'H': 2, 'S': 3}

    files = []

    # Get a list of all files in the directory (audio files, change for video files)
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(file)

    return files, data, directory, my_encoding_dict_dataset


# Debugging function to check only one file
def get_single_file(file):
    # Get a single file for debugging
    data = pd.read_csv(r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\labels_testing.csv")
    directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio_testing"
    for row in data.iterrows():
        if row[1]['File'] == file:
            data = row[1]
            break
    my_encoding_dict_dataset = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3}
    return file, data, directory, my_encoding_dict_dataset

def get_label_keys(data, my_encoding_dict_dataset):
    # Get the label keys from the dataset
    true_labels_multi = data['Emotion'] # Change this to the true labels of the multimodal model
    true_labels_audio = data['Emotion'] # Change this to the true labels of the audio model
    true_labels_visual = data['Emotion'] # Change this to the true labels of the visual model
    label_keys_multi = true_labels_multi.map(my_encoding_dict_dataset).values
    label_keys_audio = true_labels_audio.map(my_encoding_dict_dataset).values
    label_keys_visual = true_labels_visual.map(my_encoding_dict_dataset).values

    return label_keys_multi, true_labels_multi




# Separete the audio and the video of the file
def separate_audio_video(files):
    # Separate the audio and video files
    audio_files = []
    video_files = []
    audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\AudioWav"
    video_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\VideoFlash"
    for file in files:
        # Lood for audio file in AudioWav folder
        audio_files.append(os.path.join(audio_directory, file)+'.wav')
        # Look for video file in VideoFlash folder
        video_files.append(os.path.join(video_directory, file)+'.flv')
    return audio_files, video_files


# Call both models to classify the audio and the video
def save_results(results, name):
    # Save the results for debugging as a csv file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{name}.csv', index=False)

def select_final_label(final_probabilities):
    # Select the final label based on the combined probabilities
    final_label = np.argmax(final_probabilities)
    return final_label

def combine_probabilities(audio_prob, video_prob):
    # Implement your logic to combine probabilities from both models
    combined_prob = (audio_prob + video_prob) / 2  # Example logic
    return combined_prob

def process_audio(audio_file):
    audio_classifier = AudioModel()
    out_prob, score, index, text_lab = audio_classifier.classify_audio_file(audio_file)
    return out_prob, score, index, text_lab

def process_video(video_file):
    video_classifier = VisualModel()
    out_prob, score, index, text_lab = video_classifier.classify_video_file(video_file)
    return out_prob, score, index, text_lab


# Call both models to classify the audio and the video
if __name__ == '__main__':

    my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3} # Change this to include (or not) the extra classes of the visual model
    label_names = ['neu', 'ang', 'hap', 'sad'] # Same as above

    dataset = "CREMA-D" # Change this to the dataset you are using
    directory = "path_to_datasets" # Change this to the directory where you save the datasets
    file = '1001_DFA_ANG_XX' # Change this to the file you want to classify


    # files, data, directory, my_encoding_dict_dataset = get_dataset(dataset, directory)
    # files, data, directory, my_encoding_dict_dataset = get_single_file(file)
    # label_keys, true_labels = get_label_keys(data, my_encoding_dict_dataset)
    audio_input, video_input = separate_audio_video(file)
    audio_input = file
    
    with mp.Pool(2) as pool:
        audio_result = pool.apply_async(process_audio, (audio_input,))
        video_result = pool.apply_async(process_video, (video_input,))
        
        audio_probabilities, _, _, _ = audio_result.get()
        video_probabilities, _, _, _ = video_result.get()

        # Save separate results for debugging
        save_results(audio_probabilities, 'audio_results')
        save_results(video_probabilities, 'video_results')
        
        # Combine results and determine final label
        final_probabilities = combine_probabilities(audio_probabilities, video_probabilities)
        final_label = select_final_label(final_probabilities)
        
        print(f'Final Label: {final_label}', f'Final Probabilities: {final_probabilities}')
        # print(f'True Label: {true_labels}')



