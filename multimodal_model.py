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
        data = pd.read_csv(os.path.join(directory,r"CREMA-D\labels_testing.csv"))
        directory = os.path.join(directory,r"CREMA-D\audio_testing")
        my_encoding_dict_dataset = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3}

    elif dataset == "CREMA-D-voted":
        data = pd.read_csv(os.path.join(directory,r"CREMA-D\labels_v_testing.csv"))
        directory = os.path.join(directory,r"CREMA-D\audio_v_testing")
        my_encoding_dict_dataset = {'N': 0, 'A': 1, 'H': 2, 'S': 3}

    files = []

    # Get a list of all files in the directory (audio files, change for video files)
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(file)

    return files, data, directory, my_encoding_dict_dataset

def get_crema_d_dataset(audio_directory, video_directory, labels_file, voted=True):
    # Define the dataset and the directory
    if voted:
        # data = pd.read_csv(os.path.join(directory, labels_file))
        # directory = os.path.join(directory,r"CREMA-D\audio_testing")
        my_encoding_dict_dataset = {'N': 0, 'A': 1, 'H': 2, 'S': 3}
    else:
        # data = pd.read_csv(os.path.join(directory,r"CREMA-D\labels_v_testing.csv"))
        # directory = os.path.join(directory,r"CREMA-D\audio_v_testing")
        my_encoding_dict_dataset = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3}

    audio_files = []
    video_files = []

    # Get a list of all files in the directory (audio files, change for video files)
    for file in os.listdir(audio_directory):
        if file.endswith('.wav'):
            audio_files.append(file)
    
    for file in os.listdir(video_directory):
        if file.endswith('.mp4'):
            video_files.append(file)

    return audio_files, video_files, my_encoding_dict_dataset


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
def separate_audio_video(file):
    # Separate the audio and video files

    audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\AudioWav"
    video_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\VideoFlash"

    # Lood for audio file in AudioWav folder
    audio_files = (os.path.join(audio_directory, file)+'.wav')

    # Look for video file in VideoFlash folder
    video_files = (os.path.join(video_directory, file)+'.flv')

    return audio_files, video_files


# Call both models to classify the audio and the video
def save_results(results, name):
    # Save the results for debugging as a csv file
    results_df = pd.DataFrame(results)
    # Get current directory
    current_dir = os.getcwd()
    results_save_path = os.path.join(current_dir, 'multimodal_results')
    results_df.to_csv(os.path.join(results_save_path, f'{name}.csv'), index=False)
    # results_df.to_csv(f'{name}.csv', index=False)

def select_final_label(final_probabilities):
    # Select the final label based on the combined probabilities
    final_label = np.argmax(final_probabilities)
    return final_label

def combine_probabilities(audio_prob, video_prob, audio_weight=0.4, video_weight=0.6):
    # Implement your logic to combine probabilities from both models
    print()
    print()
    print("-----------------Combining Probabilities-----------------")
    print()
    print("Audio Probabilities: ", audio_prob)
    audio_weighted = audio_prob*audio_weight
    print("Audio Prob. weighted: ", audio_weighted)
    print()
    print("Video Probabilities: ", video_prob)
    video_weighted = video_prob*video_weight
    print("Video Prob. weighted: ", video_weighted)
    combined_prob = audio_weighted + video_weighted
    print()
    print("Combined Probabilities: ", combined_prob)
    return combined_prob

def process_audio(audio_file):
    audio_classifier = AudioModel()
    out_prob, score, index, text_lab = audio_classifier.classify_audio_file(audio_file)
    out_prob = out_prob.numpy()
    # Add three zeros to match the dimensions of the visual model
    out_prob = np.append(out_prob, [0, 0, 0]).reshape(1,7)
    return out_prob, score, index, text_lab

def process_video(video_file, backbone_model_path, LSTM_model_path):
    video_classifier = VisualModel()
    out_prob, score, index, text_lab = video_classifier.classify_video_file(video_file, backbone_model_path, LSTM_model_path)
    out_prob = reorder_video_probabilities(out_prob)
    return out_prob, score, index, text_lab

def reorder_video_probabilities(video_probabilities):
    # Reorder the video_probabilities to match the audio_probabilities
    # Order = ['neu', 'ang', 'hap', 'sad']
    # video_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    video_model_dict = {'neu' : 0, 'hap' : 1, 'sad' : 2, 'sur' : 3, 'fea' : 4, 'dis' : 5, 'ang' : 6}

    new_probabilities = np.array([video_probabilities[video_model_dict['neu']],
                                  video_probabilities[video_model_dict['ang']],
                                  video_probabilities[video_model_dict['hap']],
                                  video_probabilities[video_model_dict['sad']],
                                  video_probabilities[video_model_dict['sur']],
                                  video_probabilities[video_model_dict['fea']],
                                  video_probabilities[video_model_dict['dis']]])
    
    new_probabilities = new_probabilities.reshape(1,7)
    # compare old and new probabilities
    print("Old Probabilities: ")
    print(video_probabilities)
    print("New Probabilities: ")
    print(new_probabilities)
    return new_probabilities



# Call both models to classify the audio and the video
if __name__ == '__main__':

    # label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    # my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3} # Change this to include (or not) the extra classes of the visual model
    my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3, 'sur': 4, 'fea': 5, 'dis': 6}
    label_names = ['neu', 'ang', 'hap', 'sad'] # Same as above

    label_model_decoder = {0: 'Neutral', 1: 'Anger', 2: 'Happiness', 3: 'Sadness', 4: 'Surprise', 5: 'Fear', 6: 'Disgust'}

    backbone_model_path = 'ryumina_fer_model/models_fer/EmoAffectnet/weights_0_66_37_wo_gl.h5'
    LSTM_model_path = 'ryumina_fer_model/models_fer/LSTM/CREMA-D_with_config.h5'

    # dataset = "CREMA-D" # Change this to the dataset you are using
    # directory = "path_to_datasets" # Change this to the directory where you save the datasets
    # file = '1001_DFA_ANG_XX' # Change this to the file you want to classify
    # file = '1001_IEO_SAD_MD'

    audio_folder = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\AudioWAV_testing"
    # audio_folder = r"C:\_HomeDocs\Ari\DTU\00-MSc\Thesis\Data\AudioWAV_testing"
    video_folder = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\VideoMP4_testing"
    # video_folder = r"C:\_HomeDocs\Ari\DTU\00-MSc\Thesis\Data\VideoMP4_testing"


    # files, data, directory, my_encoding_dict_dataset = get_dataset(dataset, directory)
    # labels = pd.read_csv(r"C:\_HomeDocs\Ari\DTU\00-MSc\Thesis\Data\voted_combined_labels_corrected_testing.csv")
    # filenames = labels['filename'].tolist()
    # predictions_df = labels.copy()
    # predictions_df['audio_prob'] = None
    # predictions_df['video_prob'] = None

    labels_data = pd.read_csv(r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\voted_combined_labels_corrected_testing.csv")
    filenames = labels_data['filename'].tolist()
    emotions_list = labels_data['Emotion'].tolist()
    labels_list = labels_data['Label'].tolist()
    total_data_length = len(filenames)

    predictions_df = pd.DataFrame(columns=['filename', 'Emotion', 'Label', 'audio_prob', 'video_prob', 'checkpoint'])
    predictions_df['filename'] = filenames
    predictions_df['Emotion'] = emotions_list
    predictions_df['Label'] = labels_list

    # files, data, directory, my_encoding_dict_dataset = get_single_file(file)
    # label_keys, true_labels = get_label_keys(data, my_encoding_dict_dataset)
    # audio_input, video_input = separate_audio_video(file)
    
    # audio_probs_list = []
    # audio_probs_list = []
    with mp.Pool(2) as pool:
        debug_counter = 0
        for file in filenames:
            if debug_counter>=2:
                break
            audio_input = os.path.join(audio_folder, file + '.wav')
            video_input = os.path.join(video_folder, file + '.mp4')
            # audio_input = file
            print(f'Audio Input: {audio_input}', f'Video Input: {video_input}')
            # with mp.Pool(2) as pool:
            audio_result = pool.apply_async(process_audio, (audio_input,))
            video_result = pool.apply_async(process_video, (video_input, backbone_model_path, LSTM_model_path))
            
            audio_probabilities, _, _, _ = audio_result.get()
            video_probabilities, _, _, _ = video_result.get()

            # Save separate results for debugging
            save_results(audio_probabilities, 'audio_results')
            save_results(video_probabilities, 'video_results')

            # Let's investigate the results
            # First audio_probabilities: Type, shape, values
            # print("Audio Probabilities")
            # print(type(audio_probabilities))
            # print(audio_probabilities.shape)
            # print(audio_probabilities)

            # Second video_probabilities: Type, shape, values
            # print("Video Probabilities")
            # print(type(video_probabilities))
            # print(video_probabilities.shape)
            # print(video_probabilities)

            # # Convert audio_probabilities (tensor) to a numpy array of dimensions (1,4)
            # audio_probabilities = audio_probabilities.numpy()
            # # Add three zeros to match the dimensions of the visual model
            # audio_probabilities = np.append(audio_probabilities, [0, 0, 0]).reshape(1,7)
            # print(audio_probabilities.shape)
            # print(audio_probabilities)

            # Convert video_probabilities to a numpy array of dimensions (1,7)
            # video_probabilities = video_probabilities.reshape(1,7)
            # video_probabilities = reorder_video_probabilities(video_probabilities)
            print("We have now reordered the video probabilities")
            print(video_probabilities.shape)
            print(video_probabilities)

            # Combine results and determine final label
            final_probabilities = combine_probabilities(audio_probabilities, video_probabilities, audio_weight=0.5, video_weight=0.5)
            final_label = select_final_label(final_probabilities)
            final_label_name = label_model_decoder[final_label]
            
            print(f'Final Label: {final_label}', f'Final Probabilities: {final_probabilities}')
            print(f'Final Label Name: {final_label_name}')
            # print(f'True Label: {true_labels}')
            # audio_probs_list.append
            # video_probs_list       
            # predictions_df.loc[predictions_df['filename'] == file, 'audio_prob'] = audio_probabilities
            # predictions_df.loc[predictions_df['filename'] == file, 'video_prob'] = video_probabilities
            predictions_df.at[predictions_df[predictions_df['filename'] == file].index[0], 'audio_prob'] = audio_probabilities
            predictions_df.at[predictions_df[predictions_df['filename'] == file].index[0], 'video_prob'] = video_probabilities
     
            debug_counter += 1
    
    print(predictions_df.head())

    print()
    print()
    print("------------------------------------------------------------------------------")
    # save predictions_df to a csv file
    # get cwd
    predictions_df.to_csv(os.path.join(os.getcwd(), 'multimodal_results', 'multimodal_predictions.csv'), index=False)
    print("Now inspect that the probabilites have been stored correctly in the df")
    print()
    list_of_probabilities = predictions_df['audio_prob']
    for prob in list_of_probabilities:
        if prob is not None:
            print("Probabilities:")
            print(prob)
            print(f"Type: {type(prob)}")
            print(f"Shape: {prob.shape}")
            print()
            print(prob[0][0])
            print(f"Type: {type(prob[0])}")