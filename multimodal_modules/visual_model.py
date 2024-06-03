from ryumina_fer_model.functions import get_face_areas
from ryumina_fer_model.functions.get_models import load_weights_EE, load_weights_LSTM
from ryumina_fer_model.functions import sequences
import os
import time
import pickle
import numpy as np
from scipy import stats
import pandas as pd
from ryumina_fer_model.select_video_subset import select_video_subset

# define class
class VisualModel:
    
    def classify_video_file(self, file):
        """Classify a file using the classifier

        Args:
            file (str): Path to the file to classify

        Returns:
            out_prob (np.array): Array of probabilities for each class
            score (float): Score of the predicted class
            index (int): Index of the predicted class
            text_lab (list): List with the name of the predicted class
        """
        # classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
        # out_prob, score, index, text_lab = classifier.classify_file(file)

        # return out_prob, score, index, text_lab

        # Initialize parameters
        conf_d = 0.7
        backbone_model_path = 'ryumina_fer_model/models_fer/EmoAffectnet/weights_0_66_37_wo_gl.h5'
        LSTM_model_path = 'ryumina_fer_model/models_fer/LSTM/CREMA-D_with_config.h5'
        emotion_dict = {"NEU": "Neutral", "HAP": "Happiness", "SAD": "Sadness", "SUR": "Surprise", "FEA": "Fear", "DIS": "Disgust", "ANG": "Anger"}
        # true_label = emotion_dict[video_file.split("_")[2]]

        # start_time = time.time()
        label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
        detect = get_face_areas.VideoCamera(path_video=file, conf=conf_d) # args.conf_d
        dict_face_areas, total_frame = detect.get_frame()
        name_frames = list(dict_face_areas.keys())
        face_areas = list(dict_face_areas.values())
        # print("Number of frames after sampling: ", len(name_frames))
        EE_model = load_weights_EE(backbone_model_path) # args.path_FE_model
        # print("Backbone model: ", backbone_model_path)
        LSTM_model = load_weights_LSTM(LSTM_model_path) # args.path_LSTM_model
        # print("LSTM model: ", LSTM_model_path)
        features = EE_model(np.stack((face_areas)))
        # Features are tensor of 512 elements for each frame
        # print("Shape of features: ", features.shape)
        step_size = 5
        window_size = 10
        seq_paths, seq_features = sequences.sequences(name_frames, features, win=window_size, step=step_size)
        # print("Step size: ", step_size)
        # print("Window size: ", window_size)
        # print("Sampled frames ", list(range(0,len(name_frames)+1,step_size)))
        # print("Number of steps/windows: ", len(seq_paths))
        pred = LSTM_model(np.stack(seq_features)).numpy()

        # let's inspect the characteristics of the prediction
        # print("Pred shape")
        # print(pred.shape)
        # print("Pred")
        # print(pred)

        # Take the average of all len(seq_paths) predictions
        avg_pred = np.mean(pred, axis=0)
         # Let's inspect the characteristics of the average prediction
        # print("Avg pred shape")
        # print(avg_pred.shape)
        # print("Avg pred")
        # print(avg_pred)

        out_prob = avg_pred
        # Get the max of the average prediction, and the index of the max
        score = np.max(avg_pred)
        index = np.argmax(avg_pred)

        # Get the text label of the max
        text_lab = label_model[index]

        # print results
        # print("out_prob")
        # print(out_prob)
        # print("score")
        # print(score)
        # print("index")
        # print(index)
        # print("text_lab")
        # print(text_lab)

        return out_prob, score, index, text_lab

if __name__ == "__main__":
    root_path = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST\HI"
    video_file = select_video_subset(root_path,1)
    video_path = os.path.join(root_path, video_file[0])
    my_classifier = VisualModel()
    my_classifier.classify_video_file(video_path)


