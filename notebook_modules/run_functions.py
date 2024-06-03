import os
import numpy as np
import time
from scipy import stats
import sequences
import get_face_areas
from get_models import load_weights_EE, load_weights_LSTM
# import pickle

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

def pred_one_video(directory, video_file, backbone_model_path, LSTM_model_path):
    # Initialize parameters
    conf_d = 0.7
    # backbone_model_path = 'models/EmoAffectnet/weights_0_66_37_wo_gl.h5'
    # LSTM_model_path = 'models/LSTM/CREMA-D_with_config.h5'

    path = os.path.join(directory, video_file)
    emotion_dict = {"NEU": "Neutral", "HAP": "Happiness", "SAD": "Sadness", "SUR": "Surprise", "FEA": "Fear", "DIS": "Disgust", "ANG": "Anger"}
    true_label = emotion_dict[video_file.split("_")[2]]

    start_time = time.time()
    label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    detect = get_face_areas.VideoCamera(path_video=path, conf=conf_d) # args.conf_d
    dict_face_areas, total_frame = detect.get_frame()
    # with open('face_areas.pkl', 'wb') as file:
        # pickle.dump(dict_face_areas, file)
    name_frames = list(dict_face_areas.keys())
    face_areas = list(dict_face_areas.values())
    print("Number of frames after sampling: ", len(name_frames))
    EE_model = load_weights_EE(backbone_model_path) # args.path_FE_model
    print("Backbone model: ", backbone_model_path)
    LSTM_model = load_weights_LSTM(LSTM_model_path) # args.path_LSTM_model
    print("LSTM model: ", LSTM_model_path)
    features = EE_model(np.stack((face_areas)))
    # Features are tensor of 512 elements for each frame
    print("Shape of features: ", features.shape)
    step_size = 5
    window_size = 10
    seq_paths, seq_features = sequences.sequences(name_frames, features, win=window_size, step=step_size)
    print("Step size: ", step_size)
    print("Window size: ", window_size)
    print("Sampled frames ", list(range(0,len(name_frames)+1,step_size)))
    print("Number of steps/windows: ", len(seq_paths))
    pred = LSTM_model(np.stack(seq_features)).numpy()
    end_time = time.time() - start_time

    print("Shape of pred: ", pred.shape)

    calculate_argmax = np.argmax(pred, axis=1)
    print("Shape of argmax: ", calculate_argmax.shape)
    print("argmax: ", calculate_argmax)
    # Convert calculate_argmax to a String
    return_argmax = np.array2string(calculate_argmax)

    mode_result = stats.mode(np.argmax(pred, axis=1))
    if mode_result.mode.shape == ():
        # Scalar scenario: mode is a single value scalar
        mode = mode_result.mode
    else:
        # Array scenario: mode is an array
        mode = mode_result.mode[0]
    
    print('Predicted emotion: ', label_model[mode])
    print('True emotion: ', true_label)
    print('Lead time: {} s'.format(np.round(end_time, 2)))
    print()
    return label_model[mode], return_argmax