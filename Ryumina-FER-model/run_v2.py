# import argparse
import numpy as np
import pandas as pd
import os
import time
from scipy import stats
from functions import sequences
from functions import get_face_areas
from functions.get_models import load_weights_EE, load_weights_LSTM

import pickle
from select_video_subset import select_video_subset

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

# parser = argparse.ArgumentParser(description="run")

# parser.add_argument('--path_video', type=str, default='video/', help='Path to all videos')
# parser.add_argument('--path_save', type=str, default='report/', help='Path to save the report')
# parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
# parser.add_argument('--path_FE_model', type=str, default='models/EmoAffectnet/weights_0_66_37_wo_gl.h5',
                    # help='Path to a model for feature extraction')
# parser.add_argument('--path_LSTM_model', type=str, default='models/LSTM/CREMA-D_with_config.h5',
                    # help='Path to a model for emotion prediction')

# args = parser.parse_args()

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
    with open('face_areas.pkl', 'wb') as file:
        pickle.dump(dict_face_areas, file)
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
    step_size = 4
    window_size = 5
    seq_paths, seq_features = sequences.sequences(name_frames, features, win=window_size, step=step_size)
    print("Step size: ", step_size)
    print("Window size: ", window_size)
    print("Sampled frames ", list(range(0,len(name_frames)+1,step_size)))
    print("Number of steps/windows: ", len(seq_paths))
    pred = LSTM_model(np.stack(seq_features)).numpy()
    # all_pred = []
    # all_path = []
    # for id, c_p in enumerate(seq_paths):
    #     c_f = [str(i).zfill(6) for i in range(int(c_p[0]), int(c_p[-1])+1)]
    #     c_pr = [pred[id]]*len(c_f)
    #     all_pred.extend(c_pr)
    #     all_path.extend(c_f)    
    # m_f = [str(i).zfill(6) for i in range(int(all_path[-1])+1, total_frame+1)] 
    # m_p = [all_pred[-1]]*len(m_f)
    
    # df=pd.DataFrame(data=all_pred+m_p, columns=label_model)
    # df['frame'] = all_path+m_f
    # df = df[['frame']+ label_model]
    # df = sequences.df_group(df, label_model)
    
    # if not os.path.exists(args.path_save):
    #     print("Let's create a folder to save the report!")
    #     os.makedirs(args.path_save)
        
    # filename = os.path.basename(path)[:-4] + '.csv'
    # df.to_csv(os.path.join(args.path_save,filename), index=False)
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
    
    # print('Report saved in: ', os.path.join(args.path_save,filename))
    print('Predicted emotion: ', label_model[mode])
    print('True emotion: ', true_label)
    print('Lead time: {} s'.format(np.round(end_time, 2)))
    print()
    return label_model[mode], return_argmax

# def pred_all_video():
#     path_all_videos = os.listdir(args.path_video)
#     for id, cr_path in enumerate(path_all_videos):
#         print('{}/{}'.format(id+1, len(path_all_videos)))
#         pred_one_video(os.path.join(args.path_video,cr_path))
        
        
if __name__ == "__main__":
    # pred_all_video()
    # pred_one_video(r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST\1001_IEO_ANG_HI.mp4")
    # pred_one_video(r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST_MP4\1003_IEO_SAD_HI.mp4")
    emotion_dict = {"NEU": "Neutral", "HAP": "Happiness", "SAD": "Sadness", "SUR": "Surprise", "FEA": "Fear", "DIS": "Disgust", "ANG": "Anger"}
    root_path = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST\HI"

    backbone_model_path = 'models/EmoAffectnet/weights_0_66_37_wo_gl.h5'
    LSTM_model_path = 'models/LSTM/CREMA-D_with_config.h5'

    # pred_one_video(os.path.join(root_path, "1021_IEO_ANG_HI.mp4"),"Anger")
    video_files = select_video_subset(root_path,5)
    print("video_files: ", video_files)
    print()

    data = pd.DataFrame(columns=['filename', 'emotion', 'predicted', 'argmax'])
    data['filename'] = video_files
    data['emotion'] = [emotion_dict[video_file.split("_")[2]] for video_file in video_files]
    progress = 1
    for video_file in video_files:
        print(f"Processing video {progress}/{len(video_files)}")
        # Get emotion from substring in file name
        true_emotion = emotion_dict[video_file.split("_")[2]]
        predicted_emotion, predicted_argmax = pred_one_video(root_path, video_file, backbone_model_path, LSTM_model_path)
        data.loc[data['filename'] == video_file, 'predicted'] = predicted_emotion
        data.loc[data['filename'] == video_file, 'argmax'] = predicted_argmax
        progress += 1
    
    # save the DataFrame to a csv file
    data.to_csv('video_labels_predicted.csv', index=False)
    print(data.head())
    
    