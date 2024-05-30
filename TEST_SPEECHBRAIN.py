from audio_model import AudioModel
import numpy as np

if __name__ == "__main__":
    audio_path = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\AudioWAV\1001_DFA_ANG_XX.wav"
    my_classifier = AudioModel()
    out_prob, score, index, text_lab = my_classifier.classify_audio_file(audio_path)
    # Results are tensors, convert to numpy arrays
    out_prob = np.array(out_prob)
    score = np.array(score)
    index = np.array(index)
    text_lab = np.array(text_lab)
    
    # Inspect results now: Type, shape, and values
    print(f"out_prob type: {type(out_prob)}")
    print(f"out_prob shape: {out_prob.shape}")
    print(f"out_prob values: {out_prob}")
    print(f"score type: {type(score)}")
    print(f"score shape: {score.shape}")
    print(f"score values: {score}")
    print(f"index type: {type(index)}")
    print(f"index shape: {index.shape}")
    print(f"index values: {index}")
    print(f"text_lab type: {type(text_lab)}")
    print(f"text_lab shape: {text_lab.shape}")
    print(f"text_lab values: {text_lab}")
