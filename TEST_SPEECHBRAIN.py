from audio_model import classify_audio_file

if __name__ == "__main__":
    audio_path = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\AudioWAV\1001_DFA_ANG_XX.wav"
    result = classify_audio_file(audio_path)

    # let's inspect the characteristics of result
    print("Result shape")
    print(result.shape)
    print("Result type")
    print(type(result))
    print("Result")
    print(result)
    
