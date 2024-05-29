from speechbrain.inference.interfaces import foreign_class

class AudioModel:

    def classify_audio_file(self, file):
        """Classify a file using the classifier

        Args:
            file (str): Path to the file to classify

        Returns:
            out_prob (np.array): Array of probabilities for each class
            score (float): Score of the predicted class
            index (int): Index of the predicted class
            text_lab (list): List with the name of the predicted class
        """
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
        out_prob, score, index, text_lab = classifier.classify_file(file)

        return out_prob, score, index, text_lab


