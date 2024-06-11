import torch
import os
import torchaudio
from speechbrain.inference.interfaces import foreign_class, Pretrained
from hyperpyyaml import load_hyperpyyaml


class AudioClassifier:

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
    
class AudioRegressor(Pretrained):
    HPARAMS_NEEDED = ["modules", "pretrainer"]
    MODULES_NEEDED = ["encoder", "pooling", "regressor"]

    def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        super().__init__(modules, hparams, run_opts, freeze_params)
        
    def forward(self, wavs, wav_lens):
        """Forward pass for inference."""
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        print(f"Shape of wav before encoder: {wavs.shape}")
        print(f"Shape of wav_lens before encoder: {wav_lens.shape}")
        features = self.mods.encoder(wavs, wav_lens)
        pooled_features = self.hparams.avg_pool(features, wav_lens)  # Ensure pooling is included
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        predictions = self.mods.regressor(pooled_features)
        return predictions

    def predict_file(self, wav_path):
        """Predict emotion from an audio file."""
        wavs = self.load_audio(wav_path)
        print(f"Original wavs shape: {wavs.shape}")
        if wavs.dim() == 3 and wavs.shape[0] == 1:
            wavs = wavs.squeeze(0)  # Remove batch dimension if present
        print(f"Shape after squeeze(0): {wavs.shape}")
        if wavs.dim() == 2 and wavs.shape[0] == 2:
            wavs = wavs.mean(dim=0)  # Average the channels to get a single channel
        print(f"Shape after mean(dim=0): {wavs.shape}")
        if wavs.dim() == 1:
            wavs = wavs.unsqueeze(0)  # Add batch dimension
        print(f"Shape after unsqueeze(0): {wavs.shape}")
        wav_lens = torch.tensor([wavs.size(1)], dtype=torch.float32) / wavs.size(1)  # Normalize the length
        print(f"Shape of wav_lens: {wav_lens.shape}")
        print(f"Shape of wav before forward pass: {wavs.shape}")
        predictions = self.forward(wavs, wav_lens)
        return predictions.squeeze().cpu().numpy()

    @staticmethod
    def load_audio(wav_path, sample_rate=16000):
        import torchaudio
        sig, fs = torchaudio.load(wav_path)
        if fs != sample_rate:
            sig = torchaudio.transforms.Resample(fs, sample_rate)(sig)
        return sig
    
class AudioMultiobjetive(Pretrained):
    HPARAMS_NEEDED = ["modules", "pretrainer"]
    MODULES_NEEDED = ["encoder", "pooling", "regressor", "classifier"]

    def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        super().__init__(modules, hparams, run_opts, freeze_params)
        
    def forward(self, wavs):
        """Forward pass for inference."""
        wavs, wav_lens = wavs.to(self.device), torch.tensor([1.0]).to(self.device)
        features = self.mods.encoder(wavs, wav_lens)
        pooled_features = self.hparams.avg_pool(features, wav_lens)  # Ensure pooling is included
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        regressor_predictions = self.mods.regressor(pooled_features)
        classifier_predictions = self.mods.classifier(pooled_features)
        return regressor_predictions, classifier_predictions

    def predict_file(self, wav_path):
        """Predict emotion from an audio file."""
        wavs = self.load_audio(wav_path)
        wavs = wavs.squeeze(0) # Remove batch dimension if present
        wavs = torch.tensor(wavs).unsqueeze(0)
        if wavs.dim() == 4:
            wavs = wavs.squeeze(1)  # Ensure the input is [batch_size, channels, length]
        regression_predictions, classifier_predictions = self.forward(wavs)
        classifier_predictions = torch.log_softmax(classifier_predictions, dim=-1)
        return regression_predictions.squeeze().cpu().numpy(), classifier_predictions.squeeze().cpu().numpy()

    @staticmethod
    def load_audio(wav_path, sample_rate=16000):
        """Load an audio file and resample if needed."""
        print(wav_path)
        sig, fs = torchaudio.load(wav_path)
        if fs != sample_rate:
            sig = torchaudio.transforms.Resample(fs, sample_rate)(sig)
        return sig
    
def load_checkpoint_with_renamed_keys(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    renamed_checkpoint = {}
    
    for key, value in checkpoint.items():
        new_key = key.replace('0.w.', 'w.')  # Renaming logic
        renamed_checkpoint[new_key] = value
    
    model.load_state_dict(renamed_checkpoint, strict=False)

    missing_keys, unexpected_keys = model.load_state_dict(renamed_checkpoint, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return model


# Custom function to load the model from local files
def load_regression_model(model_dir, hparams_file):
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Add device to hparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams["device"] = device

    modules = hparams["modules"]
    for module in modules.values():
        if module is not None:
            module.to(device)

    pretrainer = hparams.get("pretrainer", None)
    if pretrainer is not None:
        pretrainer.set_collect_in(model_dir)
        pretrainer.load_collected()

    # Load the regressor with renamed keys
    model = AudioRegressor(modules, hparams)
    load_checkpoint_with_renamed_keys(os.path.join(model_dir, 'model.ckpt'), model.mods.regressor, device)
    
    return model

def load_multiobjective_model(model_dir, hparams_file):
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams["device"] = device

    modules = hparams["modules"]

    # Instantiate the model
    model = AudioMultiobjetive(modules, hparams)

    # Load checkpoint of the heads
    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
    checkpoint_heads = torch.load(checkpoint_path, map_location=device)
    
    # Prepare a new state dictionary for the regressor and classifier
    new_state_dict = {}
    for key, param in checkpoint_heads.items():
        if key.startswith('0.'):  # Weights for the regressor
            new_key = key.replace('0.', '')  # Remove the '0.' prefix
            new_state_dict['mods.regressor.' + new_key] = param
        elif key.startswith('1.'):  # Weights for the classifier
            new_key = key.replace('1.', '')  # Remove the '1.' prefix
            new_state_dict['mods.classifier.' + new_key] = param

    # Load the new state dict into the model
    model.load_state_dict(new_state_dict, strict=False)

    # Load checkpoint of the encoder
    checkpoint_path = os.path.join(model_dir, 'wav2vec2.ckpt')
    checkpoint_encoder = torch.load(checkpoint_path, map_location=device)

    # Load the encoder state dict
    model.mods.encoder.load_state_dict(checkpoint_encoder, strict=False)

    missing_keys, unexpected_keys = model.mods.encoder.load_state_dict(checkpoint_encoder, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print()
    for name, param in model.named_parameters():
        print(name)


    return model


