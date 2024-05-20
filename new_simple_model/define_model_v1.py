import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch import nn

# Load pre-trained Wav2Vec 2.0 model
pretrained_model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
wav2vec2_model = Wav2Vec2Model.from_pretrained(pretrained_model_name)

# Define the number of emotion classes
num_classes = 7

# Create a new model class that adds a linear layer for emotion classification
class Wav2Vec2ForEmotionRecognition(nn.Module):
    def __init__(self, base_model, num_classes):
        super(Wav2Vec2ForEmotionRecognition, self).__init__()
        self.wav2vec2 = base_model
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)
        
        # Initialize weights of the classifier
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, input_values):
        # Get the output from Wav2Vec 2.0 model
        outputs = self.wav2vec2(input_values)
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Pooling: Mean pooling over the time dimension
        pooled_output = hidden_states.mean(dim=1)
        
        # Classify pooled output
        logits = self.classifier(pooled_output)
        
        return logits

# Instantiate the new model
model = Wav2Vec2ForEmotionRecognition(wav2vec2_model, num_classes)

# Print model architecture
print(model)
