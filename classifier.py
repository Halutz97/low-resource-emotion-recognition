import torch

class MyClassifier(torch.nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.fc = torch.nn.Linear(768, num_classes)  # Assuming `wav2vec2-base-960h` model

    def forward(self, x):
        # Pooling strategy, e.g., mean pooling
        x = x.mean(dim=1)  # Mean pooling over the sequence dimension
        x = self.fc(x)
        return x
