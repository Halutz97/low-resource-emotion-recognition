import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import DataLoader

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')

