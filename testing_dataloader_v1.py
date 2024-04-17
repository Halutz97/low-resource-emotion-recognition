from torch.utils.data import DataLoader
from ser_dataloader import AudioDataset
import torch

max_length = 16000 * 10  # 10 seconds

def collate_fn(batch):
    """ Collate function to trim or pad audio samples to a fixed length """
    waveforms, sample_rates = zip(*batch)
    waveforms_padded = torch.stack([
        torch.nn.functional.pad(torch.tensor(wf)[:max_length], (0, max_length - len(wf[0])), "constant", 0)
        for wf in waveforms
    ])
    return waveforms_padded, torch.tensor(sample_rates)

def main():
    audio_dataset = AudioDataset("C:/MyDocs/DTU/MSc/Thesis/Data/MELD/MELD_fine_tune_v1_train_data")
    data_loader = DataLoader(audio_dataset, batch_size=4, collate_fn=collate_fn)
    print()
    print(f"Number of batches per epoch: {len(data_loader)}")
    print()

    i = 0
    for batch in data_loader:
        print(batch)
        # Dimensions of the batch
        print(batch[0].shape)
        print(batch[1].shape)
        print()
        

        i = i + 1
        if i >= 1:
            break
    

    print(data_loader.dataset)

    # Training loop here

if __name__ == "__main__":
    main()



  


