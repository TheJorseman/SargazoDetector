from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn import MSELoss
from torch.optim import Adam
import torch

from deeplabv3.dataset import get_dataloader_single_folder
from deeplabv3.model import get_model
from deeplabv3.train import train_model


def main():
    model = get_model()
    criterion = MSELoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=1e-4)

    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
    data_directory="dataset"
    exp_directory="experiment"
    data_directory = Path(data_directory)
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()
    BATCH_SIZE = 2
    EPOCHS = 100
    # Create the dataloader
    dataloaders = get_dataloader_single_folder(data_directory, batch_size=BATCH_SIZE)
    model = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=EPOCHS)
    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')

if __name__ == "__main__":
    main()