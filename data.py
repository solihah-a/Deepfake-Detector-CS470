from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import DATASET_NAME, IMAGE_COLUMN, LABEL_COLUMN, IMAGE_SIZE, BATCH_SIZE


trainTransforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

evalTransforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


class DeepfakeDataset(Dataset):
    def __init__(self, hfSplit, transform=None):
        self.hfSplit = hfSplit
        self.transform = transform

    def __len__(self):
        return len(self.hfSplit)

    def __getitem__(self, idx):
        item = self.hfSplit[idx]

        img = item[IMAGE_COLUMN].convert("RGB")
        label = int(item[LABEL_COLUMN])

        if self.transform:
            img = self.transform(img)

        return img, label


def getLoaders():
    ds = load_dataset(DATASET_NAME)

    trainSet = DeepfakeDataset(ds["train"], transform=trainTransforms)
    valSet = DeepfakeDataset(ds["validation"], transform=evalTransforms)
    testSet = DeepfakeDataset(ds["test"], transform=evalTransforms)

    trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valLoader = DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return trainLoader, valLoader, testLoader