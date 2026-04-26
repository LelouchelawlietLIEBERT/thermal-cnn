import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_COLS = [
    "LBL_SinkMarks",
    "LBL_SprueCircle",
    "LBL_Underfilled",
    "LBL_OldGranulate",
    "LBL_StreaksLevel1",
    "LBL_StreaksLevel2",
    "LBL_StreaksLevel3",
    "LBL_NOK",
]


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class ThermalDefectDataset(Dataset):
    def __init__(self, parquet_path: str, image_dir: str, train: bool = True):
        df = pd.read_parquet(parquet_path)

        # drop rows with missing labels or filenames
        df = df.dropna(subset=["IR_Image1Name"] + LABEL_COLS)

        # map csv → png
        df["image_path"] = df["IR_Image1Name"].str.replace(".csv", ".png", regex=False)
        df["image_path"] = df["image_path"].apply(lambda f: os.path.join(image_dir, f))

        # keep only existing images
        df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

        self.image_paths = df["image_path"].tolist()
        self.labels = df[LABEL_COLS].values.astype("float32")
        self.transform = get_transforms(train)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]