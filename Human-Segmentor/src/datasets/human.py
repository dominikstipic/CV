from PIL import Image
from pathlib import Path

from src.sat_dataset import Sat_Dataset

class HumanDataset(Sat_Dataset):
    def __init__(self, root, transforms=None):
        super().__init__(root=root, transforms=transforms, split=None)
        self.path = Path(root)
        paths = [x for x in list(self.path.iterdir())]
        self.data   = [p for p in paths if p.stem.startswith("img-")]
        self.labels = [p for p in paths if p.stem.startswith("mask-")]
        self.data = sorted(self.data, key=lambda p: p.stem.split("-")[1])
        self.labels = sorted(self.labels, key=lambda p: p.stem.split("-")[1])

    def get_paths(self):
        paths = [str(p) for p in self.data]
        paths = sorted(paths)
        return paths

    def get_examples(self):
        return self.data

    def get(self, idx):
        img_name, label_name = self.data[idx], self.labels[idx]
        img, mask = Image.open(img_name), Image.open(label_name)
        if len(img.size) == 2:
          img = img.convert("RGB")
        return img, mask 

    def copy(self):
        return HumanDataset(self.path, self.transforms)