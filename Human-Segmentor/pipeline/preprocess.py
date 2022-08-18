from torch.utils import data
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
import numpy as np
from src.utils.hashes import get_digest

import src.utils.pipeline_repository as pipeline_repository
from src.utils.common import h_concatenate_images

def scale_mask(mask: Image):
    return Image.fromarray(np.array(mask, dtype=np.uint8)*255)

def save_alignment(img, mask, in_art_dir, in_art_name):
    artifact_path = pipeline_repository.get_path(in_art_dir)
    mask_scaled = scale_mask(mask)
    subplot = h_concatenate_images(img, mask_scaled)
    pipeline_repository.push_images(artifact_path, [subplot], [in_art_name])

def save_dataset_hash(dataset: Dataset, out_path: Path):
    paths = dataset.get_paths()
    paths_str = " ".join(paths)
    digest = get_digest(paths_str)
    result = {"hash": digest}
    pipeline_repository.push_json(out_path, "hash.json", result)


def process(dataset: Dataset, format: str, output_dir: Path, in_alignment: bool, out_alignment: bool) -> None:
    path = pipeline_repository.get_path("preprocess/artifacts")
    save_dataset_hash(dataset, path)
    img, mask = dataset[0]
    with tqdm(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            if in_alignment:
                x, y = dataset.get(i)
                x, y = x.resize((500, 500)), y.resize((500, 500))
                in_art_name = Path(dataset.get_paths()[i]).stem
                in_align_dir = "preprocess/artifacts/in_alignment"
                save_alignment(x, y, in_align_dir, in_art_name)
            img_name  = dataset.data[i].stem
            mask_name = dataset.labels[i].stem
            img, mask = dataset[i]
            image_path = Path(f"{output_dir}/{img_name}.{format}")
            mask_path  = Path(f"{output_dir}/{mask_name}.{format}")
            pipeline_repository.push_images(output_dir, [img, mask], [image_path.name, mask_path.name])
            if out_alignment:
                _,idx = img_name.split("-")
                ext = dataset.data[i].name.split(".")[-1]
                out_art_name = f"{idx}.{ext}"
                out_align_dir = "preprocess/artifacts/out_alignment"
                save_alignment(img, mask, out_align_dir, out_art_name)
            pbar.update()
    
    
                


