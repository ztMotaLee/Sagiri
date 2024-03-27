import numpy as np
from PIL import Image
import time
import torch.utils.data as data
from typing import Dict, Union, Sequence
import cv2
import random
from utils.file import load_file_list
from utils.image import center_crop_arr, augment
from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)
def apply_jpeg_compression(img, quality=85):
    """ Apply JPEG compression to the image. """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, img_encoded = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    return img

def apply_gaussian_blur(img, kernel_size=3):
    """ Apply Gaussian blur to the image. """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def load_file_list(file_list: str):
    with open(file_list, 'r') as file:
        paths = file.readlines()
    return [line.strip().split() for line in paths]
def load_captions(captions_file: str):
    captions = {}
    with open(captions_file, 'r') as file:
        for line in file:
            path, caption = line.strip().split(': ')
            captions[path] = caption
    return captions
def random_crop_pair(lq_pil, gt_pil, crop_size):
    lq_img = np.array(lq_pil)
    gt_img = np.array(gt_pil)

    h, w, _ = lq_img.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)

    lq_crop = lq_img[top:top + crop_size, left:left + crop_size]
    gt_crop = gt_img[top:top + crop_size, left:left + crop_size]

    return lq_crop, gt_crop


def augment_pair(lq_img, gt_img, hflip=True, rotation=False):
    if hflip and random.random() > 0.5:
        lq_img = np.flip(lq_img, axis=1)
        gt_img = np.flip(gt_img, axis=1)

    # Add more augmentations as needed

    return lq_img, gt_img
class DarkEnhanceDataset(data.Dataset):
    
    def __init__(self, file_list: str, captions_file: str, out_size: int, crop_type: str, use_hflip: bool) -> "DarkEnhanceDataset":
        super(DarkEnhanceDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.captions = load_captions(captions_file)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # self.load_and_process_image = load_and_process_image
    def load_and_process_image(self, path: str) -> Image.Image:
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {path}"

        return pil_img

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        # Each item in self.paths is a pair [lq_path, gt_path]
        lq_path, gt_path = self.paths[index]

        lq_img = self.load_and_process_image(lq_path)
        gt_img = self.load_and_process_image(gt_path)

        caption = self.captions.get(gt_path, '')
        # print(gt_path)
        # print(caption)
        lq_img, gt_img = self.apply_augmentations(lq_img, gt_img)
        gt_img = gt_img * 2 - 1
        return {'hint': lq_img, 'jpg': gt_img, 'txt':caption}

    def apply_augmentations(self, lq_img, gt_img):
        if self.crop_type == "center":
            lq_img = center_crop_arr(lq_img, self.out_size)
            gt_img = center_crop_arr(gt_img, self.out_size)
        elif self.crop_type == "random":
            # Ensure the same random crop is applied to both
            # Implement this function to apply the same crop
            lq_img, gt_img = random_crop_pair(lq_img, gt_img, self.out_size)

        # Convert to float and scale to [0, 1]
        lq_img = (lq_img / 255.0).astype(np.float32)
        gt_img = (gt_img / 255.0).astype(np.float32)

        # Apply the same horizontal flip to both
        lq_img, gt_img = augment_pair(lq_img, gt_img, hflip=self.use_hflip, rotation=False)

        # Apply JPEG compression and Gaussian blur to lq_img
        # lq_img = apply_jpeg_compression((lq_img * 255).astype(np.uint8))
        # lq_img = apply_gaussian_blur(lq_img)
        # lq_img = (lq_img / 255.0).astype(np.float32)
        return lq_img, gt_img

    def __len__(self) -> int:
        return len(self.paths)
