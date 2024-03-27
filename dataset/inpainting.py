import numpy as np
from PIL import Image, ImageFilter
import time
import torch.utils.data as data
from typing import Dict, Union, Sequence
import cv2
import random
import deeplake
import sys
from utils.file import load_file_list
from utils.image import center_crop_arr_bP, augment
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


def augment_pair(lq_img, gt_img, binary_mask, hflip=True, rotation=False):
    if hflip and random.random() > 0.5:
        lq_img = np.flip(lq_img, axis=1)
        gt_img = np.flip(gt_img, axis=1)
        binary_mask = np.flip(binary_mask, axis=1)
    # Add more augmentations as needed

    return lq_img, gt_img, binary_mask
class DarkEnhanceDataset(data.Dataset):
    
    def __init__(self, captions_file: str, out_size: int, crop_type: str, use_hflip: bool):
        super(DarkEnhanceDataset, self).__init__()
        self.ds = deeplake.load("hub://activeloop/places205") ###For inpainting pretraining.
        self.captions = None
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        self.image_indices = self._get_image_indices()
        # self.load_and_process_image = load_and_process_image
    def load_and_process_image(self, url: str) -> Image.Image:
        response = requests.get(url)
        pil_img = Image.open(BytesIO(response.content)).convert("RGB")
        return pil_img
      
  ##Check in list
    # def _get_image_indices(self):
    #     all_indices = list(range(len(self.ds)))
    #     print(len(self.ds))
    #     selected_indices = random.sample(all_indices, 250000)

    #     image_indices = []
    #     for i in selected_indices:
    #         # print(i)
    #         item = self.ds[i]
    #         if 'images' in item and item['images'].ndim == 3 and item['images'].shape[2] == 3:
    #             image_indices.append(i)

    #     with open('image_indices.txt', 'w') as f:
    #         for index in image_indices:
    #             f.write(f"{index}\n")

    #     return image_indices

###No check in list in current version.
    def _get_image_indices(self):
      all_indices = list(range(len(self.ds)))
      selected_indices = random.sample(all_indices, 250000)
      return selected_indices
    ##Random Mask.
    def apply_mask(self, img: Image.Image) -> (Image.Image, Image.Image):
          state = random.getstate()
          random.seed(time.time())
          width, height = img.size

          img_array = np.array(img)

          mask = np.zeros((height, width), dtype=np.uint8)

          num_shapes = random.randint(2, 5)
          for _ in range(num_shapes):
              x1, y1 = random.randint(0, width), random.randint(0, height)
              x2, y2 = random.randint(0, width), random.randint(0, height)
              thickness = random.randint(5, 15)
              cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

          kernel_size = random.randint(2, 4)
          kernel = np.ones((kernel_size, kernel_size), np.uint8)
          mask = cv2.dilate(mask, kernel, iterations=random.randint(2, 4))

          blurred_mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=0, sigmaY=0)

          binary_mask = (blurred_mask > 0).astype(np.uint8)

          alpha_mask = blurred_mask.astype(float) / 255

          degraded_img_array = cv2.GaussianBlur(img_array, (0, 0), sigmaX=20, sigmaY=20, borderType=cv2.BORDER_DEFAULT)

          img_array = (1 - alpha_mask)[:, :, None] * img_array + alpha_mask[:, :, None] * degraded_img_array

          random.setstate(state)
          return Image.fromarray(img_array.astype(np.uint8)), Image.fromarray(binary_mask * 255)
    ##Square Mask.
    # def apply_mask(self, img: Image.Image) -> Image.Image:
    #     width, height = img.size
    #     mask_percentage = random.uniform(0.2, 0.6)
    #     mask_area = width * height * mask_percentage
    #     mask_width = int(np.sqrt(mask_area))
    #     mask_height = mask_width

    #     x = random.randint(0, width - mask_width)
    #     y = random.randint(0, height - mask_height)

    #     mask = Image.new("RGB", (mask_width, mask_height), (255, 255, 255))
    #     img.paste(mask, (x, y))
    #     return img
###No check.
    # def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
    #     img_index = self.image_indices[index]
    #     item = self.ds[img_index]
    #     gt_img = Image.fromarray(item['images'].numpy())
    #     lq_img = gt_img.copy()
    #     caption = ''
    #     lq_img, gt_img = self.apply_augmentations(lq_img, gt_img)
    #     gt_img = gt_img * 2 - 1
    #     return {'hint': lq_img, 'jpg': gt_img, 'txt': caption}
##Check in items.
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
            attempts = 0
            max_attempts = 10

            while attempts < max_attempts:
                try:
                    actual_index = self.image_indices[index]
                    item = self.ds[actual_index]

                    if 'images' in item and item['images'].ndim == 3 and item['images'].shape[2] == 3:
                        gt_img = Image.fromarray(item['images'].numpy())
                        lq_img = gt_img.copy()

                        caption = ''
                        lq_img, gt_img, binary_mask = self.apply_augmentations(lq_img, gt_img)
                        gt_img = gt_img * 2 - 1

                        return {'hint': lq_img, 'jpg': gt_img, 'txt': "", 'mask': binary_mask}
                    else:
                        raise ValueError(f"Invalid data format at index {actual_index}")

                except Exception as e:
                    print(f"Error at index {actual_index}: {e}")
                    index += 1 
                    if index >= len(self.image_indices):
                        index = 0
                    attempts += 1

    def apply_augmentations(self, lq_img, gt_img):
        if self.crop_type == "center":
            lq_img = center_crop_arr_bP(lq_img, self.out_size)
            gt_img = center_crop_arr_bP(gt_img, self.out_size)
        elif self.crop_type == "random":
            # Ensure the same random crop is applied to both
            # Implement this function to apply the same crop
            lq_img, gt_img = random_crop_pair(lq_img, gt_img, self.out_size)

        lq_img, binary_mask = self.apply_mask(lq_img)
        lq_img = (np.array(lq_img).astype(np.float32) / 255.0)
        gt_img = (np.array(gt_img).astype(np.float32) / 255.0)
        binary_mask = (np.array(binary_mask).astype(np.float32) / 255.0)
        lq_img, gt_img, binary_mask = augment_pair(lq_img, gt_img, binary_mask, hflip=self.use_hflip, rotation=False)

        # Apply JPEG compression and Gaussian blur to lq_img
        # lq_img = apply_jpeg_compression((lq_img * 255).astype(np.uint8))
        # lq_img = apply_gaussian_blur(lq_img)        
        return lq_img, gt_img, binary_mask


    def __len__(self) -> int:
        return len(self.image_indices)

##DarkEnhanceDataset_val

class DarkEnhanceDataset_val(data.Dataset):
    
    def __init__(self, captions_file: str, out_size: int, crop_type: str, use_hflip: bool):
        super(DarkEnhanceDataset_val, self).__init__()
        self.ds = deeplake.load("hub://activeloop/places205")
        self.captions = None
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        self.image_indices = self._get_image_indices()
        # self.load_and_process_image = load_and_process_image
    def load_and_process_image(self, url: str) -> Image.Image:
        response = requests.get(url)
        pil_img = Image.open(BytesIO(response.content)).convert("RGB")
        return pil_img
      
  ##Check in list
    # def _get_image_indices(self):
    #     all_indices = list(range(len(self.ds)))
    #     print(len(self.ds))
    #     selected_indices = random.sample(all_indices, 250000)

    #     image_indices = []
    #     for i in selected_indices:
    #         # print(i)
    #         item = self.ds[i]
    #         if 'images' in item and item['images'].ndim == 3 and item['images'].shape[2] == 3:
    #             image_indices.append(i)

    #     with open('image_indices.txt', 'w') as f:
    #         for index in image_indices:
    #             f.write(f"{index}\n")

    #     return image_indices
    
###No check in list.
    def _get_image_indices(self):
      all_indices = list(range(len(self.ds)))
      selected_indices = random.sample(all_indices, 100)
      return selected_indices
###Maskv3:obvious boundary.
    # def apply_mask(self, img: Image.Image) -> Image.Image:
    #     state = random.getstate()
    #     random.seed(time.time())
    #     width, height = img.size

    #     img_array = np.array(img)

    #     mask = np.zeros((height, width), dtype=np.uint8)

    #     num_shapes = random.randint(1, 10)
    #     for _ in range(num_shapes):
    #         x1, y1 = random.randint(0, width), random.randint(0, height)
    #         x2, y2 = random.randint(0, width), random.randint(0, height)
    #         thickness = random.randint(1, 40)
    #         cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    #     kernel_size = random.randint(1, 10)
    #     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #     mask = cv2.dilate(mask, kernel, iterations=random.randint(1, 5))

    #     masked_img = cv2.bitwise_and(img_array, img_array, mask=mask)

    #     degraded_area = cv2.GaussianBlur(masked_img, (0, 0), sigmaX=20, sigmaY=20, borderType=cv2.BORDER_DEFAULT)

    #     img_array[mask > 0] = degraded_area[mask > 0]
    #     degraded_img = Image.fromarray(img_array)

    #     random.setstate(state)
    #     return degraded_img
    def apply_mask(self, img: Image.Image) -> (Image.Image, Image.Image):
          state = random.getstate()
          random.seed(time.time())
          width, height = img.size

          img_array = np.array(img)

          mask = np.zeros((height, width), dtype=np.uint8)

          num_shapes = random.randint(2, 5)
          for _ in range(num_shapes):
              x1, y1 = random.randint(0, width), random.randint(0, height)
              x2, y2 = random.randint(0, width), random.randint(0, height)
              thickness = random.randint(5, 15)
              cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

          kernel_size = random.randint(2, 4)
          kernel = np.ones((kernel_size, kernel_size), np.uint8)
          mask = cv2.dilate(mask, kernel, iterations=random.randint(2, 4))

          blurred_mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=0, sigmaY=0)

          binary_mask = (blurred_mask > 0).astype(np.uint8)

          alpha_mask = blurred_mask.astype(float) / 255

          degraded_img_array = cv2.GaussianBlur(img_array, (0, 0), sigmaX=20, sigmaY=20, borderType=cv2.BORDER_DEFAULT)

          img_array = (1 - alpha_mask)[:, :, None] * img_array + alpha_mask[:, :, None] * degraded_img_array

          random.setstate(state)
          return Image.fromarray(img_array.astype(np.uint8)), Image.fromarray(binary_mask * 255)
##Check in items.
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
            attempts = 0
            max_attempts = 10

            while attempts < max_attempts:
                try:
                    actual_index = self.image_indices[index]
                    item = self.ds[actual_index]

                    if 'images' in item and item['images'].ndim == 3 and item['images'].shape[2] == 3:
                        gt_img = Image.fromarray(item['images'].numpy())
                        lq_img = gt_img.copy()

                        caption = ''
                        lq_img, gt_img, binary_mask = self.apply_augmentations(lq_img, gt_img)
                        gt_img = gt_img * 2 - 1

                        return {'hint': lq_img, 'jpg': gt_img, 'txt': "", 'mask': binary_mask}
                    else:
                        raise ValueError(f"Invalid data format at index {actual_index}")

                except Exception as e:
                    print(f"Error at index {actual_index}: {e}")
                    index += 1 
                    if index >= len(self.image_indices):
                        index = 0
                    attempts += 1

    def apply_augmentations(self, lq_img, gt_img):
        if self.crop_type == "center":
            lq_img = center_crop_arr_bP(lq_img, self.out_size)
            gt_img = center_crop_arr_bP(gt_img, self.out_size)
        elif self.crop_type == "random":
            # Ensure the same random crop is applied to both
            # Implement this function to apply the same crop
            lq_img, gt_img = random_crop_pair(lq_img, gt_img, self.out_size)

        lq_img, binary_mask = self.apply_mask(lq_img)
        lq_img = (np.array(lq_img).astype(np.float32) / 255.0)
        gt_img = (np.array(gt_img).astype(np.float32) / 255.0)
        binary_mask = (np.array(binary_mask).astype(np.float32) / 255.0)
        lq_img, gt_img, binary_mask = augment_pair(lq_img, gt_img, binary_mask, hflip=self.use_hflip, rotation=False)

        # Apply JPEG compression and Gaussian blur to lq_img
        # lq_img = apply_jpeg_compression((lq_img * 255).astype(np.uint8))
        # lq_img = apply_gaussian_blur(lq_img)        
        return lq_img, gt_img, binary_mask

    def __len__(self) -> int:
        return len(self.image_indices)