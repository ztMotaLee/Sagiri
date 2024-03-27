import matplotlib.pyplot as plt
from inpaintingv2 import DarkEnhanceDataset_val
# def test_dataset(dataset):
#     for index in range(len(dataset)):
#         try:
#             print(index)
#             print("index")
#             data = dataset[index]
#             lq_img = data['hint']
#             gt_img = data['jpg']
#             # Just a simple operation to check if images are loaded correctly
#             if lq_img is None or gt_img is None:
#                 print(f"Missing data at index {index}")
#         except Exception as e:
#             print(f"Error at index {index}: {e}")
#             break  # Stop the loop if an error occurs

# dataset = DarkEnhanceDataset(captions_file=None, out_size=128, crop_type='center', use_hflip=False)

# test_dataset(dataset)
import os
def save_images(dataset, lq_dir, gt_dir, mask_dir):
    os.makedirs(lq_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for index in range(0,100):
        data = dataset[index]
        lq_img = data['hint']
        gt_img = data['jpg']
        mask = data['mask']

        lq_img = (lq_img * 255).astype('uint8')
        gt_img = ((gt_img + 1) / 2 * 255).astype('uint8')
        mask_img = (mask*255).astype('uint8')
        lq_path = os.path.join(lq_dir, f'{index}.png')
        gt_path = os.path.join(gt_dir, f'{index}.png')
        mask_path = os.path.join(mask_dir, f'{index}.png')
        plt.imsave(lq_path, lq_img)
        plt.imsave(gt_path, gt_img)
        plt.imsave(mask_path, mask_img)

dataset = DarkEnhanceDataset_val(captions_file=None, out_size=1024, crop_type='none', use_hflip=False)

lq_dir = "results/lq"
gt_dir = "results/gt"
mask_dir = "results/mask"
save_images(dataset, lq_dir, gt_dir, mask_dir)