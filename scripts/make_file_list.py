import sys
import os
from argparse import ArgumentParser
import sys
sys.path.append("/mnt/petrelfs/liboang/models/hdr_eccv24/DiffBIR-main")
from utils.file import list_image_files

parser = ArgumentParser()
parser.add_argument("--train_lq_folder", type=str, required=True)
parser.add_argument("--train_gt_folder", type=str, required=True)
parser.add_argument("--val_lq_folder", type=str, required=True)
parser.add_argument("--val_gt_folder", type=str, required=True)
parser.add_argument("--save_folder", type=str, required=True)
parser.add_argument("--follow_links", action="store_true")
args = parser.parse_args()

def process_folder(lq_folder, gt_folder, save_path):
    lq_files = set(os.path.basename(p.split(".")[0]) for p in list_image_files(lq_folder, exts=(".jpg", ".png", ".jpeg"), follow_links=args.follow_links))
    # print(lq_files)
    gt_files = set(os.path.basename(p.split(".")[0]) for p in list_image_files(gt_folder, exts=(".hdr"), follow_links=args.follow_links))
    # print(gt_files)
    common_files = lq_files.intersection(gt_files)
    # print(common_files)
    print(f"find {len(common_files)} matching LQ and GT images in {lq_folder} and {gt_folder}")

    with open(save_path, "w") as fp:
        for file_name in common_files:
            lq_path = os.path.join(lq_folder, file_name+".jpg")
            gt_path = os.path.join(gt_folder, file_name+".hdr")
            fp.write(f"{lq_path} {gt_path}\n")

os.makedirs(args.save_folder, exist_ok=True)

# Process training data
process_folder(args.train_lq_folder, args.train_gt_folder, os.path.join(args.save_folder, "train_hdr.list"))

# Process validation data
process_folder(args.val_lq_folder, args.val_gt_folder, os.path.join(args.save_folder, "val_hdr.list"))
