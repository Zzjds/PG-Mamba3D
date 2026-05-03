import os
import numpy as np
import tifffile as tiff
from pathlib import Path
from PIL import Image
import cv2
import math
import mvtec3d_util as mvt_util
import argparse


def preprocess_pc(tiff_path, canny_param=(5, 10)):
    image = cv2.imread(tiff_path, cv2.IMREAD_COLOR)
    edges = cv2.Canny(image, canny_param[0], canny_param[1])  # 调整阈值参数

    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 创建一个空白的二值图像，用于绘制分割结果
    segmentation_result = np.zeros_like(image)

    # 4. 绘制分割区域
    for contour in contours:
        cv2.drawContours(segmentation_result, [contour], 0, (255, 255, 255), -1)  # 填充轮廓

    segmentation_result = dilated_edges = cv2.erode(segmentation_result, kernel, iterations=2)
    cv2.imwrite(tiff_path.replace('depth', 'fgmask'), segmentation_result[:, :, 0])
    # cv2.imwrite("/home/v-kecenli/EasyNet/fg.png", segmentation_result[:, :, 0])
    # cv2.imwrite("/home/kecen/Easynet/mask_candy/"+, image)
    print()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MVTec 3D-AD')
    parser.add_argument('--dataset_path', type=str, default="/home/kecen/data/Eyecandies/Eyecandies" , help='The root path of the MVTec 3D-AD. The preprocessing is done inplace (i.e. the preprocessed dataset overrides the existing one)')
    args = parser.parse_args()

    params = {"CandyCane": (10,50), "ChocolateCookie": (5,50), "ChocolatePraline": (18,50) , "Confetto": (10,50), 
              "GummyBear": (5,50), "HazelnutTruffle": (18,50), "LicoriceSandwich": (20,50), "Lollipop": (5,50), 
              "Marshmallow": (18,50), "PeppermintCandy": (10,50)}
    root_path = args.dataset_path
    category_list = os.listdir(root_path)
    for category in category_list:
        # if category != "LicoriceSandwich":
        #     continue
        # os.makedirs("/home/kecen/Easynet/mask_candy/{}".format(category), exist_ok=True)
        cate_path = os.path.join(root_path, category)
        paths = Path(cate_path).rglob('*depth.png')
        print(f"Found {len(list(paths))} tiff files in {cate_path}")
        processed_files = 0
        for path in Path(cate_path).rglob('*depth.png'):
            if 'train' not in path.__str__():
                continue
            preprocess_pc(path.__str__(), params[category])
            processed_files += 1
            if processed_files % 50 == 0:
                print(f"Processed {processed_files} tiff files...")