import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import cv2
import imgaug.augmenters as iaa
import tifffile as tiff
from scipy.ndimage import gaussian_filter, convolve

from utils.perlin import rand_perlin_2d_np
from utils.skew_gaussian import generate_skew_kernel


def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]


class MVTec3D(Dataset):

    def __init__(self, split, class_name, img_size, dataset_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((img_size[0], img_size[1]), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

class MVTec3DTrain(MVTec3D):
    def __init__(self, class_name, img_size, dataset_path, anomaly_source_path, fusion=False, args=None):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.resize_shape = img_size
        self.fusion = fusion

        self.drop_p = args.drop_p

        self.aug_num = [0, 0]
        self.perlin_t = args.perlin_t
        self.low_peak = args.low_peak
        self.high_peak = args.high_peak
        self.min_noise = args.min_noise
        self.aug_type = args.aug_type
        self.mask_type = args.mask_type
        self.mask_t = args.mask_t
        self.skew = args.skew
 
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
        # There is a chance of rotation between -90 and 90 degrees
        self.rot_perlin = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        # Path of noise dataset
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))


    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def random_rotate(self, rgb, zzz, rot=45):
        data = np.concatenate([rgb, zzz], axis=2)
        data = np.transpose(data, (2, 0, 1))
        data = torch.from_numpy(data).unsqueeze(0)
        data = transforms.RandomRotation(rot)(data).squeeze(0)
        data = np.transpose(data.numpy(), (1, 2, 0))
        return data[:, :, :3], data[:, :, 3:]

    def skew_filter(self, mask, sigma=2., truncate=4.):
        radius = round(truncate * sigma)
        bias_x = np.random.rand() * 0.5
        bias_y = np.random.rand() * 0.5
        kernel = generate_skew_kernel(radius=radius, sigma=sigma, bias=(bias_x, bias_y))
        kernel = np.expand_dims(kernel, axis=2)
        mask = convolve(mask, kernel)
        return mask

    def augment_image_gaussian(self, image, anomaly_source_path, depth):
        # print(self.aug_num)
        # Random rotation with three variations
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        threshold_msk = 0.001
        # image, depth = self.random_rotate(image, depth)

        nonzero_ind = depth > threshold_msk
        depth_mask = np.where(nonzero_ind, np.ones_like(depth), np.zeros_like(depth))

        # load noise image
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # Generate even numbers from 0 to 12
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        # control aug ratio
        while True: 
            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_thr = np.where(np.abs(perlin_noise) > self.perlin_t, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_noise = perlin_noise.astype(np.float32)
            perlin_noise = np.expand_dims(perlin_noise, axis=2)
            perlin_thr = np.expand_dims(perlin_thr, axis=2)
            # convert perlin_thr intp RGB


            mask_zzz = depth_mask.astype(np.float32) * perlin_thr        # anomaly mask
            # if self.aug_num[0] < self.aug_num[1] * 2 and np.sum(mask_zzz) < 1e-3:
            #     continue
            break
        
        perlin_thr_red = np.where(perlin_noise > self.perlin_t, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr_blue = np.where(perlin_noise < -self.perlin_t, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))

        b_channel = np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8) * 1
        g_channel = np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8) * 0
        r_channel = np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8) * 0
        blue_image = cv2.merge((b_channel, g_channel, r_channel))
        b_channel = np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8) * 0
        g_channel = np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8) * 0
        r_channel = np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8) * 1
        red_image = cv2.merge((b_channel, g_channel, r_channel))
        perlin_thr = perlin_thr_red * red_image + perlin_thr_blue * blue_image


        beta = torch.rand(1).numpy()[0] * 0.4

        high_noise_pink = np.random.rand() * (self.high_peak - self.min_noise) + self.min_noise
        low_noise_pink = np.random.rand() * (self.low_peak - self.min_noise) + self.min_noise
        zzz_noise = mask_zzz * perlin_noise
        # high_noise_pink = low_noise_pink = self.min_noise
        zzz_noise[zzz_noise>0.] = high_noise_pink
        zzz_noise[zzz_noise<0.] = -low_noise_pink
        if self.skew:
            zzz_noise = self.skew_filter(zzz_noise, sigma=3.)
            deltaz = self.skew_filter(perlin_thr * depth_mask, sigma=3.)
        else:
            zzz_noise = gaussian_filter(zzz_noise[:, :, 0], sigma=2)
        # zzz_noise = np.expand_dims(zzz_noise, axis=2)

        # augmented_zzz = depth * (1 - mask_zzz) + mask_zzz * (depth + perlin_noise)
        augmented_zzz = depth + zzz_noise
        augmented_zzz = np.clip(augmented_zzz, 0., 1.)

        mask_zzz = np.where(np.abs(zzz_noise) > self.mask_t, np.ones_like(zzz_noise), np.zeros_like(zzz_noise))
        img_thr = anomaly_source_img.astype(np.float32) * mask_zzz / 255.0
        augmented_image = image * (1 - mask_zzz) + (1 - beta) * img_thr + beta * image * (mask_zzz)
        
        return augmented_image, augmented_zzz.astype(np.float32), mask_zzz, perlin_noise, perlin_thr, depth_mask, deltaz, anomaly_source_img

    def transform_image(self, image_path, tiff_path, anomaly_source_path):
        # Generate numpy format functions for raw and noisy images
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        xyz = tiff.imread(tiff_path)
        xyz = np.array(xyz)

        zzz = np.copy(xyz)[:,:,2]
        zzz = np.expand_dims(zzz,axis=2)
        
        zzz = torch.from_numpy(np.transpose(zzz, (2, 0, 1))).unsqueeze(0)
        # zzz_min = zzz.min()
        # zzz = (zzz - zzz_min)/(zzz.max() - zzz_min)
        zzz_min = zzz[zzz>0.5].min()
        zzz = 1 - (zzz - zzz_min)/(zzz.max() - zzz_min)
        zzz[zzz>1.] = 0.
        zzz = F.interpolate(zzz, size=[self.resize_shape[1], self.resize_shape[0]], mode="bilinear").squeeze(0)
        zzz = zzz.numpy()
        zzz = np.transpose(zzz, (1, 2, 0))

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0

        augmented_image, augmented_zzz, mask, perlin_noise, perlin_thr, depth_mask, zzz_noise, anomaly_source_img = self.augment_image_gaussian(image, anomaly_source_path, zzz)
        
        return (image, augmented_image, zzz, augmented_zzz, mask, perlin_noise, perlin_thr, depth_mask, zzz_noise, anomaly_source_img)

    def __getitem__(self, idx):
        # idx = torch.randint(0, len(self.image_paths), (1,)).item()
        # choose a random picture to generate noise
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        result = self.transform_image(self.img_paths[idx][0],
                                      self.img_paths[idx][1], self.anomaly_source_paths[anomaly_source_idx])

        return result, self.img_paths[idx][0]


class MVTec3DTest(MVTec3D):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="test", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1]), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.img_size = img_size

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]

        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        xyz = tiff.imread(tiff_path)
        xyz = np.array(xyz)

        zzz = np.copy(xyz)[:,:,2]
        zzz = np.expand_dims(zzz,axis=2)
        
        zzz = torch.from_numpy(np.transpose(zzz, (2, 0, 1))).unsqueeze(0)
        # zzz_min = zzz.min()
        # zzz = (zzz - zzz_min)/(zzz.max() - zzz_min)
        zzz_min = zzz[zzz>0.5].min()
        zzz = 1 - (zzz - zzz_min)/(zzz.max() - zzz_min)
        zzz[zzz>1.] = 0.
        zzz = F.interpolate(zzz, size=[self.img_size[1], self.img_size[0]], mode="bilinear").squeeze(0)
        zzz = zzz.numpy()
        zzz = np.transpose(zzz, (1, 2, 0))

        # load mask
        if gt == 0:
            has_anomaly = np.array([0], dtype=np.float32)
            mask = np.zeros((self.img_size[0],self.img_size[1]))
        else:
            has_anomaly = np.array([1], dtype=np.float32)
            mask = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)


        if self.img_size != None:
            image = cv2.resize(image, dsize=(self.img_size[1], self.img_size[0]))
            mask = cv2.resize(mask, dsize=(self.img_size[1], self.img_size[0]))

        image = image / 255.0
        mask = mask / 255.0

        zzz = np.repeat(zzz, 3, axis=2)

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        zzz = np.array(zzz).reshape((zzz.shape[0], zzz.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)


        image = np.transpose(image, (2, 0, 1))
        tiff_image_zzz = np.transpose(zzz, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        sample = {'image': image,'mask': mask ,'has_anomaly': has_anomaly,'zzz': tiff_image_zzz, "file_path": self.img_paths[idx][0]}

        return sample


def get_data_loader(args, split, class_name, img_size,batch_size=1, num_workers=1,shuffle=False, is_fusion=False):
    if split in ['train']:
        dataset = MVTec3DTrain(class_name=class_name, img_size=img_size, dataset_path=args.data_dir,
                               anomaly_source_path=args.anomaly_source_path,fusion=is_fusion, args=args)
    elif split in ['test']:
        dataset = MVTec3DTest(class_name=class_name, img_size=img_size, dataset_path=args.data_dir)
    datas_len = len(dataset)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, 
                             shuffle=shuffle, num_workers=num_workers, 
                             drop_last=False)
    return data_loader,datas_len
