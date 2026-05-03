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
        # Randomly perform three aug
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Generate even numbers from 0 to 12
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        # control aug ratio
        while True: 
            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot_perlin(image=perlin_noise)
            perlin_thr = np.where(np.abs(perlin_noise) > self.perlin_t, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_noise = perlin_noise.astype(np.float32)
            perlin_noise = np.expand_dims(perlin_noise, axis=2)
            perlin_thr = np.expand_dims(perlin_thr, axis=2)
            mask_zzz = depth_mask.astype(np.float32) * perlin_thr        # anomaly mask
            # if self.aug_num[0] < self.aug_num[1] * 2 and np.sum(mask_zzz) < 1e-3:
            #     continue
            break

        beta = torch.rand(1).numpy()[0] * 0.8

        high_noise_pink = np.random.rand() * (self.high_peak - self.min_noise) + self.min_noise
        low_noise_pink = np.random.rand() * (self.low_peak - self.min_noise) + self.min_noise
        zzz_noise = mask_zzz * perlin_noise
        # high_noise_pink = low_noise_pink = self.min_noise
        zzz_noise[zzz_noise>0.] = high_noise_pink
        zzz_noise[zzz_noise<0.] = -low_noise_pink
        if self.skew:
            zzz_noise = self.skew_filter(zzz_noise[:, :, 0], sigma=2.)
        else:
            zzz_noise = gaussian_filter(zzz_noise[:, :, 0], sigma=2)
        zzz_noise = np.expand_dims(zzz_noise, axis=2)

        # augmented_zzz = depth * (1 - mask_zzz) + mask_zzz * (depth + perlin_noise)
        augmented_zzz = depth + zzz_noise
        augmented_zzz = np.clip(augmented_zzz, 0., 1.)

        if self.mask_type == "rgb_mask":
            img_thr = anomaly_img_augmented.astype(np.float32) * mask_zzz / 255.0
            augmented_image = image * (1 - mask_zzz) + (1 - beta) * img_thr + beta * image * (mask_zzz)
        else:
            mask_zzz = np.where(np.abs(zzz_noise) > self.mask_t, np.ones_like(zzz_noise), np.zeros_like(zzz_noise))
            img_thr = anomaly_img_augmented.astype(np.float32) * mask_zzz / 255.0
            augmented_image = image * (1 - mask_zzz) + (1 - beta) * img_thr + beta * image * (mask_zzz)

        if self.fusion and self.drop_p != 0:
            no_anomaly_depth = np.random.rand()
            no_anomaly_rgb = np.random.rand()
            t = 1-self.drop_p
            if no_anomaly_rgb > t and no_anomaly_depth > t:
                image = image.astype(np.float32)
                self.aug_num[1] += 1
                return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
            elif no_anomaly_rgb > t and no_anomaly_depth <= t:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                    self.aug_num[1] += 1
                else:
                    has_anomaly = 1.0
                    self.aug_num[0] += 1
                return image, augmented_zzz.astype(np.float32), np.zeros_like(perlin_thr, dtype=np.float32), mask_zzz, np.array([has_anomaly],dtype=np.float32)
            elif no_anomaly_rgb <= t and no_anomaly_depth > t:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                    self.aug_num[1] += 1
                else:
                    has_anomaly = 1.0
                    self.aug_num[0] += 1
                return augmented_image, depth, mask_zzz, np.zeros_like(perlin_thr, dtype=np.float32), np.array([has_anomaly],dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                    self.aug_num[1] += 1
                else:
                    has_anomaly = 1.0
                    self.aug_num[0] += 1
                return augmented_image, augmented_zzz.astype(np.float32), mask_zzz, mask_zzz, np.array([has_anomaly],dtype=np.float32)
        else:
            no_anomaly = np.random.rand()
            if no_anomaly > 0.5:
                image = image.astype(np.float32)
                return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                has_anomaly = 1.0
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                return augmented_image, augmented_zzz.astype(np.float32), mask_zzz, mask_zzz, np.array([has_anomaly],dtype=np.float32)
    
    def augment_image(self, image, anomaly_source_path,depth):
        # Random rotation with three variations
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        threshold_msk = 0.001
        depth_mask = np.where(depth > threshold_msk, np.ones_like(depth), np.zeros_like(depth))

        # load noise image
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # Randomly perform three aug
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Generate even numbers from 0 to 12
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot_perlin(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_noise = perlin_noise.astype(np.float32)
        perlin_noise = np.expand_dims(perlin_noise, axis=2)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        mask_zzz = depth_mask.astype(np.float32) * perlin_thr        # anomaly mask

        img_thr = anomaly_img_augmented.astype(np.float32) * mask_zzz / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        # augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
        augmented_image = image * (1 - mask_zzz) + (1 - beta) * img_thr + beta * image * (mask_zzz)
        augmented_zzz = depth * (1 - mask_zzz) + mask_zzz * perlin_noise

        no_anomaly = torch.rand(1).numpy()[0]
        if self.fusion and self.drop_p != 0:
            no_anomaly_depth = np.random.rand()
            no_anomaly_rgb = np.random.rand()
            t = 1-self.drop_p
            if no_anomaly_rgb > t and no_anomaly_depth > t:
                image = image.astype(np.float32)
                self.aug_num[1] += 1
                return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
            elif no_anomaly_rgb > t and no_anomaly_depth <= t:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                    self.aug_num[1] += 1
                else:
                    has_anomaly = 1.0
                    self.aug_num[0] += 1
                return image, augmented_zzz.astype(np.float32), np.zeros_like(perlin_thr, dtype=np.float32), mask_zzz, np.array([has_anomaly],dtype=np.float32)
            elif no_anomaly_rgb <= t and no_anomaly_depth > t:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                    self.aug_num[1] += 1
                else:
                    has_anomaly = 1.0
                    self.aug_num[0] += 1
                return augmented_image, depth, mask_zzz, np.zeros_like(perlin_thr, dtype=np.float32), np.array([has_anomaly],dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                    self.aug_num[1] += 1
                else:
                    has_anomaly = 1.0
                    self.aug_num[0] += 1
                return augmented_image, augmented_zzz.astype(np.float32), mask_zzz, mask_zzz, np.array([has_anomaly],dtype=np.float32)
        else:
            if no_anomaly > 0.5:
                image = image.astype(np.float32)
                return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)
                has_anomaly = 1.0
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                return augmented_image, augmented_zzz.astype(np.float32), mask_zzz, mask_zzz, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, tiff_path, anomaly_source_path):
        # Generate numpy format functions for raw and noisy images
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        xyz = tiff.imread(tiff_path)
        xyz = np.array(xyz)

        zzz = np.copy(xyz)[:,:,2]
        zzz = np.expand_dims(zzz,axis=2)
        
        # zzz_min = zzz.min()
        # zzz = (zzz - zzz_min)/(zzz.max() - zzz_min)
        zzz_min = np.min(zzz[zzz>0.5])
        zzz = 1 - (zzz - zzz_min)/(np.max(zzz) - zzz_min)
        zzz[zzz>1.] = 0.
        zzz = cv2.resize(zzz, dsize=(self.img_size[1], self.img_size[0]))

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0

        if self.aug_type == "gaussian":
            augmented_image, augmented_zzz, mask_image, mask_zzz, has_anomaly = self.augment_image_gaussian(image, anomaly_source_path, zzz)
        else:
            augmented_image, augmented_zzz, mask_image, mask_zzz, has_anomaly = self.augment_image(image, anomaly_source_path, zzz)

        normal = depth2normal((1-zzz[..., 0]) * 65535.0) * 0.5 + 0.5
        augmented_normal = depth2normal((1-augmented_zzz[..., 0]) * 65535.0) * 0.5 + 0.5

        # Adjusting dimensions
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        normal = np.transpose(normal, (2, 0, 1)).astype(np.float32)
        augmented_normal = np.transpose(augmented_normal, (2, 0, 1)).astype(np.float32)

        mask_image = np.transpose(mask_image, (2, 0, 1))
        mask_zzz = np.transpose(mask_zzz, (2, 0, 1))
        
        return (image, augmented_image, normal, augmented_normal, has_anomaly, mask_image, mask_zzz)

    def __getitem__(self, idx):
        # idx = torch.randint(0, len(self.image_paths), (1,)).item()
        # choose a random picture to generate noise
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        result = self.transform_image(self.img_paths[idx][0],
                                      self.img_paths[idx][1], self.anomaly_source_paths[anomaly_source_idx])

        sample = {'image': result[0], 'augmented_image': result[1],
                    'zzz': result[2], 'augmented_zzz': result[3],
                  "has_anomaly":result[4], "mask_rgb": result[5], "mask_d": result[6],
                  "file_path": self.img_paths[idx][0]}
        # output size: 3,256,256  0~1 numpy
        #'image': b,3,256,256, 'augmented_image': b,3,256,256,"mask": b,1,256,256,
                #   'zzz': b,3,256,256, 'augmented_zzz': b,3,256,256
        return sample


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
        
        # zzz_min = zzz.min()
        # zzz = (zzz - zzz_min)/(zzz.max() - zzz_min)
        zzz_min = np.min(zzz[zzz>0.5])
        zzz = 1 - (zzz - zzz_min)/(np.max(zzz) - zzz_min)
        zzz[zzz>1.] = 0.
        zzz = cv2.resize(zzz, dsize=(self.img_size[1], self.img_size[0]))

        normal = depth2normal((1-zzz[..., 0]) * 65535.0) * 0.5 + 0.5

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

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)


        normal = np.transpose(normal, (2, 0, 1)).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        sample = {'image': image,'mask': mask ,'has_anomaly': has_anomaly,'zzz': normal, "file_path": self.img_paths[idx][0]}

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
