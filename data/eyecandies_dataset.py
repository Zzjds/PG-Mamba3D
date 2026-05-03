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
from utils.perlin import rand_perlin_2d_np
import yaml
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter, convolve
from utils.skew_gaussian import generate_skew_kernel


# DATASETS_PATH = "/data1/chenruitao/eyecandies/Eyecandies"

def eyecandies_classes():
    return [
        "CandyCane" ,
        "ChocolateCookie",
        "ChocolatePraline",
        "Confetto",
        "GummyBear" ,
        "HazelnutTruffle",
        "LicoriceSandwich",
        "Lollipop",
        "Marshmallow",
        "PeppermintCandy"
    ]


class Eyecandies(Dataset):

    def __init__(self, split, class_name, img_size,dataset_path):
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split,"data")
    def load_and_convert_depth(self, depth_img, info_depth):
        with open(info_depth) as f:
            data = yaml.safe_load(f)
        mind, maxd = data["normalization"]["min"], data["normalization"]["max"]

        dimg = iio.imread(depth_img)
        dimg = dimg.astype(np.float32)
        dimg = dimg / 65535.0 * (maxd - mind) + mind
        
        medd = np.median(dimg)
        th = 2*medd - mind
        dimg[dimg>th] = th
        return dimg


class EyecandiesTrain(Eyecandies):
    def __init__(self, class_name, img_size,dataset_path,anomaly_source_path, fusion=False, args=None):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.dataset_len = 1000
        self.drop_p = args.drop_p
        self.flip = args.flip
        self.whole_set = args.whole_set
        self.fgsg = args.fgsg
        self.perlin_t = args.perlin_t
        self.low_peak = args.low_peak
        self.high_peak = args.high_peak
        self.min_noise = args.min_noise
        self.aug_type = args.aug_type
        self.mask_type = args.mask_type
        self.mask_t = args.mask_t
        self.resize_shape = img_size
        self.skew = args.skew
        self.fusion = fusion
        
        self.img_paths, self.path_depth, self.path_info_depth, self.path_fgmask, self.path_normal = self.load_dataset()  # self.labels => good : 0, anomaly : 1
 
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
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        # Path of noise dataset
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))


    def load_dataset(self):
        path_img = []
        path_depth = []
        path_info_depth = []
        path_fgmask = []
        path_normal = []
        for num in range(self.dataset_len):
            if self.whole_set:
                for i in range(6):
                    path_img.append(os.path.join(self.img_path,str(num).zfill(3)+"_image_"+str(i)+".png"))
                    path_info_depth.append(os.path.join(self.img_path,str(num).zfill(3)+"_info_depth"+".yaml"))
                    path_depth.append(os.path.join(self.img_path,str(num).zfill(3)+"_depth"+".png"))
                    path_fgmask.append(os.path.join(self.img_path,str(num).zfill(3)+"_fgmask"+".png"))
                    path_normal.append(os.path.join(self.img_path,str(num).zfill(3)+"_normals"+".png"))
            else:
                # i = np.random.randint(6)
                i = 4
                path_img.append(os.path.join(self.img_path,str(num).zfill(3)+"_image_"+str(i)+".png"))
                path_info_depth.append(os.path.join(self.img_path,str(num).zfill(3)+"_info_depth"+".yaml"))
                path_depth.append(os.path.join(self.img_path,str(num).zfill(3)+"_depth"+".png"))
                path_fgmask.append(os.path.join(self.img_path,str(num).zfill(3)+"_fgmask"+".png"))
                path_normal.append(os.path.join(self.img_path,str(num).zfill(3)+"_normals"+".png"))
        return path_img,path_depth,path_info_depth,path_fgmask,path_normal
    


    def __len__(self):
        return self.dataset_len
    
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
    
    def random_flip(self, rgb, zzz, p=0.5):
        data = np.concatenate([rgb, zzz], axis=2)
        if np.random.rand() > p:
            data = data[:, ::-1, :].copy()
        return data[:, :, :3], data[:, :, 3:]
    
    def skew_filter(self, mask, sigma=2., truncate=4.):
        radius = round(truncate * sigma)
        bias_x = np.random.rand() * 0.5
        bias_y = np.random.rand() * 0.5
        kernel = generate_skew_kernel(radius=radius, sigma=sigma, bias=(bias_x, bias_y))
        mask = convolve(mask, kernel)
        return mask


    def augment_image_gaussian(self, image, depth, anomaly_source_path, fg_mask):
        # Random rotation with three variations
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        threshold_msk = 0.001
        if self.flip:
            image, depth = self.random_flip(image, depth)
        # image, depth = self.random_rotate(image, depth)

        nonzero_ind = fg_mask > threshold_msk
        if np.sum(fg_mask) > np.prod(fg_mask.shape)*0.01 and self.fgsg:
            depth_mask = np.where(nonzero_ind, np.ones_like(depth), np.zeros_like(depth))
        else:
            depth_mask = np.ones_like(depth)

        # load noise image
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # Randomly perform three aug
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Generate even numbers from 0 to 12
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)

        perlin_thr = np.where(np.abs(perlin_noise) > self.perlin_t, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_noise = perlin_noise.astype(np.float32)
        perlin_noise = np.expand_dims(perlin_noise, axis=2)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        mask_zzz = depth_mask.astype(np.float32) * perlin_thr        # anomaly mask

        high_noise_pink = np.random.rand() * (self.high_peak - self.min_noise) + self.min_noise
        low_noise_pink = np.random.rand() * (self.low_peak - self.min_noise) + self.min_noise
        zzz_noise = mask_zzz * perlin_noise
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

        beta = torch.rand(1).numpy()[0] * 0.8
        if self.mask_type == "rgb_mask":
            img_thr = anomaly_img_augmented.astype(np.float32) * mask_zzz / 255.0
            augmented_image = image * (1 - mask_zzz) + (1 - beta) * img_thr + beta * image * (mask_zzz)
        else:
            mask_zzz = np.where(np.abs(zzz_noise) > self.mask_t, np.ones_like(zzz_noise), np.zeros_like(zzz_noise))
            img_thr = anomaly_img_augmented.astype(np.float32) * mask_zzz / 255.0
            augmented_image = image * (1 - mask_zzz) + (1 - beta) * img_thr + beta * image * (mask_zzz)

        if self.fusion:
            no_anomaly_depth = torch.rand(1).numpy()[0]
            no_anomaly_rgb = torch.rand(1).numpy()[0]
            t = 1-self.drop_p
            if no_anomaly_rgb > t and no_anomaly_depth > t:
                image = image.astype(np.float32)
                return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
            elif no_anomaly_rgb > t and no_anomaly_depth <= t:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                has_anomaly = 1.0
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                return image, augmented_zzz.astype(np.float32), np.zeros_like(perlin_thr, dtype=np.float32), mask_zzz, np.array([has_anomaly],dtype=np.float32)
            elif no_anomaly_rgb <= t and no_anomaly_depth > t:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                has_anomaly = 1.0
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                return augmented_image, depth, mask_zzz, np.zeros_like(perlin_thr, dtype=np.float32), np.array([has_anomaly],dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                has_anomaly = 1.0
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                return augmented_image, augmented_zzz.astype(np.float32), mask_zzz, mask_zzz, np.array([has_anomaly],dtype=np.float32)
        else:
            no_anomaly = torch.rand(1).numpy()[0]
            t = 0.5
            if no_anomaly > t:
                image = image.astype(np.float32)
                return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)
                # augmented_image = msk * augmented_image + (1-msk)*image
                has_anomaly = 1.0
                if np.sum(mask_zzz) == 0:
                    has_anomaly=0.0
                return augmented_image, augmented_zzz.astype(np.float32), mask_zzz, mask_zzz, np.array([has_anomaly],dtype=np.float32)
            
    def augment_image(self, image, depth, anomaly_source_path, fg_mask):
        # Random rotation with three variations
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        # Generate even numbers from 0 to 12
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)

        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        perlin_noise = np.expand_dims(perlin_noise, axis=2)


        # load noise image
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # Randomly perform three aug
        anomaly_img_augmented = aug(image=anomaly_source_img)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        
        beta = torch.rand(1).numpy()[0] * 0.8


        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
        augmented_zzz = depth * (1 - perlin_thr) + perlin_thr * perlin_noise

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = perlin_thr
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, augmented_zzz.astype(np.float32), msk, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, depth_path, depth_info_path, anomaly_source_path, fgmask_path):
        # Generate numpy format functions for raw and noisy images

        rgb_img = iio.imread(image_path)
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = rgb_img / 255.0
        rgb_img = cv2.resize(rgb_img, dsize=(self.resize_shape[1], self.resize_shape[0]))


        depth_img = self.load_and_convert_depth(depth_path,depth_info_path)

        depth_img = cv2.resize(depth_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        depth_img = 1-(depth_img - np.min(depth_img)) / (np.max(depth_img)-np.min(depth_img))
        depth_img = np.expand_dims(depth_img, axis=2)

        # depth_img = torch.from_numpy(np.transpose(depth_img, (2, 0, 1))).unsqueeze(0)
        # depth_img = F.interpolate(depth_img, size=[self.resize_shape[1], self.resize_shape[0]], mode="bilinear").squeeze(0)
        # depth_img = depth_img.numpy()
        # depth_img = np.transpose(depth_img, (1, 2, 0))

        # depth_img = cv2.resize(depth_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        fg_mask = iio.imread(fgmask_path)
        fg_mask = fg_mask.astype(np.float32) / 255.0
        fg_mask = cv2.resize(fg_mask, dsize=(self.resize_shape[1], self.resize_shape[0]))
        fg_mask = np.expand_dims(fg_mask, axis=2)

        if self.aug_type == "gaussian":
            augmented_image, augmented_depth, mask_image, mask_zzz, has_anomaly = self.augment_image_gaussian(rgb_img,depth_img,anomaly_source_path, fg_mask)
        else:
            augmented_image, augmented_depth, mask_image, mask_zzz, has_anomaly = self.augment_image(rgb_img, depth_img, anomaly_source_path, fg_mask)
        # print("augmented_image",augmented_image.shape)

        # Adjusting dimensions
        depth_img = np.repeat(depth_img, 3, axis=2)
        augmented_depth = np.repeat(augmented_depth, 3, axis=2)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        depth_img = np.transpose(depth_img, (2, 0, 1))

        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        augmented_depth = np.transpose(augmented_depth, (2, 0, 1))

        mask_image = np.transpose(mask_image, (2, 0, 1))
        mask_zzz = np.transpose(mask_zzz, (2, 0, 1))

        # mask = np.transpose(mask, (2, 0, 1))
        
        # return (rgb_img, augmented_image, depth_img, augmented_depth, has_anomaly,mask)
        return (rgb_img, augmented_image, depth_img, augmented_depth, has_anomaly, mask_image, mask_zzz)

    def __getitem__(self, idx):
        # idx = torch.randint(0, len(self.image_paths), (1,)).item()
        # choose a random picture to generate noise
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        result = self.transform_image(self.img_paths[idx], self.path_depth[idx], self.path_info_depth[idx],self.anomaly_source_paths[anomaly_source_idx],self.path_fgmask[idx])

        # sample = {'image': result[0], 'augmented_image': result[1],
        #             'zzz': result[2], 'augmented_zzz': result[3],
        #           "has_anomaly":result[4],"mask": result[5], "file_path": self.img_paths[idx]}
        sample = {'image': result[0], 'augmented_image': result[1],
                    'zzz': result[2], 'augmented_zzz': result[3],
                  "has_anomaly":result[4], "mask_rgb": result[5], "mask_d": result[6],
                  "file_path": self.img_paths[idx]}
        return sample


class EyecandiesTest(Eyecandies):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="test_public", class_name=class_name, img_size=img_size,dataset_path=dataset_path)
        self.dataset_len = 50
        self.img_paths, self.depth_paths, self.gt_paths, self.info_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.resize_shape = img_size

    def load_dataset(self):
        path_img = []
        path_depth = []
        path_mask = []
        path_info_depth = []
        for num in range(self.dataset_len):
            # for i in range(6):
            #     path_img.append(os.path.join(self.img_path,str(num).zfill(2)+"_image_"+str(i)+".png"))
            #     path_info_depth.append(os.path.join(self.img_path,str(num).zfill(2)+"_info_depth"+".yaml"))
            #     path_depth.append(os.path.join(self.img_path,str(num).zfill(2)+"_depth"+".png"))
            #     path_mask.append(os.path.join(self.img_path,str(num).zfill(2)+"_mask"+".png"))
            i = 4
            path_img.append(os.path.join(self.img_path,str(num).zfill(2)+"_image_"+str(i)+".png"))
            path_info_depth.append(os.path.join(self.img_path,str(num).zfill(2)+"_info_depth"+".yaml"))
            path_depth.append(os.path.join(self.img_path,str(num).zfill(2)+"_depth"+".png"))
            path_mask.append(os.path.join(self.img_path,str(num).zfill(2)+"_mask"+".png"))
        return path_img, path_depth, path_mask, path_info_depth

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        image_path, depth_path, mask_path, depth_info_path = self.img_paths[idx], self.depth_paths[idx], self.gt_paths[idx],self.info_paths[idx]

        rgb_img = iio.imread(image_path)
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = rgb_img / 255.0
        rgb_img = cv2.resize(rgb_img, dsize=(self.resize_shape[1], self.resize_shape[0]))


        depth_img = self.load_and_convert_depth(depth_path,depth_info_path)
        depth_img = cv2.resize(depth_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        depth_img = 1-(depth_img - np.min(depth_img)) / (np.max(depth_img)-np.min(depth_img))

        depth_img = np.expand_dims(depth_img,axis=2)

        # depth_img = torch.from_numpy(np.transpose(depth_img, (2, 0, 1))).unsqueeze(0)
        # depth_img = F.interpolate(depth_img, size=[self.resize_shape[1], self.resize_shape[0]], mode="bilinear").squeeze(0)
        # depth_img = depth_img.numpy()
        # depth_img = np.transpose(depth_img, (1, 2, 0))

        depth_img = np.repeat(depth_img,3,axis=2)
        # depth_img = cv2.resize(depth_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        mask_img = iio.imread(mask_path)
        mask_img = mask_img.astype(np.float32)
        mask_img = mask_img / 255.0
        mask_img = cv2.resize(mask_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        mask_img = np.expand_dims(mask_img, axis=0)

        # load mask
        if np.sum(mask_img) == 0:
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            has_anomaly = np.array([1], dtype=np.float32)

        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        depth_img = np.transpose(depth_img, (2, 0, 1))


        sample = {'image': rgb_img,'mask': mask_img ,'has_anomaly': has_anomaly,'zzz': depth_img, "file_path": self.img_paths[idx]}

        return sample


def get_data_loader(args,split, class_name, img_size, batch_size=1, num_workers=1,shuffle=False, is_fusion=False):
    if split in ['train']:
        dataset = EyecandiesTrain(class_name=class_name, img_size=img_size,dataset_path=args.data_dir,
                                  anomaly_source_path=args.anomaly_source_path, fusion=is_fusion, args=args)
    elif split in ['test']:
        dataset = EyecandiesTest(class_name=class_name, img_size=img_size,dataset_path=args.data_dir)
    datas_len = len(dataset)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, 
                             shuffle=shuffle, num_workers=num_workers, 
                             drop_last=False)
    return data_loader,datas_len
