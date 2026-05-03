import os

import torch
from torch import optim
from torch import nn
from utils.loss import FocalLoss, SSIM
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import cv2
import random
import importlib
import yaml
from scipy.ndimage import gaussian_filter
from utils.au_pro_util import calculate_au_pro


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_file(args):
    for path in [
        args.weight_save_path,
        os.path.join(args.weight_save_path, args.mode_type + '_' + args.exp_name),
        args.pic_path_train,
        os.path.join(args.pic_path_train, args.mode_type + '_' + args.exp_name),
        args.record_path,
        os.path.join(args.record_path, args.mode_type + '_' + args.exp_name),
    ]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_sample_grid(save_path, gray_image, gray_depth, gray_rec_image, gray_rec_depth,
                     true_mask_cv, out_mask_sm, mode_type):
    img_rgb   = (gray_image.detach().cpu().numpy() * 255.0).astype(np.uint8)[0]
    img_rgb   = np.transpose(img_rgb, (1, 2, 0))
    img_depth = (gray_depth.detach().cpu().numpy() * 255.0).astype(np.uint8)[0]
    img_depth = np.transpose(img_depth, (1, 2, 0))
    img_mask  = np.repeat(true_mask_cv[0] * 255.0, 3, axis=2)
    img_score = np.repeat(
        np.expand_dims(out_mask_sm[0, 1, :, :].detach().cpu().numpy() * 255.0, axis=2),
        3, axis=2)

    if mode_type == 'RGB':
        img_rec = (gray_rec_image.detach().cpu().numpy() * 255.0).astype(np.uint8)[0]
        img_rec = np.transpose(img_rec, (1, 2, 0))
        grid = np.concatenate((img_rgb, img_rec, img_mask, img_score), axis=1)
    elif mode_type == 'Depth':
        img_rec = (gray_rec_depth.detach().cpu().numpy() * 255.0).astype(np.uint8)[0]
        img_rec = np.transpose(img_rec, (1, 2, 0))
        grid = np.concatenate((img_depth, img_rec, img_mask, img_score), axis=1)
    else:
        img_rec_rgb = (gray_rec_image.detach().cpu().numpy() * 255.0).astype(np.uint8)[0]
        img_rec_rgb = np.transpose(img_rec_rgb, (1, 2, 0))
        img_rec_dep = (gray_rec_depth.detach().cpu().numpy() * 255.0).astype(np.uint8)[0]
        img_rec_dep = np.transpose(img_rec_dep, (1, 2, 0))
        grid = np.concatenate((img_rgb, img_rec_rgb, img_depth, img_rec_dep,
                                img_mask, img_score), axis=1)

    cv2.imwrite(save_path, grid)


class Wrap_model(nn.Module):
    def __init__(self, wrap_model, ngpu) -> None:
        super().__init__()
        self.wrap_model = wrap_model
        self.ngpu = ngpu

    def forward(self, rgb, depth=None):
        if self.ngpu > 1:
            if depth is not None:
                x = torch.cat([rgb, depth], dim=1)
            else:
                x = rgb
            output = nn.parallel.data_parallel(self.wrap_model, (x,), range(self.ngpu))
        else:
            if depth is not None:
                output = self.wrap_model(rgb, depth)
            else:
                output = self.wrap_model(rgb)
        return output

    def _reinit_mamba_zeros(self):
        if hasattr(self.wrap_model, '_reinit_mamba_zeros'):
            self.wrap_model._reinit_mamba_zeros()


def train_on_device(obj_names, args, dataset_checkpoint):
    create_file(args)

    for obj_name in obj_names:
        run_name = (
            "model_" + args.mode_type + "_"
            + str(args.lr) + '_' + str(args.epochs)
            + '_bs' + str(args.bs) + "_" + obj_name + '_'
        )

        model = EasyNet()
        checkpoint_rgb_path = dataset_checkpoint['checkpoint_rgb'][obj_name]

        if args.pretrain:
            model.load_state_dict(torch.load(
                os.path.join(args.weight_save_path,
                             args.mode_type,
                             run_name + "best.pckl")))
            model.cuda()
        else:
            model.cuda()
            model.apply(weights_init)

        model = Wrap_model(model, args.ngpu)
        model._reinit_mamba_zeros()

        if args.mode_type == "Fusion2":
            module = importlib.import_module(args.Model_type[args.mode_type + 'RGB'])
            EasyNet_rgb = getattr(module, 'ReconstructiveSubNetwork')
            model_rgb = EasyNet_rgb(in_channels=3, out_channels=3)
            model_rgb = Wrap_model(model_rgb, args.ngpu)
            model_rgb.load_state_dict(
                torch.load(checkpoint_rgb_path, map_location=torch.device('cuda:0')),
                strict=True)
            model_rgb.cuda()
            model_rgb.eval()

        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2, last_epoch=-1)

        loss_l2    = torch.nn.modules.loss.MSELoss()
        loss_ssim  = SSIM()
        loss_focal = FocalLoss()

        img_dim = 256

        train_loader, _ = get_data_loader(
            args, "train", class_name=obj_name, img_size=(256, 256),
            batch_size=args.bs, num_workers=2, shuffle=True,
            is_fusion='Fusion' in args.mode_type)

        test_loader, datas_len = get_data_loader(
            args, "test", class_name=obj_name, img_size=[img_dim, img_dim],
            batch_size=args.test_bs, num_workers=2, shuffle=False)

        n_iter = 0

        for epoch in range(args.start_epoch, args.epochs):
            loss_all = 0
            model.train()

            for i_batch, sample_batched in enumerate(train_loader):
                gray_aug_image = sample_batched["augmented_image"].cuda()
                gray_aug_depth = sample_batched["augmented_zzz"].cuda()
                gray_image     = sample_batched["image"].cuda()
                gray_depth     = sample_batched["zzz"].cuda()

                if "Fusion" in args.mode_type:
                    mask_rgb = sample_batched["mask_rgb"].cuda()
                    mask_d   = sample_batched["mask_d"].cuda()
                    mask     = 1 - (1 - mask_rgb) * (1 - mask_d)
                else:
                    mask = sample_batched["mask_rgb"].cuda()

                if args.mode_type == 'RGB':
                    gray_rec_image, out_mask, _ = model(gray_aug_image)
                    out_mask_sm   = torch.softmax(out_mask, dim=1)
                    loss = loss_l2(gray_rec_image, gray_image) + \
                           loss_ssim(gray_rec_image, gray_image) + \
                           loss_focal(out_mask_sm, mask)

                elif args.mode_type == 'Depth':
                    gray_rec_depth, out_mask, _ = model(gray_aug_depth)
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                    loss = loss_l2(gray_rec_depth, gray_depth) + \
                           loss_focal(out_mask_sm, mask)

                elif args.mode_type == 'XYZ':
                    xyz     = sample_batched["xyz"].cuda()
                    aug_xyz = sample_batched["augmented_xyz"].cuda()
                    rec_xyz, out_mask, _ = model(aug_xyz)
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                    loss = loss_l2(rec_xyz, xyz) + loss_focal(out_mask_sm, mask)

                elif args.mode_type == 'RGBD':
                    gray_rec_image, gray_rec_depth, out_mask = model(gray_aug_image, gray_aug_depth)
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                    loss = loss_l2(gray_rec_image, gray_image) + \
                           loss_ssim(gray_rec_image, gray_image) + \
                           loss_l2(gray_rec_depth, gray_depth) + \
                           loss_focal(out_mask_sm, mask)

                elif args.mode_type in ('Fusion0', 'Fusion6'):
                    B  = gray_aug_image.shape[0]
                    dI = (torch.rand(B, device=gray_aug_image.device) < args.drop_p).float().view(B, 1, 1, 1)
                    dZ = (torch.rand(B, device=gray_aug_image.device) < args.drop_p).float().view(B, 1, 1, 1)
                    aug_image_feed = dI * gray_image + (1 - dI) * gray_aug_image
                    aug_depth_feed = dZ * gray_depth + (1 - dZ) * gray_aug_depth
                    mask_drop = (1 - dI * dZ) * mask
                    gray_rec_image, gray_rec_depth, out_mask = model(aug_image_feed, aug_depth_feed)
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                    loss = loss_l2(gray_rec_image, gray_image) + \
                           loss_ssim(gray_rec_image, gray_image) + \
                           loss_l2(gray_rec_depth, gray_depth) + \
                           loss_focal(out_mask_sm, mask_drop)

                elif args.mode_type == 'Fusion1':
                    gray_rec_image, gray_rec_depth, out_mask, out_mask_rgb, _, _ = \
                        model(gray_aug_image, gray_aug_depth)
                    out_mask_sm     = torch.softmax(out_mask, dim=1)
                    out_mask_sm_rgb = torch.softmax(out_mask_rgb, dim=1)
                    loss = loss_l2(gray_rec_image, gray_image) + \
                           loss_ssim(gray_rec_image, gray_image) + \
                           loss_l2(gray_rec_depth, gray_depth) + \
                           loss_focal(out_mask_sm_rgb, mask) + \
                           loss_focal(out_mask_sm, mask)

                elif args.mode_type == 'Fusion2':
                    with torch.no_grad():
                        gray_rec_image, out_mask_rgb, merge_rgb = model_rgb(gray_aug_image)
                    gray_rec_depth, out_mask, _, _ = model(gray_aug_depth, merge_rgb)
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                    loss = loss_l2(gray_rec_depth, gray_depth) + loss_focal(out_mask_sm, mask)

                elif args.mode_type == 'Fusion3':
                    gray_aug_depth = gray_aug_depth[:, :1, ...]
                    gray_depth     = gray_depth[:, :1, ...]
                    output, out_mask = model(gray_aug_image, gray_aug_depth)
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                    target = torch.cat([gray_image, gray_depth], dim=1)
                    loss = loss_l2(output, target) + \
                           loss_ssim(output, target) + \
                           loss_focal(out_mask_sm, mask)

                elif args.mode_type == 'Fusion4':
                    gray_rec_image, gray_rec_depth, out_mask_d, out_mask_rgb = \
                        model(gray_aug_image, gray_aug_depth)
                    out_mask_sm_d   = torch.softmax(out_mask_d, dim=1)
                    out_mask_sm_rgb = torch.softmax(out_mask_rgb, dim=1)
                    loss = loss_l2(gray_rec_image, gray_image) + \
                           loss_ssim(gray_rec_image, gray_image) + \
                           loss_l2(gray_rec_depth, gray_depth) + \
                           loss_focal(out_mask_sm_d, mask) + \
                           loss_focal(out_mask_sm_rgb, mask)

                loss_all += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            scheduler.step()

            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.weight_save_path,
                    args.mode_type + '_' + args.exp_name,
                    run_name + ".pckl"))

            if epoch % args.test_freq != 0:
                continue

            total_pixel_scores    = []
            total_gt_pixel_scores = []
            mask_cnt              = 0
            anomaly_score_gt         = []
            anomaly_score_prediction = []
            predictions = []
            gts         = []

            model.eval()

            with torch.no_grad():
                for i_batch, sample_batched in enumerate(test_loader):
                    gray_image = sample_batched["image"].cuda()
                    gray_depth = sample_batched["zzz"].cuda()
                    is_normal  = list(sample_batched["has_anomaly"].detach().flatten().numpy())
                    anomaly_score_gt.extend(is_normal)

                    true_mask    = sample_batched["mask"]
                    true_mask_cv = true_mask.detach().numpy().transpose((0, 2, 3, 1))

                    gray_rec_image = gray_image
                    gray_rec_depth = gray_depth

                    if args.mode_type == 'RGB':
                        gray_rec_image, out_mask, _ = model(gray_image)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    elif args.mode_type == 'Depth':
                        gray_rec_depth, out_mask, _ = model(gray_depth)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    elif args.mode_type == 'XYZ':
                        xyz = sample_batched["xyz_"].cuda()
                        _, out_mask, _ = model(xyz)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    elif args.mode_type == 'RGBD':
                        gray_rec_image, gray_rec_depth, out_mask = model(gray_image, gray_depth)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    elif args.mode_type in ('Fusion0', 'Fusion6'):
                        gray_rec_image, gray_rec_depth, out_mask = model(gray_image, gray_depth)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    elif args.mode_type == 'Fusion1':
                        gray_rec_image, gray_rec_depth, out_mask, _, _, _ = \
                            model(gray_image, gray_depth)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    elif args.mode_type == 'Fusion2':
                        gray_rec_image, out_mask_rgb, merge_rgb = model_rgb(gray_image)
                        gray_rec_depth, out_mask, _, _ = model(gray_depth, merge_rgb)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    elif args.mode_type == 'Fusion3':
                        gray_depth_1ch = gray_depth[:, :1, ...]
                        output, out_mask = model(gray_image, gray_depth_1ch)
                        gray_rec_image = output[:, :3, ...]
                        gray_rec_depth = output[:, 3:, ...]
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    elif args.mode_type == 'Fusion4':
                        gray_rec_image, gray_rec_depth, out_mask_d, out_mask_rgb = \
                            model(gray_image, gray_depth)
                        out_mask_sm_d   = torch.softmax(out_mask_d, dim=1)
                        out_mask_sm_rgb = torch.softmax(out_mask_rgb, dim=1)
                        out_mask_sm = torch.where(
                            out_mask_sm_d[:, 1:, ...] > out_mask_sm_rgb[:, 1:, ...],
                            out_mask_sm_d[:, 1:, ...],
                            out_mask_sm_rgb[:, 1:, ...])
                        out_mask_sm = torch.cat([out_mask_sm_rgb[:, :1, ...], out_mask_sm], dim=1)

                    else:
                        gray_rec_image, gray_rec_depth, out_mask = model(gray_image, gray_depth)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                    if epoch % 10 == 0 and i_batch == 0:
                        save_sample_grid(
                            os.path.join(
                                args.pic_path_train,
                                args.mode_type + '_' + args.exp_name,
                                obj_name + str(epoch) + ".png"),
                            gray_image, gray_depth,
                            gray_rec_image, gray_rec_depth,
                            true_mask_cv, out_mask_sm,
                            args.mode_type)

                    out_mask_cv = out_mask_sm[:, 1, :, :].detach().cpu().numpy()
                    out_mask_averaged = torch.nn.functional.avg_pool2d(
                        out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2
                    ).cpu().detach().numpy()

                    image_score = list(
                        np.max(out_mask_averaged.reshape((out_mask_sm.shape[0], -1)), axis=1).reshape(-1))
                    anomaly_score_prediction.extend(image_score)

                    total_pixel_scores.extend(list(out_mask_cv.flatten()))
                    total_gt_pixel_scores.extend(list(true_mask_cv.flatten()))

                    for i in range(out_mask_cv.shape[0]):
                        m = out_mask_cv[i]
                        if args.sigma != 0:
                            map_max = np.max(m)
                            m = gaussian_filter(m / (map_max + 1e-8), sigma=args.sigma) * map_max
                        predictions.append(m.squeeze())
                        gts.append(true_mask_cv[i].squeeze())

                    mask_cnt += 1

            anomaly_score_prediction  = np.array(anomaly_score_prediction)
            anomaly_score_gt          = np.array(anomaly_score_gt)
            total_pixel_scores        = np.array(total_pixel_scores)
            total_gt_pixel_scores     = np.array(total_gt_pixel_scores).astype(np.uint8)

            auroc       = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
            ap          = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
            auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
            ap_pixel    = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
            aupro, _    = calculate_au_pro(gts, predictions)
            metric      = (aupro + auroc_pixel + auroc) / 3

            if epoch == args.start_epoch:
                best_epoch  = epoch
                best_metric = metric

            with open(os.path.join(
                args.record_path,
                args.mode_type + '_' + args.exp_name,
                obj_name + "run.txt"), "a") as f:
                f.write(
                    f"Epoch: {epoch}  loss: {loss_all}  "
                    f"AUC Image: {auroc}  AP Image: {ap}  "
                    f"AUC Pixel: {auroc_pixel}  AP Pixel: {ap_pixel}  "
                    f"AUPRO-0.3: {aupro}\n")

            if epoch % 50 == 0:
                print(f"Epoch: {epoch}  loss: {loss_all}  "
                      f"AUC Image: {auroc}  AP Image: {ap}  "
                      f"AUC Pixel: {auroc_pixel}  AP Pixel: {ap_pixel}  "
                      f"AUPRO-0.3: {aupro}")

            if epoch > 1 and metric >= best_metric:
                torch.save(model.state_dict(), os.path.join(
                    args.weight_save_path,
                    args.mode_type + '_' + args.exp_name,
                    run_name + "best.pckl"))
                best_metric = metric
                best_epoch  = epoch
                print("==============================")
                print(f"{obj_name} best epoch: {best_epoch}")
                print(f"AUC Image:  {auroc}")
                print(f"AP Image:   {ap}")
                print(f"AUC Pixel:  {auroc_pixel}")
                print(f"AP Pixel:   {ap_pixel}")
                print(f"AUPRO-0.3:  {aupro}")
                print("==============================")
                with open(os.path.join(
                    args.record_path,
                    args.mode_type + '_' + args.exp_name,
                    obj_name + "best.txt"), "a") as f:
                    f.write(f"best epoch: {best_epoch}\n")
                    f.write(f"AUC Image:  {auroc}\n")
                    f.write(f"AP Image:   {ap}\n")
                    f.write(f"AUC Pixel:  {auroc_pixel}\n")
                    f.write(f"AP Pixel:   {ap_pixel}\n")
                    f.write(f"AUPRO-0.3:  {aupro}\n")
                    f.write("==============================\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', type=int, nargs='+', default=-1)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--test_bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=65)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--dataset_type', default='Eyecandies', type=str,
                        choices=['Mvtec3D_AD', 'Eyecandies'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--anomaly_source_path', type=str, default='./dtd/images')
    parser.add_argument('--weight_save_path', type=str, default='./weights')
    parser.add_argument('--pic_path_train', type=str, default='./pictures')
    parser.add_argument('--record_path', type=str, default='./records')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--whole_set', action='store_true')
    parser.add_argument('--fgsg', action='store_true')
    parser.add_argument('--drop_p', type=float, default=0.5)
    parser.add_argument('--perlin_t', type=float, default=0.65)
    parser.add_argument('--low_peak', type=float, default=1.5)
    parser.add_argument('--high_peak', type=float, default=0.4)
    parser.add_argument('--min_noise', type=float, default=0.1)
    parser.add_argument('--aug_type', type=str, default="gaussian")
    parser.add_argument('--mask_type', type=str, default="depth_mask")
    parser.add_argument('--mask_t', type=float, default=0.01)
    parser.add_argument('--skew', type=bool, default=True)
    parser.add_argument('--noise_type', type=str, default="perlin")
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--sigma', type=float, default=4)
    parser.add_argument('--mode_type', default='Fusion0', type=str,
                        choices=['RGB', 'Depth', 'Fusion0', 'Fusion1', 'Fusion2',
                                 'Fusion3', 'Fusion6', 'XYZ'])
    parser.add_argument('--model_variant', default='multimodal', type=str,
                        choices=['multimodal', 'unimodal'])
    parser.add_argument('--checkpoint_yaml', type=str, default="./checkpoint/checkpoint.yaml")
    args = parser.parse_args()

    with open(args.checkpoint_yaml, 'r') as file:
        yaml_data = yaml.safe_load(file)

    args_dict = vars(args)
    args_dict.update(yaml_data)
    args = argparse.Namespace(**args_dict)

    setup_seed(args.seed)

    if args.model_variant == 'multimodal':
        from model.easynet_pfdt_fusion_fixed import ReconstructiveSubNetwork as EasyNet
    else:
        from model.easynet_pfdt import ReconstructiveSubNetwork as EasyNet

    if args.dataset_type == 'Mvtec3D_AD':
        if args.normal:
            from data.mvtec3d_dataset_normal import get_data_loader, mvtec3d_classes
        else:
            from data.mvtec3d_dataset import get_data_loader, mvtec3d_classes
        dataset_checkpoint = args.mvtec3d_ad
        obj_batch = mvtec3d_classes()
    elif args.dataset_type == 'Eyecandies':
        if args.normal:
            from data.eyecandies_dataset_normal import get_data_loader, eyecandies_classes
        else:
            from data.eyecandies_dataset import get_data_loader, eyecandies_classes
        dataset_checkpoint = args.eyecandies
        obj_batch = eyecandies_classes()

    if int(args.obj_id[0]) == -1:
        picked_classes = obj_batch
    else:
        picked_classes = [obj_batch[int(i)] for i in args.obj_id]

    print('Training classes:', picked_classes)
    train_on_device(picked_classes, args, dataset_checkpoint)