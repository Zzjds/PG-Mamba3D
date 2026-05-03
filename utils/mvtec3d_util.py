import tifffile as tiff
import numpy as np
import torch


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0)
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]


def normalizeVector(v):
    length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    v = v/length
    return v

def depth2normal(depth):
    h, w = np.shape(depth)
    normals = np.zeros((h, w, 3))

    for x in range(1, h-1):
        for y in range(1, w-1):

            dzdx = (float(depth[x+1, y]) - float(depth[x-1, y])) / 2.0
            dzdy = (float(depth[x, y+1]) - float(depth[x, y-1])) / 2.0

            d = (-dzdx, -dzdy, 1.0)

            n = normalizeVector(d)

            normals[x,y] = n
    return normals
