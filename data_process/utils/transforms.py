import torch
import numpy as np

def cam2pixel_o(cam_coord, f, c):
    x = cam_coord[:, 0] * f[0] + c[0]
    y = cam_coord[:, 1] * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def cam2pixel_o(cam_coord, f, c):
    x = cam_coord[:, 0] * f[0] + c[0]
    y = cam_coord[:, 1] * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def cam2pixel_t(cam_coord0, f, c):
    for i in range(cam_coord0.shape[0]):
        cam_coord=cam_coord0[i]
        cam_coord_d=cam_coord0[i].clone().detach()
        x = cam_coord[:, 0] / (cam_coord_d[:, 2] + 1e-8) * f[i,0] + c[i,0]
        y = cam_coord[:, 1] / (cam_coord_d[:, 2] + 1e-8) * f[i,1] + c[i,1]
        z = cam_coord[:, 2]
        img_coord = torch.cat((x[:,None], y[:,None], z[:,None]),1)
        cam_coord0[i]=img_coord
    return cam_coord0

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T.reshape([3,1]))
    return cam_coord

def cam2world(cam_coord, R, T):
    world_coord = np.dot(np.linalg.inv(R), cam_coord) + T
    return world_coord

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped
