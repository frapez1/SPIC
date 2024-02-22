"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import sys

import torch as th
import torch.distributed as dist
import torchvision as tv
import torch.nn.functional as F
import random
from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
print(th.cuda.is_available())

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    print("Setting seeds ...... \n")
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic=  True

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        generated_semantic=args.generated_semantic,
        image_path=args.image_path,
        coarse_path=args.coarse_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False,
        no_instance=args.no_instance 
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    compressed_path = os.path.join(args.results_path, 'compressed')
    os.makedirs(compressed_path, exist_ok=True)
    

    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        image = batch.cuda()#((batch + 1.0) / 2.0).cuda()
        
        label = (cond['label_ori'].float() / 255.0).cuda()
        model_kwargs = preprocess_input(image, cond, num_classes=args.num_classes, large_size=args.large_size, small_size=args.small_size, compression_type=args.compression_type, compression_level=args.compression_level)
        compressed_img = (model_kwargs['compressed']).cuda()

        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.max_iter_time != args.diffusion_steps:
            sample = sample_fn(
                model,
                (args.batch_size, 3, image.shape[2], image.shape[3]),
                noise = F.interpolate(compressed_img, (image.shape[2], image.shape[3]), mode="bilinear"),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True
            )
        else:
            sample = sample_fn(
                model,
                (args.batch_size, 3, image.shape[2], image.shape[3]),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True
            )
        sample = (sample + 1) / 2.0

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.compression_type == 'down+bpg':
            compressed_img = F.interpolate(image, (args.small_size, args.large_size), mode="bilinear")
        
        for j in range(image.shape[0]):
            save_compressed(((compressed_img[j] + 1.0) / 2.0), os.path.join(compressed_path, cond['path'][j].split('/')[-1].split('.')[0]), args)
            tv.utils.save_image(((image[j] + 1.0) / 2.0), os.path.join(image_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(sample[j], os.path.join(sample_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(label[j]*255.0/35.0, os.path.join(label_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

        if len(all_samples) * args.batch_size > args.num_samples:
            break

    dist.barrier()
    logger.log("sampling complete")

def gaussian_kernel(kernel_size=3, sigma=1.0):
    # Create a 1D Gaussian kernel
    x = th.linspace(-sigma, sigma, kernel_size)
    one_d_kernel = th.exp(-0.5 * x**2)
    one_d_kernel /= one_d_kernel.sum()
    
    # Create a 2D Gaussian kernel
    two_d_kernel = one_d_kernel[:, None] * one_d_kernel[None, :]
    two_d_kernel = two_d_kernel / two_d_kernel.sum()
    
    return two_d_kernel


def preprocess_input(image, comp_, num_classes, large_size, small_size, compression_type, compression_level):
    # move to GPU and change data types
    comp_['label'] = comp_['label'].long()

    # create one-hot label map
    label_map = comp_['label']
    bs, _, h, w = label_map.size()
    if num_classes == 19:
            nc = num_classes+1
    else:
        nc = num_classes
    input_label = th.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    if num_classes == 19:
            input_semantics = input_semantics[:, :-1, :, :] 

    # concatenate instance map if it exists
    if 'instance' in comp_:
        inst_map = comp_['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    cond = {key: value for key, value in comp_.items() if key not in ['label', 'instance', 'path', 'label_ori', 'coarse']}
    cond['y'] = input_semantics

    if 'coarse' in comp_:
        cond["compressed"] = comp_["coarse"]
    else:
        if compression_type == 'down+bpg':
            # print(image.shape)
            image = F.interpolate(((image + 1.0) / 2.0), (small_size, large_size), mode="area")
            batch_size, _, new_height, new_width = image.shape
            bpg_image_list = []
            for i in range(batch_size):
                cv2.imwrite("compressed_bpg.png", cv2.cvtColor(image[i].cpu().numpy().transpose(1,2,0)*255, cv2.COLOR_BGR2RGB))
                os.system(f"bpgenc -c ycbcr -q  {int(compression_level)} -o compressed_bpg.bpg compressed_bpg.png")
                os.system("bpgdec -o compressed_bpg.png compressed_bpg.bpg")
                decompressed_image = cv2.imread("compressed_bpg.png")
                decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_BGR2RGB)
                tensor_image = th.from_numpy(decompressed_image).permute(2, 0, 1)/255.0
                bpg_image_list.append(tensor_image)
            compressed = th.stack(bpg_image_list)
            cond['compressed'] = (2*compressed-1).cuda()
        else:
            cond['compressed'] = F.interpolate(image, (small_size, large_size), mode="area")

    return cond


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0,
        input_img=None,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def save_compressed(compressed, path, args):
    if args.compression_type == 'down+bpg':
        tv.utils.save_image(compressed, path + '.png')
        os.system(f"bpgenc -c ycbcr -q  {int(args.compression_level)} -o {path}.bpg {path}.png")
        os.system(f"bpgdec -o {path}.png {path}.bpg")
    else:
        tv.utils.save_image(compressed, path + '.png')

if __name__ == "__main__":
    set_random_seed(0)
    main()
