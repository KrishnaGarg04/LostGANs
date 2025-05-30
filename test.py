import argparse
from collections import OrderedDict
import numpy as np
import imageio  # NEW
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_v2 import *
from utils.util import *


def get_dataloader(dataset='coco', img_size=128):
    if dataset == 'coco':
        dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/val2017/',
                                        instances_json='./datasets/coco/annotations/instances_val2017.json',
                                        stuff_json='./datasets/coco/annotations/stuff_val2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        with open("./datasets/vg/vocab.json", "r") as read_file:
            vocab = json.load(read_file)
        dataset = VgSceneGraphDataset(vocab=vocab,
                                      h5_path='./datasets/vg/val.h5',
                                      image_dir='./datasets/vg/images/',
                                      image_size=(128, 128), left_right_flip=False, max_objects=30)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31

    dataloader = get_dataloader(args.dataset)

    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()

    if not os.path.isfile(args.model_path):
        print(f"Model path {args.model_path} does not exist.")
        return

    state_dict = torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    thres = 2.0
    for idx, data in enumerate(dataloader):
        real_images, label, bbox = data
        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()
        fake_images = netG.forward(z_obj, bbox.cuda(), z_im, label.squeeze(dim=-1))

        img = (fake_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        imageio.imwrite(f"{args.sample_path}/sample_{idx}.jpg", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', help='Dataset to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--sample_path', type=str, default='samples', help='Where to save generated samples')
    args = parser.parse_args()
    main(args)
