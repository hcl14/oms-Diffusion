import os
import random
from PIL import Image
import torch
import numpy as np
from PIL import ImageDraw
import cv2
import pycocotools.mask as maskUtils
import math

import torch.utils.data as data
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    #flip = random.random() > 0.5
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform_resize(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
    # osize = [256,192]
    osize = [512, 384]
    # osize = [1024, 768]
    transform_list.append(transforms.Resize(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        # osize = [256,192]
        osize = [512, 384]
        # osize = [1024, 768]
        transform_list.append(transforms.Resize(osize, method))  
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class AlignedDataset(BaseDataset):
    def initialize(self, opt, mode='train', stage='warp'):
        self.opt = opt
        self.root = opt.dataroot
        self.warproot = opt.warproot
        self.resolution = opt.resolution
        self.stage = stage

        if self.resolution == 512:
            self.fine_height=512
            self.fine_width=384
            self.radius=8
        else:
            self.fine_height=1024
            self.fine_width=768
            self.radius=16  

        pair_txt_path = os.path.join(self.root, opt.image_pairs_txt)
        if mode == 'train' and 'train' in opt.image_pairs_txt:
            self.mode = 'train'
        else:
            self.mode = 'test'
        with open(pair_txt_path, 'r') as f:
            lines = f.readlines()

        self.P_paths = []
        self.C_paths = []
        self.C_types = []
        for line in lines:
            p_name, c_name, c_type = line.strip().split()
            p_name = p_name.replace('.png', '.jpg')
            P_path = os.path.join(self.root, c_type, 'image', p_name)
            C_path = os.path.join(self.root, c_type, 'cloth_align', c_name)
            if self.resolution == 1024:
                P_path = P_path.replace('.png', '.jpg')
            self.P_paths.append(P_path)
            self.C_paths.append(C_path)
            self.C_types.append(c_type)

        ratio_dict = None
        if self.mode == 'train':
            ratio_dict = {}
            person_clothes_ratio_txt = os.path.join(self.root, '/workspace/DressCode_train/DressCode_1024/person_clothes_ratio_upper_train.txt')
            with open(person_clothes_ratio_txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                c_name, ratio = line.strip().split()
                ratio = float(ratio)
                ratio_dict[c_name] = ratio
        self.ratio_dict = ratio_dict
        self.dataset_size = len(self.P_paths)

    ############### get palm mask ################
    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[..., np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps = [x1, y1, x2, y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / \
            (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / \
            (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1 < cos2:
            kps.extend([x3, y3, x4, y4])
        else:
            kps.extend([x4, y4, x3, y3])

        kps = np.array(kps).reshape(1, -1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask

    def get_hand_mask(self, hand_keypoints, h, w):
        # shoulder, elbow, wrist
        s_x, s_y, s_c = hand_keypoints[0]
        e_x, e_y, e_c = hand_keypoints[1]
        w_x, w_y, w_c = hand_keypoints[2]

        up_mask = np.ones((h, w, 1), dtype=np.float32)
        bottom_mask = np.ones((h, w, 1), dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            if self.resolution == 512:
                kernel = np.ones((50, 50), np.uint8)
            else:
                kernel = np.ones((100, 100), np.uint8)
            up_mask = cv2.dilate(up_mask, kernel, iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[..., np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            if self.resolution == 512:
                kernel = np.ones((30, 30), np.uint8)
            else:
                kernel = np.ones((60, 60), np.uint8)
            bottom_mask = cv2.dilate(bottom_mask, kernel, iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[..., np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)
                             == 2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, parsing, keypoints):
        h, w = parsing.shape[0:2]

        left_hand_keypoints = keypoints[[5, 6, 7], :].copy()
        right_hand_keypoints = keypoints[[2, 3, 4], :].copy()

        left_hand_up_mask, left_hand_bottom_mask = self.get_hand_mask(
            left_hand_keypoints, h, w)
        right_hand_up_mask, right_hand_bottom_mask = self.get_hand_mask(
            right_hand_keypoints, h, w)

        # mask refined by parsing
        left_hand_mask = (parsing == 15).astype(np.float32)
        right_hand_mask = (parsing == 16).astype(np.float32)

        left_palm_mask = self.get_palm_mask(
            left_hand_mask, left_hand_up_mask, left_hand_bottom_mask)
        right_palm_mask = self.get_palm_mask(
            right_hand_mask, right_hand_up_mask, right_hand_bottom_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############### get palm mask ################

    def __getitem__(self, index):
        C_type = self.C_types[index]

        # person image
        P_path = self.P_paths[index]
        print(f"P_path: {P_path}")
        P = Image.open(P_path).convert('RGB')
        P_np = np.array(P)
        params = get_params(self.opt, P.size)
        transform_for_rgb = get_transform(self.opt, params)
        P_tensor = transform_for_rgb(P)

        # person 2d pose
        pose_path = P_path.replace('/image/', '/pose_25/')+'.npy'
        print(f"pose_path .npy: {pose_path}")
        pose_data = np.load(pose_path)[0]
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx + r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = transform_for_rgb(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        Pose_tensor = pose_map

        # person openpose keypoint pose
        # openpose_path = P_path.replace('/image/', '/skeletons/')[:-4]+'.jpg'
        # openpose_mask = Image.open(openpose_path).convert('L')
        # transform_for_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # openpose_mask_tensor = transform_for_mask(openpose_mask) * 255.0
        # openpose_mask_tensor = openpose_mask_tensor[0:1, ...]

        # pose_name = img_name.replace('.jpg', '_rendered.png')
        # pose_rgb = Image.open(osp.join(self.data_path, 'openpose_img', pose_name))
        # pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        # pose_rgb = self.transform(pose_rgb)  # [-1,1]
        # pose_rgb = (pose_rgb + 1) /2. #[0,1]  <--- somewhow, works better without it

        openpose_path = P_path.replace('/image/', '/skeletons/')[:-6]+'_5.jpg'
        print(f"openpose_path: {openpose_path}")
        openpose_img = Image.open(openpose_path).convert('RGB')
        openpose_img_tensor = transform_for_rgb(openpose_img)

        # person 3d pose
        densepose_path = P_path.replace('/image/', '/densepose/')[:-4]+'.png'
        print(f"densepose_path: {densepose_path}")
        dense_mask = Image.open(densepose_path).convert('L')
        transform_for_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        dense_mask_tensor = transform_for_mask(dense_mask) * 255.0
        dense_mask_tensor = dense_mask_tensor[0:1, ...]

        # person parsing
        parsing_path = P_path.replace('/image/', '/parse-bytedance/')[:-4]+'.png'
        print(f"parsing_path: {parsing_path}")
        parsing = Image.open(parsing_path).convert('L')
        parsing_tensor = transform_for_mask(parsing) * 255.0

        parsing_np = (parsing_tensor.numpy().transpose(1, 2, 0)[..., 0:1]).astype(np.uint8)
        palm_mask_np = self.get_palm(parsing_np, pose_data)

        person_clothes_left_sleeve_mask_np = np.zeros_like(parsing_np)
        person_clothes_torso_mask_np = np.zeros_like(parsing_np)
        person_clothes_right_sleeve_mask_np = np.zeros_like(parsing_np)
        person_clothes_left_pants_mask_np = np.zeros_like(parsing_np)
        person_clothes_right_pants_mask_np = np.zeros_like(parsing_np)
        person_clothes_skirts_mask_np = np.zeros_like(parsing_np)
        neck_mask_np = np.zeros_like(parsing_np)
        left_hand_mask_np = np.zeros_like(parsing_np)
        right_hand_mask_np = np.zeros_like(parsing_np)
        hand_mask_np = np.zeros_like(parsing_np)

        if C_type == 'upper' or C_type == 'dresses':
            person_clothes_left_sleeve_mask_np = (parsing_np==21).astype(int) + \
                                                (parsing_np==24).astype(int)
            person_clothes_torso_mask_np = (parsing_np==5).astype(int) + \
                                        (parsing_np==6).astype(int)
            person_clothes_right_sleeve_mask_np = (parsing_np==22).astype(int) + \
                                                (parsing_np==25).astype(int)
            person_clothes_mask_np = person_clothes_left_sleeve_mask_np + \
                                  person_clothes_torso_mask_np + \
                                  person_clothes_right_sleeve_mask_np
            left_hand_mask_np = (parsing_np==15).astype(int)
            right_hand_mask_np = (parsing_np==16).astype(int)
            hand_mask_np = left_hand_mask_np + right_hand_mask_np
            neck_mask_np = (parsing_np==11).astype(int)
        else:
            person_clothes_left_pants_mask_np = (parsing_np==9).astype(int)
            person_clothes_right_pants_mask_np = (parsing_np==10).astype(int)
            person_clothes_skirts_mask_np = (parsing_np==13).astype(int)
            person_clothes_mask_np = person_clothes_left_pants_mask_np + \
                                  person_clothes_right_pants_mask_np + \
                                  person_clothes_skirts_mask_np

        person_clothes_mask_tensor = torch.tensor(person_clothes_mask_np.transpose(2, 0, 1)).float()
        person_clothes_left_sleeve_mask_tensor = torch.tensor(person_clothes_left_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_torso_mask_tensor = torch.tensor(person_clothes_torso_mask_np.transpose(2, 0, 1)).float()
        person_clothes_right_sleeve_mask_tensor = torch.tensor(person_clothes_right_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_left_pants_mask_tensor =  torch.tensor(person_clothes_left_pants_mask_np.transpose(2, 0, 1)).float()
        person_clothes_skirts_mask_tensor =  torch.tensor(person_clothes_skirts_mask_np.transpose(2, 0, 1)).float()
        person_clothes_right_pants_mask_tensor =  torch.tensor(person_clothes_right_pants_mask_np.transpose(2, 0, 1)).float()
        left_hand_mask_tensor = torch.tensor(left_hand_mask_np.transpose(2, 0, 1)).float()
        right_hand_mask_tensor = torch.tensor(right_hand_mask_np.transpose(2, 0, 1)).float()
        neck_mask_tensor = torch.tensor(neck_mask_np.transpose(2, 0, 1)).float()

        seg_gt_tensor = person_clothes_left_sleeve_mask_tensor * 1 + person_clothes_torso_mask_tensor * 2 + \
                        person_clothes_right_sleeve_mask_tensor * 3 +  person_clothes_left_pants_mask_tensor * 4 + \
                        person_clothes_skirts_mask_tensor * 5 + person_clothes_right_pants_mask_tensor * 6 + \
                        left_hand_mask_tensor * 7 + right_hand_mask_tensor * 8 + neck_mask_tensor * 9
        background_mask_tensor = 1 - (person_clothes_left_sleeve_mask_tensor + person_clothes_torso_mask_tensor + \
                                      person_clothes_right_sleeve_mask_tensor + person_clothes_left_pants_mask_tensor + \
                                      person_clothes_right_pants_mask_tensor + person_clothes_skirts_mask_tensor + \
                                      left_hand_mask_tensor + right_hand_mask_tensor + neck_mask_tensor)
        seg_gt_onehot_tensor = torch.cat([background_mask_tensor, person_clothes_left_sleeve_mask_tensor, \
                                         person_clothes_torso_mask_tensor, person_clothes_right_sleeve_mask_tensor, \
                                         person_clothes_left_pants_mask_tensor, person_clothes_skirts_mask_tensor, \
                                         person_clothes_right_pants_mask_tensor,  left_hand_mask_tensor, \
                                         right_hand_mask_tensor, neck_mask_tensor],0)

        if C_type == 'upper' or C_type == 'dresses':
            person_clothes_left_mask_tensor = person_clothes_left_sleeve_mask_tensor
            person_clothes_middle_mask_tensor = person_clothes_torso_mask_tensor
            person_clothes_right_mask_tensor = person_clothes_right_sleeve_mask_tensor
        else:
            person_clothes_left_mask_tensor = person_clothes_left_pants_mask_tensor
            person_clothes_middle_mask_tensor = person_clothes_skirts_mask_tensor
            person_clothes_right_mask_tensor = person_clothes_right_pants_mask_tensor

        ### preserve region mask
        ### preserve_mask1_np and preserve_mask2_np are only used for the training of warping module
        ### preserve_mask3_np is a bit different for the warping module and the try-on module
        if C_type == 'upper':
            if self.ratio_dict is None:
                preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
            else:
                pc_ratio = self.ratio_dict[self.C_paths[index].split('/')[-1][:-4]+'.png']
                if pc_ratio < 0.95:
                    preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
                elif pc_ratio < 1.0:
                    if random.random() < 0.5:
                        preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                    else:
                        preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,12,14,23,26,27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
                else:
                    if random.random() < 0.1:
                        preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                    else:
                        preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,12,14,23,26,27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)

            preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
            preserve_mask_np = np.sum(preserve_mask_np,axis=0)

            preserve_mask1_np = preserve_mask_for_loss_np + palm_mask_np
            preserve_mask2_np = preserve_mask_for_loss_np + hand_mask_np
            preserve_mask3_np = preserve_mask_np + palm_mask_np
        elif C_type == 'dresses':
            preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,12,14,23]])
            if self.stage == 'gen':
                preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,12,14,23,8,19,20]])
            preserve_mask_np = np.sum(preserve_mask_np,axis=0)
            preserve_mask_for_loss_np = preserve_mask_np

            preserve_mask1_np = preserve_mask_for_loss_np + palm_mask_np
            preserve_mask2_np = preserve_mask_for_loss_np + hand_mask_np
            preserve_mask3_np = preserve_mask_np + palm_mask_np
        else:
            preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,5,6,7,11,12,14,15,16,21,22,23,24,25,26,27,28]])
            if self.stage == 'gen':
                preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,5,6,7,11,12,14,15,16,21,22,23,24,25,26,27,28,8,19,20]])
            preserve_mask_for_loss_np = preserve_mask_np

            preserve_mask1_np = np.sum(preserve_mask_for_loss_np,axis=0)
            preserve_mask2_np = np.sum(preserve_mask_for_loss_np, axis=0)
            preserve_mask3_np = np.sum(preserve_mask_np,axis=0)

        preserve_mask1_tensor = torch.tensor(preserve_mask1_np.transpose(2,0,1)).float()
        preserve_mask2_tensor = torch.tensor(preserve_mask2_np.transpose(2,0,1)).float()
        preserve_mask3_tensor = torch.tensor(preserve_mask3_np.transpose(2,0,1)).float()

        ### used for gradient truncation during training
        preserve_legs_mask_np = np.zeros_like(parsing_np)
        preserve_left_pants_mask_np = np.zeros_like(parsing_np)
        preserve_right_pants_mask_np = np.zeros_like(parsing_np)

        pants_mask_np = (parsing_np==9).astype(np.uint8) + (parsing_np==10).astype(np.uint8)
        skirts_mask_np = (parsing_np==13).astype(np.uint8)
        if C_type == 'lower':
            if np.sum(skirts_mask_np) > np.sum(pants_mask_np):
                preserve_legs_mask_np = (parsing_np==17).astype(np.uint8) + (parsing_np==18).astype(np.uint8) + \
                                        (parsing_np==19).astype(np.uint8) + (parsing_np==20).astype(np.uint8)
            else:
                preserve_left_pants_mask_np = (parsing_np==9).astype(np.uint8)
                preserve_right_pants_mask_np = (parsing_np==10).astype(np.uint8)
        elif C_type == 'dresses':
            preserve_legs_mask_np = (parsing_np==17).astype(np.uint8) + (parsing_np==18).astype(np.uint8) + \
                                    (parsing_np==19).astype(np.uint8) + (parsing_np==20).astype(np.uint8)
        
        preserve_legs_mask_tensor = torch.tensor(preserve_legs_mask_np.transpose(2,0,1)).float()
        preserve_left_pants_mask_tensor = torch.tensor(preserve_left_pants_mask_np.transpose(2,0,1)).float()
        preserve_right_pants_mask_tensor = torch.tensor(preserve_right_pants_mask_np.transpose(2,0,1)).float()
        

        ### clothes
        C_path = self.C_paths[index]
        print(f"C_path: {C_path}")
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_for_rgb(C)

        CM_path = C_path.replace('/cloth_align/', '/cloth_align_mask-bytedance/')
        print(f"CM_path: {CM_path}")
        CM = Image.open(CM_path).convert('L')
        CM_tensor = transform_for_mask(CM)

        cloth_parsing_path = C_path.replace('/cloth_align/', '/cloth_align_parse-bytedance/')
        print(f"cloth_parsing_path: {cloth_parsing_path}")
        cloth_parsing = Image.open(cloth_parsing_path).convert('L')
        cloth_parsing_tensor = transform_for_mask(cloth_parsing) * 255.0
        cloth_parsing_tensor = cloth_parsing_tensor[0:1, ...]

        cloth_parsing_np = (cloth_parsing_tensor.numpy().transpose(1,2,0)).astype(int)
        if C_type == 'upper' or C_type == 'dresses':
            flat_clothes_left_mask_np = (cloth_parsing_np==21).astype(int)
            flat_clothes_middle_mask_np = (cloth_parsing_np==5).astype(int) + \
                                          (cloth_parsing_np==24).astype(int) + \
                                          (cloth_parsing_np==13).astype(int)
            flat_clothes_right_mask_np = (cloth_parsing_np==22).astype(int)
            flat_clothes_label_np = flat_clothes_left_mask_np * 1 + flat_clothes_middle_mask_np * 2 + flat_clothes_right_mask_np * 3
        else:
            flat_clothes_left_mask_np = (cloth_parsing_np==9).astype(int)
            flat_clothes_middle_mask_np = (cloth_parsing_np==13).astype(int)
            flat_clothes_right_mask_np = (cloth_parsing_np==10).astype(int)
            flat_clothes_label_np = flat_clothes_left_mask_np * 4 + flat_clothes_middle_mask_np * 5 + flat_clothes_right_mask_np * 6
        flat_clothes_label_np = flat_clothes_label_np / 6

        cloth_type_np = np.zeros_like(parsing_np)
        if C_type == 'upper':
            cloth_type_np = cloth_type_np + 1.0
        elif C_type == 'lower':
            cloth_type_np = cloth_type_np + 2.0
        else:
            cloth_type_np = cloth_type_np + 3.0
        cloth_type_np = cloth_type_np / 3.0
        
        flat_clothes_left_mask_tensor = torch.tensor(flat_clothes_left_mask_np.transpose(2, 0, 1)).float()
        flat_clothes_middle_mask_tensor = torch.tensor(flat_clothes_middle_mask_np.transpose(2, 0, 1)).float()
        flat_clothes_right_mask_tensor = torch.tensor(flat_clothes_right_mask_np.transpose(2, 0, 1)).float()

        flat_clothes_label_tensor = torch.tensor(flat_clothes_label_np.transpose(2, 0, 1)).float()
        cloth_type_tensor = torch.tensor(cloth_type_np.transpose(2,0,1)).float()

        WC_tensor = None
        WE_tensor = None
        AMC_tensor = None
        ANL_tensor = None
        if self.warproot:
            ### skin color
            face_mask_np = (parsing_np==14).astype(np.uint8)
            neck_mask_np = (parsing_np==11).astype(np.uint8)
            hand_mask_np = (parsing_np==15).astype(np.uint8) + (parsing_np==16).astype(np.uint8)
            leg_mask_np = (parsing_np==17).astype(int) + (parsing_np==18).astype(int)
            skin_mask_np = (face_mask_np+hand_mask_np+neck_mask_np+leg_mask_np)
            skin = skin_mask_np * P_np
            skin_r = skin[..., 0].reshape((-1))
            skin_g = skin[..., 1].reshape((-1))
            skin_b = skin[..., 2].reshape((-1))
            skin_r_valid_index = np.where(skin_r > 0)[0]
            skin_g_valid_index = np.where(skin_g > 0)[0]
            skin_b_valid_index = np.where(skin_b > 0)[0]

            skin_r_median = np.median(skin_r[skin_r_valid_index])
            skin_g_median = np.median( skin_g[skin_g_valid_index])
            skin_b_median = np.median(skin_b[skin_b_valid_index])

            arms_r = np.ones_like(parsing_np[...,0:1]) * skin_r_median
            arms_g = np.ones_like(parsing_np[...,0:1]) * skin_g_median
            arms_b = np.ones_like(parsing_np[...,0:1]) * skin_b_median
            arms_color = np.concatenate([arms_r,arms_g,arms_b],2).transpose(2,0,1)
            AMC_tensor = torch.FloatTensor(arms_color)
            AMC_tensor = AMC_tensor / 127.5 - 1.0

            # warped clothes
            warped_name = C_type + '___' + P_path.split('/')[-1] + '___' + C_path.split('/')[-1][:-4]+'.png'
            warped_path = os.path.join(self.warproot, warped_name)
            print(f"warped_path: {warped_path}")
            warped_result = Image.open(warped_path).convert('RGB')
            warped_result_np = np.array(warped_result)

            if self.resolution == 512:
                w = 384
            else:
                w = 768
            warped_cloth_np = warped_result_np[:,-2*w:-w,:]
            warped_parse_np = warped_result_np[:,-w:,:]

            warped_cloth = Image.fromarray(warped_cloth_np).convert('RGB')
            WC_tensor = transform_for_rgb(warped_cloth)

            warped_edge_np = (warped_parse_np==1).astype(np.uint8) + \
                             (warped_parse_np==2).astype(np.uint8) + \
                             (warped_parse_np==3).astype(np.uint8) + \
                             (warped_parse_np==4).astype(np.uint8) + \
                             (warped_parse_np==5).astype(np.uint8) + \
                             (warped_parse_np==6).astype(np.uint8)
            warped_edge = Image.fromarray(warped_edge_np).convert('L')
            WE_tensor = transform_for_mask(warped_edge) * 255.0
            WE_tensor = WE_tensor[0:1,...]
            preserve_mask3_tensor = preserve_mask3_tensor * (1-WE_tensor)

            arms_neck_label = (warped_parse_np==7).astype(np.uint8) * 1 + \
                              (warped_parse_np==8).astype(np.uint8) * 2 + \
                              (warped_parse_np==9).astype(np.uint8) * 3

            arms_neck_label = Image.fromarray(arms_neck_label).convert('L')
            ANL_tensor = transform_for_mask(arms_neck_label) * 255.0 / 3.0
            ANL_tensor = ANL_tensor[0:1,...]

        if C_type == 'upper':
            category_id = torch.tensor([0], dtype=torch.int32)
        elif C_type == 'lower':
            category_id = torch.tensor([1], dtype=torch.int32)
        else:
            category_id = torch.tensor([2], dtype=torch.int32)

        rand_num = random.Random(os.urandom(4)).random()
        if rand_num < 0.05: #i_drop_rate:
            drop_image_embed = 1
        else:
            drop_image_embed = 0

        if rand_num < 0.05: #i_drop_rate:
            drop_prompt_embed = 1
        else:
            drop_prompt_embed = 0

        input_dict = {
            'image': P_tensor,
            'pose': Pose_tensor,
            'openpose': openpose_img_tensor,
            'densepose': dense_mask_tensor,
            'seg_gt': seg_gt_tensor,
            'seg_gt_onehot': seg_gt_onehot_tensor,
            'person_clothes_mask': person_clothes_mask_tensor,
            'person_clothes_left_mask': person_clothes_left_mask_tensor,
            'person_clothes_middle_mask': person_clothes_middle_mask_tensor,
            'person_clothes_right_mask': person_clothes_right_mask_tensor,
            'preserve_mask': preserve_mask1_tensor,
            'preserve_mask2': preserve_mask2_tensor,
            'preserve_mask3': preserve_mask3_tensor,
            'color': C_tensor,
            'edge': CM_tensor,
            'flat_clothes_left_mask': flat_clothes_left_mask_tensor,
            'flat_clothes_middle_mask': flat_clothes_middle_mask_tensor,
            'flat_clothes_right_mask': flat_clothes_right_mask_tensor,
            'flat_clothes_label': flat_clothes_label_tensor,
            'flat_clothes_type': cloth_type_tensor,
            'c_type': C_type,
            'category_id': category_id,
            'color_path': C_path,
            'img_path': P_path,
            'preserve_legs_mask': preserve_legs_mask_tensor,
            'preserve_left_pants_mask': preserve_left_pants_mask_tensor,
            'preserve_right_pants_mask': preserve_right_pants_mask_tensor,
            'drop_image_embed': torch.tensor([drop_image_embed], dtype=torch.int32),
            'drop_prompt_embed': torch.tensor([drop_prompt_embed], dtype=torch.int32),
        }
        if WC_tensor is not None:
            input_dict['warped_cloth'] = WC_tensor
            input_dict['warped_edge'] = WE_tensor
            input_dict['arms_color'] = AMC_tensor
            input_dict['arms_neck_lable'] = ANL_tensor

        return input_dict

    def __len__(self):
        if self.mode == 'train':
            return len(self.P_paths) // (self.opt.batchSize * self.opt.num_gpus) * (self.opt.batchSize * self.opt.num_gpus)
        else:
            return len(self.P_paths)

    def name(self):
        return 'AlignedDataset'

from torch import FloatTensor
import argparse
import os
import torch

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='flow', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=1024, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=20, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str,default='/workspace/DressCode_train/DressCode_1024/')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--warproot', type=str, default='')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=1024,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=4, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        
        self.parser.add_argument('--tv_weight', type=float, default=0.1, help='weight for TV loss')

        self.parser.add_argument('--image_pairs_txt', type=str, default='/workspace/DressCode_train/DressCode_1024/train_pairs_230729.txt')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument(
            '--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
        self.parser.add_argument('--local_rank', type=int, default=0)

        self.parser.add_argument('--write_loss_frep', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int,
                                 default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=20,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument(
            '--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument(
            '--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument(
            '--lr', type=float, default=0.00005, help='initial learning rate for adam')
        self.parser.add_argument(
            '--lr_D', type=float, default=0.00005, help='initial learning rate for adam')
        self.parser.add_argument('--pretrain_checkpoint_D', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PFAFN_warp_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PFAFN_gen_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_warp_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_gen_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')

        self.parser.add_argument('--CPM_checkpoint', type=str)
        self.parser.add_argument('--CPM_D_checkpoint', type=str)

        self.parser.add_argument('--write_loss_frep_eval', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_freq_eval', type=int, default=100,
                                 help='frequency of showing training results on screen')

        self.parser.add_argument('--add_mask_tvloss', action='store_true',
                                 help='if specified, use employ tv loss for the predicted composited mask')

        # for discriminators
        self.parser.add_argument(
            '--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument(
            '--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument(
            '--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument(
            '--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true',
                                 help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true',
                                 help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0,
                                 help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--debug_test', action='store_true')
        self.parser.add_argument(
            '--image_test_pairs_txt', type=str, default='')
        self.parser.add_argument(
            '--image_pairs_txt_eval', type=str, default='')
        self.parser.add_argument('--use_preserve_mask_refine', action='store_true',
                                 help='if specified, use preserve mask to refine to the warp clothes')

        self.parser.add_argument('--repeat_num', type=int, default=6)
        self.parser.add_argument('--loss_ce', type=float, default=1)
        self.parser.add_argument('--loss_gan', type=float, default=1)

        self.parser.add_argument('--debug_train', action='store_true')
        self.parser.add_argument('--test_flip', action='store_true')

        self.parser.add_argument(
            '--first_order_smooth_weight', type=float, default=0.01)
        self.parser.add_argument(
            '--squaretv_weight', type=float, default=1)

        self.parser.add_argument('--mask_epoch', type=int, default=-1)
        self.parser.add_argument('--no_dynamic_mask', action='store_true')

        self.parser.add_argument('--resolution', type=int, default=512)
        # self.parser.add_argument('--resolution', type=int, default=1024)
        self.parser.add_argument('--dataset', type=str, default='dresscode')

        self.isTrain = True


from torch.utils import data
from torchvision.utils import save_image

class DressCodeDataLoader:
    def __init__(self, dataset):
        super(DressCodeDataLoader, self).__init__()

        train_sampler = data.sampler.RandomSampler(dataset)
        self.data_loader = data.DataLoader(
                dataset, batch_size=1, shuffle=(train_sampler is None),
                num_workers=1, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == '__main__':
    train_dataset = AlignedDataset()
    opt = TrainOptions().parse()
    train_dataset.initialize(opt, mode='train', stage='gen')
    dataloader = DressCodeDataLoader(train_dataset)

    for i in range(1):
        batch = dataloader.next_batch()
        print(batch['image'].shape)
        save_image(batch['image'], 'tests/image.png')

        # print(batch['pose'].shape)

        print(batch['openpose'].shape)
        save_image(batch['openpose'], 'tests/openpose.png')

        print(batch['densepose'].shape)
        save_image(batch['densepose'], 'tests/densepose.png')
        print(batch['seg_gt'].shape)
        save_image(batch['seg_gt'], 'tests/seg_gt.png')
        print(batch['seg_gt_onehot'].shape)
        # save_image(batch['seg_gt_onehot'], 'tests/seg_gt_onehot.png')
        print(batch['person_clothes_mask'].shape)
        save_image(batch['person_clothes_mask'], 'tests/person_clothes_mask.png')
        print(batch['person_clothes_left_mask'].shape)
        save_image(batch['person_clothes_left_mask'], 'tests/person_clothes_left_mask.png')
        print(batch['person_clothes_middle_mask'].shape)
        save_image(batch['person_clothes_middle_mask'], 'tests/person_clothes_middle_mask.png')
        print(batch['person_clothes_right_mask'].shape)
        save_image(batch['person_clothes_right_mask'], 'tests/person_clothes_right_mask.png')
        print(batch['preserve_mask'].shape)
        save_image(batch['preserve_mask'], 'tests/preserve_mask.png')
        print(batch['preserve_mask2'].shape)
        save_image(batch['preserve_mask2'], 'tests/preserve_mask2.png')
        print(batch['preserve_mask3'].shape)
        save_image(batch['preserve_mask3'], 'tests/preserve_mask3.png')
        print(batch['color'].shape)
        save_image(batch['color'], 'tests/color.png')
        print(batch['edge'].shape)
        save_image(batch['edge'], 'tests/edge.png')
        print(batch['flat_clothes_left_mask'].shape)
        save_image(batch['flat_clothes_left_mask'], 'tests/flat_clothes_left_mask.png')
        print(batch['flat_clothes_middle_mask'].shape)
        save_image(batch['flat_clothes_middle_mask'], 'tests/flat_clothes_middle_mask.png')
        print(batch['flat_clothes_right_mask'].shape)
        save_image(batch['flat_clothes_right_mask'], 'tests/flat_clothes_right_mask.png')
        print(batch['flat_clothes_label'].shape)
        save_image(batch['flat_clothes_label'], 'tests/flat_clothes_label.png')
        print(batch['flat_clothes_type'].shape)
        save_image(batch['flat_clothes_type'], 'tests/flat_clothes_type.png')

        print(batch['c_type'])
        print(batch['color_path'])
        print(batch['img_path'])
    
        print(batch['preserve_legs_mask'].shape)
        save_image(batch['preserve_legs_mask'], 'tests/preserve_legs_mask.png')
        print(batch['preserve_left_pants_mask'].shape)
        save_image(batch['preserve_left_pants_mask'], 'tests/preserve_left_pants_mask.png')
        print(batch['preserve_right_pants_mask'].shape)
        save_image(batch['preserve_right_pants_mask'], 'tests/preserve_right_pants_mask.png')
        print('---')
