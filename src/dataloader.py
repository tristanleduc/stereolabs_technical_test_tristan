import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl
import cv2
import json

def rgb(triplet):
    _NUMERALS = '0123456789abcdefABCDEF'
    _HEXDEC = {v: int(v, 16) for v in (x+y for x in _NUMERALS for y in _NUMERALS)}
    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]

def loadAde20K(file):
    fileseg = file.replace('.jpg', '_seg.png')
    with Image.open(fileseg) as io:
        seg = np.array(io)

    R = seg[:, :, 0]
    G = seg[:, :, 1]
    B = seg[:, :, 2]
    ObjectClassMasks = (R // 10).astype(np.int32) * 256 + (G.astype(np.int32))

    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat

    level = 0
    PartsClassMasks = []
    PartsInstanceMasks = []
    while True:
        level += 1
        file_parts = file.replace('.jpg', '_parts_{}.png'.format(level))
        if os.path.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io)
            R = partsseg[:, :, 0]
            G = partsseg[:, :, 1]
            PartsClassMasks.append((np.int32(R) // 10) * 256 + np.int32(G))
            PartsInstanceMasks = PartsClassMasks
        else:
            break

    objects = {}
    parts = {}
    attr_file_name = file.replace('.jpg', '.json')
    if os.path.isfile(attr_file_name):
        try:
            with open(attr_file_name, 'r', errors='ignore') as f:
                input_info = json.load(f)
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError: Could not decode {attr_file_name}. Skipping file.")
            input_info = None

        contents = input_info['annotation']['object']
        instance = np.array([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name = [x['name'] for x in contents]
        partlevel = np.array([int(x['parts']['part_level']) for x in contents])
        ispart = np.array([p > 0 for p in partlevel])
        iscrop = np.array([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = np.array(p['x'])
            p['y'] = np.array(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in list(np.where(ispart == 0)[0])]

        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {'img_name': file, 'segm_name': fileseg,
            'class_mask': ObjectClassMasks, 'instance_mask': ObjectInstanceMasks, 
            'partclass_mask': PartsClassMasks, 'part_instance_mask': PartsInstanceMasks, 
            'objects': objects, 'parts': parts}

class CityscapesSkyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform
        self.target_transform = target_transform

        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('_leftImg8bit.png'):
                    self.image_paths.append(os.path.join(root, file))
                    mask_file = file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    mask_path = os.path.join(mask_dir, os.path.relpath(root, image_dir), mask_file)
                    self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        sky_mask = (mask == 23).astype(np.uint8) * 255
        sky_mask = Image.fromarray(sky_mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            sky_mask = self.target_transform(sky_mask)

        return image, sky_mask

class ADE20KSkyDataset(Dataset):
    def __init__(self, image_dir, index_file, transform=None, target_transform=None, split='train'):
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.mask_paths = []

        root_dir = os.path.dirname(os.path.dirname(image_dir))
        with open(os.path.join(root_dir, index_file), 'rb') as f:
            index_ade20k = pkl.load(f)

        sky_label = 'sky'
        sky_label_index = None

        for idx, name in enumerate(index_ade20k['objectnames']):
            if name == sky_label:
                sky_label_index = idx
                break

        if sky_label_index is None:
            raise ValueError("Sky label not found in objectnames.")
        else:
            sky_files = []
            for i in range(len(index_ade20k['filename'])):
                if index_ade20k['objectPresence'][sky_label_index, i] > 0:
                    root_dir = os.path.dirname(os.path.dirname(image_dir))
                    root_dir = os.path.dirname(root_dir)
                    file_path = os.path.join(root_dir, index_ade20k['folder'][i], index_ade20k['filename'][i])
                    if split == 'train' and 'train' in file_path:
                        sky_files.append(file_path)
                    elif split == 'val' and 'val' in file_path:
                        sky_files.append(file_path)

            print(f"Number of files containing skies in {split} set: ", len(sky_files))

        for filename in sky_files:
            self.image_paths.append(filename)
            mask_path = filename.replace('.jpg', '_seg.png')
            self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        info = loadAde20K(image_path)
        image = Image.open(image_path).convert("RGB")
        class_mask = info['class_mask']

        sky_label_index = 2420
        sky_mask = np.where(class_mask == sky_label_index, 255, 0).astype(np.uint8)

        if sky_mask.ndim == 3:
            sky_mask = sky_mask[:, :, 0]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            sky_mask = self.target_transform(Image.fromarray(sky_mask))

        return image, sky_mask