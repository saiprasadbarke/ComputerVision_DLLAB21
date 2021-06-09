import os
import json
import random
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image, ImageDraw

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
selected_class_ids = {"1": "person", "2": "bycicle", "3": "car", "17": "cat", "18": "dog"}
statistics =  {"mean": [0.4654, 0.4446, 0.4120], "std": [0.2484, 0.2416, 0.2439]}


class DataReaderBinarySegmentation:
    """ Loads images and produces masks for binary segmentation. """
    def __init__(self, images_folder, annotations_file, transform=None, target_transform=None, no_target=False):
        print("loading dataset...")
        self.annotations = json.load(open(annotations_file, 'r'))
        self.transform = transform
        self.target_transform = target_transform
        self.root = images_folder
        self.no_target = no_target

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        size = img.size

        # the ugly seed hack is to ensure we have same random augmentation (eg random crop) for both image and maks
        new_seed = random.randint(0, 99999999)
        if self.transform is not None:
            random.seed(new_seed)
            img = self.transform(img)

        if not self.no_target:
            mask = self.make_segmentation(ann, size)

            if self.target_transform is not None:
                random.seed(new_seed)
                mask = self.target_transform(mask)

            return img, mask

        return img

    def make_segmentation(self, ann, size, to_view=False):
        mask = Image.new('L', size, 0)
        maskdraw = ImageDraw.Draw(mask)
        drawn = False
        for id in ann["annotations"].keys():
            for instance in ann["annotations"][id]:
                for segm in instance["segmentation"]:
                    if not isinstance(segm, list):
                        continue
                    pos_val = 1
                    if to_view:
                        pos_val = 255
                    maskdraw.polygon(segm, outline=pos_val, fill=pos_val)
                    drawn = True
        assert drawn
        return mask

    def view_sample(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        size = img.size
        mask = self.make_segmentation(ann, size, to_view=True)

        img.show()
        mask.show()

    def __len__(self):
        return len(self.annotations)


class DataReaderSemanticSegmentation:
    """ Loads images and produces masks for semantic segmentation. """
    def __init__(self, images_folder, annotations_file, transform=None, target_transform=None, coco_ids=False):
        print("loading dataset...")
        self.annotations = json.load(open(annotations_file, 'r'))
        self.transform = transform
        self.target_transform = target_transform
        self.root = images_folder
        if coco_ids:
            self.class_ids = {id: int(id) for indx, id in enumerate(selected_class_ids.keys())}
        else:
            self.class_ids = {id: indx+1 for indx, id in enumerate(selected_class_ids.keys())}


    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        size = img.size

        mask = self.make_segmentation(ann, size)

        # the ugly seed hack is to ensure we have same random augmentation (eg random crop) for both image and maks
        new_seed = random.randint(0, 99999999)

        if self.transform is not None:
            random.seed(new_seed)
            img = self.transform(img)

        if self.target_transform is not None:
            # masks = {id: self.target_transform(m) for id, m in masks.items()}
            random.seed(new_seed)
            mask = self.target_transform(mask)
            if not isinstance(mask, (np.ndarray, np.generic)) or not torch.is_tensor(mask):
                mask = np.asarray(mask)

        return img, mask

    def make_segmentation(self, ann, size, to_view=False):
        mask = Image.new('L', size, 0)
        if to_view:
            mask = Image.new('RGB', size, (0, 0, 0))
        maskdraw = ImageDraw.Draw(mask)
        for id, c in zip(ann["annotations"].keys(), colors):
            for instance in ann["annotations"][id]:
                for segm in instance["segmentation"]:
                    if not isinstance(segm, list):
                        continue
                    pos_val = self.class_ids[id]
                    if to_view:
                        pos_val = c
                    maskdraw.polygon(segm, outline=pos_val, fill=pos_val)
                    drawn = True
        assert drawn
        return mask

    def view_sample(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        size = img.size
        mask = self.make_segmentation(ann, size, to_view=True)

        img.show()
        mask.show()

    def __len__(self):
        return len(self.annotations)


class DataReaderCrops:
    def __init__(self, images_folder, annotations_file, transform=None, target_transform=None):
        print("loading dataset...")
        self.annotations = json.load(open(annotations_file, 'r'))
        self.transform = transform
        self.target_transform = target_transform
        self.root = images_folder

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')

        # crop image
        box = ann["bbox"][:2]+[ann["bbox"][i]+ann["bbox"][i+2] for i in range(2)]
        img = img.crop(box)
        segm = ann["relative_segmentation"]

        # make mask
        mask = Image.new('L', img.size, 0)
        maskdraw = ImageDraw.Draw(mask)
        for segm_part in segm:
            maskdraw.polygon(segm_part, outline=1, fill=1)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def view_sample(self, idx):
        img, mask = self.__getitem__(idx)

        img.show()
        Image.fromarray(np.array(mask)*255).show()

    def __len__(self):
        return len(self.annotations)


class DataReaderSingleClassSemanticSegmentationVector:
    """ Loads images and produces masks for semantic segmentation with class hot vector. """
    def __init__(self, images_folder, annotations_file, transform=None, vec_transform=None, target_transform=None, coco_ids=False):
        print("loading dataset...")
        self.annotations = json.load(open(annotations_file, 'r'))
        self.transform = transform
        self.vec_transform = vec_transform
        self.target_transform = target_transform
        self.root = images_folder
        if coco_ids:
            self.class_ids = {id: int(id) for indx, id in enumerate(selected_class_ids.keys())}
        else:
            self.class_ids = {id: indx for indx, id in enumerate(selected_class_ids.keys())}

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        size = img.size

        if random.random()<0.5:
            # Can include classes not present in the image, then mask is black
            selected_ann = random.choice([*self.class_ids.keys()])
        else:
            # Choose from classes present in the image, then mask is not black
            selected_ann = random.choice([*ann["annotations"].keys()])
        if not isinstance(selected_ann, list):
            selected_ann = [selected_ann]
        mask = self.make_segmentation(ann, size, selected_ann)

        vec = np.zeros((1, len(self.class_ids)))
        for id in selected_ann:
            vec[0, self.class_ids[id]] = 1.0

        # the ugly seed hack is to ensure we have same random augmentation (eg random crop) for both image and maks
        new_seed = random.randint(0, 99999999)

        if self.transform is not None:
            random.seed(new_seed)
            img = self.transform(img)

        if self.target_transform is not None:
            random.seed(new_seed)
            mask = self.target_transform(mask)

        if self.vec_transform is not None:
            vec = self.vec_transform(vec)

        return img, vec, mask

    def make_segmentation(self, ann, size, selected_ann, to_view=False):
        mask = Image.new('L', size, 0)
        maskdraw = ImageDraw.Draw(mask)
        drawn = False
        for id in selected_ann:
            if id in ann["annotations"].keys():
                for instance in ann["annotations"][id]:
                    for segm in instance["segmentation"]:
                        if not isinstance(segm, list):
                            continue
                        pos_val = 1
                        if to_view:
                            pos_val = 255
                        maskdraw.polygon(segm, outline=pos_val, fill=pos_val)
                        drawn = True
                assert drawn
        return mask

    def view_sample(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        size = img.size

        for selected_ann in self.class_ids.keys():
            if not isinstance(selected_ann, list):
                selected_ann = [selected_ann]
            mask = self.make_segmentation(ann, size, selected_ann, to_view=True)

            vec = np.zeros((1, len(self.class_ids)))
            for id in selected_ann:
                vec[0, self.class_ids[id]] = 1.0

            print(vec)
            mask.show(title='Mask ind:{} id:{} class:{}'.format(self.class_ids[id], id, selected_class_ids[id]))

        img.show()

    def __len__(self):
        return len(self.annotations)


class DataReaderSemanticSegmentationVector:
    """ Loads images and produces binary masks for semantic segmentation. """
    def __init__(self, images_folder, annotations_file, transform=None, vec_transform=None, target_transform=None, coco_ids=False):
        print("loading dataset...")
        self.annotations = json.load(open(annotations_file, 'r'))
        self.transform = transform
        self.vec_transform = vec_transform
        self.target_transform = target_transform
        self.root = images_folder
        if coco_ids:
            self.class_ids = {id: int(id) for indx, id in enumerate(selected_class_ids.keys())}
        else:
            self.class_ids = {id: indx for indx, id in enumerate(selected_class_ids.keys())}

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        size = img.size
        # the ugly seed hack is to ensure we have same random augmentation (eg random crop) for both image and maks
        new_seed = random.randint(0, 99999999)
        if self.transform is not None:
            random.seed(new_seed)
            img = self.transform(img)
        img = [img]*len(self.class_ids)
        img = torch.stack(img)

        masks = []
        vecs = []
        for selected_id in self.class_ids.keys():
            mask = self.make_segmentation(ann, size, selected_id)
            if self.target_transform is not None:
                random.seed(new_seed)
                mask = self.target_transform(mask)
                if not isinstance(mask, (np.ndarray, np.generic)) or not torch.is_tensor(mask):
                    mask = np.asarray(mask)
            masks.append(mask)

            vec = np.zeros((1, len(self.class_ids)))
            vec[0, self.class_ids[selected_id]] = 1.0
            if self.vec_transform is not None:
                vec = self.vec_transform(vec)
            vecs.append(vec)

        mask = np.stack(masks).squeeze(1)
        vec = torch.stack(vecs).squeeze(1)

        return img, vec, mask

    def make_segmentation(self, ann, size, id, to_view=False):
        mask = Image.new('L', size, 0)
        maskdraw = ImageDraw.Draw(mask)
        drawn = False
        if id in ann["annotations"].keys():
            for instance in ann["annotations"][id]:
                for segm in instance["segmentation"]:
                    if not isinstance(segm, list):
                        continue
                    pos_val = 1
                    if to_view:
                        pos_val = 255
                    maskdraw.polygon(segm, outline=pos_val, fill=pos_val)
                    drawn = True
            assert drawn
        return mask

    def view_sample(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        img.show()
        size = img.size

        for selected_id in self.class_ids.keys():
            mask = self.make_segmentation(ann, size, selected_id, to_view=True)
            mask.show()

            vec = np.zeros((1, len(self.class_ids)))
            vec[0, self.class_ids[selected_id]] = 1.0
            print(vec)

    def __len__(self):
        return len(self.annotations)

