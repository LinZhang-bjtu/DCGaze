import os
import cv2
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

def Decode_MPII(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[0]

    anno.gaze3d, anno.head3d = line[5], line[6]
    anno.gaze2d, anno.head2d = line[7], line[8]
    return anno


def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno


def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno


def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    anno.id = line[3].split('/')[0]
    return anno


def Decode_RTGene(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[6]
    anno.head2d = line[7]
    anno.name = line[0]
    return anno


def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.ethtrain = Decode_ETH
    mapping.rtgene = Decode_RTGene
    return mapping


def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1) - i + 1):
            if j > len(substr) and (str1[i:i + j] in str2):
                substr = str1[i:i + j]
    return len(substr)


def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key = keys[score.index(max(score))]
    return mapping[key]


class trainloader(Dataset):
    def __init__(self, dataset):

        # Read source data
        self.data = edict()
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        if isinstance(dataset.label, list):

            for i in dataset.label:

                with open(i) as f:
                    line = f.readlines()

                if dataset.header: line.pop(0)

                self.data.line.extend(line)
        else:

            with open(dataset.label) as f:
                self.data.line = f.readlines()

            if dataset.header: self.data.line.pop(0)

        # build transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.data.line)

    def __getitem__(self, idx):

        # Read souce information
        line = self.data.line[idx]
        line = line.strip().split(" ")
        anno = self.data.decode(line)

        img = cv2.imread(os.path.join(self.data.root, anno.face.replace("\\", "/")))
        img = self.transforms(img)

        label = np.array(anno.gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        data = edict()
        data.face = img
        data.name = anno.name

        return data, label

class trainloader_byid(Dataset):
    def __init__(self, dataset):

        # Read source data
        self.data = edict()
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        # Dictionary to store lines grouped by id
        self.data.id_dict = {}

        if isinstance(dataset.label, list):

            for i in dataset.label:

                with open(i) as f:
                    line = f.readlines()

                if dataset.header: line.pop(0)

                self.data.line.extend(line)
        else:

            with open(dataset.label) as f:
                self.data.line = f.readlines()

            if dataset.header: self.data.line.pop(0)

        # Group lines by id
        for line in self.data.line:
            line = line.strip().split(" ")
            anno = self.data.decode(line)
            if anno.id not in self.data.id_dict:
                self.data.id_dict[anno.id] = []
            self.data.id_dict[anno.id].append(line)

        # build transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data.line)

    def __getitem__(self, idx):
        # Read souce information
        line = self.data.line[idx]
        line = line.strip().split(" ")
        anno = self.data.decode(line)

        img = cv2.imread(os.path.join(self.data.root, anno.face.replace("\\", "/")))
        img = self.transforms(img)

        label = np.array(anno.gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        data = edict()
        data.face = img
        data.name = anno.name
        data.id = anno.id

        # Get a random sample with the same id
        # same_id_samples = self.data.id_dict[anno.id]
        # random_sample = random.choice(same_id_samples)
        # random_anno = self.data.decode(random_sample)
        # random_img = cv2.imread(os.path.join(self.data.root, random_anno.face.replace("\\", "/")))
        # random_img = self.transforms(random_img)
        #
        # random_label = np.array(random_anno.gaze2d.split(",")).astype("float")
        # random_label = torch.from_numpy(random_label).type(torch.FloatTensor)
        #
        # random_data = edict()
        # random_data.face = random_img
        # random_data.name = random_anno.name
        # random_data.id = random_anno.id

        return data, label


class trainloader_pair(Dataset):
    def __init__(self, dataset):

        # Read source data
        self.data = edict()
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        if isinstance(dataset.label, list):

            for i in dataset.label:

                with open(i) as f:
                    line = f.readlines()

                if dataset.header: line.pop(0)

                self.data.line.extend(line)
        else:

            with open(dataset.label) as f:
                self.data.line = f.readlines()

            if dataset.header: self.data.line.pop(0)

        # build transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.data.line)

    def __getitem__(self, idx):

        idx2 = idx  # 设置初始值为 idx，确保进入循环
        while idx2 == idx:
            idx2 = np.random.randint(0, len(self.data))

        # Read souce information
        line = self.data.line[idx]
        line = line.strip().split(" ")
        anno = self.data.decode(line)


        line2 = self.data.line[idx2]
        line2 = line2.strip().split(" ")
        anno2 = self.data.decode(line2)

        img = cv2.imread(os.path.join(self.data.root, anno.face.replace("\\", "/")))
        img = self.transforms(img)

        img2 = cv2.imread(os.path.join(self.data.root, anno2.face.replace("\\", "/")))
        img2 = self.transforms(img2)

        label = np.array(anno.gaze2d.split(",")).astype("float")
        label1 = torch.from_numpy(label).type(torch.FloatTensor)

        label2 = np.array(anno2.gaze2d.split(",")).astype("float")
        label2 = torch.from_numpy(label2).type(torch.FloatTensor)

        data1 = edict()
        data1.face = img
        data1.name = anno.name

        data2 = edict()
        data2.face = img2
        data2.name = anno2.name

        return data1, data2, label1, label2



class testloader(Dataset):
    def __init__(self, dataset):

        # Read source data
        self.data = edict()
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        if isinstance(dataset.label, list):

            for i in dataset.label:

                with open(i) as f:
                    line = f.readlines()

                if dataset.header: line.pop(0)

                self.data.line.extend(line)
        else:

            with open(dataset.label) as f:
                self.data.line = f.readlines()

            if dataset.header: self.data.line.pop(0)

        # build transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.data.line)

    def __getitem__(self, idx):

        # Read souce information
        line = self.data.line[idx]
        line = line.strip().split(" ")
        anno = self.data.decode(line)

        img = cv2.imread(os.path.join(self.data.root, anno.face.replace("\\", "/")))
        img = self.transforms(img)

        label = np.array(anno.gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        data = edict()
        data.face = img
        data.name = anno.name

        return data, label


def loader(source, batch_size, shuffle=True, num_workers=0):
    dataset = trainloader(source)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return load

def combined_loader(source1,source2, batch_size, shuffle=True, num_workers=0):
    dataset1 = trainloader(source1)
    dataset2 = trainloader(source2)
    print(f"-- [Read Data]: Source: {source1.label};{source2.label}")
    print(f"-- [Read Data]: Total num: {len(dataset1)+len(dataset2)}")
    combined_dataset=torch.utils.data.ConcatDataset([dataset1,dataset2]);
    load = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return load

def finetune_loader(source, batch_size, shuffle=True, num_workers=0,seed=0):
    dataset = trainloader(source)
    adp_num = 100
    test_num = len(dataset) - adp_num
    lengths = [adp_num, test_num]
    adp_data, test_data = torch.utils.data.dataset.random_split(dataset, lengths,
                                                                generator=torch.Generator().manual_seed(seed))
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    # load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    adp_dataset = DataLoader(
        adp_data,
        batch_size,
        shuffle=True,
        num_workers=2
    )

    test_dataset = DataLoader(
        test_data,
        batch_size,
        shuffle=True,
        num_workers=2
    )
    return adp_dataset,test_dataset

def testloader(source, batch_size, shuffle=True, num_workers=0):
    dataset = trainloader(source)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.003 * dataset_size)) # 9张作为校准图像
    random.shuffle(indices)
    test_indices, cal_indices = indices[split:], indices[:split]
    test_dataset = Subset(dataset, test_indices)
    cal_dataset = Subset(dataset, cal_indices)
    test_load = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    cal_load = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Calibration num: {len(cal_dataset)}")
    print(f"-- [Read Data]: Test num: {len(test_dataset)}")
    return test_load,cal_load


class NameBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ids = self._group_by_id()

    def _group_by_id(self):
        # Group data indices by 'name'
        id_dict = {}
        for idx in tqdm(range(len(self.dataset))):
            data, _ = self.dataset[idx]
            id = data.id
            if id not in id_dict:
                id_dict[id] = []
            id_dict[id].append(idx)

        # Randomly shuffle the names for variety
        names = list(id_dict.keys())
        random.shuffle(names)
        return id_dict

    def __iter__(self):
        # Yield indices for each batch, ensuring all samples in the batch have the same name
        for name in self.ids:
            indices = self.ids[name]
            random.shuffle(indices)

            # Yield indices in chunks of batch_size
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        # Total number of batches
        total_samples = sum(len(indices) for indices in self.ids.values())
        return total_samples // self.batch_size

def loader_id(source, batch_size, shuffle=True, num_workers=0):
    dataset = trainloader_byid(source)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    sampler = NameBatchSampler(dataset, batch_size=batch_size)
    load = DataLoader(dataset, batch_size=1, num_workers=num_workers, batch_sampler=sampler)
    return load
