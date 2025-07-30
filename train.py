import sys, os
import time
from random import random

import numpy as np

import gtools
from thop import profile
from thop import clever_format

import importlib
import torch
import torch.optim as optim
import yaml
import ctools
import config
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from models.trainer import Trainer

clip_vis = 'RN50'



def main(config,checkpoint):

    #  ===================>> Setup <<=================================
    dataloader = importlib.import_module("reader." + train_config.reader)

    torch.cuda.set_device(train_config.device)
    cudnn.benchmark = True

    data = train_config.data
    save = train_config.save
    params = train_config.params

    print("===> Read data <===")


    data, folder = ctools.readfolder(
        data,
        [train_config.person],
        reverse=True
    )

    savename = folder[train_config.person]

    dataset = dataloader.loader(
        data,
        params.batch_size,
        shuffle=True,
        num_workers=2
    )

    print("===> Model building <===")

    Net = Trainer(train_config)
    Net.cuda()

    print("===> optimizer building <===")
    optimizer = optim.Adam(
        Net.parameters(),
        lr=params.lr,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params.decay_step,
        gamma=params.decay
    )

    if params.warmup:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=params.warmup,
            after_scheduler=scheduler
        )

    savepath = os.path.join(save.metapath, save.folder, checkpoint, f"{savename}")
    if not os.path.exists(savepath):
        os.makedirs(savepath)


    # =====================================>> Training << ====================================
    print("===> Training <===")

    length = len(dataset)
    total = length * params.epoch
    timer = ctools.TimeCounter(total)

    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(train_config) + '\n')
        errs = []

        for epoch in range(1, params.epoch + 1):
            since = time.time()
            for i, (data, anno) in enumerate(dataset):
                Net.train()
                # -------------- forward -------------
                data = data['face'].cuda()
                # loss
                anno =anno.cuda()
                _,loss =Net.loss(data,anno)

                # -------------- Backward ------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rest = timer.step() / 3600

                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " + \
                          f"loss:{loss} " + \
                          f"lr:{ctools.GetLR(optimizer)} " + \
                          f"rest time:{rest:.2f}h"

                    print(log)
                    outfile.write(log + "\n")
                    filename =  checkpoint + f'epoch{epoch}.pth.tar'
                    state =  {'epoch': epoch,
                              'model_state': Net.state_dict(),
                            }
                    ckpt_path = os.path.join(savepath, filename)
                    torch.save(state, ckpt_path)
                    sys.stdout.flush()
                    outfile.flush()
            time_elapsed = time.time() - since
            scheduler.step()
            
              
              

if __name__ == "__main__":
    args = config.get_config()
    checkpoint = "checkpoint-" + args.checkpoint
    for i in range(0, 4):
        train_conf = edict(yaml.load(open("config_diap.yaml"), Loader=yaml.FullLoader))
        train_config = train_conf.train
        train_config.params.lr = args.init_lr
        train_config.params.batch_size = args.batch_size
        train_config.is_AFU = True
        train_config.person = i
        train_config.device = args.device
        train_config.grade = args.grade
        train_config.a = args.a
        train_config.b = args.b
        savepath_all = os.path.join(train_config.save.metapath, train_config.save.folder, checkpoint)
        print(f"=====================>> (Begin) Training params  person:{i} << =======================")
        print(ctools.DictDumps(train_config))
        print("=====================>> (End) Traning params << =======================")
        main(train_config, checkpoint)
