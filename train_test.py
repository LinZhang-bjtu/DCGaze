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

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main(train_config, person):

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
        test_outfile = open(os.path.join(savepath, "test_log"), 'w')
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
                    sys.stdout.flush()
                    outfile.flush()
            time_elapsed = time.time() - since
            scheduler.step()

            Net.eval()
            err = test(train_config, person, Net, epoch, test_outfile)
            errs.append(err)
        test_outfile.close()
        return errs


def test(train, i, net, epoch, test_outfile):
    test_conf = edict(yaml.load(open('config_diap.yaml'), Loader=yaml.FullLoader))
    test = test_conf.test
    test.device = train.device
    test.person = i
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test))
    print("=======================>(End) Config for test<======================")
    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    data = test.data
    load = test.load

    # ===============================> Read Data <=========================
    data, folder = ctools.readfolder(data, [test.person])

    testname = folder[test.person]

    print(f"==> Test: {data.label} <==")
    dataset = reader.loader(data, 1, num_workers=0, shuffle=False)

    logpath = os.path.join(train.save.metapath,
                           train.save.folder, f"{test.savename}")

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <=============================

    begin = load.begin_step;
    end = load.end_step;
    step = load.steps

    length = len(dataset);
    accs = 0;
    count = 0

    with torch.no_grad():
        for j, (data, label) in enumerate(dataset):

            for key in data:
                if key != 'name': data[key] = data[key].cuda()

            names = data["name"]
            gts = label.cuda()

            gazes = net(data['face'])


            for k, gaze in enumerate(gazes):
                since = time.time()
                gaze = gaze.cpu().detach().numpy()
                gt = gts.cpu().numpy()[k]
                # print(gaze, gt)

                count += 1
                accs += gtools.angular(
                    gtools.gazeto3d(gaze),
                    gtools.gazeto3d(gt)
                )

        loger = f"[epoch:{epoch}] Total Num: {count}, avg: {accs / count}"
        test_outfile.write(loger+ "\n")
        sys.stdout.flush()
        test_outfile.flush()
        print(loger)
        return accs / count



if __name__ == "__main__":
    args = config.get_config()
    checkpoint = "checkpoint-" + args.checkpoint

    errs = []
    savepath_all = ""
    for i in range(0, 4):
        train_conf = edict(yaml.load(open("config_diap.yaml"), Loader=yaml.FullLoader))
        train_config = train_conf.train
        train_config.params.lr = args.init_lr
        train_config.params.batch_size = args.batch_size
        train_config.is_mask = True
        train_config.person = i
        train_config.device = args.device
        train_config.grade = args.grade
        train_config.a = args.a
        train_config.b = args.b
        train_config.learn_prompt = args.learn_prompt
        train_config.adapter = args.adapter
        train_config.basemodel=args.basemodel
        train_config.memorybank = args.memorybank
        savepath_all = os.path.join(train_config.save.metapath, train_config.save.folder, checkpoint)
        print(f"=====================>> (Begin) Training params  person:{i} << =======================")
        print(ctools.DictDumps(train_config))
        print("=====================>> (End) Traning params << =======================")
        err = main(train_config, i)
        errs.append(err)

    combined_errs = np.array(errs)
    averages = np.mean(combined_errs, axis=0)
    min_value = np.min(averages)
    min_index = np.argmin(averages)
    outfile = open(os.path.join(savepath_all, "log"), 'w')
    for idx, avg in enumerate(averages):
        log1 = f"epoch: {idx+1}, avg_err: {avg.item()}"
        print(log1)
        outfile.write(log1 + "\n")
        sys.stdout.flush()
        outfile.flush()
    log2 = f"min_err: {min_value}, epoch: {min_index+1}"
    print(log2)
    outfile.write(log2 + "\n")
    sys.stdout.flush()
    outfile.close()



