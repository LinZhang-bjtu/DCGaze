import os, sys

import importlib
import torch
import yaml
from easydict import EasyDict as edict
import ctools, gtools
from models.trainer import Trainer

def main(checkpoint_path, train, test):
    test.device = train.device
    test.person = i
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test))
    print("=======================>(End) Config for test<======================")
    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)
    
    net = Trainer(train)
    net.cuda()
    state_dict = torch.load(
            checkpoint_path,
            map_location="cpu"
        )
    net.load_state_dict(state_dict['model_state'])
    net.eval()

    data = test.data
    load = test.load

    # ===============================> Read Data <=========================
    data, folder = ctools.readfolder(data, [test.person])

    testname = folder[test.person]

    print(f"==> Test: {data.label} <==")
    dataset = reader.loader(data, 1, num_workers=0, shuffle=False)

    logpath = os.path.join(train.save.metapath,
                           train.save.folder, f"{test.savename}")
    test_outfile = open(os.path.join(logpath, "test_log"), 'w')

    if not os.path.exists(test_outfile):
        os.makedirs(logpath)

    # =============================> Test <=============================

    begin = load.begin_step
    end = load.end_step
    step = load.steps

    length = len(dataset)
    accs = 0
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
        
if __name__ == "__main__":
    # Read model from train config and Test data in test config.
    args = config.get_config()
    conf = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))
    conf.test.person = args.person
    checkpoint_path = args.path

    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test_conf))
    print("=======================>(End) Config for test<======================")

    main(checkpoint_path, conf.train, conf.test)


 
