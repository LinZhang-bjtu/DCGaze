# Differential Contrastive Learning for Gaze Estimation (ACM MM 2025)

## Please follow the steps to use our codes 
    1、 Prepare the datasets which contians Image folder and Label folder
    2、 Modify the "config_xxx.yaml" file
    3、 Run the following commands.

### For train:
    python train.py --config config_xxx.yaml --checkpoint save_name

### For test:
    python test.py --config config_xxx.yaml --path checkpoint_path --person test_person

### If you want to eval every epoch, you can run
    python train_test.py --config config_xxx.yaml --checkpoint save_name


## Two versions
    We announced two versions of our DCGaze: **DCGaze-Base** and **DCGaze-AFU**
    If you want to use DCGaze-Base, set the is_AFU of args false.
    If you want to use DCGaze-AFU, set the is_AFU of args true.

