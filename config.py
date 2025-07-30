import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--batch_size', type=int, default=16, 
                      help='# number of images in each batch of data')
data_arg.add_argument('--config', type=int, default="config_diap.yaml", 
                      help='# config file path')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--grade', type=int, default=5,
                      help='numbers of similarity grades')
train_arg.add_argument('--a', type=float, default=0.1,
                      help='loss_mask')
train_arg.add_argument('--b', type=float, default=0.1,
                      help='loss_align')
train_arg.add_argument('--is_pretrain', type=str2bool, default=True,
                       help='Whether to use the pre-trained model')
train_arg.add_argument('--is_AFU', type=str2bool, default=True,
                       help='Whether to add clip image feature')
train_arg.add_argument('--init_lr', type=float, default=0.0001,  
                       help='Initial learning rate value')
train_arg.add_argument('--checkpoint', type=str, default='test',
                       help='Savename of the checkpoint')
train_arg.add_argument('--device', type=int, default=0,
                       help='Number of GPU to use')

# test params
test_arg = add_argument_group('Test Params')
test_arg.add_argument('--path', type=str, default='/',
                       help='Savepath of the checkpoint')
test_arg.add_argument('--person', type=int, default=0,
                       help='Id of the test person')
def get_config():
    config, unparsed = parser.parse_known_args()
    return config
