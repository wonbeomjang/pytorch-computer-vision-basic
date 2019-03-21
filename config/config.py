import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00004, help='learning rate')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--dataset', default='dataset', help="")

def get_config():
    return parser.parse_args()