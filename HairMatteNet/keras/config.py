import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='./dataset', help='path to your dataset')
parser.add_argument('--image_size', type=int, default=224, help='height/width of the input image to network')
parser.add_argument('--nf', type=int, default=32, help='number of filters for the first layer of the network')
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--outf', default='checkpoints', help='folder to save model checkpoints')
parser.add_argument('--model_path', default='./model', help="path to saved model checkpoints (to continue training)")


def get_config():
    return parser.parse_args()
