from model.backbone.vgg import vgg16
import torch
import os
from glob import glob


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Tester:
    def __init__(self, config, test_loader, num_class=10):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.num_class = num_class
        self.learning_rate = config.lr
        self.test_loader = test_loader
        self.epoch = config.epoch
        self.build_model()

    def build_model(self):
        self.net = vgg16(self.num_class)
        self.net.apply(weights_init)
        self.net.to(self.device)
        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        if not os.listdir(self.checkpoint_dir):
            print("[!] No checkpoint in ", str(self.checkpoint_dir))
            os.mkdir(self.checkpoint_dir)
            return

        model = glob(os.path.join(self.checkpoint_dir, "vgg16*.pth"))
        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print("[*] Load Model from %s: " % str(self.checkpoint_dir), str(model[-1]))

    def test(self):
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(self.test_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self.net(images)

            _, predicted = torch.max(preds.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('testing... {}/{}'.format(step, len(self.test_loader)))

        print('[*] Test accuracy on %d test images: %.2f%%' % (total, 100 * correct / total))
