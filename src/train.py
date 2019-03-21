from model.backbone.vgg import vgg16
import torch
import os
from glob import glob
import torch.nn as nn

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer:
    def __init__(self, config, train_loader, num_class=10):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.num_class = num_class
        self.learning_rate = config.lr
        self.train_loader = train_loader
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
            return

        model = glob(os.path.join(self.checkpoint_dir, "vgg16*.pth"))
        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print("[*] Load Model from %s: " % str(self.checkpoint_dir), str(model[-1]))

    def train(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        total_step = len(self.train_loader)
        num_epoch = self.epoch

        for epoch in range(self.epoch):
            for step, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epoch, step + 1, total_step, loss.item()))

            torch.save(self.net.state_dict(), '%s/vgg16-%d.pth' % (self.checkpoint_dir, epoch))
