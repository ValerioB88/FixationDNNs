import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from src.utils.misc import imshow_batch

class ConvInputModel(nn.Module):
    def __init__(self, encoder_output_size):
        super(ConvInputModel, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 5, stride=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(64, 128, 5, stride=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(128, 128, 5, stride=2, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU())

        self.lin = nn.Sequential(nn.Linear(128, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, encoder_output_size),
                                 nn.Sigmoid())

    def forward(self, img):
        """convolution"""
        x = self.features(img)
        x = x.flatten(1)
        x = self.lin(x)
        return x

class FixationNet(nn.Module):
    def __init__(self, num_colors):
        super().__init__()
        self.lstm_hidden_size = 256
        self.encoder_output_size = 64
        self.encoder_img = ConvInputModel(self.encoder_output_size)
        question_size = 5 + num_colors
        self.lstm = torch.nn.LSTM(self.encoder_output_size + 2 + question_size, self.lstm_hidden_size, num_layers=1, batch_first=True)

        self.answer_net = nn.Sequential(nn.Linear(self.lstm_hidden_size, 256),
                                        nn.LeakyReLU(),
                                        nn.Linear(256, 256),
                                        nn.LeakyReLU(),
                                        nn.Linear(256, 256),
                                        nn.LeakyReLU(),
                                        nn.Linear(256, 10),
                                        nn.LogSoftmax(1))

    def forward(self, crops, coords, questions):
        batch_size, num_crops, nc, h, w = crops.shape
        c = crops.view(batch_size * num_crops, nc, h, w)
        encoded_img = self.encoder_img(c).view(batch_size, num_crops,  self.encoder_output_size)
        qq = questions.unsqueeze(dim=1)
        qq = qq.repeat(1, num_crops, 1)
        img_coords = torch.cat((encoded_img, coords, qq), dim=2)
        out, (h, c) = self.lstm(img_coords)
        # answers = self.answer_net(out.flatten(1))
        answers = self.answer_net(out[:, -1, :])

        return answers