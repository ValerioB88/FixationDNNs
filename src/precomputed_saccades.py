import matplotlib.pyplot as plt
import torch
from src.external.condconv.condconv import CondConv2d
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import PIL.Image as Image
import itertools
import numpy as np

torch.manual_seed(0)

def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


class Generator(nn.Module):
    def __init__(self, n_hid_feat_fact=64, nc=3, nz=100):
        super().__init__()
        # self.ngpu = ngpu
        self.linear = nn.Sequential(
            nn.Linear(nz, 128),
            nn.Linear(128, 256),
            nn.Linear(256, nz))

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, n_hid_feat_fact * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(n_hid_feat_fact * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_hid_feat_fact * 8, n_hid_feat_fact * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(n_hid_feat_fact * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_hid_feat_fact * 4, n_hid_feat_fact * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_hid_feat_fact * 2, nc, 4, 2, 1, bias=True),
            # (ngf) x 32 x 32
        )

    def forward(self, input):
        x = self.linear(input)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)
        x = self.main(x)
        return x


lstm_hidden_size = 1024
lstm_h_vis_mem = 1024
lstm_cell_size = 1024
channels = 3
vgg_output_size = 1000
num_latent_fovea = 1024

fovea_to_z = torchvision.models.vgg11(pretrained=True, progress=True, num_classes=1000)
fovea_to_z.features = nn.Sequential(*[fovea_to_z.features[i] for i in range(16)])
fovea_to_z.classifier[-1] = nn.Linear(fovea_to_z.classifier[-1].in_features, num_latent_fovea)
saccade_generator = torch.nn.LSTM(vgg_output_size, lstm_hidden_size, num_layers=1, batch_first=True)
vision_memory = torch.nn.LSTM(num_latent_fovea + 2, lstm_h_vis_mem, num_layers=1, batch_first=True)
gen = Generator(nz=lstm_h_vis_mem)

if torch.cuda.is_available():
    saccade_generator.cuda()
    gen.cuda()
    fovea_to_z.cuda()
    vision_memory.cuda()


mean = [0.491, 0.482, 0.44]
std = [0.247, 0.243, 0.262]

transform_w_resize = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean ,
                                     std=std)
])
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean ,
                                     std=std)
])

optimizer = torch.optim.Adam(itertools.chain(*[fovea_to_z.parameters(), vision_memory.parameters(), gen.parameters()]), lr=0.0001)



img_original = Image.open('./data/hinton_face.png').resize((224, 224))
img_full_downgraded = img_original.resize((32, 32))
img_full_downgraded_t = transform(img_full_downgraded)



lstm_hidden = torch.zeros(1, 1, lstm_hidden_size).cuda()
lstm_cell = (0,torch.zeros(1, 1, lstm_cell_size).cuda())

##
# saccade_img_t = torch.rand(1, 3, 32, 32).cuda()


for i in range(10000):
    optimizer.zero_grad()
    x, y = torch.tensor([np.random.uniform(0.2, 0.8)]).cuda(), torch.tensor([np.random.uniform(0.2, 0.8)]).cuda()
    h, w = 32, 32
    saccade_img = img_original.crop((int(x*img_original.size[0]), int(y*img_original.size[1]),int(x*img_original.size[0]) + h, int(y*img_original.size[1]) + w))

    saccade_img_t = transform(saccade_img).unsqueeze(0).cuda()
    z = fovea_to_z(saccade_img_t)

    latent_input = torch.cat([z[0], x, y]).cuda()
    (lstm_hidden, lstm_cell) = vision_memory(latent_input.detach().unsqueeze(dim=0).unsqueeze(0), (lstm_hidden.detach(), lstm_cell[1].detach()))
    rec_img = gen(lstm_hidden.squeeze(0))

    diff = torch.square(rec_img - img_full_downgraded_t.unsqueeze(0).cuda())

    loss = torch.sum(diff)
    loss.backward()
    optimizer.step()

    print(f'{i}:{loss}')
    if False:
        img_plot = conver_tensor_to_plot(rec_img.cpu().squeeze(0).detach(), mean, std)
        plt.imshow(img_plot)
        plt.show()
        plt.imshow(conver_tensor_to_plot(img_full_downgraded_t, mean, std))
        plt.show()
        plt.savefig('./prova.png')

