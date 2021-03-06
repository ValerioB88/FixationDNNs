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
    # mean = np.array([0.485, 0.456, 0.406])
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
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(n_hid_feat_fact * 8, n_hid_feat_fact * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(n_hid_feat_fact * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(n_hid_feat_fact * 4, n_hid_feat_fact * 2, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(n_hid_feat_fact * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(n_hid_feat_fact * 2, nc, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(n_hid_feat_fact),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(n_hid_feat_fact, nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            # state size. (nc) x 64 x 64
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

decide_saccades = torchvision.models.vgg11(pretrained=True, progress=True, num_classes=1000)

fovea_to_z = torchvision.models.vgg11(pretrained=True, progress=True, num_classes=1000)
fovea_to_z.features = nn.Sequential(*[fovea_to_z.features[i] for i in range(16)])
fovea_to_z.classifier[-1] = nn.Linear(fovea_to_z.classifier[-1].in_features, num_latent_fovea)
saccade_generator = torch.nn.LSTM(vgg_output_size, lstm_hidden_size, num_layers=1, batch_first=True)
vision_memory = torch.nn.LSTM(num_latent_fovea + 2, lstm_h_vis_mem, num_layers=1, batch_first=True)
gen = Generator(nz=lstm_h_vis_mem)

if torch.cuda.is_available():
    decide_saccades.cuda()
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
img_full_input = img_original.resize((32, 32))

img_input_resize = transform_w_resize(img_full_input).unsqueeze(0)
img_input = transform(img_full_input)
lstm_hidden = torch.zeros(1, 1, lstm_hidden_size).cuda()
lstm_cell = (0,torch.zeros(1, 1, lstm_cell_size).cuda())

##
# saccade_img_t = torch.rand(1, 3, 32, 32).cuda()


for i in range(10000):
    optimizer.zero_grad()
    # vgg_output = net(img_input_resize.cuda()).unsqueeze(dim=0)
    # (h1, c1) = saccade_generator(vgg_output, (lstm_hidden.cuda(), lstm_cell.cuda()))
    # # mapping function from hidden to x, y, w, h
    # x, y = [torch.tanh((torch.sigmoid(torch.sum(i)))).item() for i in h1[0][0].split(1024//2)]
    # x, y = torch.tensor([0.5]).cuda(), torch.tensor([0.5]).cuda()  #np.random.uniform(0.49, 0.50), np.random.uniform(0.49, 0.50)

    x, y = torch.tensor([np.random.uniform(0.2, 0.8)]).cuda(), torch.tensor([np.random.uniform(0.2, 0.8)]).cuda()
    h, w = 32, 32
    saccade_img = img_original.crop((int(x*img_original.size[0]), int(y*img_original.size[1]),int(x*img_original.size[0]) + h, int(y*img_original.size[1]) + w))

    saccade_img_t = transform(saccade_img).unsqueeze(0).cuda()
    z = fovea_to_z(saccade_img_t)

    latent_input = torch.cat([z[0], x, y]).cuda()
    (lstm_hidden, lstm_cell) = vision_memory(latent_input.detach().unsqueeze(dim=0).unsqueeze(0), (lstm_hidden.detach(), lstm_cell[1].detach()))
    rec_img = gen(lstm_hidden.squeeze(0))

    diff = torch.square(rec_img - img_input.unsqueeze(0).cuda())

    loss = torch.sum(diff)
    loss.backward()
    optimizer.step()

    print(f'{i}:{loss}')
    if False:
        img_plot = conver_tensor_to_plot(rec_img.cpu().squeeze(0).detach(), mean, std)
        plt.imshow(img_plot)
        plt.show()
        plt.imshow(conver_tensor_to_plot(img_input, mean, std))
        plt.show()
        plt.savefig('./prova.png')

