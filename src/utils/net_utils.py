import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
from sty import fg, ef, rs
import os
from src.external.vonenet.vonenet import vonenet

class GrabNet():
    @classmethod
    def get_net(cls, network_name, imagenet_pt=False, num_classes=None, **kwargs):
        """
        @num_classes = None indicates that the last layer WILL NOT be changed.
        """
        if imagenet_pt:
            print(fg.red + "Loading ImageNet" + rs.fg)

        nc = 1000 if imagenet_pt else num_classes
        kwargs = dict(num_classes=nc) if nc is not None else dict()
        if network_name == 'vgg11':
            net = torchvision.models.vgg11(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg11bn':
            net = torchvision.models.vgg11_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16':
            net = torchvision.models.vgg16(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16bn':
            net = torchvision.models.vgg16_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg19':
            net = torchvision.models.vgg19(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg19bn':
            net = torchvision.models.vgg19_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'resnet18':
            net = torchvision.models.resnet18(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet50':
            net = torchvision.models.resnet50(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet152':
            net = torchvision.models.resnet152(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'alexnet':
            net = torchvision.models.alexnet(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'inception_v3':  # nope
            net = torchvision.models.inception_v3(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'densenet121':
            net = torchvision.models.densenet121(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'densenet201':
            net = torchvision.models.densenet201(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'googlenet':
            net = torchvision.models.googlenet(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = cls.get_other_nets(network_name, imagenet_pt, **kwargs)
            assert False if net is False else True, f"Network name {network_name} not recognized"

        return net

    @staticmethod
    def get_other_nets(network_name, num_classes, imagenet_pt, **kwargs):
        pass


def prepare_network(net, config, train=True):
    from src.utils.train_utils import load_pretraining

    pretraining_file = 'vanilla' if config.pretraining == 'ImageNet' else config.pretraining
    net = load_pretraining(net, pretraining_file, config.use_cuda)
    net.cuda() if config.use_cuda else None
    cudnn.benchmark = True
    net.train() if train else net.eval()


def make_cuda(fun, is_cuda):
    return fun.cuda() if is_cuda else fun


def print_net_info(net):
    num_trainable_params = 0
    tmp = ''
    print(fg.yellow)
    print("Params to learn:")
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            tmp += "\t" + name + "\n"
            print("\t" + name)
            num_trainable_params += len(param.flatten())
    print(f"Trainable Params: {num_trainable_params}")

    print('***Network***')
    print(net)
    print(ef.inverse + f"Network is in {('~train~' if net.training else '~eval~')} mode." + rs.inverse)
    print(rs.fg)
    print()



class MyGrabNet(GrabNet):
    @staticmethod
    def get_other_nets(network_name, imagenet_pt, **kwargs):
        net = None
        if network_name == 'cornet-rt':
            net = cornet.cornet_rt(pretrained=True, map_location='cpu', times=5)
        elif network_name == 'cornet-s':
            net = cornet.cornet_s(pretrained=True, map_location='cpu')
        elif network_name == 'clip':
            import clip
            net, _ = clip.load("ViT-B/32", device='cpu')
        if 'vonenet' in network_name:
            os.environ['HOME'] = './'
            if network_name == 'vonenet-resnet50':
                net = vonenet.get_model(model_arch='resnet50', pretrained=True)
            elif network_name == 'vonenet-cornets':
                net = vonenet.get_model(model_arch='cornets', pretrained=True)
            elif network_name == 'vonenet-alexnet':
                net = vonenet.get_model(model_arch='alexnet', pretrained=True)
            elif network_name == 'vonenet-resnet50-non-stoch':
                net = vonenet.get_model(model_arch='resnet50', pretrained=True, noise_level=0)

        if not net:
            assert False, f"Network {network_name} not recognised"
        return net




class GrabNet():
    @classmethod
    def get_net(cls, network_name, imagenet_pt=False, num_classes=None, **kwargs):
        """
        @num_classes = None indicates that the last layer WILL NOT be changed.
        """
        if imagenet_pt:
            print(fg.red + "Loading ImageNet" + rs.fg)

        nc = 1000 if imagenet_pt else num_classes
        kwargs = dict(num_classes=nc) if nc is not None else dict()
        if network_name == 'vgg11':
            net = torchvision.models.vgg11(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg11bn':
            net = torchvision.models.vgg11_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16':
            net = torchvision.models.vgg16(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16bn':
            net = torchvision.models.vgg16_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg19bn':
            net = torchvision.models.vgg19_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'resnet18':
            net = torchvision.models.resnet18(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet50':
            net = torchvision.models.resnet50(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet152':
            net = torchvision.models.resnet152(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'alexnet':
            net = torchvision.models.alexnet(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'inception_v3':  # nope
            net = torchvision.models.inception_v3(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'densenet121':
            net = torchvision.models.densenet121(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'densenet201':
            net = torchvision.models.densenet201(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'googlenet':
            net = torchvision.models.googlenet(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = cls.get_other_nets(network_name, imagenet_pt, **kwargs)
            assert False if net is False else True, f"Network name {network_name} not recognized"

        return net



