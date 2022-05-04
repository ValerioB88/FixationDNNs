import os
# import cornet
import src.external.vonenet.vonenet as vonenet
import numpy as np
from src.utils.net_utils import GrabNet
import torch
import matplotlib.pyplot as plt
from sty import fg, bg, ef, rs
from torchvision.transforms import transforms
import neptune.new as neptune
from neptune.new.types import File
import cv2

class RandomBackground(torch.nn.Module):
    def __init__(self, color_to_randomize=0):
        super().__init__()
        self.color_to_randomize = color_to_randomize

    def forward(self, input):
        i = np.array(input)
        s = len(i[i == self.color_to_randomize])

        i[i == self.color_to_randomize] = np.repeat([np.random.randint(0, 255, 3)], s/3, axis=0).flatten()
        return transforms.ToPILImage()(i)


def save_fig_pair(path, set, n=4):
    fig, ax = plt.subplots(n, 2)
    if np.ndim(ax) == 1:
        ax = np.array([ax])
    for idx, axx in enumerate(ax):
        axx[0].imshow(set[idx][0])
        axx[1].imshow(set[idx][1])
    [x.axis('off') for x in ax.flatten()]
    plt.gcf().set_size_inches([2.4, 5])
    plt.savefig(path)


# def imshow_batch(inp, stats=None, labels=None, title_more='', maximize=True, ax=None):
#     if stats is None:
#         mean = np.array([0, 0, 0])
#         std = np.array([1, 1, 1])
#     else:
#         mean = stats['mean']
#         std = stats['std']
#     """Imshow for Tensor."""
#
#     cols = np.min([5, len(inp)])
#     rows = int(np.ceil(np.shape(inp)[0] / cols))
#     if ax is None:
#         fig, ax = plt.subplots(rows, cols)
#     if not isinstance(ax, np.ndarray):
#         ax = np.array(ax)
#     ax = ax.flatten()
#     mng = plt.get_current_fig_manager()
#     try:
#         mng.window.showMaximized() if maximize else None
#     except AttributeError:
#         print("Tkinter can't maximize. Skipped")
#     big_image = np.zeros((inp.shape[-1]*rows+cols-1, inp.shape[-1]*cols+cols-1, 3))
#     for idx, image in enumerate(inp):
#         img = conver_tensor_to_plot(image, mean, std)
#         big_image[idx % cols *inp.shape[-1]:idx % cols*inp.shape[-1] + inp.shape[-1], idx % rows * inp.shape[-1]: idx % rows * inp.shape[-1] + inp.shape[-1]] = img
#
#


def imshow_batch(inp, stats=None, labels=None, title_more='', maximize=True, ax=None):
    if stats is None:
        mean = np.array([0, 0, 0])
        std = np.array([1, 1, 1])
    else:
        mean = stats['mean']
        std = stats['std']
    """Imshow for Tensor."""

    cols =  int(np.ceil(np.sqrt(len(inp))))
    if ax is None:
        fig, ax = plt.subplots(cols, cols)
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)
    ax = ax.flatten()
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized() if maximize else None
    except AttributeError:
        print("Tkinter can't maximize. Skipped")
    for idx, image in enumerate(inp):
        image = conver_tensor_to_plot(image, mean, std)
        ax[idx].clear()
        ax[idx].axis('off')
        if len(np.shape(image)) == 2:
            ax[idx].imshow(image, cmap='gray', vmin=0, vmax=1)
        else:
            ax[idx].imshow(image)
        if labels is not None and len(labels) > idx:
            if isinstance(labels[idx], torch.Tensor):
                t = labels[idx].item()
            else:
                t = labels[idx]
            ax[idx].set_title(str(labels[idx]) + ' ' + (title_more[idx] if title_more != '' else ''))
    plt.subplots_adjust(top=1,
                        bottom=0.01,
                        left=0,
                        right=1,
                        hspace=0.01,
                        wspace=0.01)
    plt.tight_layout()
    plt.pause(0.1)

    return ax


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image

def config_to_path_train_RN(config):
    return f"{config.network_name}" #_{config.type_rel}"

def config_to_path_train_FixationNet(config):
    return f"{config.network_name}_{config.type_rel}"


def config_to_path_train(config):
    return f"{config.folder}//" \
           + f'{config.network_name}'

class RandomPixels(torch.nn.Module):
    def __init__(self, background_color=(0, 0, 0), line_color=(255, 255, 255)):
        super().__init__()
        self.background_color = background_color
        self.line_color = line_color

    def forward(self, input):
        i = np.array(input)
        i = i.astype(np.int16)
        s_line = len(i[i == self.line_color])
        i[i == self.line_color] = np.repeat([1000, 1000, 1000], s_line/3, axis=0).flatten()

        s = len(i[i == self.background_color])
        i[i == self.background_color] = np.random.randint(0, 255, s)

        s_line = len(i[i == [1000, 1000, 1000]])
        i[i == [1000, 1000, 1000]] = np.repeat([0, 0, 0], s_line / 3, axis=0).flatten()
        i = i.astype(np.uint8)

        return transforms.ToPILImage()(i)


def weblog_dataset_info(dataloader, log_text='', dataset_name=None, weblogger=1, plotter=None, num_batches_to_log=2):
    stats = {}
    if plotter is None:
        plotter = plot_images_on_weblogger
    if 'stats' in dir(dataloader.dataset):
        dataset = dataloader.dataset
        dataset_name = dataset.name_ds
        stats = dataloader.dataset.stats
    else:
        dataset_name = 'no_name' if dataset_name is None else dataset_name
        stats['mean'] = [0.5, 0.5, 0.5]
        stats['std'] = [0.2, 0.2, 0.2]
        Warning('MEAN, STD AND DATASET_NAME NOT SET FOR NEPTUNE LOGGING. This message is not referring to normalizing in PyTorch')

    if isinstance(weblogger, neptune.Run):
        weblogger['Logs'] = f'{dataset_name} mean: {stats["mean"]}, std: {stats["std"]}'

    for idx, data in enumerate(dataloader):
        images, labels, *more = data
        plotter(dataset_name, stats=stats,  images=images,  more=more, weblogger=weblogger, log_text=log_text, text=labels, data=data, batch_num=idx)
        if idx + 1 >= num_batches_to_log:
            break

def convert_normalized_tensor_to_plottable_array(tensor, mean, std, text):
    image = conver_tensor_to_plot(tensor, mean, std)

    canvas_size = np.shape(image)

    font_scale = np.ceil(canvas_size[1])/150
    font = cv2.QT_FONT_NORMAL
    umat = cv2.UMat(image * 255)
    umat = cv2.putText(img=cv2.UMat(umat), text=text, org=(0, int(canvas_size[1] - 3)), fontFace=font, fontScale=font_scale, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=6)
    umat = cv2.putText(img=cv2.UMat(umat), text=text, org=(0, int(canvas_size[1] - 3)),
                fontFace=font, fontScale=font_scale, color=[255, 255, 255], lineType=cv2.LINE_AA, thickness=1)
    image = cv2.UMat.get(umat)
    image = np.array(image, np.uint8)
    return image



def plot_images_on_weblogger(dataset_name, stats, weblogger=2, **kwargs):# images, labels, more, log_text, weblogger=2):
    images, text, more, log_text = kwargs['images'], kwargs['text'], kwargs['more'], kwargs['log_text']
    plot_images = images[0:np.max((4, len(images)))]
    metric_str = 'Debug/{} example images: [{}]'.format(log_text, dataset_name)

    if isinstance(weblogger, neptune.Run):
        [weblogger[metric_str].log
                           (File.as_image(convert_normalized_tensor_to_plottable_array(im, stats['mean'], stats['std'], text=t)/255))
         for im, t in zip(plot_images, text)]

from src.train_relations.create_datasets.generate_sort_of_clevr import translate_answer, translate_question
def plot_images_on_weblogger_crop(dataset_name, stats,  weblogger, *args, **kwargs):
    data, text, log_text, batch_num = kwargs['data'], kwargs['text'], kwargs['log_text'], kwargs['batch_num']
   ##
    crops, coords, questions, answers, image = data
    for i in range(np.min((10, len(crops)))):
        seq = crops[i]
        img = image[i]
        crd = coords[i]
        q = questions[i]
        a = answers[i]
        fig = plot_sequence(stats, seq, img, crd, q, a)
        metric_str = f'Debug/{log_text} [{dataset_name}] bn{batch_num}:{i}'
        weblogger[metric_str].upload(neptune.types.File.as_image(fig))
    plt.close('all')

def plot_sequence(stats, seq, img, crd, q, a, predicted=None):
    sequence = [convert_normalized_tensor_to_plottable_array(img, stats['mean'], stats['std'], text='')] + [convert_normalized_tensor_to_plottable_array(s, stats['mean'], stats['std'], text='') for s, c in zip(seq, crd)]
    text = [""] + ["\n".join([f'{xy}: {i:.2f}' for i, xy in zip(c.numpy(), ['x', 'y'])]) for c in crd]
    fig, ax = plt.subplots(1, len(seq) + 1)
    [axx.imshow(seq) for axx, seq in zip(ax, sequence)]
    [axx.axes.yaxis.set_visible(False) for axx in ax]
    [axx.set_xticks([]) for axx in ax]
    [axx.set_title(t) for axx, t in zip(ax, text)]
    ax[3].set_xlabel(f'Question: {translate_question(q, print_color=False)},\nAnswer: {translate_answer(a)}' + (f'\nPredicted: {translate_answer(predicted)}' if predicted else ''))
    return fig
##

def plot_crop_corr_incorr(self, logs):
    def plot_idxs(idx, metric_str):
        counter = 0
        for i in idx:
            fig = plot_sequence(self.dataset.stats, crops[i], img[i], coords[i], questions[i], answers[i], predicted=logs['y_pred'][i].item())

            self.weblogger[metric_str + f':{counter}'].upload(neptune.types.File.as_image(fig)) if self.weblogger else None
            counter += 1
            if counter >= self.max_images:
                break
            plt.close('all')

    crops, coords, questions, answers, img = logs['data']

    corr = logs['y_true'] == logs['y_pred']
    metric_str = f'Debug/EPOCH {logs["epoch"]}, test n.{self.counter}/CORR [{self.dataset.name_ds}] {self.text}'
    plot_idxs((corr).nonzero().squeeze(), metric_str)

    metric_str = f'Debug/EPOCH {logs["epoch"]}, test n. {self.counter}/INCORR [{self.dataset.name_ds}] {self.text}'
    plot_idxs((~corr).nonzero().squeeze(), metric_str)

