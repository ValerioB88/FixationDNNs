import os

import torch
from sty import fg, rs
from src.utils.misc import plot_images_on_weblogger_crop
from src.utils.callbacks import *
from typing import List
from src.utils.net_utils import make_cuda
from src.utils.misc import plot_sequence

class Logs():
    value = None

    def __repl__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __copy__(self):
        return self.value

    def __deepcopy__(self, memodict={}):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rfloordiv__(self, other):
        return other // self.value

    def __rtruediv__(self, other):
        return other / self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __floordiv__(self, other):
        return self.value // other

    def __truediv__(self, other):
        return self.value / other

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __int__(self):
        return int(self.value)

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other

    def __float__(self):
        return float(self.value)

    def __pow__(self, power, modulo=None):
        return self.value ** power

    def __format__(self, format_spec):
        return format(self.value, format_spec)

class ExpMovingAverage(Logs):
    value = None
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def add(self, *args):
        if self.value is None:
            self.value = args[0]
        else:
            self.value = self.alpha * args[0] + (1 -    self.alpha) * self.value
        return self


class CumulativeAverage(Logs):
    value = None
    n = 0

    def add(self, *args):
        if self.value is None:
            self.value = args[0]

        else:
            self.value = (args[0] + self.n*self.value) / (self.n+1)
        self.n += 1
        return self



def run(data_loader, use_cuda, net, callbacks: List[Callback] = None, optimizer=None, loss_fn=None, iteration_step=None, logs=None, **kwargs):
    if logs is None:
        logs = {}
    torch.cuda.empty_cache()

    make_cuda(net, use_cuda)

    callbacks = CallbackList(callbacks)
    callbacks.set_model(net)
    callbacks.set_optimizer(optimizer)
    callbacks.set_loss_fn(loss_fn)

    callbacks.on_train_begin()

    tot_iter = 0
    epoch = 0
    logs['tot_iter'] = 0
    while True:
        callbacks.on_epoch_begin(epoch)
        logs['epoch'] = epoch

        for batch_index, data in enumerate(data_loader, 0):

            callbacks.on_batch_begin(batch_index, logs)

            loss, y_true, y_pred, logs = iteration_step(data, net, loss_fn, optimizer, use_cuda, logs, **kwargs)
            logs.update({
                # 'y_pred': y_pred,
                'loss': loss.item(),
                # 'y_true': y_true,
                'tot_iter': tot_iter,
                'stop': False})

            callbacks.on_training_step_end(batch_index, logs)
            callbacks.on_batch_end(batch_index, logs)
            if logs['stop']:
                break
            tot_iter += 1

        callbacks.on_epoch_end(epoch, logs)
        epoch += 1
        if logs['stop']:
            break

    callbacks.on_train_end(logs)
    return net, logs



def run_multidataset(data_loaders, use_cuda, net, callbacks: List[Callback] = None, optimizer=None, loss_fn=None, iteration_step=None, logs=None, **kwargs):
    if logs is None:
        logs = {}
    torch.cuda.empty_cache()

    make_cuda(net, use_cuda)

    callbacks = CallbackList(callbacks)
    callbacks.set_model(net)
    callbacks.set_optimizer(optimizer)
    callbacks.set_loss_fn(loss_fn)

    callbacks.on_train_begin()

    tot_iter = 0
    epoch = 0
    logs['tot_iter'] = 0
    while True:
        callbacks.on_epoch_begin(epoch)
        logs['epoch'] = epoch
        iters = [iter(dl) for dl in data_loaders]
        for batch_index  in range(len(iters[0])):
            datas = [next(i) for i in iters]
            callbacks.on_batch_begin(batch_index, logs)
            for it in iters:
                data = next(it)
                loss, y_true, y_pred, logs = iteration_step(data, net, loss_fn, optimizer, use_cuda, logs, **kwargs)
                logs.update({
                    # 'y_true': y_true,
                    # 'y_pred': y_pred,
                    'loss': loss.item(),
                    'tot_iter': tot_iter,
                    'stop': False})

                logs[f'{it._dataset.name_ds}__ema_acc'].add(torch.mean((y_pred == y_true).float()).item())
                logs[f'{it._dataset.name_ds}__ca_acc'].add(torch.mean((y_pred == y_true).float()).item())

            if logs['stop']:
                break
            tot_iter += 1
            callbacks.on_training_step_end(batch_index, logs)
            callbacks.on_batch_end(batch_index, logs)

        callbacks.on_epoch_end(epoch, logs)
        epoch += 1
        if logs['stop']:
            break

    callbacks.on_train_end(logs)
    return net, logs


def load_pretraining(net, pretraining, use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    if pretraining != 'vanilla':
        if os.path.isfile(pretraining):
            print(fg.red + f"Loading.. full model from {pretraining}..." + rs.fg, end="")
            ww = torch.load(pretraining, map_location='cuda' if use_cuda else 'cpu')
            if 'full' in ww:
                ww = ww['full']
            net.load_state_dict(ww)
            print(fg.red + " Done." + rs.fg)
        else:
            assert False, f"Pretraining path not found {pretraining}"

    return net

from src.train_relations.create_datasets.generate_sort_of_clevr import SortOfCLEVRGenerator, translate_answer, translate_question
from src.utils.misc import imshow_batch

def standard_step(data, model, loss_fn, optimizer, use_cuda, logs, train, **kwargs):
    images, questions, answers = data
    images = make_cuda(images, use_cuda)
    questions = make_cuda(questions, use_cuda)
    answers = make_cuda(answers, use_cuda)
    optimizer.zero_grad() if train else None
    output_batch = model(images, questions)

    loss = loss_fn(output_batch,
                   answers)
    logs['output'] = output_batch
    predicted = torch.argmax(output_batch, -1)

    # n = np.random.randint(0, len(questions))
    # imshow_batch(images[n].unsqueeze(0).cpu(), stats=kwargs['stats'])
    # print(translate_question(questions[n].cpu().numpy()))
    # print(translate_answer(answers[n]))
    # print(translate_answer(predicted[n].item()))

    logs['ema_loss'].add(loss.item())
    logs['ema_acc'].add(torch.mean((predicted == answers).float()).item())
    logs['ca_acc'].add(torch.mean((predicted == answers).float()).item())
    logs['y_true'] = answers
    logs['y_pred'] = predicted
    if 'collect_data' in kwargs and kwargs['collect_data']:
        logs['images'] = images
    if train:
        loss.backward()
        optimizer.step()

    return loss, answers, predicted, logs


def standard_step_crops(data, model, loss_fn, optimizer, use_cuda, logs, train, **kwargs):
    # ['yes', 'no', 'square', 'circle', '1', '2', '3', '4', '5', '6'][a]
    crops, coords, questions, answers, img = data
    crops = make_cuda(crops, use_cuda)
    coords = make_cuda(coords, use_cuda)
    questions = make_cuda(questions, use_cuda)
    answers = make_cuda(answers, use_cuda)
    optimizer.zero_grad() if train else None
    output_batch = model(crops, coords, questions)


############
    # model.eval()
    # model(crops, coords, questions)
    # model.train()
    # model(crops, coords, questions)
    #
    # model.eval()
    # batch_size, num_crops, nc, h, w = crops.shape
    # c = crops.view(batch_size * num_crops, nc, h, w)
    # encoded_img1 = model.encoder_img(c).view(batch_size, num_crops, model.encoder_output_size)
    # qq1 = questions.unsqueeze(dim=1)
    # qq1 = qq1.repeat(1, num_crops, 1)
    # img_coords1 = torch.cat((encoded_img1, coords, qq1), dim=2)
    # out1, (h, c) = model.lstm(img_coords1)
    # # answers = self.answer_net(out.flatten(1))
    # answers1 = model.answer_net(out1[:, -1, :])
    #
    #
    # model.train()
    # batch_size, num_crops, nc, h, w = crops.shape
    # c = crops.view(batch_size * num_crops, nc, h, w)
    # encoded_img2 = model.encoder_img(c).view(batch_size, num_crops, model.encoder_output_size)
    # qq2 = questions.unsqueeze(dim=1)
    # qq2 = qq2.repeat(1, num_crops, 1)
    # img_coords2 = torch.cat((encoded_img2, coords, qq2), dim=2)
    # out2, (h, c) = model.lstm(img_coords2)
    # # answers = self.answer_net(out.flatten(1))
    # answers2 = model.answer_net(out2[:, -1, :])

    ############
    loss = loss_fn(output_batch,
                   answers)
    logs['output'] = output_batch
    predicted = torch.argmax(output_batch, -1)

    # i = 25
    # plot_sequence(kwargs['stats'], seq=crops[i].detach().cpu(), img=img[i].detach().cpu(),
    #               crd=coords[i].detach().cpu(),
    #               q=questions[i].detach().cpu(), a=answers[i].detach().cpu(), predicted=predicted[i])
    # plt.show()

    logs['ema_loss'].add(loss.item())
    logs['ema_acc'].add(torch.mean((predicted == answers).float()).item())
    logs['ca_acc'].add(torch.mean((predicted == answers).float()).item())
    logs['y_true'] = answers
    logs['y_pred'] = predicted
    if 'collect_data' in kwargs and kwargs['collect_data']:
        logs['data'] = data
    if train:
        loss.backward()
        optimizer.step()

    return loss, answers, predicted, logs
