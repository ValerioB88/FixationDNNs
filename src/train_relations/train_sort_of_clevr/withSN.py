import torch.nn
from src.utils.train_utils import run
from src.utils.Config import Config
from src.utils.net_utils import GrabNet, prepare_network
from src.utils.dataset_utils import add_compute_stats, MyImageFolder, RunTimeSOCLEVR, StaticDataSOCLEVR, StaticDataSOCLVR2, StaticDataSOCLEVR_saccades
from src.utils.train_utils import ExpMovingAverage, CumulativeAverage, standard_step_crops
from torch.utils.data import DataLoader
from src.utils.callbacks import *
from src.utils.misc import *
from src.models.FixationNet import FixationNet
import torch.nn.functional as F
import pickle
import torchvision

config = Config(stop_when_train_acc_is=99,
                cuda_device=0,
                project_name='Train-Sort-of-CLEVR',
                network_name='FixationNet',
                batch_size=128,
                weblogger=2,  #set to "2" if you want to log into neptune client
                pretraining='vanilla',
                continue_train=False,
                learning_rate=0.001,
                num_colors=6,
                # clip_grad_norm=0.5,
                weight_decay=0.0,
                img_size=128,
                type_rel='rel',  # 'rel', 'non_rel', 'both'
                is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False)

torch.cuda.set_device(config.cuda_device)
config.net = FixationNet(num_colors=config.num_colors)
config.model_output_filename = './models/' + config_to_path_train_FixationNet(config) + '.pt'

if config.continue_train:
    config.pretraining = config.model_output_filename

prepare_network(config.net, config)

config.loss_fn = torch.nn.functional.nll_loss#torch.nn.CrossEntropyLoss()
config.optimizer = torch.optim.Adam(config.net.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)

# train_dataset = add_compute_stats(RunTimeSOCLEVR)(img_size=config.img_size, num_colors=config.num_colors, stats={'mean':[0, 0, 0], 'std':[1, 1, 1]})

#stats={'mean':[0, 0, 0], 'std':[1, 1, 1]})
# train_dataset = add_compute_stats(StaticDataSOCLVR2)(folder='./data/sort-of-clevr2/sort-of-clevr.pickle', stats={'mean':[0, 0, 0], 'std':[1, 1, 1]})#size=9800, img_size=config.img_size, num_colors=config.num_colors) #stats={'mean':[0, 0, 0], 'std':[1, 1, 1]})
#
#
train_dataset = StaticDataSOCLEVR_saccades(path=None, type_rel=config.type_rel, num_q_per_img=-1, num_images=9800, img_size=config.img_size, num_colors=config.num_colors)
train_dataset.stats = {'mean': [0, 0, 0], 'std': [1, 1, 1]}
train_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset.name_ds = config.type_rel

train_loader = DataLoader(train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                            timeout=0 if config.use_cuda and not config.is_pycharm else 0)

weblog_dataset_info(train_loader, plotter=plot_images_on_weblogger_crop, weblogger=config.weblogger, num_batches_to_log=1) if config.weblogger else None
test_datasets = []


test_rel = StaticDataSOCLEVR_saccades(type_rel=config.type_rel, num_q_per_img=-1, num_images=200, img_size=config.img_size, num_colors=config.num_colors)
test_rel.stats = {'mean': [0, 0, 0], 'std': [1, 1, 1]}
test_rel.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_rel.name_ds = config.type_rel

test_datasets.append(test_rel)

# test_datasets.append(add_compute_stats(StaticDataSOCLEVR_saccades)(name_ds='relational', type_rel='rel', num_q_per_img=-1, num_images=200, img_size=config.img_size, num_colors=config.num_colors, stats=train_dataset.stats))

# test_dataset = add_compute_stats(StaticDataSOCLVR2)(train=False, folder='./data/sort-of-clevr2/sort-of-clevr.pickle', stats=train_dataset.stats)#size=9800, img_size=config.img_size, num_colors=config.num_colors) #stats={'mean':[0, 0, 0], 'std':[1, 1, 1]})


test_loaders = [DataLoader(td,
                          batch_size=config.batch_size,
                          drop_last=True,
                          num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                          timeout=0 if config.use_cuda and not config.is_pycharm else 0,
                          pin_memory=True) for td in test_datasets]

# [weblog_dataset_info(td, weblogger=config.weblogger, num_batches_to_log=np.max((1, round(20 / config.batch_size)))) for td in test_loaders]

config.step = standard_step_crops


def call_run(loader, train, callbacks, **kwargs):
    logs = {'ema_loss': ExpMovingAverage(0.2),
            'ema_acc': ExpMovingAverage(0.2),
            'ca_acc': CumulativeAverage(),
            }

    return run(loader,
               use_cuda=config.use_cuda,
               net=config.net,
               callbacks=callbacks,
               loss_fn=config.loss_fn,
               optimizer=config.optimizer,
               iteration_step=config.step,
               train=train,
               logs=logs,
               collect_data=kwargs['collect_data'] if 'collect_data' in kwargs else False,
               stats=train_dataset.stats)


def stop(logs, cb):
    logs['stop'] = True
    print('Early Stopping')

all_cb = [
    StopFromUserInput(),
    ProgressBar(l=len(train_dataset), batch_size=config.batch_size, logs_keys=['ema_loss', 'ema_acc']),
    PrintNeptune(id='ema_loss', plot_every=10, weblogger=config.weblogger),
    PrintNeptune(id='ema_acc', plot_every=10, weblogger=config.weblogger),
    # Either train for 10 epochs (which is more than enough for convergence):
    # TriggerActionWhenReachingValue(mode='max', patience=1, value_to_reach=10, check_every=10, metric_name='epoch', action=stop, action_name='10epochs'),

    # Or explicitely traing until 90% accuracy or convergence:
    TriggerActionWhenReachingValue(mode='max', patience=20, value_to_reach=config.stop_when_train_acc_is/100, check_every=10, metric_name='ema_acc', action=stop, action_name=f'goal{config.stop_when_train_acc_is}%'),

    # TriggerActionWithPatience(mode='min', min_delta=0.01,
    #                           patience=config.patience_stagnation,
    #                           min_delta_is_percentage=False,
    #                           metric_name='ema_loss',
    #                           check_every=10,
    #                           triggered_action=stop,
    #                           action_name='Early Stopping',
    #                           weblogger=config.weblogger),
    #

    # PlateauLossLrScheduler(config.optimizer, patience=1000, check_batch=True, loss_metric='ema_loss'),

    *[DuringTrainingTest(testing_loaders=test_ds, auto_increase=1, weblogger=config.weblogger, log_text='test during train', use_cuda=config.use_cuda, call_run=call_run, plot_samples_corr_incorr=True,  callbacks=[
        PrintNeptune(id='ca_acc', plot_every=np.inf, log_prefix='test_EVAL&TRAIN_', weblogger=config.weblogger),
         PrintConsole(id='ca_acc', endln=" / ", plot_every=np.inf, plot_at_end=True),
         PlotImagesEveryOnceInAWhile(config.weblogger, test_ds.dataset,  plotting_fun=plot_crop_corr_incorr, plot_every=np.inf, plot_only_n_times=1, plot_at_the_end=True, max_images=10, text='')]) for test_ds in test_loaders]



    # DuringTrainingTest(testing_loaders=test_loaders[1], auto_increase=1, weblogger=config.weblogger, log_text='test during train', use_cuda=config.use_cuda, call_run=call_run, plot_samples_corr_incorr=True,  callbacks=[
    #     PrintNeptune(id='ca_acc', plot_every=np.inf, log_prefix='test_EVAL&TRAIN_', weblogger=config.weblogger),
    #     PrintConsole(id='ca_acc', endln=" / ", plot_every=np.inf, plot_at_end=True)]),

]

# all_cb.append(ClipGradNorm(config.net, config.clip_grad_norm)) if config.clip_grad_norm is not None else None
all_cb.append(SaveModel(config.net, config.model_output_filename, epsilon_loss=0.001, loss_metric_name='ema_loss', max_iter=5000)) if not config.is_pycharm else None

net, logs = call_run(train_loader, True, all_cb)
config.weblogger.stop() if config.weblogger else None

# network_names = ['vonenet-resnet50'] #alexnet', 'inception_v3', 'densenet201', 'vgg19bn', 'resnet152', 'vonenet-resnet50', 'cornet-s', 'vonenet-cornets']  #
# pt = ['vanilla']
# all_exps = (product(network_names, pt))
# arguments = list((dict(network_name=i[0],
#                        pt=i[1]) for i in all_exps))
# [train(**a) for a in arguments]
