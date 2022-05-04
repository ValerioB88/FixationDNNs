import torch.nn
from src.utils.train_utils import run
from src.utils.Config import Config
from src.utils.net_utils import GrabNet, prepare_network
from src.utils.dataset_utils import add_compute_stats, ImageFolder
from src.utils.train_utils import ExpMovingAverage, CumulativeAverage
from torch.utils.data import DataLoader
from src.utils.callbacks import *
from src.utils.misc import *
from itertools import product
from torchvision.transforms import transforms

def train(network_name, pt):
    config = Config(stop_when_train_acc_is=95,
                    patience_stagnation=500,
                    project_name='Train-Relations',
                    network_name=network_name,
                    batch_size=8 if not torch.cuda.is_available() else 64,
                    weblogger=2,  #set to "2" if you want to log into neptune client
                    pretraining=pt,
                    continue_train=False,
                    learning_rate=0.0005,
                    folder='simple_relations_redness_random_bk_xy_shape',
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False)

    if config.network_name == 'inception_v3':
        pil_t = [transforms.Resize(299)]
    else:
        pil_t = None

    config.net = MyGrabNet().get_net(config.network_name, imagenet_pt=True if config.pretraining == 'ImageNet' else False, num_classes=2)
    # utils.print_net_info(config.net)

    config.additional_tags = f'{config.network_name}'

    config.model_output_filename = './models/' + config_to_path_train(config) + '.pt'
    # config.run_id = get_run_id(config)
    if config.continue_train:
        config.pretraining = config.model_output_filename

    prepare_network(config.net, config)

    config.loss_fn = torch.nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(config.net.parameters(),
                                        lr=config.learning_rate)
    if config.pretraining == 'ImageNet':
        if 'vonenet' in config.network_name:
            stats = {'mean': [0.5, 0.5, 0.5], 'std':  [0.5, 0.5, 0.5]}
        else:
            stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    else:
        stats = None

    train_dataset = add_compute_stats(MyImageFolder)(root=f'./data/{config.folder}/train/', name_generator='train', add_PIL_transforms=pil_t, stats=stats)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                              num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                              timeout=0 if config.use_cuda and not config.is_pycharm else 0)
    weblog_dataset_info(train_loader, weblogger=config.weblogger, num_batches_to_log=np.max((1, round(20 / config.batch_size))))

    test_dataset = add_compute_stats(MyImageFolder)(root=f'./data/{config.folder}/test', name_generator='test', add_PIL_transforms=pil_t, stats=train_dataset.stats)

    test_loaders = [DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size,
                               num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                               timeout=0 if config.use_cuda and not config.is_pycharm else 0)]
    [weblog_dataset_info(i, weblogger=config.weblogger, num_batches_to_log=np.max((1, round(20 / config.batch_size)))) for i in test_loaders]

    config.step = standard_step


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
                   collect_data=kwargs['collect_data'] if 'collect_data' in kwargs else False)


    def stop(logs, cb):
        logs['stop'] = True
        print('Early Stopping')

    all_cb = [
        StopFromUserInput(),
        ProgressBar(l=len(train_loader), batch_size=config.batch_size, logs_keys=['ema_loss', 'ema_acc']),
        PrintNeptune(id='ema_loss', plot_every=10, weblogger=config.weblogger),
        PrintNeptune(id='ema_acc', plot_every=10, weblogger=config.weblogger),
        # Either train for 10 epochs (which is more than enough for convergence):
        # TriggerActionWhenReachingValue(mode='max', patience=1, value_to_reach=10, check_every=10, metric_name='epoch', action=stop, action_name='10epochs'),

        # Or explicitely traing until 90% accuracy or convergence:
        TriggerActionWhenReachingValue(mode='max', patience=20, value_to_reach=config.stop_when_train_acc_is/100, check_every=10, metric_name='ema_acc', action=stop, action_name=f'goal{config.stop_when_train_acc_is}%'),

        TriggerActionWithPatience(mode='min', min_delta=0.01,
                                  patience=config.patience_stagnation,
                                  min_delta_is_percentage=False,
                                  metric_name='ema_loss',
                                  check_every=10,
                                  triggered_action=stop,
                                  action_name='Early Stopping',
                                  weblogger=config.weblogger),


        PlateauLossLrScheduler(config.optimizer, patience=1000, check_batch=True, loss_metric='ema_loss'),

        DuringTrainingTest(testing_loaders=test_loaders, auto_increase=True, weblogger=config.weblogger, log_text='test during train', use_cuda=config.use_cuda, call_run=call_run, plot_samples_corr_incorr=True,  callbacks=[
            PrintNeptune(id='ca_acc', plot_every=np.inf, log_prefix='test_EVAL&TRAIN_', weblogger=config.weblogger),
            PrintConsole(id='ca_acc', endln=" / ", plot_every=np.inf, plot_at_end=True)]),

    ]

    all_cb.append(SaveModel(config.net, config.model_output_filename, epsilon_loss=0.01, loss_metric_name='ema_loss', max_iter=100)) if not config.is_pycharm else None

    net, logs = call_run(train_loader, True, all_cb)
    config.weblogger.stop()

network_names = ['vonenet-resnet50'] #alexnet', 'inception_v3', 'densenet201', 'vgg19bn', 'resnet152', 'vonenet-resnet50', 'cornet-s', 'vonenet-cornets']  #
pt = ['vanilla']
all_exps = (product(network_names, pt))
arguments = list((dict(network_name=i[0],
                       pt=i[1]) for i in all_exps))
[train(**a) for a in arguments]
