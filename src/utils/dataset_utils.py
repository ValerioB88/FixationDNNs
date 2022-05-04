import sty
import os
import pathlib
import pickle
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import ImageStat
from sty import fg, rs
from torch.utils.data import IterableDataset
from torchvision import transforms as tf
from torchvision.datasets import ImageFolder
from src.train_relations.create_datasets.generate_sort_of_clevr import SortOfCLEVRGenerator, translate_answer, translate_question


def add_compute_stats(obj_class):
    class ComputeStatsUpdateTransform(obj_class):
        # This class basically is used for normalize Dataset Objects such as ImageFolder in order to be used in our more general framework
        def __init__(self, name_ds='dataset', add_PIL_transforms=None, add_tensor_transforms=None, num_image_calculate_mean_std=70, stats=None, save_stats_file=None, **kwargs):
            """

            @param add_tensor_transforms:
            @param stats: this can be a dict (previous stats, which will contain 'mean': [x, y, z] and 'std': [w, v, u], a path to a pickle file, or None
            @param save_stats_file:
            @param kwargs:
            """
            self.verbose = True
            print(fg.yellow + f"\n**Creating Dataset [" + fg.cyan + f"{name_ds}" + fg.yellow + "]**" + rs.fg)
            super().__init__(**kwargs)
            # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            if add_PIL_transforms is None:
                add_PIL_transforms = []
            if add_tensor_transforms is None:
                add_tensor_transforms = []

            self.transform = torchvision.transforms.Compose([*add_PIL_transforms, torchvision.transforms.ToTensor(), *add_tensor_transforms])

            self.name_ds = name_ds
            self.additional_transform = add_PIL_transforms
            self.num_image_calculate_mean_std = num_image_calculate_mean_std

            compute_stats = False

            if isinstance(stats, dict):
                self.stats = stats
                print(fg.red + f"Using precomputed stats: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)

            elif isinstance(stats, str):
                if os.path.isfile(stats):
                    self.stats = pickle.load(open(stats, 'rb'))
                    print(fg.red + f"Using stats from file [{Path(stats).name}]: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)
                    if stats == save_stats_file:
                        save_stats_file = None
                else:
                    print(fg.red + f"File [{Path(stats).name}] not found, stats will be computed." + rs.fg)
                    compute_stats = True

            if stats is None or compute_stats is True:
                self.stats = self.call_compute_stats()

            if save_stats_file is not None:
                print(f"Stats saved in {save_stats_file}")
                pathlib.Path(os.path.dirname(save_stats_file)).mkdir(parents=True, exist_ok=True)
                pickle.dump(self.stats, open(save_stats_file, 'wb'))

            normalize = torchvision.transforms.Normalize(mean=self.stats['mean'],
                                                         std=self.stats['std'])
            # self.stats = {}
            # self.stats['mean'] = [0.491, 0.482, 0.44]
            # self.stats['std'] = [0.247, 0.243, 0.262]
            # normalize = torchvision.transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

            self.transform.transforms += [normalize]

        def call_compute_stats(self):
            return compute_mean_and_std_from_dataset(self, None, max_iteration=self.num_image_calculate_mean_std, verbose=self.verbose)

    return ComputeStatsUpdateTransform


class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, np.array(self.h) / 255, np.array(other.h) / 255)))


def compute_mean_and_std_from_dataset(dataset, dataset_path=None, max_iteration=100, data_loader=None, verbose=True):
    if max_iteration < 30:
        print(f'Max Iteration in Compute Mean and Std for dataset is lower than 30! This could create unrepresentative stats!') if verbose else None
    start = time()
    stats = {}
    transform_save = dataset.transform
    if data_loader is None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0)

    statistics = None
    c = 0
    stop = False
    while stop is False:
        for data, _, _ in data_loader:
            for b in range(data.shape[0]):
                if c % 10 == 9 and verbose:
                    print(f'{c}/{max_iteration}, m: {np.around(np.array(statistics.mean) / 255, 4)}, std: {np.around(np.array(statistics.stddev) / 255, 4)}')
                c += 1
                if statistics is None:
                    statistics = Stats(tf.ToPILImage()(data[b]))
                else:
                    statistics += Stats(tf.ToPILImage()(data[b]))
                if c > max_iteration:
                    stop = True
                    break
            if stop:
                break

    stats['time_one_iter'] = (time() - start) / max_iteration
    stats['mean'] = np.array(statistics.mean) / 255
    stats['std'] = np.array(statistics.stddev) / 255
    stats['iter'] = max_iteration
    print(fg.cyan + f'mean={np.around(stats["mean"], 4)}, std={np.around(stats["std"], 4)}, time1it: {np.around(stats["time_one_iter"], 4)}s' + rs.fg) if verbose else None
    if dataset_path is not None:
        print('Saving in {}'.format(dataset_path))
        with open(dataset_path, 'wb') as f:
            pickle.dump(stats, f)

    dataset.transform = transform_save
    return stats


from src.train_relations.create_datasets.generate_sort_of_clevr import SortOfCLEVRGenerator


class StaticDataSOCLEVR():
    def __init__(self, type_rel='rel', num_q_per_img=-1, img_size=175, num_colors=6, num_images=9800):
        self.gen = SortOfCLEVRGenerator(img_size=img_size, num_colors=num_colors)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.images = []
        self.questions = []
        self.answers = []
        self.size = num_images
        all_q = self.gen.generate_all_questions(type_rel=type_rel)

        for i in range(self.size):
            img, rep = self.gen.generate_sample(p_circle=0.5)
            if num_q_per_img != -1:
                qq = [all_q[i] for i in np.random.choice(len(all_q), num_q_per_img, replace=False)]
            else:
                qq = all_q
            for q in qq:
                a = self.gen.generate_answers(rep, [q])[0]

                self.images.append(img)
                self.questions.append(q)
                self.answers.append(a)
        if not hasattr(self, 'transform'):
            self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]) if self.transform else self.images, \
               torch.tensor(self.questions[idx]), \
               torch.tensor(self.answers[idx]).long()


class StaticDataSOCLEVR_saccades():
    """
    The coordinates are given in absolute value
    """

    def shuffle_in_unison(self, a, b):

        assert len(a) == len(b)
        c = np.arange(len(a))
        np.random.shuffle(c)

        return [a[i] for i in c], [b[i] for i in c]

    def __init__(self, path=None, type_rel='rel', num_q_per_img=-1, img_size=175, num_colors=6, num_images=9800):
        self.gen = SortOfCLEVRGenerator(img_size=img_size, num_colors=num_colors)
        if path and os.path.isfile(path):
            print(sty.fg.yellow + f"Dataset Loaded from {path}" + sty.rs.fg)
            data = pickle.load(open(path, 'rb'))
            self.images, self.coords, self.crops, self.questions, self.answers = data['images'], data['coords'], data['crops'], data['questions'], data['answers']
        else:
            self.images = []
            self.questions = []
            self.answers = []
            self.size = num_images
            self.size_sample = 32
            self.crops = []
            self.coords = []
            all_q = self.gen.generate_all_questions(type_rel=type_rel)
            for i in range(self.size):
                img, rep = self.gen.generate_sample(p_circle=0.5)
                noise = np.random.normal(0, 0, len(rep))
                coords_img = [np.array((r[0][0] + n, r[0][1] + n)) for r, n in zip(rep, noise)]
                crops_img = [img.crop((r[0] - self.size_sample // 2,
                                       r[1] - self.size_sample // 2,
                                       r[0] + self.size_sample // 2,
                                       r[1] + self.size_sample // 2
                                       )) for r in coords_img]
                crops_img, coords_img = self.shuffle_in_unison(crops_img, coords_img)
                coords_img = [c / (img_size / 2) - 1 for c in coords_img]
                if num_q_per_img != -1:
                    qq = [all_q[i] for i in np.random.choice(len(all_q), np.min((num_q_per_img, len(all_q))), replace=False)]
                else:
                    qq = all_q
                for q in qq:
                    a = self.gen.generate_answers(rep, [q])[0]
                    ### use .astype(np.float32)
                    self.images.append(img)
                    self.crops.append(crops_img)
                    self.coords.append(np.array(coords_img))
                    self.questions.append(np.array([float(i) for i in q]))
                    self.answers.append(a)
            if path:
                pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

                pickle.dump({'images': self.images,
                             'coords': self.coords,
                             'crops': self.crops,
                             'questions': self.questions,
                             'answers': self.answers}, open(path, 'wb'))
                print(sty.fg.yellow + f"Dataset written in {path}" + sty.rs.fg)
        if not hasattr(self, 'transform'):
            self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.cat([self.transform(c).unsqueeze(0) for c in self.crops[idx]]) if self.transform else self.crops, \
               torch.tensor(self.coords[idx]).float(), \
               torch.tensor(self.questions[idx]).float(), \
               torch.tensor(self.answers[idx]).long(), \
               self.transform(self.images[idx]) if self.transform else self.images[idx]

class StaticDataSOCLVR2():
    """
    from here
    https://github.com/kimhc6028/relational-networks
    """

    def __init__(self, type, folder, train=True):
        self.ds = pickle.load(open(folder, 'rb'))
        if train:
            self.ds = self.ds[0]
        else:
            self.ds = self.ds[1]
        self.images = []
        self.questions = []
        self.answers = []
        for dd in self.ds:
            # rel = np.random.randint(2)
            rel = 0 if type == 'rel' else 1 ## fix to non rel
            for idx, q_set in enumerate(dd[2 + rel][0]):
                self.images.append(dd[0].astype(np.float32))
                self.questions.append(q_set)
                self.answers.append(dd[2 + rel][1][idx])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        return self.transform(self.images[idx]) if self.transform else self.images, \
               torch.tensor(self.questions[idx]), torch.tensor(self.answers[idx]).long()


class RunTimeSOCLEVR(IterableDataset):
    def __init__(self, img_size, batch_size=64, num_colors=6):
        self.batch_size = batch_size
        self.img_size = img_size
        self.gen = SortOfCLEVRGenerator(img_size=self.img_size, num_colors=num_colors)
        self.all_questions = self.gen.generate_all_questions()
        if not hasattr(self, 'transform'):
            self.transform = None

    def __iter__(self):
        # rnd = np.random.randint(0, len(self.all_questions), 1000)
        # counter = 0
        while True:
            # for i in range(self.batch_size):
            img, rep = self.gen.generate_sample(p_circle=0.5)
            # plt.imshow(img); plt.show()
            # q = self.all_questions[rnd[counter]]
            q = self.gen.generate_questions(rep, 1)[0]

            # print(translate_question(q))
            a = self.gen.generate_answers(rep, [q])[0]
            # print(translate_answer(a))

            # batch_img_t[counter] = self.transform(img)
            # batch_questions.append(q)
            # batch_answers.append(a)
            # counter += 1
            # if counter >= 1000:
            #     rnd = np.random.randint(0, len(self.all_questions), 1000)
            #     counter = 0
            yield self.transform(img) if self.transform else img, torch.tensor(q), torch.tensor(a).long()


class MyImageFolder(ImageFolder):
    def finalize_getitem(self, path, sample, labels, info=None):
        if info is None:
            info = {}
        return sample, labels, info

    def __init__(self, name_classes=None, verbose=True, *args, **kwargs):
        print(fg.red + ef.inverse + "ROOT:  " + kwargs['root'] + rs.inverse + rs.fg)
        self.name_classes = np.sort(name_classes) if name_classes is not None else None
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def _find_classes(self, dir: str):
        if self.name_classes is None:
            return super()._find_classes(dir)
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and (d.name in self.name_classes)]
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = self.finalize_getitem(path=path, sample=sample, labels=target)
        return output

# from src.train_relations.create_datasets.generate_sort_of_clevr import SortOfCLEVRGenerator, translate_question,  translate_answer
# import matplotlib.pyplot as plt
# gg = SortOfCLEVRGenerator()
# all_q = gg.generate_all_questions()
# img, r = gg.generate_sample()
# answs = gg.generate_answers(r, all_q)
# plt.imshow(img)
# plt.show()
# for qq in all_q:
#     print(translate_question(qq))
#     answs = gg.generate_answers(r, [qq])[0]
#     print(translate_answer(answs))

##

