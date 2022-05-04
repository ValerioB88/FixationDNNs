from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pathlib
import shutil
folder = './data/simple_relations_lr_half/'
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
shutil.rmtree(folder)

## Train
pathlib.Path(folder + '/train/blue_left').mkdir(parents=True, exist_ok=True)
pathlib.Path(folder + '/train/blue_right').mkdir(parents=True, exist_ok=True)

img_size = np.array([128, 128])

size_target = 10
size_relation = 10

color_target = (50, 50, 255)
color_relation = (255, 50, 50)

for i in range(8000):
    canvas = Image.new('RGB', tuple(img_size), 'black')
    draw = ImageDraw.Draw(canvas)
    random_xy_target = np.array([np.random.randint(0, img_size[0]//2 - 2*size_target), np.random.randint(0, img_size[0]-size_target)])
    draw.rectangle((*random_xy_target, *random_xy_target + size_target), fill=color_target)

    random_xy_relation = np.array([np.random.randint(0, img_size[0]//2 - 2*size_relation), np.random.randint(0, img_size[0]-size_relation)])
    draw.rectangle((*random_xy_relation, *random_xy_relation + size_relation), fill=color_relation)
    # canvas.show()
    if random_xy_target[0] < random_xy_relation[0]:
        type = 'blue_left'
    else:
        type = 'blue_right'

    canvas.save(f'{folder}/train/{type}/{i}.png')


pathlib.Path(folder + '/test/blue_left').mkdir(parents=True, exist_ok=True)
pathlib.Path(folder + '/test/blue_right').mkdir(parents=True, exist_ok=True)

img_size = np.array([128, 128])

size_target = 10
size_relation = 10

color_target = (50, 50, 255)
color_relation = (255, 50, 50)

for i in range(1000):
    canvas = Image.new('RGB', tuple(img_size), 'black')
    draw = ImageDraw.Draw(canvas)
    random_xy_target = np.array([np.random.randint(img_size[0]//2 + 2*size_target, img_size[0] - size_target), np.random.randint(0, img_size[0]-size_target)])
    draw.rectangle((*random_xy_target, *random_xy_target + size_target), fill=color_target)

    random_xy_relation = np.array([np.random.randint(img_size[0]//2 + 2*size_target, img_size[0] - size_relation), np.random.randint(0, img_size[0]-size_relation)])
    draw.rectangle((*random_xy_relation, *random_xy_relation + size_relation), fill=color_relation)
    # canvas.show()
    if random_xy_target[0] < random_xy_relation[0]:
        type = 'blue_left'
    else:
        type = 'blue_right'

    canvas.save(f'{folder}/test/{type}/{i}.png')

