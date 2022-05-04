from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pathlib
import shutil


folder = './data/simple_relations_redness_random_bk_xy_shape/'
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
shutil.rmtree(folder)

## Train
pathlib.Path(folder + '/train/square_red').mkdir(parents=True, exist_ok=True)
pathlib.Path(folder + '/train/circle_red').mkdir(parents=True, exist_ok=True)

img_size = np.array([128, 128])

size_target = 10
size_relation = 10

color_target = (50, 50, 255)
color_relation = (255, 50, 50)

for i in range(8000):
    canvas = Image.new('RGB', tuple(img_size),(np.random.randint(150, 255), np.random.randint(100, 255), np.random.randint(150, 255)))

    draw = ImageDraw.Draw(canvas)
    random_xy_target = np.array([np.random.randint(0, img_size[0]-size_target), np.random.randint(0, img_size[0]-size_target)])
    # xy_target = np.array([img_size[0]*1//3, img_size[0]//2])
    color_target = (np.random.randint(50, 255), 50, 50)
    draw.ellipse((*random_xy_target, *random_xy_target + size_target), fill=color_target)
    # draw.rectangle((*random_xy_target, *random_xy_target + size_target), fill=color_target)

    color_relation = (np.random.randint(50, 255), 50, 50)

    random_xy_relation = np.array([np.random.randint(0, img_size[0] - size_relation), np.random.randint(0, img_size[0]-size_relation)])
    # xy_relation = np.array([img_size[0]*2//3, img_size[0]//2])

    draw.rectangle((*random_xy_relation, *random_xy_relation + size_relation), fill=color_relation)
    # canvas.show()
    if color_target[0] < color_relation[0]:
        type = 'square_red'
    else:
        type = 'circle_red'

    canvas.save(f'{folder}/train/{type}/{i}.png')

##
pathlib.Path(folder + '/test/square_red').mkdir(parents=True, exist_ok=True)
pathlib.Path(folder + '/test/circle_red').mkdir(parents=True, exist_ok=True)

for i in range(1000):
    canvas = Image.new('RGB', tuple(img_size),(np.random.randint(0, 1), np.random.randint(0, 1), np.random.randint(0, 1)))

    draw = ImageDraw.Draw(canvas)
    random_xy_target = np.array([np.random.randint(0, img_size[0]-size_target), np.random.randint(0, img_size[0]-size_target)])
    # xy_target = np.array([img_size[0]*1//3, img_size[0]//2])
    color_target = (np.random.randint(50, 255), 50, 50)
    draw.ellipse((*random_xy_target, *random_xy_target + size_target), fill=color_target)
    # draw.rectangle((*random_xy_target, *random_xy_target + size_target), fill=color_target)

    color_relation = (np.random.randint(50, 255), 50, 50)

    random_xy_relation = np.array([np.random.randint(0, img_size[0] - size_relation), np.random.randint(0, img_size[0]-size_relation)])
    # xy_relation = np.array([img_size[0]*2//3, img_size[0]//2])

    draw.rectangle((*random_xy_relation, *random_xy_relation + size_relation), fill=color_relation)
    # canvas.show()
    if color_target[0] < color_relation[0]:
        type = 'square_red'
    else:
        type = 'circle_red'

    canvas.save(f'{folder}/test/{type}/{i}.png')


