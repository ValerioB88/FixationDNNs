from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pathlib
import shutil


def draw_line_with_circled_edge(draw, xy, **kwargs):
    draw.line(xy, width=width)
    add_circles_to_loc(xy, draw)

def add_circles_to_loc(l, draw):
        circle(draw, l[0], width // 2)
        circle(draw, l[1], width // 2)

def circle(draw, center, radius):
        draw.ellipse((center[0] - radius + 1,
                      center[1] - radius + 1,
                      center[0] + radius - 1,
                      center[1] + radius - 1), fill=fill, outline=None)


def top_left(coord):
    draw_line_with_circled_edge(draw, ((coord[0], coord[1]), (coord[0] + segment_length, coord[1])),
                                width=width)


    draw_line_with_circled_edge(draw, ((coord[0], coord[1]), (coord[0], coord[1] + segment_length)),
                                width=width)

def bottom_left(coord):
    draw_line_with_circled_edge(draw, ((coord[0], coord[1]), (coord[0] + segment_length, coord[1])),
                                width=width)

    draw_line_with_circled_edge(draw, ((coord[0], coord[1]), (coord[0], coord[1] - segment_length)),
                                width=width)

def top_right(coord):
    draw_line_with_circled_edge(draw, ((coord[0], coord[1]), (coord[0] - segment_length, coord[1])),
                                width=width)

    draw_line_with_circled_edge(draw, ((coord[0], coord[1]), (coord[0], coord[1] + segment_length)),
                                width=width)


def bottom_right(coord):
    draw_line_with_circled_edge(draw, ((coord[0], coord[1]), (coord[0] - segment_length, coord[1])),
                                width=width)

    draw_line_with_circled_edge(draw, ((coord[0], coord[1]), (coord[0], coord[1] - segment_length)),
                                width=width)


## down left

folder = './data/square_scrambled/'
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
shutil.rmtree(folder)

## Train
pathlib.Path(folder + '/train/aligned').mkdir(parents=True, exist_ok=True)
pathlib.Path(folder + '/train/scrambled').mkdir(parents=True, exist_ok=True)

img_size = np.array([224, 224])

fill = 'white'
canvas = Image.new('RGB', tuple(img_size),'black') #(np.random.randint(150, 255), np.random.randint(100, 255), np.random.randint(150, 255)))
draw = ImageDraw.Draw(canvas)


width = 5
segment_length = 20
edge_size = 40

center = np.array([img_size[0]//2, img_size[1]//2])
top_left(center - edge_size)
bottom_left(np.array([center[0] - edge_size, center[1] + edge_size]))
bottom_right(center + edge_size)
top_right([center[0] + edge_size, center[1] - edge_size])
canvas.show()