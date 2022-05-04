import io
import numpy as np
# import tensorflow as tf
from PIL import Image, ImageDraw
from sty import rs, fg, bg, ef

def translate_question(q, print_color=True):
    # if len(q) != 11:
    #     return 'Not a proper question'
    colors = ['red', 'green', 'blue', 'magenta', 'yellow', 'cyan']
    idx = np.argwhere(q[:len(q)-5])[0][0]
    color = colors[idx]
    if color == 'red':
        ccc = fg.red
    if color == 'blue':
        ccc = fg.blue
    if color == 'green':
        ccc = fg.green
    if color == 'magenta':
        ccc = fg.magenta
    if color == 'yellow':
        ccc = fg.yellow
    if color == 'cyan':
        ccc = fg.cyan
    color_str = f"" + ccc + color + rs.fg + "" if  print_color else f"~{color}~"
    if q[len(q)-5]:
        if q[len(q)-5+2]:
            return f"What's the shape of the nearest object to the object in  {color_str}?"
        elif q[len(q)-5+3]:
            return f"What's the shape of the farthest object away from the object in {color_str}?"
        elif q[len(q)-5+4]:
            return f'How many objects have the same shape as the object in {color_str}?'
    else:
        if q[len(q)-5+2]:
            return f'Is the object in {color_str} a circle or a square?'
        elif q[len(q)-5+3]:
            return f'Is the object in {color_str} at the bottom of the image?'
        elif q[len(q)-5+4]:
            return f'Is the object in {color_str} on the left of the image?'


def translate_answer(a):
    return ['yes', 'no', 'square', 'circle', '1', '2', '3', '4', '5', '6'][a]


class SortOfCLEVRGenerator(object):
    def __init__(self, img_size=128, num_colors=6):
        self.img_size = img_size
        self.bg_color = (200, 200, 200) # (231, 231, 231)

        self.colors = [
            (255, 0, 0),  ##r
            (0, 255, 0),  ##g
            (0, 0, 255),  ##b
            (255, 0, 255),  ## magenta
            (255, 255, 0),  ## yellow
            (0, 255, 255)  ## cyan
        ]
        self.colors = self.colors[:num_colors]
        self.num_color = len(self.colors)
        self.num_shape = self.num_color
        # Avoid a color shared by more than one objects
        self.num_shape = min(self.num_shape, self.num_color)
        self.shape_size = int((img_size * 0.9 / 4) * 0.7 / 2)

        self.question_vector_size = self.num_color + 5
        self.answer_vector_size = 10

    def is_overlapping_1D(self, line1, line2):
        """
        line:
            (xmin, xmax)
        """
        return line1[0] <= line2[1] and line2[0] <= line1[1]

    def is_overlapping_2d(self, box1, box2):
        """
        box:
            (xmin, ymin, xmax, ymax)
        """
        return self.is_overlapping_1D([box1[0], box1[2]], [box2[0], box2[2]]) and self.is_overlapping_1D([box1[1], box1[3]], [box2[1], box2[3]])


    def generate_sample(self, p_circle=0.5):
        # Generate I: [img_size, img_size, 3]
        img = Image.new('RGB', (self.img_size, self.img_size), color=self.bg_color)
        drawer = ImageDraw.Draw(img)
        # Don't shuffle the colors! They are the criteria to identify the objects.
        idx_color_shape = np.arange(self.num_color)
        coin = np.random.rand(self.num_shape)
        X = []
        Y = []
        for i in range(self.num_shape):
            stop = False
            while not stop:
                x = np.random.randint(self.img_size)
                y = np.random.randint(self.img_size)
                box = xmin, ymin, xmax, ymax = x- self.shape_size, y - self.shape_size, x + self.shape_size, y + self.shape_size
                if xmin > 0 and xmax < self.img_size and ymin  > 0 and ymax < self.img_size:
                    is_overlapping = False
                    for prev_x, prev_y in zip(X, Y):
                        prev_xmin, prev_ymin, prev_xmax, prev_ymax = prev_x - self.shape_size, prev_y - self.shape_size, prev_x + self.shape_size, prev_y + self.shape_size

                        if self.is_overlapping_2d(box, (prev_xmin, prev_ymin, prev_xmax, prev_ymax)):
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        stop =True


            # x = idx_coor[i] % self.n_grid
            # y = (self.n_grid - np.floor(idx_coor[i] / self.n_grid) - 1).astype(np.uint8)
            # # sqaure terms are added to remove ambiguity of distance
            # position = ((x + 0.5) * self.block_size - self.shape_size + x ** 2, (y + 0.5) * self.block_ size - self.shape_size + y ** 2,
            #             (x + 0.5) * self.block_size + self.shape_size + x ** 2, (y + 0.5) * self.block_size + self.shape_size + y ** 2)
            X.append(x)
            Y.append(y)
            if coin[i] < p_circle:
                drawer.ellipse(box, outline=self.colors[idx_color_shape[i]], fill=self.colors[idx_color_shape[i]])
            else:
                drawer.rectangle(box, outline=self.colors[idx_color_shape[i]], fill=self.colors[idx_color_shape[i]])

        # Generate its representation
        color = idx_color_shape[:self.num_shape]
        shape = coin < p_circle
        shape_str = ['c' if x else 's' for x in shape]  # c=circle, s=square

        representation = []
        for i in range(len(X)):
            center = (X[i], Y[i])
            shape = shape_str[i]
            representation.append([center, shape])  # color order is constant so you don't need it.

        return img, representation

    def biased_distance(self, center1, center2):
        """center1 and center2 are lists [x, y]"""
        return (center1[0] - center2[0]) ** 2 + 1.1 * (center1[1] - center2[1]) ** 2

    def generate_questions(self, rep, number_questions=10):
        """
        Given a queried color, all the possible questions are as follows.
        Non-relational questions:
            Is it a circle or a rectangle?
            Is it on the bottom of the image?
            Is it on the left of the image
        Relational questions:
            The shape of the nearest object from the COL obj?
            The shape of the farthest object from the COL obj?
            How many objects have the same shape as the COL obj?
        Questions are encoded into a one-hot vector of size 11:
        [red, blue, green, orange, yellow, gray, relational, non-relational, question 1, question 2, question 3]
        [reg green blue magenta yellow cyan rel non-rel q1 q2 q3]
        """
        questions = []
        for q in range(number_questions):
            # for r in range(2):
            question = [0] * self.question_vector_size
            color = np.random.randint(self.num_color)
            question[color] = 1
            question[self.num_color + np.random.randint(2)] = 1
            question_type = np.random.randint(3)
            question[self.num_color + 2 + question_type] = 1
            questions.append(question)
        return questions

    def get_all_rel_questions(self, rep):
        questions = []
        # Iterate over color indexes.
        for i in range(self.num_color):
            # Iterate over question number.
            for j in range(3):
                # Initialize question vec.
                question = [0] * self.question_vector_size
                # Fill color
                question[i] = 1
                # Fill relational question type.
                question[self.num_color] = 1
                # Fill question number.
                question[self.num_color + 2 + j] = 1
                # Append to list.
                questions.append(question)
        return questions

    def get_all_nonrel_questions(self, rep):
        questions = []
        # Iterate over color indexes.
        for i in range(6):
            # Iterate over question number.
            for j in range(3):
                # Initialize question vec.
                question = [0] * self.question_vector_size
                # Fill color
                question[i] = 1
                # Fill non-relational question type.
                question[7] = 1
                # Fill question number.
                question[8 + j] = 1
                # Append to list.
                questions.append(question)
        return questions

    def generate_all_questions(self, type_rel=True):
        # [red, blue, green, orange, yellow, gray, relational, non - relational, question 1, question 2, question 3]
        from torch.nn.functional import one_hot
        import torch
        from itertools import product
        color = one_hot(torch.arange(0, self.num_color)).tolist()
        if type_rel == 'rel':
            type = torch.tensor([[1, 0]]).tolist()
        if type_rel == 'non_rel':
            type = torch.tensor([[0, 1]]).tolist()
        if type_rel == 'both':
            type = torch.tensor([[0, 1], [1, 0]]).tolist()

        quest = one_hot(torch.arange(0, 3)).tolist()
        from functools import reduce
        import operator
        all_questions = [reduce(operator.concat, q) for q in product(color, type, quest)]
        return all_questions

    def generate_answers(self, rep, questions):
        """
        The possible answer is a fixed length one-hot vector whose elements represent:
        [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]
        """
        answers = []
        for question in questions:
            answer = [0] * self.answer_vector_size
            color = question[:self.num_color].index(1)
            if question[self.num_color]:
                if question[self.num_color + 2]:  # The shape of the nearest object?
                    # dist = [((rep[color][0]-obj[0])**2).sum() for obj in rep]
                    # dist = [self.biased_distance(rep[color][0], obj[0]) for obj in rep]
                    dist = [np.linalg.norm(np.array(rep[color][0]) -np.array(obj[0])) for obj in rep]

                    dist[dist.index(0)] = float('inf')
                    closest = dist.index(min(dist))
                    if rep[closest][1] == 's':
                        # answer[2] = 1
                        answer = 2
                    else:
                        # answer[3] = 1
                        answer = 3
                elif question[self.num_color + 3]:  # The shape of the farthest object?
                    dist = [np.linalg.norm(np.array(rep[color][0]) -np.array(obj[0])) for obj in rep]
                    # dist = [self.biased_distance(rep[color][0], obj[0]) for obj in rep]
                    furthest = dist.index(max(dist))
                    if rep[furthest][1] == 's':
                        # answer[2] = 1
                        answer = 2
                    else:
                        # answer[3] = 1
                        answer = 3

                else:  # How many objects have the same shape?
                    count = -1
                    shape = rep[color][1]
                    for obj in rep:
                        if obj[1] == shape:
                            count += 1
                    # answer[count + 4] = 1
                    answer = 4 + count
            else:
                if question[self.num_color + 2]:  # Is it a circle or a rectangle?
                    if rep[color][1] == 's':
                        # answer[2] = 1
                        answer = 2
                    else:
                        # answer[3] = 1
                        answer = 3
                elif question[self.num_color + 3]:  # Is it on the bottom of the image?
                    if rep[color][0][1] > self.img_size / 2:
                        # answer[0] = 1
                        answer = 0
                    else:
                        # answer[1] = 1
                        answer = 1
                else:  # Is it on the left of the image?
                    if rep[color][0][0] > self.img_size / 2:
                        # answer[1] = 1
                        answer = 1
                    else:
                        # answer[0] = 1
                        answer = 0
            answers.append(answer)
        return answers
