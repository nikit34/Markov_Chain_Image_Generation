from PIL import Image
import numpy as np
import pyprind
import random
import os
import sys
import cv2
import pygame
from collections import defaultdict, Counter


class MarkChain:
    def __init__(self, bucket_size=10, four_nb=True):
        self.weights = defaultdict(Counter)
        self.bucket_size = bucket_size
        self.four_nb = four_nb
        self.directional = True

    def normalize(self, pixel):
        # нормализуем писель (r, g, b), (делим без остатка)
        return pixel // self.bucket_size

    def denormalize(self, pixel):
        # возвращаем нормализованный пиксель в близкое к начальному, дискретное, значение
        return pixel * self.bucket_size

    def get_neighbours(self, x, y):
        if self.four_nb:
            return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        else:
            return [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
                (x + 1, y + 1),
                (x - 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y - 1)
            ]

    def get_neighbours_dir(self, x, y):
        if self.four_nb:
            return {'r': (x + 1, y), 'l': (x - 1, y), 'b': (x, y + 1), 't': (x, y - 1)}
        else:
            return dict(zip(
                ['r', 'l', 'b', 't', 'br', 'tl', 'tr', 'bl'],
                [
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                    (x + 1, y + 1),
                    (x - 1, y - 1),
                    (x - 1, y + 1),
                    (x + 1, y - 1)
                ]
            ))

    def train(self, img):
        width, height = img.size
        img = np.array(img)
        prog = pyprind.ProgBar((width * height), width=64, stream=1)

        if self.directional:
            self.weights = defaultdict(lambda: defaultdict(Counter))

        for x in range(height):
            for y in range(width):
                pix = tuple(self.normalize(img[x, y]))
                prog.update()
                if self.directional:
                    for dir, neighbour in self.get_neighbours_dir(x, y).items():
                        try:
                            # записываем веса в формате [(нормализованный пиксель)][имя соседа][(нормализованный сосед)] = счетчик
                            self.weights[pix][dir][tuple(self.normalize(img[neighbour]))] += 1
                        except IndexError:
                            continue
                else:
                    # берем все соседние клетки в формате [(x, y), ...]
                    for neighbor in self.get_neighbours(x, y):
                        try:
                            # записываем веса в формате [(нормализованный пиксель)][(нормализованный сосед)] = счетчик
                            self.weights[pix][tuple(self.normalize(img[neighbor]))] += 1
                        except IndexError:
                            continue

    def check_status_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                sys.exit()

    def generate(self, init_state=None, width=512, height=512):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('markov_img.mp4', fourcc, 24, (width, height))
        pygame.init()

        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Markov Image')
        screen.fill((0, 0, 0))

        if init_state is None:
            init_state = random.choice(list(self.weights.keys()))
        if type(init_state) is not tuple and len(init_state) != 3:
            raise ValueError("Initial State must be a 3-tuple")
        img = Image.new('RGB', (width, height), 'white')
        img = np.array(img)
        img_out = np.array(img.copy())

        init_pos = (np.random.randint(0, width), np.random.randint(0, height))
        img[init_pos] = init_state
        # заводим стек с первым рандомным элементом (x, y)
        stack = [init_pos]
        coloured = set()
        i = 0
        prog = pyprind.ProgBar((width * height), width=64, stream=1)
        while stack:
            self.check_status_events()

            # удаляем очередной элемент из стека и получаем его значение
            x, y = stack.pop()
            if (x, y) in coloured:
                continue
            else:
                # добавляем значение, которого уже нет в стеке в сет, содержащий пройденные пиксели в формате (x, y)
                coloured.add((x, y))
            try:
                # берем значения текущего пикселя из изображения
                cpixel = img[x, y]
                # берем у текущего пикселя [имя соседа][(нормализованный сосед)] = счетчик или [(нормализованный сосед)] = счетчик
                node = self.weights[tuple(cpixel)]
                img_out[x, y] = self.denormalize(cpixel)
                prog.update()
                i += 1
                # устанавливаем цвет текущему пикселю
                screen.set_at((x, y), img_out[x, y])
                if i % 128 == 0:
                    pygame.display.flip()
                    writer.write(cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
                    pass
            except IndexError:
                continue

            neighbours = []
            if self.directional:
                # берем [имя соседа]: [(нормализованный сосед)]
                normalized_neighbour = {dir: list(node[dir].keys()) for dir in node}
                # берем [имя соседа]: [веса]
                weights_neighbour = {dir: np.array(list(node[dir].values()), dtype=np.float32) for dir in normalized_neighbour}
                # берем [имя соседа]: [0, 1, 2, ..., len([(нормализованный сосед): вес])]
                sequence_count_neighbors = {dir: np.arange(len(node[dir])) for dir in normalized_neighbour}
                np.seterr(divide='ignore', invalid='ignore')
                # делим вес на сумму весов
                relative_weight = {dir: weights_neighbour[dir] / weights_neighbour[dir].sum() for dir in normalized_neighbour}

                neighbours = list(self.get_neighbours_dir(x, y).items())
            else:
                # берем [(нормализованный сосед)]
                normalized_neighbour = list(node.keys())
                # берем [веса]
                weights_neighbour = np.array(list(node.values()), dtype=np.float32)
                # берем [0, 1, 2, ..., len([(нормализованный сосед): вес])]
                sequence_count_neighbors = np.arange(len(normalized_neighbour))
                # делим вес на сумму весов
                relative_weight = weights_neighbour / weights_neighbour.sum()

                neighbours = self.get_neighbours(x, y)

            # перемешиваем соседей
            np.random.shuffle(neighbours)
            for neighbour in neighbours:
                try:
                    if self.directional:
                        direction = neighbour[0]
                        neighbour = neighbour[1]
                        if neighbour not in coloured:
                            col_idx = np.random.choice(sequence_count_neighbors[direction], p=relative_weight[direction])
                            img[neighbour] = normalized_neighbour[direction][col_idx]
                    else:
                        col_idx = np.random.choice(sequence_count_neighbors, p=relative_weight)
                        if neighbour not in coloured:
                            img[neighbour] = normalized_neighbour[col_idx]
                except IndexError:
                    pass
                except ValueError:
                    continue
                if 0 <= neighbour[0] < width and 0 <= neighbour[1] < height:
                    stack.append(neighbour)
        writer.release()
        return Image.fromarray(img_out)


if __name__  == "__main__":
    chain = MarkChain(bucket_size=16, four_nb=True)
    fname = 'АВА.jpg'
    im = Image.open(fname)
    im.show()
    print("Training " + fname)
    chain.train(im)
    print("\nGenerating")
    chain.generate().show()
