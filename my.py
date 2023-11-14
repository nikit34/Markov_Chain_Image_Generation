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
        self.directional = False

    def normalize(self, pixel):
        return pixel // self.bucket_size

    def denormalize(self, pixel):
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
        img = np.array(img)[:, :, :3]
        prog = pyprind.ProgBar((width * height), width=64, stream=1)

        for x in range(height):
            for y in range(width):
                pix = tuple(self.normalize(img[x, y]))
                prog.update()
                if self.directional:
                    self.weights = defaultdict(lambda: defaultdict(Counter))
                    for dir, neighbour in self.get_neighbours_dir(x, y).items():
                        try:
                            self.weights[pix][dir][tuple(self.normalize(img[neighbour]))] += 1
                        except IndexError:
                            continue
                else:
                    for neighbor in self.get_neighbours(x, y):
                        try:
                            self.weights[pix][tuple(self.normalize(img[neighbor]))] += 1
                        except IndexError:
                            continue

    def generate(self, init_state=None, width=512, height=512):
        fourcc = cv2.VideoWriter_fourcc(*'MP4v')
        writer = cv2.VideoWriter('markov_img.mp4', fourcc, 24, (width, height))
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
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
        stack = [init_pos]
        coloured = set()
        i = 0
        prog = pyprind.ProgBar((width * height), width=64, stream=1)
        while stack:
            x, y = stack.pop()
            if (x, y) in coloured:
                continue
            else:
                coloured.add((x, y))
            try:
                cpixel = img[x, y]
                node = self.weights[tuple(cpixel)]
                img_out[x, y] = self.denormalize(cpixel)
                prog.update()
                i += 1
                screen.set_at((x, y), img_out[x, y])
                if i % 128 == 0:
                    pygame.display.flip()
                    pass
            except IndexError:
                continue

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit()

            if self.directional:
                keys = {dir: list(node[dir].keys()) for dir in node}
                neighbours = self.get_neighbours_dir(x, y).items()
                counts = {dir:np.arange(len(node[dir])) for dir in keys}
                ps = {dir: counts[dir] / counts[dir].sum() for dir in keys}
            else:
                keys = list(node.keys())
                heighbours = self.get_neighbours(x, y)
                counts = np.array(list(node.values()), dtype=np.float32)
                key_idxs = np.arange(len(keys))
                ps = counts / counts.sum()
            np.random.shuffle(neighbours)
            for neightbour in neighbours:
                try:
                    if self.directional:
                        direction = neighbour[0]
                        neighbour = neighbour[1]
                        if neighbour not in coloured:
                            col_idx = np.random.choice(key_idxs[direction], p=ps[direction])
                            img[neighbour] = keys[direction][col_idx]
                        else:
                            col_idx = np.random.choice(key_idxs, p=ps)
                            if neighbour not in coloured:
                                img[neighbour] = keys[col_idx]
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
    fnames = ['АВА.jpg']
    for fname in fnames:
        im = Image.open(fname)
        im.show()
        print("Training " + fname)
        chain.train(im)
    print("\nGenerating")
    chain.generate().show()
