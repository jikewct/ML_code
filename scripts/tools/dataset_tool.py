import pickle
import sys

sys.path.append("..")
sys.path.append(".")

import imageio
import matplotlib.pyplot as plt
import numpy as np
from absl import app
from pipeline.datasets import datasetsHelper


def unpack_cifar10():
    rootpath = "E:\jikewct\Dataset\cifar10"

    train_loader, test_loader = datasetsHelper.get_loader("cifar10", rootpath, 1)

    i = 0
    for x, y in train_loader:
        x = x.cpu().numpy()
        x = x.reshape(3, 32, 32)
        x = np.uint8(x.transpose(1, 2, 0) * 256)
        imageio.imwrite("{}\cifar-10-images\\train\{:05}.png".format(rootpath, i), x)
        i += 1
    i = 0
    for x, y in test_loader:
        x = x.cpu().numpy()
        x = x.reshape(3, 32, 32)
        x = np.uint8(x.transpose(1, 2, 0) * 256)
        imageio.imwrite("{}\cifar-10-images\\test\{:05}.png".format(rootpath, i), x)
        i += 1


if __name__ == "__main__":
    app.run(unpack_cifar10())
