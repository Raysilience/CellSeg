import re
from os import PathLike
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import torch
import torchvision.io.image
from PIL import Image


def check_file_format(filename: Union[str, PathLike], *suffixes):
    filepath = Path(filename)
    pat = re.compile(f"^\.({'|'.join(suffixes)})$")
    flag = filepath.exists()
    flag &= pat.match(filepath.suffix) is not None
    return flag


def load_img(filename: Union[str, PathLike], mode: str = "numpy"):
    res = None
    if check_file_format(filename, "tif", "tiff"):
        if mode == "numpy":
            res = tif.imread(filename)
        elif mode == "torch":
            res = torch.as_tensor(res)

    elif check_file_format(filename, "png", "jpg"):
        if mode == "numpy":
            res = np.asarray(Image.open(filename))
        elif mode == "torch":
            res = torchvision.io.image.read_image(filename)

    return res


if __name__ == '__main__':
    label_file = "/root/CellSeg/data/Train_Labeled/labels/cell_00001_label.tiff"
    img = load_img(label_file, "numpy")

    plt.imshow(img)
    plt.show()
