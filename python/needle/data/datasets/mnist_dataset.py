from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # decompression
    # reference: https://docs.python.org/3/library/gzip.html#gzip.open
    with gzip.open(image_filename, 'rb') as img_obj:
        img = img_obj.read()
    with gzip.open(label_filename, 'rb') as lbl_obj:
        lbl = lbl_obj.read()
    # determine dataset size
    # reference: https://blog.csdn.net/lindorx/article/details/94639183
    num_img, H, W = struct.unpack_from('>iii', # big endian
                                       img,
                                       offset=4)
    num_lbl = struct.unpack_from('>i', # big endian
                                 lbl,
                                 offset=4)
    # `num_lbl` is a tuple.
    assert num_img == num_lbl[0], '# of images and labels do not match'
    # load data to NumPy
    img = struct.unpack_from('>' + 'B' * H * W * num_img, # big endian
                             img,
                             offset=16)
    img = np.array(img, dtype=np.float32).reshape(num_img, H * W)
    img = np.clip(img / 255. ,
                  0., 1.)
    lbl = struct.unpack_from('>' + 'B' * num_lbl[0], # big endian
                             lbl,
                             offset=8)
    lbl = np.array(lbl, dtype=np.uint8)

    return img, lbl
    ### END YOUR CODE


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.h = self.w = 28
        self.transforms = transforms
        self.transforms_fn = lambda I : np.reshape(self.apply_transforms(np.reshape(I,
                                                                                    (-1, self.h, self.w))[..., None]),
                                                   (-1, self.h * self.w))
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        try:
            img, lbl = self.img[index], self.lbl[index] # image shape: (B, H x W)
        except AttributeError:
            self.img, self.lbl = parse_mnist(self.image_filename, self.label_filename)
            return self.transforms_fn(self.img[index]), self.lbl[index]
        else:
            return self.transforms_fn(img), lbl
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        try:
            return len(self.lbl)
        except AttributeError:
            self.img, self.lbl = parse_mnist(self.image_filename, self.label_filename)
            return len(self.lbl)
        ### END YOUR SOLUTION