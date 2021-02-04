import numpy as np
import io
from PIL import Image
import torch
from model.VAE import ConvVAE
from sklearn.neighbors import NearestNeighbors
from database.sql import SQLiter
from sqlite3 import DatabaseError
import urllib.request
from config import Config


def load_img(f: io.BytesIO, size: tuple = Config.img_size):
    """ Load image from BytesIO to numpy array

    :param f: input stream
    :param size: size of output image
    :return: image from stream as numpy array
    """
    image = Image.open(f)
    image = image.resize(size)
    return np.asarray(image, dtype=np.double)


def to_tensor(arr: np.array):
    """ Convert numpy array to torch tensor with roll images axis
    (W,H,C) -> (C,W,H)

    :param arr: input numpy array
    :return: numpy array as tensor
    """
    arr = np.rollaxis(arr, 2, 0)
    t = torch.from_numpy(arr.copy())
    return t


def model_apply(vae: ConvVAE, t: torch.Tensor):
    """ Create and apply VAE to the image

    :param vae: ConvVAE instance
    :param t: input image tensor
    :return: latent vector representation
    """
    mu, logsigma = vae.encode(t)
    z = vae.gaussian_sampler(mu, logsigma)
    return z


def get_code(f: io.BytesIO, vae: ConvVAE):
    """Get latent representation of image

    :param f: input stream
    :param vae: ConvVAE instance
    :return: latent code in numpy array
    """
    with torch.no_grad():
        img = load_img(f)
        img = to_tensor(img)
        img = img.unsqueeze(0)
        code = model_apply(vae, img.float())
        code = code.numpy()
    return code


def get_similar(code: np.array, nn_model: NearestNeighbors, n_neighbors: int = Config.n_neighbors):
    """ Get similar image in LFW dataset

    :param code: latent code
    :param nn_model: pretrained sklearn NearestNeighbors model
    :param n_neighbors: number of nearest neighbors
    :return:
    """
    (distances,), (idx,) = nn_model.kneighbors(code, n_neighbors=n_neighbors)

    return distances, idx


def get_data(idx: object):
    """ Get data from database using index array

    :param idx: rows id for return
    :return: list of rows
    """
    idx = idx.tolist()
    result = []
    db_worker = SQLiter(Config.DATABASE_NAME)
    try:
        for i in idx:
            selected = db_worker.select_single(i)
            result.append(selected[2:])

    except DatabaseError as err:
        pass

    finally:
        db_worker.close()

    return result


def extract_url(selected: tuple):
    return Config.URL + selected[1]


def extract_caption(selected: tuple, distance: object):
    string = f"You look like {selected[0]}!!!"

    if Config.display_distance:
        string += f" Distance is {distance:.3f}"

    if Config.search:
        pass

    return string


def get_similar_photo(f: io.BytesIO, vae: ConvVAE, nn: NearestNeighbors):
    """ Get url to send a similar photo

    :param f: input stream
    :param vae:
    :param nn: pretrained sklearn NearestNeighbors model
    :return:
    """
    code = get_code(f, vae)
    dist, ind = get_similar(code, nn)
    url = get_data(ind)
    distance = dist.tolist()
    return url, distance
