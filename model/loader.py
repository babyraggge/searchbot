from model.VAE import ConvVAE

import torch
import pickle
from sklearn.neighbors import NearestNeighbors

def vae_loader(model: ConvVAE, path: str = "model/storage/CONVvae20.model"):
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        return model
    except VaeLoadError:
        pass


def nn_loader(path: str = "model/storage/NearestNeighbors.pkl"):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except NNLoadError:
        pass


def load_models(vae_path: str = "model/storage/CONVvae20.model",
                nn_path: str = "model/storage/NearestNeighbors.pkl") -> (ConvVAE, NearestNeighbors):
    vae = ConvVAE(20)
    vae = vae_loader(vae, path=vae_path)
    nn = nn_loader(path=nn_path)

    return vae, nn


class VaeLoadError(Exception):
    pass


class NNLoadError(Exception):
    pass
