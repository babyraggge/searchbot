import os


class Config:
    TOKEN = os.environ.get("TOKEN_BOT")
    DATABASE_NAME = os.environ.get("DATABASE_NAME")
    URL = "http://vis-www.cs.umass.edu/lfw/images/"

    img_size = (45, 45)
    n_neighbors = 2
    display_distance = True
    search = False


class Messages:
    START_MESSAGE = "I'm a searchFace bot. Please send a human face images and I will find a similar person."

    HELP_MESSAGE = "Send a human face images and I will find a similar person. " \
                   "All information about the bot can be obtained by clicking on the corresponding buttons " \
                   "on the keyboard below"

    MODEL_INFO = "It is based on the variational autoencoder model with the dimension latent space of 20. " \
                 "Convolutional neural network model is used for feature extraction."

    BOT_INFO = f"SearchFace bot was created as part of the final project of the Deep Learning School. " \
               "It was trained on the open dataset \"Marked faces in the wild\". " \
               "A database of face photographs designed for studying the problem of unconstrained face recognition. " \
               "The data set contains more than 13,000 images of faces collected from the web. " \
               "Each face has been labeled with the name of the person pictured. " \
               "http://vis-www.cs.umass.edu/lfw/"
