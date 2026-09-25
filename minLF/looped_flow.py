import torch
import torch.nn as nn


class Denoiser(torch.nn.Module):
    def __init__(self,):
        self.__init__()
        self.state = None

        # Define denoising model
        self.model = None

    def forward(self, c, z_ti, x_ti):
        # c -> problem, z_ti -> reccurent state, x_ti -> flow state

        # The model predicts a sequence of probability vectors x_hat = (x_hat^l)^L_l=1

        # Each vector in the sequence is rouded := onehot(argmax(x_hat^l))

        # return a sequence of one hot encodings
        return


class LoopedFlow:
    def __init__(self,):
        pass

    def train(self,):
        pass

    def infer(self,):
        # Multiple steps of the flow
        pass

    def step(self,):
        # Euler step of the flow
        pass
