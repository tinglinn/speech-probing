import os
import itertools
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class LinearProbe(torch.nn.Module):
    """
    Single layer linear probe.
    """
    def __init__(self, args): 
        super(LinearProbe, self).__init__()
        input_size = args["probe"]["input size"]
        output_size = args["probe"]["output size"]
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out
