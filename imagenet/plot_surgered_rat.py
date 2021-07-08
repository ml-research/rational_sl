import pickle
import sys
import torch
from rational.torch import Rational

model = pickle.load(open(sys.argv[1], 'rb'))
rats = []
for child in model.children():
    if type(child) is torch.nn.modules.container.Sequential:
        for subc in child.children():
            if type(subc) is Rational:
                # rats.append(subc)
                rat = subc


if 'best_fitted_function' not in dir(rat):
    rat.best_fitted_function = None
rat.show()
