import pickle
import sys
from rational.torch import Rational

model = pickle.load(open(sys.argv[1], "rb"))
for el in model.modules():
    if type(el) is Rational:
        # import ipdb; ipdb.set_trace()
        print(el)
