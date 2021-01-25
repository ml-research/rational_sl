import torch
from setproctitle import setproctitle
import time

setproctitle('@FelFri_TransfProto#20s:')
t = torch.tensor([1.]).cuda()
time.sleep(20)
