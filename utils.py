import torch

def ema(source, target, decay):
    sd, td = source.state_dict(), target.state_dict()
    for k in sd.keys():
        td[k].copy_(td[k] * decay + sd[k] * (1 - decay))
    

