import json
from collections import OrderedDict

import torch

from csl.net import CSLNet


def save_model_json(model, path):
    actual_dict = OrderedDict()
    for k, v in model.state_dict().items():
      actual_dict[k] = v.tolist()
    with open(path, 'w') as f:
      json.dump(actual_dict, f)

if __name__ == '__main__':
    checkpoint = torch.load("/media/linux_hdd/Documents/kp_crc/out/csl/quant/csl.pth")
    model = CSLNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    save_model_json(model, "/media/linux_hdd/Documents/kp_crc/out/csl/quant/csl.json")