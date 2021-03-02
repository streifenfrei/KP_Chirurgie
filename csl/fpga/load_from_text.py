import json
from collections import OrderedDict

import torch

from csl.net import CSLNet


def load_model_json(model, path):
  data_dict = OrderedDict()
  with open(path, 'r') as f:
    data_dict = json.load(f)
  own_state = model.state_dict()
  for k, v in data_dict.items():
    print('Loading parameter:', k)
    if not k in own_state:
      print('Parameter', k, 'not found in own_state!!!')
    if type(v) == list or type(v) == int:
      v = torch.tensor(v)
    own_state[k].copy_(v)
  model.load_state_dict(own_state)
  print('Model loaded')

if __name__ == '__main__':
    model = CSLNet()
    load_model_json(model,"/media/linux_hdd/Documents/kp_crc/out/csl/quant/csl.json")
    torch.save({'model_state_dict': model.state_dict()}, "/media/linux_hdd/Documents/kp_crc/out/csl/quant/csl_old.pth")