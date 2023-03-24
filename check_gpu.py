from typing import List

import pkg_resources
import torch
from opensoundscape.torch.architectures import cnn_architectures
from opensoundscape.torch.models.cnn import CNN

if __name__ == '__main__':
    print(f'PyTorch version (git): {torch.version.__version__}, cuda version: {torch.version.cuda}')
    if torch.cuda.is_available():
        print(f'Current Torch GPU device: {torch.cuda.current_device()}')
        dc: int = torch.cuda.device_count()
        print(f'Torch device count: {dc}:')
        for i in range(0, dc):
            print(f'Device #{i}: [{torch.cuda.device(i)}], device name [{torch.cuda.get_device_name(i)}]')
    else:
        print(f'No Torch GPU devices available')

    model = CNN('resnet18', ["t1", "t2", "t3"], 3.0, single_target=False)
    print(f'OpneSoundccape {pkg_resources.get_distribution("opensoundscape").version}, available architectures:')
    archs: List = cnn_architectures.list_architectures()
    for a in archs:
        print(a)









