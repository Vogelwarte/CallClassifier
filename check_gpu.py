import torch

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







