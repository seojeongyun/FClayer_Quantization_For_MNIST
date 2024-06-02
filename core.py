import torch

from config.config import get_config_dict
from core.engine import Trainer, Tester, Compressor

# from core.engine import Compressor
if __name__ == '__main__':
    from setproctitle import *
    setproctitle('FCN_model_compression')

    # Get configuration
    config = get_config_dict()

    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train or compress
    if config['task'] == 'train':
        trainer = Trainer(config, device)
        trainer.start_train()
    elif config['task'] == 'test':
        tester = Tester(config, device)
        tester.start_test()
    elif config['task'] == 'compress':
        compressor = Compressor(config, device)
        compressor.start_compress()


