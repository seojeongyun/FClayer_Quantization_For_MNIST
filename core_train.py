from config.config import get_config_dict
from core.engine import Trainer

if __name__ == '__main__':
    # Get configuration
    config = get_config_dict()

    # Get Trainer
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(config, device)

    # Start train
    trainer.start_train()
