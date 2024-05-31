from config.config import get_config_dict
from core.engine import Trainer

if __name__ == '__main__':
    # Get configuration
    config = get_config_dict()

    # Get Trainer
    trainer = Trainer(config)

    # Start train
    trainer.start_train()
