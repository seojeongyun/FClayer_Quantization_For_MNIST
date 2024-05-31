import torch

def build_optimizer(cfg, model):
    if cfg['solver']['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg['solver']['lr0'],
            momentum=cfg['solver']['momentum'],
            nesterov=True,
            weight_decay=5e-4
        )

    elif cfg['solver']['name'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg['solver']['lr0'],
            weight_decay=cfg['solver']['weight_decay']
        )

    else:
        raise NotImplementedError
    return optimizer