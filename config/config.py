def get_config_dict():
    dataset_info = dict(
        name='mnist',
        train_path='/storage/jysuh/dataset',
        val_path='/storage/jysuh/dataset',
        height=28,
        width=28,
        channel=1,
        batch_size=10,
        num_workers=0,
    )

    path = dict(
        save_base_path='runs'
    )

    model = dict(
        name='linear_network_for_mnist',
        type='FCN',
        layer_dim=[784, 98, 10],
        dropout_pos=0,
    )

    solver = dict(
        name='adam',
        gpu_id=0,
        lr0=1e-4,
        momentum=0.937,
        weight_decay=5e-4,
        max_epoch=25,
        dropout=0.5,
    )

    scheduler = dict(
        name='cycliclr'
    )

    compression = dict(
        type='nothing',
        #
        # the value of self.compression has only 'quantization', 'pruning', 'knowledge_distillation',
        # 'quantization+pruning', 'quantization+knowledge_distillation', 'pruning+knowledge_distillation'
        # 'quantization+pruning+knowledge_distillation'
        # the default value is 'nothing'
        #
        pruning_ratio=0.0,
    )


    # Merge all information into a dictionary variable
    config = dict(
        dataset=dataset_info,
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler,
        compression=compression,
    )

    return config

