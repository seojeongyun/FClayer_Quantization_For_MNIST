def get_config_dict():
    dataset_info = dict(
        name='mnist',
        train_path='/storage/jysuh/dataset',
        val_path='/storage/jysuh/dataset',
        height=28,
        width=28,
        channel=1,
        batch_size=50,
        num_workers=0,
    )

    path = dict(
        save_base_path='runs'
    )

    model = dict(
        name='linear_network_for_mnist',
        type='FCN',
        layer_dim=[784, 98, 60, 10],
        t_layer_dim=[784, 98, 60, 30, 10],
        dropout_pos=0,          #if the first layer is [784, 98]0,1,2 ...
    )

    solver = dict(
        name='adam',
        gpu_id=0,
        lr0=1e-4,
        momentum=0.937,
        weight_decay=5e-4,
        max_epoch=30,
        dropout=0.2,
    )

    scheduler = dict(
        name='cycliclr'
    )

    compression = dict(
        type='pruning',
        pruning_type='random_unstructured', # random_unstructured, l1_unstructured, ln_structured, global_unstructured
        pruning_n='1', # 1 or 2
        #
        # the value of self.compression has only 'quantization', 'pruning', 'knowledge_distillation',
        # 'quantization+pruning', 'quantization+knowledge_distillation', 'pruning+knowledge_distillation'
        # 'quantization+pruning+knowledge_distillation'
        # the default value is 'nothing'
        #
        pruning_ratio=0.5,
    )


    # Merge all information into a dictionary variable
    config = dict(
        task='compress',            # 'train' or 'test' or 'compress'
        dataset=dataset_info,
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler,
        compression=compression,
    )

    return config

#       acc      : dropout = 0.2 / pruning = 0.5 > dropout = 0.5 and pruning = 0.5
# inference time : normal model is faster than pruned model in pruning_ratio = 0.5