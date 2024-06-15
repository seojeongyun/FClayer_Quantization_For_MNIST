def get_config_dict():
    dataset_info = dict(
        name='mnist',
        train_path='/storage/jysuh/dataset',
        val_path='/storage/jysuh/dataset',
        height=28,
        width=28,
        channel=1,
        batch_size=100,
        num_workers=0,
    )

    path = dict(
        save_base_path='runs'
    )

    model = dict(
        name='linear_network_for_mnist',
        type='FCN',
        layer_dim=[784, 700, 600, 500, 400, 300, 200, 100, 50, 25, 10],
        dropout_pos=8,          #if the first layer is [784, 98]0,1,2 ...
    )

    solver = dict(
        name='adam',
        gpu_id=0,
        lr0=1e-4,
        momentum=0.937,
        weight_decay=5e-4,
        max_epoch=50,
        dropout=0.2,
    )

    scheduler = dict(
        name='cycliclr'
    )

    compression = dict(
        type=['normal', 'ptq', 'qat', 'kd', 'pruning', 'kd+pruning', 'ptq+pruning', 'qat+pruning', 'qat+kd', 'qat+kd+pruning'],
        # 'normal', 'ptq', 'qat', 'kd', 'pruning', 'kd+pruning',
        # ===== parameters for pruning =====
        pruning_type='global_unstructured', # random_unstructured, l1_unstructured, ln_structured, global_unstructured
        pruning_n='1', # 1 or 2
        pruning_ratio=0.5,

        # ===== parameters for knowledge distillation =====
        teacher_model_path='/home/jysuh/PycharmProjects/FClayer_Quantization_For_MNIST/runs/linear_network_for_mnist/weights/teacher_model.pth',
        t_layer_dim=[784, 700, 600, 500, 400, 300, 200, 100, 50, 25, 10],
        kd_epoch=1,
        t=20.0,
        alpha=0.7,

        # ===== parameters for quantization =====
        qat_epoch=1,
        # the value of self.compression has only 'quantization', 'pruning', 'knowledge_distillation',
        # 'quantization+pruning', 'quantization+knowledge_distillation', 'pruning+knowledge_distillation'
        # 'quantization+pruning+knowledge_distillation'
        # the default value is 'nothing'
        #
        distill_type='qat', # qat or ptq or nothing
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