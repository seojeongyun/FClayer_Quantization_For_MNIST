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
        name='linear_network_for_mnist'
    )

    solver = dict(
        name='adam',
        gpu_id=0,
        lr0=1e-4,
        momentum=0.937,
        weight_decay=5e-4,
        max_epoch=10,
    )

    scheduler = dict(
        name='cycliclr'
    )
    # Merge all information into a dictionary variable
    config = dict(
        dataset=dataset_info,
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler,
    )

    return config
