from torch.utils.tensorboard import SummaryWriter
from utils.events import write_tbimg, write_tbloss, write_tbPR
from tqdm import tqdm
import time
import os
import torch
import numpy as np

from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device

        # ===== save config =====
        self.save_path = self.make_save_path()
        self.save_file_path = self.make_save_file_path()

        # ===== TensorBoard =====
        self.tblogger = SummaryWriter(self.save_path)

        # ===== DataLoader ======
        self.train_loader, self.val_loader = self.get_dataloader()

        # ===== Model ======
        self.model = self.build_model()

        # ===== Optimizeer ======
        self.optimizer = self.build_optimizer()

        # ===== Scheduler ======
        self.scheduler = self.build_scheduler(self.optimizer)

        # ===== Loss ======
        self.compute_loss = self.set_criterion()

        # ===== Parameters ======
        self.max_epoch = self.cfg['solver']['max_epoch']
        self.max_stepnum = len(self.train_loader) # 1 epoch 내에 몇 번을 도는지

    def make_save_file_path(self):

        # ===== consider model layer =====
        # what kinds of model ?
        # how many stacked layer ?
        # how much dimension of each layer ?

        # ===== consider dropout =====
        # where apply dropout in layers ?
        # how much dropout ratio ?

        # ===== consider model compression technologies =====
        # what kind of model compression technologies ?
        # how much pruning ratio ?
        # other hyperparameters in quantization or knowledge distillation

        what_kind_of_model = self.cfg['model']['type'] + '_'
        how_many_stacked_layer = str(len(self.cfg['model']['layer_dim']) - 1) + '_'
        how_much_dimension_of_each_layer = ''
        for dim in self.cfg['model']['layer_dim']:
            how_much_dimension_of_each_layer += str(dim) + '_'

        where_apply_dropout_in_layers = str(self.cfg['model']['dropout_pos']) + '_'
        how_much_dropout_ratio = str(self.cfg['solver']['dropout']) + '_'

        what_kind_of_model_compression_technologies = self.cfg['compression']['type'] + '_'
        how_much_pruning_ratio = str(self.cfg['compression']['pruning_ratio'])

        save_file_path = what_kind_of_model + \
                         how_many_stacked_layer + \
                         how_much_dimension_of_each_layer + \
                         where_apply_dropout_in_layers + \
                         how_much_dropout_ratio + \
                         what_kind_of_model_compression_technologies + \
                         how_much_pruning_ratio + '.pth'

        return save_file_path
    # 패스 만들 때는 os.path.join 을 많이 사용함
    def make_save_path(self):
        save_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name']) # self.cfg의 ['path']['save_base_path'] 에 self.cfg['model]['name']을 붙임
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def build_scheduler(self, optimizer):
        if self.cfg['scheduler']['name'] == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.9, step_size=5)

        elif self.cfg['scheduler']['name'] == 'cycliclr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4,
                                                         cycle_momentum=False, step_size_up=20, step_size_down=2,
                                                         mode='triangular2')
        else:
            raise NotImplementedError
        return scheduler

    def set_criterion(self):
        return torch.nn.BCEWithLogitsLoss(reduction='sum').to(self.device)

    def build_optimizer(self):
        from solver.fn_optimizer import build_optimizer
        return build_optimizer(self.cfg, self.model)

    def build_model(self):
        model_name = self.cfg['model']['name']
        if model_name == 'lenet':
            raise ValueError('LeNet not implemented in ./model')
            # from model.LeNet import LeNet
            # model = LeNet().to(self.device)
        elif model_name == 'alexnet':
            raise ValueError('AlexNet not implemented in ./model')
            # from model.AlexNet import AlexNet
            # model = AlexNet().to(self.device)
        elif model_name == 'resnet':
            raise ValueError('ResNet not implemented in ./model')
            # from model.ResNet import ResNet
            # model = ResNet().to(self.device)
        elif model_name == 'linear_network_for_mnist':
            from model.linear_network import ClassifierModule
            model = ClassifierModule(layer_dim=self.cfg['model']['layer_dim'],
                                     dropout=self.cfg['solver']['dropout'],
                                     dropout_pos=self.cfg['model']['dropout_pos']).to(self.device)
        else:
            raise NotImplementedError

        return model
    def get_dataloader(self):
        if self.cfg['dataset']['name'] == 'wdm':
            raise ValueError('WDM dataset not exist in ./dataset')
            # from dataset.wdm import Data_loader

        elif self.cfg['dataset']['name'] == 'mnist':
            from dataset.mnist import data_loader

        else:
            raise ValueError('Invalid dataset name,' 'currently supported [wdm]')

        #
        train_path = self.cfg['dataset']['train_path']
        val_path = self.cfg['dataset']['val_path']
        batch_size = self.cfg['dataset']['batch_size']
        num_workers = self.cfg['dataset']['num_workers']
        height, width = self.cfg['dataset']['height'], self.cfg['dataset']['width']
        #

        train_object = data_loader(
            path=train_path,
            height=height,
            width=width,
            augmentation=True,
            task='train'
        )
        #
        train_loader = DataLoader(
            train_object,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_loader.collate_fn
        )

        val_object = data_loader(
            path=val_path,
            height=height,
            width=width,
            augmentation=True,
            task='train'
        )
        #
        val_loader = DataLoader(
            val_object,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_loader.collate_fn
        )

        return train_loader, val_loader

    def start_train(self):
        try:
            for epoch in range(self.max_epoch):
                text = " epoch : {} ".format(epoch+1)
                total_width = 50
                formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
                print(formatted_text)

                self.train_one_epoch(epoch)
            print("Model's state_dict:")
            for param_tensor in self.model.state_dict():
                print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

            print("Save model...")
            save_dir_name = 'weights' if self.cfg['compression']['compress'] == 'off' else 'compressed_weights'
            torch.save(self.model.state_dict(), self.save_path + '/' + save_dir_name + '/' + self.save_file_name)

        except:
            print('ERROR in training loop...')


    def train_one_epoch(self, epoch):
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        #
        # TP = np.zeros(8)
        # FP = np.zeros(8)
        # FN = np.zeros(8)
        #
        pred = []
        true = []
        #
        for step, batch_data in pbar:
            imgs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            #
            out_net = self.model(imgs)
            # Calculate Loss
            loss = self.compute_loss(out_net, labels.float())
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Get statistics
            # TP, FP, FN = self.get_statistics(
            #     self.model.predict(out_net.detach()), labels,
            #     TP, FP, FN
            # )
            if step % 2 == 0:
                write_tbloss(self.tblogger, loss.detach().cpu(),
                             (epoch * self.max_epoch + step))
            #
            pred.append(out_net.argmax(dim=1))
            true.append(labels.argmax(dim=1))

            #
            # if step % 200 == 0:
            #     write_tbPR(self.tblogger, TP, FP, FN, epoch, 'train')

        self.scheduler.step()

        pred = torch.cat(pred, dim=0)
        true = torch.cat(true, dim=0)

        acc = self.accuracy(true, pred).detach().cpu()

        text = " acc : {} ".format(acc)
        total_width = 50
        formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
        print(formatted_text)


    @staticmethod
    def get_statistics(pred, true, TP, FP, FN):
        for defect_idx in range(pred.shape[1]):
            pred_per_defect = pred[:, defect_idx].cpu().detach().numpy()
            true_per_defect = true[:, defect_idx].cpu().detach().numpy()

            TP[defect_idx] += np.sum(pred_per_defect * true_per_defect)
            FP[defect_idx] += np.sum(pred_per_defect * (1 - true_per_defect))
            FN[defect_idx] += np.sum((1 - pred_per_defect) * true_per_defect)

        return TP, FP, FN

    @staticmethod
    def accuracy(true, pred):
        return (true == pred).sum() / true.shape[0]


class Tester():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device

        # ===== save config =====
        self.save_path = self.make_save_path()
        self.save_file_path = self.make_save_file_path()

        # ===== DataLoader ======
        self.train_loader, self.val_loader = self.get_dataloader()

        # ===== Model ======
        self.model = self.build_model()

        # ===== Parameters ======
        self.max_epoch = self.cfg['solver']['max_epoch']
        self.max_stepnum = len(self.train_loader) # 1 epoch 내에 몇 번을 도는지

    def make_save_file_path(self):

        # ===== consider model layer =====
        # what kinds of model ?
        # how many stacked layer ?
        # how much dimension of each layer ?

        # ===== consider dropout =====
        # where apply dropout in layers ?
        # how much dropout ratio ?

        # ===== consider model compression technologies =====
        # what kind of model compression technologies ?
        # how much pruning ratio ?
        # other hyperparameters in quantization or knowledge distillation

        what_kind_of_model = self.cfg['model']['type'] + '_'
        how_many_stacked_layer = str(len(self.cfg['model']['layer_dim']) - 1) + '_'
        how_much_dimension_of_each_layer = ''
        for dim in self.cfg['model']['layer_dim']:
            how_much_dimension_of_each_layer += str(dim) + '_'

        where_apply_dropout_in_layers = str(self.cfg['model']['dropout_pos']) + '_'
        how_much_dropout_ratio = str(self.cfg['solver']['dropout']) + '_'

        what_kind_of_model_compression_technologies = self.cfg['compression']['type'] + '_'
        how_much_pruning_ratio = str(self.cfg['compression']['pruning_ratio'])

        save_file_path = what_kind_of_model + \
                         how_many_stacked_layer + \
                         how_much_dimension_of_each_layer + \
                         where_apply_dropout_in_layers + \
                         how_much_dropout_ratio + \
                         what_kind_of_model_compression_technologies + \
                         how_much_pruning_ratio + '.pth'

        return save_file_path
    # 패스 만들 때는 os.path.join 을 많이 사용함
    def make_save_path(self):
        save_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name']) # self.cfg의 ['path']['save_base_path'] 에 self.cfg['model]['name']을 붙임
        os.makedirs(save_path, exist_ok=True)
        return save_path


    def build_model(self):
        model_name = self.cfg['model']['name']
        if model_name == 'lenet':
            raise ValueError('LeNet not implemented in ./model')
            # from model.LeNet import LeNet
            # model = LeNet().to(self.device)
        elif model_name == 'alexnet':
            raise ValueError('AlexNet not implemented in ./model')
            # from model.AlexNet import AlexNet
            # model = AlexNet().to(self.device)
        elif model_name == 'resnet':
            raise ValueError('ResNet not implemented in ./model')
            # from model.ResNet import ResNet
            # model = ResNet().to(self.device)
        elif model_name == 'linear_network_for_mnist':
            from model.linear_network import ClassifierModule
            model = ClassifierModule(layer_dim=self.cfg['model']['layer_dim'],
                                     dropout=self.cfg['solver']['dropout'],
                                     dropout_pos=self.cfg['model']['dropout_pos']).to(self.device)
        else:
            raise NotImplementedError

        model.load_state_dict(torch.load(self.save_file_path))

        return model
    def get_dataloader(self):
        if self.cfg['dataset']['name'] == 'wdm':
            raise ValueError('WDM dataset not exist in ./dataset')
            # from dataset.wdm import Data_loader

        elif self.cfg['dataset']['name'] == 'mnist':
            from dataset.mnist import data_loader

        else:
            raise ValueError('Invalid dataset name,' 'currently supported [wdm]')

        #
        train_path = self.cfg['dataset']['train_path']
        val_path = self.cfg['dataset']['val_path']
        batch_size = self.cfg['dataset']['batch_size']
        num_workers = self.cfg['dataset']['num_workers']
        height, width = self.cfg['dataset']['height'], self.cfg['dataset']['width']
        #

        train_object = data_loader(
            path=train_path,
            height=height,
            width=width,
            augmentation=True,
            task='train'
        )
        #
        train_loader = DataLoader(
            train_object,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_loader.collate_fn
        )

        val_object = data_loader(
            path=val_path,
            height=height,
            width=width,
            augmentation=True,
            task='train'
        )
        #
        val_loader = DataLoader(
            val_object,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_loader.collate_fn
        )

        return train_loader, val_loader

    def start_train(self):
        try:
            for epoch in range(self.max_epoch):
                text = " epoch : {} ".format(epoch+1)
                total_width = 50
                formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
                print(formatted_text)

                self.train_one_epoch(epoch)
            print("Model's state_dict:")
            for param_tensor in self.model.state_dict():
                print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

            print("Save model...")
            torch.save(self.model.state_dict(), self.save_path + '/weights/' + self.save_file_name)

        except:
            print('ERROR in training loop...')


    def train_one_epoch(self, epoch):
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        #
        # TP = np.zeros(8)
        # FP = np.zeros(8)
        # FN = np.zeros(8)
        #
        pred = []
        true = []
        #
        for step, batch_data in pbar:
            imgs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            #
            out_net = self.model(imgs)
            # Calculate Loss
            loss = self.compute_loss(out_net, labels.float())
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Get statistics
            # TP, FP, FN = self.get_statistics(
            #     self.model.predict(out_net.detach()), labels,
            #     TP, FP, FN
            # )
            if step % 2 == 0:
                write_tbloss(self.tblogger, loss.detach().cpu(),
                             (epoch * self.max_epoch + step))
            #
            pred.append(out_net.argmax(dim=1))
            true.append(labels.argmax(dim=1))

            #
            # if step % 200 == 0:
            #     write_tbPR(self.tblogger, TP, FP, FN, epoch, 'train')

        self.scheduler.step()

        pred = torch.cat(pred, dim=0)
        true = torch.cat(true, dim=0)

        acc = self.accuracy(true, pred).detach().cpu()

        text = " acc : {} ".format(acc)
        total_width = 50
        formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
        print(formatted_text)


    @staticmethod
    def get_statistics(pred, true, TP, FP, FN):
        for defect_idx in range(pred.shape[1]):
            pred_per_defect = pred[:, defect_idx].cpu().detach().numpy()
            true_per_defect = true[:, defect_idx].cpu().detach().numpy()

            TP[defect_idx] += np.sum(pred_per_defect * true_per_defect)
            FP[defect_idx] += np.sum(pred_per_defect * (1 - true_per_defect))
            FN[defect_idx] += np.sum((1 - pred_per_defect) * true_per_defect)

        return TP, FP, FN

    @staticmethod
    def accuracy(true, pred):
        return (true == pred).sum() / true.shape[0]
# class Compressor():