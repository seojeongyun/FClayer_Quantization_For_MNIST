from torch.utils.tensorboard import SummaryWriter
from utils.events import write_tbimg, write_tbloss, write_tbPR
from tqdm import tqdm

import time
import os
import torch
import numpy as np
import time
import torch.nn.functional as F
import json

from torch import nn
from copy import deepcopy
from torch.utils.data import DataLoader
from model.linear_network import ClassifierModule
from model.quantized_linear_model import quantizedLinearModule
from model.quantized_linear_model_for_inference import quantized_model_for_inf

def printf(say_something:str, total_width:int=50, format=None):
    if format is None:
        text = say_something
        total_width = total_width
        formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
        print(formatted_text)
    else:
        text = " student_acc : {} ".format(format)
        total_width = 50
        formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
        print(formatted_text)

class Trainer():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device

        # ===== save config =====
        self.base_path = self.make_base_path()
        self.weight_file_name = self.weight_file_name()
        self.save_dir_name = 'weights'

        # # ===== TensorBoard =====
        # self.tblogger = SummaryWriter(self.save_path)

        # ===== DataLoader ======
        self.train_loader = self.get_dataloader()

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

    def weight_file_name(self):

        # ===== consider model layer =====
        # what kinds of model ?
        # how many stacked layer ?
        # how much dimension of each layer ?

        # ===== consider dropout =====
        # where apply dropout in layers ?
        # how much dropout ratio ?

        # ===== consider model compression technologies =====
        # what type of pruning ?
        # how much pruning ratio ?
        # the_number_of_n
        # other hyperparameters in quantization or knowledge distillation

        # ===== consider other hyperparameters =====
        # how much epoch ?
        # how_much_qat_epoch ?
        # how_much_qat_epoch ?
        # how much batch_size ?

        what_kind_of_model = self.cfg['model']['type'] + '_'                            # FNC or CNN
        #
        if self.cfg['model']['atfc'] == 'ReLU6':
            what_type_of_atfc = self.cfg['model']['atfc'] + str(self.cfg['solver']['relu6_alpha']) + '_'
        else:
            what_type_of_atfc = self.cfg['model']['atfc'] + '_'
        #
        how_many_stacked_layer = str(len(self.cfg['model']['layer_dim']) - 1) + '_'     # the number of layers
        how_much_dimension_of_each_layer = ''
        for dim in self.cfg['model']['layer_dim']:
            how_much_dimension_of_each_layer += str(dim) + '_'                          # print each layer

        where_apply_dropout_in_layers = str(self.cfg['model']['dropout_pos']) + '_'
        how_much_dropout_ratio = str(self.cfg['solver']['dropout']) + '_'

        what_type_of_pruning = str(self.cfg['compression']['pruning_type']) + '_'
        how_much_pruning_ratio = str(self.cfg['compression']['pruning_ratio']) + '_'
        the_number_of_n = str(self.cfg['compression']['pruning_n']) + '_'

        how_much_epoch = str(self.cfg['solver']['max_epoch']) + '_'
        how_much_kd_epoch = str(self.cfg['compression']['kd_epoch']) + '_'
        how_much_qat_epoch = str(self.cfg['compression']['qat_epoch']) + '_'
        how_much_batch_size = str(self.cfg['dataset']['batch_size'])
        if self.cfg['compression']['pruning_type'] == 'ln_structured':
            weight_file_name = what_kind_of_model + \
                               what_type_of_atfc + \
                               how_many_stacked_layer + \
                               how_much_dimension_of_each_layer + \
                               where_apply_dropout_in_layers + \
                               how_much_dropout_ratio + \
                               what_type_of_pruning + \
                               how_much_pruning_ratio + \
                               the_number_of_n + \
                               how_much_epoch + \
                               how_much_qat_epoch + \
                               how_much_kd_epoch + \
                               how_much_batch_size + '.pth'
        else:
            weight_file_name = what_kind_of_model + \
                               what_type_of_atfc + \
                               how_many_stacked_layer + \
                               how_much_dimension_of_each_layer + \
                               where_apply_dropout_in_layers + \
                               how_much_dropout_ratio + \
                               what_type_of_pruning + \
                               how_much_pruning_ratio + \
                               how_much_epoch + \
                               how_much_qat_epoch + \
                               how_much_kd_epoch + \
                               how_much_batch_size + '.pth'

        return weight_file_name

    # 패스 만들 때는 os.path.join 을 많이 사용함
    def make_base_path(self):
        base_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name']) # self.cfg의 ['path']['save_base_path'] 에 self.cfg['model]['name']을 붙임
        os.makedirs(base_path, exist_ok=True)
        return base_path

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
            if self.cfg['compression']['type'] == 'knowledge_distillation':
                model = ClassifierModule(layer_dim=self.cfg['model']['t_layer_dim'],
                                         atfc=self.cfg['model']['atfc'],
                                         alpha=self.cfg['solver']['relu6_alpha'],
                                         dropout=self.cfg['solver']['dropout'],
                                         dropout_pos=self.cfg['model']['dropout_pos'])
            else:
                model = ClassifierModule(layer_dim=self.cfg['model']['layer_dim'],
                                         atfc=self.cfg['model']['atfc'],
                                         alpha=self.cfg['solver']['relu6_alpha'],
                                         dropout=self.cfg['solver']['dropout'],
                                         dropout_pos=self.cfg['model']['dropout_pos'])
        else:
            raise NotImplementedError

        return model.to(self.device)
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

        return train_loader

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
            torch.save(self.model.state_dict(), self.base_path + '/' + self.save_dir_name + '/' + self.weight_file_name)

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
        self.base_path = self.make_base_path()
        self.weight_file_name = self.weight_file_name()
        self.load_dir_name = 'weights'

        # ===== DataLoader ======
        self.val_loader = self.get_dataloader()

        # ===== Model ======
        self.model = self.build_model()

        # ===== Parameters ======
        self.max_epoch = self.cfg['solver']['max_epoch']
        self.max_stepnum = len(self.val_loader) # 1 epoch 내에 몇 번을 도는지

    def weight_file_name(self):

        # ===== consider model layer =====
        # what kinds of model ?
        # how many stacked layer ?
        # how much dimension of each layer ?

        # ===== consider dropout =====
        # where apply dropout in layers ?
        # how much dropout ratio ?

        # ===== consider model compression technologies =====
        # what type of pruning ?
        # how much pruning ratio ?
        # the_number_of_n
        # other hyperparameters in quantization or knowledge distillation

        # ===== consider other hyperparameters =====
        # how much epoch ?
        # how_much_qat_epoch ?
        # how_much_qat_epoch ?
        # how much batch_size ?

        what_kind_of_model = self.cfg['model']['type'] + '_'                            # FNC or CNN
        what_type_of_atfc = self.cfg['model']['atfc'] + '_'
        how_many_stacked_layer = str(len(self.cfg['model']['layer_dim']) - 1) + '_'     # the number of layers
        how_much_dimension_of_each_layer = ''
        for dim in self.cfg['model']['layer_dim']:
            how_much_dimension_of_each_layer += str(dim) + '_'                          # print each layer

        where_apply_dropout_in_layers = str(self.cfg['model']['dropout_pos']) + '_'
        how_much_dropout_ratio = str(self.cfg['solver']['dropout']) + '_'

        what_type_of_pruning = str(self.cfg['compression']['pruning_type']) + '_'
        how_much_pruning_ratio = str(self.cfg['compression']['pruning_ratio']) + '_'
        the_number_of_n = str(self.cfg['compression']['pruning_n']) + '_'

        how_much_epoch = str(self.cfg['solver']['max_epoch']) + '_'
        how_much_kd_epoch = str(self.cfg['compression']['kd_epoch']) + '_'
        how_much_qat_epoch = str(self.cfg['compression']['qat_epoch']) + '_'
        how_much_batch_size = str(self.cfg['dataset']['batch_size'])
        if self.cfg['compression']['pruning_type'] == 'ln_structured':
            weight_file_name = what_kind_of_model + \
                               what_type_of_atfc + \
                               how_many_stacked_layer + \
                               how_much_dimension_of_each_layer + \
                               where_apply_dropout_in_layers + \
                               how_much_dropout_ratio + \
                               what_type_of_pruning + \
                               how_much_pruning_ratio + \
                               the_number_of_n + \
                               how_much_epoch + \
                               how_much_qat_epoch + \
                               how_much_kd_epoch + \
                               how_much_batch_size + '.pth'
        else:
            weight_file_name = what_kind_of_model + \
                               what_type_of_atfc + \
                               how_many_stacked_layer + \
                               how_much_dimension_of_each_layer + \
                               where_apply_dropout_in_layers + \
                               how_much_dropout_ratio + \
                               what_type_of_pruning + \
                               how_much_pruning_ratio + \
                               how_much_epoch + \
                               how_much_qat_epoch + \
                               how_much_kd_epoch + \
                               how_much_batch_size + '.pth'

        return weight_file_name
    # 패스 만들 때는 os.path.join 을 많이 사용함
    def make_base_path(self):
        base_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name']) # self.cfg의 ['path']['save_base_path'] 에 self.cfg['model]['name']을 붙임
        os.makedirs(base_path, exist_ok=True)
        return base_path


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
            if self.cfg['compression']['type'] == 'knowledge_distillation':
                model = ClassifierModule(layer_dim=self.cfg['model']['t_layer_dim'],
                                         atfc=self.cfg['model']['atfc'],
                                         alpha=self.cfg['solver']['relu6_alpha'],
                                         dropout=self.cfg['solver']['dropout'],
                                         dropout_pos=self.cfg['model']['dropout_pos'])
            else:
                model = ClassifierModule(layer_dim=self.cfg['model']['layer_dim'],
                                         atfc=self.cfg['model']['atfc'],
                                         alpha=self.cfg['solver']['relu6_alpha'],
                                         dropout=self.cfg['solver']['dropout'],
                                         dropout_pos=self.cfg['model']['dropout_pos'])
            print("Load model..")
            model.load_state_dict(torch.load(self.base_path + '/' + self.load_dir_name + '/' + self.weight_file_name))
            print("Model load success")
        else:
            raise NotImplementedError

        print("Check the model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        return model.to(self.device)

    def get_dataloader(self):
        if self.cfg['dataset']['name'] == 'wdm':
            raise ValueError('WDM dataset not exist in ./dataset')
            # from dataset.wdm import Data_loader

        elif self.cfg['dataset']['name'] == 'mnist':
            from dataset.mnist import data_loader

        else:
            raise ValueError('Invalid dataset name,' 'currently supported [wdm]')

        #
        val_path = self.cfg['dataset']['val_path']
        batch_size = self.cfg['dataset']['batch_size']
        num_workers = self.cfg['dataset']['num_workers']
        height, width = self.cfg['dataset']['height'], self.cfg['dataset']['width']
        #
        val_object = data_loader(
            path=val_path,
            height=height,
            width=width,
            augmentation=True,
            task='test'
        )

        #
        val_loader = DataLoader(
            val_object,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_loader.collate_fn
        )

        return val_loader

    def start_test(self):
        try:
            self.model.eval()
            text = " Test start "
            total_width = 50
            formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
            print(formatted_text)

            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            #
            pred = []
            true = []
            #
            # ============= test start =============
            normal_model_time_start = time.time()
            for step, batch_data in pbar:
                imgs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                #
                out_net = self.model(imgs)

                #
                pred.append(out_net.argmax(dim=1))
                true.append(labels.argmax(dim=1))
                #
            normal_model_time_end = time.time()
            # ============= test end =============

            pred = torch.cat(pred, dim=0)
            true = torch.cat(true, dim=0)

            acc = self.accuracy(true, pred).detach().cpu()

            text = " acc : {} ".format(acc)
            total_width = 50
            formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
            print(formatted_text)

            perf_time_of_normal_model = normal_model_time_end - normal_model_time_start
            print(f"{perf_time_of_normal_model:.5f} sec\n")

        except:
            print('ERROR in test ...')

    @staticmethod
    def accuracy(true, pred):
        return (true == pred).sum() / true.shape[0]


class Compressor():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device

        # ===== load and save config =====
        self.base_path = self.make_base_path()
        #
        self.weight_file_name = self.weight_file_name()
        #
        self.load_dir_name = 'weights'
        self.save_dir_name = 'compressed_weights'

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
        self.qat_epoch = self.cfg['compression']['qat_epoch']
        self.kd_epoch = self.cfg['compression']['kd_epoch']

        self.max_stepnum = len(self.val_loader)  # 1 epoch 내에 몇 번을 도는지

        # ===== method dict =====
        self.method_dict = self.get_method_dict()

    def get_method_dict(self):
        dict = {
            'ptq': self.fn_ptq,
            'qat': self.fn_qat,
            'kd': self.fn_kd,
            'pruning': self.fn_prune
        }
        return dict

    def weight_file_name(self):

        # ===== consider model layer =====
        # what kinds of model ?
        # how many stacked layer ?
        # how much dimension of each layer ?

        # ===== consider dropout =====
        # where apply dropout in layers ?
        # how much dropout ratio ?

        # ===== consider model compression technologies =====
        # what type of pruning ?
        # how much pruning ratio ?
        # the_number_of_n
        # other hyperparameters in quantization or knowledge distillation

        # ===== consider other hyperparameters =====
        # how much epoch ?
        # how_much_qat_epoch ?
        # how_much_qat_epoch ?
        # how much batch_size ?

        what_kind_of_model = self.cfg['model']['type'] + '_'                            # FNC or CNN
        what_type_of_atfc = self.cfg['model']['atfc'] + '_'
        how_many_stacked_layer = str(len(self.cfg['model']['layer_dim']) - 1) + '_'     # the number of layers
        how_much_dimension_of_each_layer = ''
        for dim in self.cfg['model']['layer_dim']:
            how_much_dimension_of_each_layer += str(dim) + '_'                          # print each layer

        where_apply_dropout_in_layers = str(self.cfg['model']['dropout_pos']) + '_'
        how_much_dropout_ratio = str(self.cfg['solver']['dropout']) + '_'

        what_type_of_pruning = str(self.cfg['compression']['pruning_type']) + '_'
        how_much_pruning_ratio = str(self.cfg['compression']['pruning_ratio']) + '_'
        the_number_of_n = str(self.cfg['compression']['pruning_n']) + '_'

        how_much_epoch = str(self.cfg['solver']['max_epoch']) + '_'
        how_much_kd_epoch = str(self.cfg['compression']['kd_epoch']) + '_'
        how_much_qat_epoch = str(self.cfg['compression']['qat_epoch']) + '_'
        how_much_batch_size = str(self.cfg['dataset']['batch_size'])
        if self.cfg['compression']['pruning_type'] == 'ln_structured':
            weight_file_name = what_kind_of_model + \
                               what_type_of_atfc + \
                               how_many_stacked_layer + \
                               how_much_dimension_of_each_layer + \
                               where_apply_dropout_in_layers + \
                               how_much_dropout_ratio + \
                               what_type_of_pruning + \
                               how_much_pruning_ratio + \
                               the_number_of_n + \
                               how_much_epoch + \
                               how_much_qat_epoch + \
                               how_much_kd_epoch + \
                               how_much_batch_size + '.pth'
        else:
            weight_file_name = what_kind_of_model + \
                               what_type_of_atfc + \
                               how_many_stacked_layer + \
                               how_much_dimension_of_each_layer + \
                               where_apply_dropout_in_layers + \
                               how_much_dropout_ratio + \
                               what_type_of_pruning + \
                               how_much_pruning_ratio + \
                               how_much_epoch + \
                               how_much_qat_epoch + \
                               how_much_kd_epoch + \
                               how_much_batch_size + '.pth'

        return weight_file_name

        # 패스 만들 때는 os.path.join 을 많이 사용함

    def make_base_path(self):
        base_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model'][
                                     'name'])  # self.cfg의 ['path']['save_base_path'] 에 self.cfg['model]['name']을 붙임
        os.makedirs(base_path, exist_ok=True)
        return base_path

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
            if self.cfg['compression']['type'] == 'knowledge_distillation':
                model = ClassifierModule(layer_dim=self.cfg['model']['t_layer_dim'],
                                         atfc=self.cfg['model']['atfc'],
                                         alpha=self.cfg['solver']['relu6_alpha'],
                                         dropout=self.cfg['solver']['dropout'],
                                         dropout_pos=self.cfg['model']['dropout_pos'])
            else:
                model = ClassifierModule(layer_dim=self.cfg['model']['layer_dim'],
                                         atfc=self.cfg['model']['atfc'],
                                         alpha=self.cfg['solver']['relu6_alpha'],
                                         dropout=self.cfg['solver']['dropout'],
                                         dropout_pos=self.cfg['model']['dropout_pos'])

            print("Load model..")
            model.load_state_dict(torch.load(self.base_path + '/' + self.load_dir_name + '/' + self.weight_file_name))
            print("Model load success")

        else:
            raise NotImplementedError

        print("Check the model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        return model.to(self.device)

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
        #
        val_object = data_loader(
            path=val_path,
            height=height,
            width=width,
            augmentation=True,
            task='test'
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

    # ==============================================================================
    # ============================ Quantization Method =============================
    # ==============================================================================

    def build_qat_model(self, model):
        printf(say_something='building qat model ..', total_width=50)
        fused_model = deepcopy(model.to(self.device))
        # The model has to be switched to training mode before any layer fusion.
        fused_model.train()
        # TODO: we have to implement a fusing function into a convBNReLU module class.
        #       Then, the fusing function should be conducted here.
        # fused_model.eval()

        qat_model = quantizedLinearModule(model_fp32=fused_model)

        backend = "fbgemm"
        # Default qconfig (quantization configuration)
        # quantization_config = torch.quantization.get_default_qconfig(backend)

        # Custom qconfig
        quantization_config = torch.quantization.get_default_qconfig
        quantization_config = torch.quantization.QConfig(
            activation=torch.quantization.MovingAverageMinMaxObserver.with_args(dtype=torch.quint8,
                                                                                qscheme=torch.per_tensor_affine),
            weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )
        qat_model.qconfig = quantization_config
        torch.quantization.prepare_qat(qat_model, inplace=True)

        return qat_model.to('cpu')

    def build_ptq_model(self, model):
        printf(say_something='building ptq model ..', total_width=50)

        fused_model = deepcopy(model.to(self.device))
        # The model has to be switched to training mode before any layer fusion.
        fused_model.train()
        # TODO: we have to implement a fusing function into a convBNReLU module class.
        #       Then, the fusing function should be conducted here.
        fused_model.eval()

        quantized_model_ptq = quantizedLinearModule(model_fp32=fused_model)

        backend = "fbgemm"
        # Default qconfig (quantization configuration)
        # quantization_config = torch.quantization.get_default_qconfig(backend)

        # Custom qconfig
        quantization_config = torch.quantization.get_default_qconfig
        quantization_config = torch.quantization.QConfig(
            activation=torch.quantization.MovingAverageMinMaxObserver.with_args(dtype=torch.quint8,
                                                                                qscheme=torch.per_tensor_affine),
            weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )

        quantized_model_ptq.qconfig = quantization_config

        torch.quantization.prepare(quantized_model_ptq, inplace=True)

        self.calibrate_model(model=quantized_model_ptq, num_batches=5)

        torch.quantization.convert(quantized_model_ptq, inplace=True)

        printf(say_something='Check whether a trained float model is quantized')
        print(torch.int_repr(quantized_model_ptq.get_submodule('model_fp32').get_submodule('layers').get_submodule('1')._weight_bias()[0]))

        return quantized_model_ptq

    def calibrate_model(self, model, num_batches, device=torch.device("cpu:0")):
        model.to(device)
        model.eval()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))

        for i, batch_data in pbar:
            imgs = batch_data[0].to(device)
            _ = model(imgs)
            if i >= num_batches:
                break

    #
    def qat_train(self, model):
        for epoch in range(self.qat_epoch):
            text = " qat_train_epoch : {} ".format(epoch + 1)
            total_width = 50
            formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
            print(formatted_text)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

            # ======= qat_train start =======
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            #
            pred_list = []
            true_list = []
            #
            for step, batch_data in pbar:
                imgs = batch_data[0].to('cpu')
                labels = batch_data[1].to('cpu')
                #
                pred = model(imgs)

                # Calculate Loss
                loss = self.compute_loss(pred, labels.float())

                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #
                pred_list.append(pred.argmax(dim=1))
                true_list.append(labels.argmax(dim=1))

            self.scheduler.step()

            pred = torch.cat(pred_list, dim=0)
            true = torch.cat(true_list, dim=0)
            printf(say_something='     ', total_width=50)
        printf(say_something='qat_train_end', total_width=50)
            # ======= qat_train end =======

        torch.quantization.convert(model, inplace=True)

        # quantized_model = quantized_model_for_inf(quantized_model=model,
        #                       layer_dim=self.cfg['model']['layer_dim'],
        #                       dropout=self.cfg['solver']['dropout'],
        #                       dropout_pos=self.cfg['model']['dropout_pos'])

        # print("QAT_Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", quantized_model.state_dict()[param_tensor].size())

        # print("Save QAT Model...")
        # torch.save(quantized_model.state_dict(),
        #            self.base_path + '/' + self.save_dir_name + '/' + self.weight_file_name)

        return model

    def fn_qat(self, model):
        model = self.build_qat_model(model)
        model = self.qat_train(model)
        return model

    def fn_ptq(self, model):
        model = self.build_ptq_model(model)
        return model

    # =============================================================================
    # =============================== Prune Method ================================
    # =============================================================================

    def fn_prune(self, model):
        import torch.nn.utils.prune as prune
        #
        if self.weight_file_name.split('_')[0] == 'FCN':
            #
            if self.cfg['compression']['pruning_type'] == 'global_unstructured':
                layers_parameters = []
                #
                for module in model.named_modules():
                    if 'layers' in module[0] and not isinstance(module[1], torch.nn.ModuleList):
                        layers_parameters.append((module[1], 'weight'))
                #
                parameters_to_prune = tuple(layers_parameters)
                #
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.cfg['compression']['pruning_ratio'],
                )

                # prune.remove(module, 'weight')

                # for _, bias in enumerate(parameters_to_prune):
                #     prune.ln_structured(bias[0], name="bias",
                #                         amount=self.cfg['compression']['pruning_ratio'],
                #                         n=2,
                #                         dim=0)
                #
                # # for _, bias in enumerate(parameters_to_prune):
                # #     prune.ln_structured(torch.unsqueeze(bias[0].bias, dim=1), name="bias",
                # #                         amount=self.cfg['compression']['pruning_ratio'],
                # #                         n=2,
                # #                         dim=0)

            elif self.cfg['compression']['pruning_type'] == 'ln_structured' and self.cfg['compression']['pruning_n'] == '1':
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.ln_structured(module, name="weight", amount=self.cfg['compression']['pruning_ratio'], n=1, dim=0)
                        prune.remove(module, 'weight')

            elif self.cfg['compression']['pruning_type'] == 'ln_structured' and self.cfg['compression']['pruning_n'] == '2':
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.ln_structured(module, name="weight", amount=self.cfg['compression']['pruning_ratio'], n=2, dim=0)
                        prune.remove(module, 'weight')

            elif self.cfg['compression']['pruning_type'] == 'l1_unstructured':
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=self.cfg['compression']['pruning_ratio'])
                        prune.l1_unstructured(module, name='bias', amount=self.cfg['compression']['pruning_ratio'])
                        prune.remove(module, 'weight')
                        prune.remove(module, 'bias')


            elif self.cfg['compression']['pruning_type'] == 'random_unstructured':
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.random_unstructured(module, name="weight", amount=self.cfg['compression']['pruning_ratio'])
                        prune.random_unstructured(module, name="bias", amount=self.cfg['compression']['pruning_ratio'])
                        prune.remove(module, 'weight')
                        prune.remove(module, 'bias')

            else:
                raise NotImplementedError


        elif self.weight_file_name.split('_')[0] == 'CNN':
            raise NotImplementedError

        return model

    # ========================================================================================
    # ============================ Knowledge Distillation Method =============================
    # ========================================================================================

    def fn_kd(self, S_model=None):
        if self.weight_file_name.split('_')[0] == 'FCN':
            # Load Teacher model
            state_dict = torch.load(self.cfg['compression']['teacher_model_path'])
            teacher_model = ClassifierModule(layer_dim=self.cfg['compression']['t_layer_dim'],
                                             atfc=self.cfg['model']['atfc'],
                                             alpha=self.cfg['solver']['relu6_alpha'],
                                             dropout=self.cfg['solver']['dropout'],
                                             dropout_pos=self.cfg['model']['dropout_pos'])
            teacher_model.load_state_dict(state_dict)
            teacher_model.to(self.device);

            try:
                if S_model is not None:
                    student_model = S_model

            except:
                print("S_model is set None")

            if student_model == self.model:
                student_model.to(self.device)
            else:
                # student_model.to('cpu')
                student_model.to(self.device)
            # # Load Student model
            # if self.cfg['compression']['distill_type'] == 'qat':
            #     student_model = self.build_qat_model(self.model())
            #     student_model.to('cpu')
            # elif self.cfg['compression']['distill_type'] == 'ptq':
            #     student_model = self.build_ptq_model(self.model())
            #     student_model.to('cpu')
            # else:
            #     student_model = self.model()
            #     student_model.to(self.device)
            #
            # student_model.eval()
            # student_model.train()

        try:
            for epoch in range(self.kd_epoch):
                text = " distill_epoch : {} ".format(epoch + 1)
                total_width = 50
                formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
                print(formatted_text)

                optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4, weight_decay=5e-4)


                # ======= knowledge distillation start =======
                pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
                #
                pred = []
                true = []
                #
                # teacher_model = teacher_model.to('cpu')
                # student_model = student_model.to('cpu')
                for step, batch_data in pbar:
                    # imgs = batch_data[0].to(self.device)
                    # labels = batch_data[1].to(self.device)
                    imgs = batch_data[0].to(self.device)
                    labels = batch_data[1].to(self.device)
                    #
                    teacher_pred = teacher_model.to(self.device)(imgs)
                    student_pred = student_model.to(self.device)(imgs)

                    # Calculate Loss
                    loss = Compressor.distillation(student_pred, labels, teacher_pred, self.cfg['compression']['t'],
                                                   self.cfg['compression']['alpha'])

                    # Update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #
                    pred.append(student_pred.argmax(dim=1))
                    true.append(labels.argmax(dim=1))

                self.scheduler.step()

                pred = torch.cat(pred, dim=0)
                true = torch.cat(true, dim=0)

                acc = self.accuracy(true, pred).detach().cpu()

                text = " student_acc : {} ".format(acc)
                total_width = 50
                formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
                print(formatted_text)
                # ======= knowledge distillation end =======

            # print("KD Model's state_dict:")
            # for param_tensor in self.model.state_dict():
            #     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
            #
            # print("Save model...")
            # torch.save(self.model.state_dict(),
            #            self.base_path + '/' + self.save_dir_name + '/' + self.weight_file_name)

        except:
            print('ERROR in distillation loop...')

        return student_model

    def start_test(self, model, type, result_dict):
        try:
            #
            model.to(self.device)
            model.eval()
            #
            text = " " + type + " Test start "
            total_width = 50
            formatted_text = "\n{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
            print(formatted_text)

            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            #
            pred = []
            true = []
            #
            # ============= test start =============
            time_start = time.time()
            for step, batch_data in pbar:
                imgs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                #
                out_net = model.to(self.device)(imgs)
                #
                pred.append(out_net.argmax(dim=1))
                true.append(labels.argmax(dim=1))
                #
            time_end = time.time()
            # ============= test end =============

            pred = torch.cat(pred, dim=0)
            true = torch.cat(true, dim=0)

            acc = self.accuracy(true, pred).detach().cpu()

            text = " acc : {} ".format(acc)
            total_width = 50
            formatted_text = "{0:=>{width}}".format(text.center(total_width, '='), width=total_width)
            print(formatted_text)

            op_time = time_end - time_start
            text = " operation time : {} ".format(round(op_time, 3))
            formatted_text = "{0:=>{width}}\n\n".format(text.center(total_width, '='), width=total_width)
            print(formatted_text)

            # result_dict[str(type)]['model'] = model
            result_dict[str(type)]['op_time'] = round(op_time, 3)
            result_dict[str(type)]['model_acc'] = round(float(acc), 5)

            return result_dict

        except:
            print('ERROR in test ...')

    def start_compress(self):
        # 'ptq', 'qat', 'kd', 'pruning', 'kd+pruning', 'ptq+pruning', 'qat+pruning', 'qat+kd', 'qat+kd+pruning'
        # 'ptq': self.fn_ptq, 'qat': self.fn_qat, 'kd': self.fn_kd, 'pruning': self.fn_prune

        try:
            result_dict = dict()
            for method_type in range(len(self.cfg['compression']['type'])):
                method_list = []
                #
                model = deepcopy(self.model)                                                    # model deep copy
                #
                method_list = self.cfg['compression']['type'][method_type].split('+')           # comp type split
                #
                result_dict.setdefault(self.cfg['compression']['type'][method_type], {})
                #
                printf(say_something=self.cfg['compression']['type'][method_type] + " compression start")

                if method_list[0] == 'normal':
                    result_dict = self.start_test(model, self.cfg['compression']['type'][method_type], result_dict)

                else:
                    for method_idx in range(len(method_list)):
                        sel_method = method_list.pop(0)
                        #
                        if sel_method == 'ptq' or sel_method == 'qat':
                            model = self.method_dict[sel_method](model)

                            ptq2linear = ClassifierModule(layer_dim=self.cfg['model']['layer_dim'],
                                                          atfc=self.cfg['model']['atfc'],
                                                          alpha=self.cfg['solver']['relu6_alpha'],
                                                          dropout=self.cfg['solver']['dropout'],
                                                          dropout_pos=self.cfg['model']['dropout_pos'])
                            layers_parameters = []

                            for module in ptq2linear.named_modules():
                                if 'layers.' in module[0]:
                                    layers_parameters.append(module[0])

                            for idx, layer_name in enumerate(layers_parameters):
                                ptq_param = model.get_submodule('model_fp32').get_submodule('layers').get_submodule(str(idx))._weight_bias()[0]
                                int_weight = torch.int_repr(ptq_param)
                                ptq2linear.get_submodule(layer_name).weight = torch.nn.Parameter(int_weight.float())

                            model = ptq2linear
                        #
                        else:
                            model = self.method_dict[sel_method](model)
                    #
                    result_dict = self.start_test(model, self.cfg['compression']['type'][method_type], result_dict)
            with open(f'/home/jysuh/PycharmProjects/FClayer_Quantization_For_MNIST/result/{self.weight_file_name}.json', 'w') as f:
                json.dump(result_dict, f, indent=4)
        except:
            print("sibal")

    @staticmethod
    def accuracy(true, pred):
        return (true == pred).sum() / true.shape[0]

    @staticmethod
    # knowledge distillation loss
    def distillation(y, labels, teacher_scores, T, alpha):
        # distillation loss + classification loss
        # y: student
        # labels: hard label
        # teacher_scores: soft label
        student_loss = F.cross_entropy(y, labels) * (1. - alpha)  # hard loss with hard label
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1)  # soft loss with teacher logits
        ) * (T * T * 2.0 + alpha)
        return student_loss + distillation_loss