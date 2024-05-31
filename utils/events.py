#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)


def write_tbloss(tblogger, losses, step):
    tblogger.add_scalar("training/loss", losses, step + 1)

def write_tbPR(tblogger, TP, FP, FN, epoch, task):
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    for defect_idx in range(TP.shape[0]):
        tblogger.add_scalar("precision/{}/{}".format(defect_idx, task),
                            P[defect_idx], epoch + 1)
        tblogger.add_scalar("recall/{}/{}".format(defect_idx, task),
                            R[defect_idx], epoch + 1)

def write_tbimg(tblogger, imgs, step):
    for i in range(len(imgs)):
        tblogger.add_image('train_imgs/train_batch_{}'.format(i),
                           imgs[i].contiguous().permute(1, 2, 0),
                           step + 1,
                           dataformats='HWC')
