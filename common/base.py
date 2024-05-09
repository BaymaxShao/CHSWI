import os
import os.path as osp
import math
import time
import glob
import abc
import numpy as np
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from collections import OrderedDict
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel

from config import cfg
from utils.dir import make_folder
from model import get_model
from model import get_model_branch
from dataset import MultipleDatasets
from utils.human_models import smpl

# dynamic dataset import
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
for i in range(len(cfg.testset)):
    exec('from ' + cfg.testset[i] + ' import ' + cfg.testset[i])

def worker_init_fn(worder_id):
    np.random.seed(np.random.get_state()[1][0] + worder_id)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        total_params = []
        for module in model.module.trainable_modules:
            total_params += list(module.parameters())
        optimizer = torch.optim.Adam(total_params, lr=cfg.lr)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))

        # do not save smpl & smplicit layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smpl_layer' in k:
                dump_key.append(k)
            if 'smplicit_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)

        print("cur_epoch: ", cur_epoch)
        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))
       
        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []
        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []

        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model('train', cfg.type)
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset[0])(transforms.ToTensor(), "test")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True, worker_init_fn=worker_init_fn)
        # print(batch_generator)
        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        # model_path1 = os.path.join(cfg.model_dir1, 'snapshot_%d.pth.tar' % self.test_epoch)
        # model_path2 = os.path.join(cfg.model_dir2, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        # assert os.path.exists(model_path1), 'Cannot find model at ' + model_path1
        # self.logger.info('Load checkpoint from {}'.format(model_path1))
        # assert os.path.exists(model_path2), 'Cannot find model at ' + model_path2
        # self.logger.info('Load checkpoint from {}'.format(model_path2))
        
        # prepare network
        model = get_model('test', cfg.type)
        # model1, model2 = get_model_branch(cfg.type)
        model = DataParallel(model).cuda()
        # model1 = DataParallel(model1).cuda()
        # model2 = DataParallel(model2).cuda()
        # model1.cuda()
        # model2.cuda()
        ckpt = torch.load(model_path)
        # ckpt1 = torch.load(model_path1)
        # ckpt2 = torch.load(model_path2)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()
        # model1.load_state_dict(ckpt1['network'], strict=False)
        # model1.eval()
        # model2.load_state_dict(ckpt2['network'], strict=False)
        # model2.eval()

        self.model = model
        # self.model1 = model1
        # self.model2 = model2
    
    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)

def check_data_parallel(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict