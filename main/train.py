import os
import argparse
from config import cfg
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision import transforms
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"
 
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.benchmark = True

    from base import Trainer
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    losses = []

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            if cfg.type == 'vit' or cfg.type == 'vtm':
                tfms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((384,384)), transforms.ToTensor(),
                                           transforms.Normalize(0.5, 0.5), ])
                tar = torch.zeros(8,3,384,384)
                for i in range(8):
                    tar[i, :, :, :] = tfms(inputs['img'][i, :, :, :]).unsqueeze(0)
                inputs['img2'] = tar
                if 'parse' in inputs:
                    tar = torch.zeros(8, 3, 384, 384)
                    for i in range(8):
                        obj = inputs['parse'][i, :, :]
                        obj = torch.tensor(obj, dtype=torch.float32)
                        for j in range(3):
                            tar[i, j, :, :] = tfms(obj).unsqueeze(0)
                    inputs['parse'] = tar

            elif cfg.type == 'stmix' or 'swin':
                tfms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(),
                                           transforms.Normalize(0.5, 0.5), ])
                tar = torch.zeros(8,3,224,224)
                for i in range(8):
                    tar[i, :, :, :] = tfms(inputs['img'][i, :, :, :]).unsqueeze(0)
                inputs['img3'] = tar
                if 'parse' in inputs:
                    tar = torch.zeros(8, 3, 224, 224)
                    for i in range(8):
                        obj = inputs['parse'][i, :, :]
                        obj = torch.tensor(obj, dtype= torch.float32)
                        for j in range(3):
                            tar[i, j, :, :] = tfms(obj).unsqueeze(0)
                    inputs['parse'] = tar


            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            
            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train', cfg.type)
            loss = {k:loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        losses.append(['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss.items()])
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)

    with open('losses.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in losses:
            writer.writerow(row)

if __name__ == "__main__":
    main()
