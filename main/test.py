import os
import os.path as osp
from config import cfg
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import transforms
from utils.vis import save_result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--type', type=str)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    from base import Tester

    if args.type == 'cd': 
        cfg.calculate_cd = True
        cfg.testset = ['PW3D']
    elif args.type == 'bcc': 
        cfg.calculate_bcc = True
        cfg.testset = ['MSCOCO']
    else:
        assert 0, 'Test type is invalid.'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()

    
    eval_result = {}
    cur_sample_idx = 0
    i = 0
    errs = ''
    multi_errs = ''
    gs = ''
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        i += 1
        if  itr < cur_sample_idx:
            continue

        if cfg.type == 'vit' or cfg.type == 'vtm':
            tfms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((384, 384)), transforms.ToTensor(),
                                       transforms.Normalize(0.5, 0.5), ])
            tar = torch.zeros(1, 3, 384, 384)
            for i in range(1):
                tar[i, :, :, :] = tfms(inputs['img'][0, :, :, :]).unsqueeze(0)
            inputs['img2'] = tar
            if 'parse' in inputs:
                tar = torch.zeros(1, 3, 384, 384)
                for i in range(1):
                    obj = inputs['parse'][i, :, :]
                    obj = torch.tensor(obj, dtype=torch.float32)
                    for j in range(3):
                        tar[i, j, :, :] = tfms(obj).unsqueeze(0)
                inputs['parse'] = tar

        elif cfg.type == 'stmix' or 'swin':
            tfms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                                       transforms.Normalize(0.5, 0.5), ])
            tar = torch.zeros(1, 3, 224, 224)
            for i in range(1):
                tar[i, :, :, :] = tfms(inputs['img'][0, :, :, :]).unsqueeze(0)
            inputs['img3'] = tar
            if 'parse' in inputs:
                tar = torch.zeros(1, 3, 224, 224)
                obj = inputs['parse']
                obj = torch.tensor(obj, dtype=torch.float32)
                for j in range(3):
                    tar[:, j, :, :] = tfms(obj).unsqueeze(0)
                inputs['parse'] = tar

        for k,v in inputs.items():
            if type(v) is torch.Tensor: inputs[k] = v.cuda()
        for k,v in targets.items():
            if type(v) is torch.Tensor: targets[k] = v.cuda()
        for k,v in meta_info.items():
            if type(v) is torch.Tensor: meta_info[k] = v.cuda()

        # forward
        with torch.no_grad():
            # out = tester.model(inputs, targets, meta_info, 'test', cfg.type)
            out, pred, gt, err, g_e = tester.model(inputs, targets, meta_info, 'test', cfg.type)
        errs += ', %.4f' % err
        gs += ', %.4f' % g_e
        gt = gt.tolist()
        if (gt[0][0] == 1 and gt[0][1] == 1) or (gt[0][2] == 1 and gt[0][3] == 1):
            multi_errs += ', %.4f' % err

        # save output
        _out = {}
        for k,v in out.items():
            if type(v) is torch.Tensor:
                _out[k] = v.cpu().numpy()
                batch_size = v.shape[0]
            else:
                _out[k] = v
        out = _out
        # for cloth_type in cfg.cloth_types:
        #     print(out[cloth_type + '_mesh'])
        _, _ = save_result(out, osp.join(cfg.vis_dir, 'output_raw'+str(itr)+'.obj'), osp.join(cfg.vis_dir, 'output'+str(itr)+'.obj'))
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]
        
        # evaluate
#        print(out)
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k,v in cur_eval_result.items():
            if k in eval_result: eval_result[k] += v
            else: eval_result[k] = v
        cur_sample_idx += len(out)
        tester.logger.info('pred:' + str(pred.tolist()) + '  groundtruth:'+ str(gt) + '  error:%.8f' % err)

    tester.logger.info('errs: ' + str(errs))
    tester.logger.info('multi_errs: ' + str(multi_errs))
    tester.logger.info('gender_errs: ' + str(gs))
    tester._print_eval_result(eval_result)

if __name__ == "__main__":
    main()
