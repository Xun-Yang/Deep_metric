# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import torch.utils.data
from torch.backends import cudnn
import models
cudnn.benchmark = True


def main(args):
    log_dir = os.path.join('checkpoints', args.log_dir)

    paths = os.listdir(log_dir)

    model_paths = []
    pool_paths = []
    for file_ in paths:
        if file_[-3:] != 'pkl' or file_[:4] == 'pool':
            continue
        else:
            model_path = os.path.join(log_dir, file_)
            pool_path = os.path.join(log_dir, 'pool_' + file_)
            model_paths.append(model_path)
            pool_paths.append(pool_path)
            print(pool_path)

    for model_path, pool_path in zip(model_paths, pool_paths):
        pretrain_model = torch.load(model_path)
        pretrained__dict = pretrain_model.state_dict()

        model = models.create('bn', Embed_dim=args.d)
        # load part of the model
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained__dict.items() if k[:5] != 'Embed'}
        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)

        # print('initialize the FC layer orthogonally')
        # _, _, v = torch.svd(model_dict['Embed.linear.weight'])
        model_dict['Embed.linear.bias'] = torch.zeros(args.d)
        model_dict['Embed.linear.weight'] = torch.eye(args.d)
        model.load_state_dict(model_dict)

        model = model.cuda()
        torch.save(model, pool_path)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pool-5 model transfer')

    # hype-parameters
    parser.add_argument('-log_dir', type=str, default='cub_512_orth',
                        help="checkpoint directory for model")
    parser.add_argument('-d', default=1024, type=int, metavar='N',
                        help='the dimension of pool-5')
    main(parser.parse_args())