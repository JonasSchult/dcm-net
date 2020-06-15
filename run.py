import open3d
import os
import os.path
import copy
import json
import argparse
import torch
from torchvision import transforms
from torch_geometric.data import GraphLevelDataLoader
from torch_geometric.data import DataListLoader
import dataset
import transform
import sample_checker
import model as architectures
import loss.weighted_cross_entropy_loss as module_loss
from trainer import Trainer
from utils import Logger


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def get_instance_list(module, config, *args):
    return getattr(module, config['type'])(*args, **config['args'])


def main(config, resume, is_runtime, max_points, max_points_epoch, vis, not_save):

    adapted_config = copy.deepcopy(config)
    train_logger = Logger()

    transf_list_train = []
    transf_list_valid = []

    for transform_config in adapted_config['train_transform']:
        transf_list_train.append(
            get_instance_list(transform, transform_config))
    for transform_config in adapted_config['valid_transform']:
        transf_list_valid.append(
            get_instance_list(transform, transform_config))

    checker = []
    for checker_config in adapted_config['sample_checker']:
        checker.append(get_instance_list(sample_checker, checker_config))

    adapted_config['train_dataset']['args']['transform'] = transforms.Compose(
        transf_list_train)
    adapted_config['valid_dataset']['args']['transform'] = transforms.Compose(
        transf_list_valid)

    adapted_config['train_dataset']['args']['sample_checker'] = checker

    if not args.eval:
        train_dataset = get_instance(dataset, 'train_dataset', adapted_config)
        adapted_config['train_data_loader']['args']['dataset'] = train_dataset
        if adapted_config['n_gpu'] > 1:
            train_data_loader = DataListLoader(
                **adapted_config['train_data_loader']['args'])
        else:
            train_data_loader = GraphLevelDataLoader(
                **adapted_config['train_data_loader']['args'])

        train_data_loader.n_samples = lambda: len(train_data_loader.dataset)

    valid_dataset = get_instance(dataset, 'valid_dataset', adapted_config)
    adapted_config['valid_data_loader']['args']['dataset'] = valid_dataset
    if adapted_config['n_gpu'] > 1:
        valid_data_loader = DataListLoader(
            **adapted_config['valid_data_loader']['args'])
    else:
        valid_data_loader = GraphLevelDataLoader(
            **adapted_config['valid_data_loader']['args'])

    valid_data_loader.n_samples = lambda: len(valid_data_loader.dataset)

    model = get_instance(architectures, 'arch', adapted_config)
    print(model)
    loss = get_instance(module_loss, 'loss', adapted_config)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer',
                             adapted_config, trainable_params)

    lr_scheduler = get_instance(
        torch.optim.lr_scheduler, 'lr_scheduler', adapted_config, optimizer)

    trainer = Trainer(model, loss, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=train_data_loader if not args.eval else None,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger,
                      eval_mode=args.eval,
                      is_runtime=is_runtime,
                      max_points=max_points,
                      max_points_epoch=max_points_epoch,
                      vis=vis,
                      n_gpu=adapted_config['n_gpu'],
                      s3dis_gt_pcd=args.s3dis_gt_pcd,
                      not_save=not_save)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DualConvMeshNet')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-p', '--max_points', default=None, type=int,
                        help='maximal number of points in a batch')
    parser.add_argument('-o', '--max_points_epoch', default=None, type=int,
                        help='epoch when maximal number of points crash happened')
    parser.add_argument('-e', '--eval', dest='eval', action='store_true')
    parser.set_defaults(eval=False)
    parser.add_argument('-v', '--vis', dest='vis', action='store_true')
    parser.set_defaults(vis=False)
    parser.add_argument('-t', '--runtime',
                        dest='is_runtime', action='store_true')
    parser.add_argument('-s', '--s3dis_gt_pcd',
                        dest='s3dis_gt_pcd', action='store_true')
    parser.set_defaults(s3dis_gt_pcd=False)
    parser.set_defaults(is_runtime=False)
    parser.add_argument('-q', '--not_save',
                        dest='not_save', action='store_true')
    parser.set_defaults(not_save=False)
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified.")

    if args.eval and not args.vis and not args.is_runtime:
        eval_file = args.resume.replace('.pth', '.txt')
        if os.path.isfile(eval_file) and not args.not_save:
            print(f"{args.resume} was already evaluated")
            exit(0)

    if args.is_runtime:
        print(f"RUNTIME CHECK")

    main(config, args.resume, args.is_runtime, args.max_points,
         args.max_points_epoch, args.vis, args.not_save)
