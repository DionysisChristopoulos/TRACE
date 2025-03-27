import argparse
import datetime, time
import os, random, json
import math
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, random_split

from utils.datasets import read_data, labels2indices, build_datasets
from utils.models import TRACE
from utils.engine import train_one_epoch, evaluate
from utils.losses import FocalLoss
from utils import misc
from utils.sampler import CustomSampler
from utils.config import setup
from fvcore.nn import FlopCountAnalysis, flop_count_table
        
def get_args_parser():
    parser = argparse.ArgumentParser('Attention based melanoma/osc risk estimation on clinical data', add_help=True)
    parser.add_argument('--config_file', default='./config/trace_cdc22.yaml', help='config file path')
    parser.add_argument('--output_dir', default='./results/experiment_name', help='path to save the output model')

    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--checkpoint', default='/path/to/model/ckpt_best.pth', help='Load model from checkpoint')
    return parser

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device} device\n")

    # fix the seed for reproducibility
    set_seed(args.train.seed)

    with open(args.data.metadata_dir, 'r') as f:
        feature_metadata = json.load(f)

    train_dataset, test_dataset, num_indices = build_datasets(args, device=device)

    print(f"\nTrain dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    if args.data.num_labels==1:
        train_positives = (train_dataset.tensors[1] == 1).sum().item()
        train_negatives = (train_dataset.tensors[1] == 0).sum().item()
        test_positives = (test_dataset.tensors[1] == 1).sum().item()
        test_negatives = (test_dataset.tensors[1] == 0).sum().item()

        print(f"\nTrain positives: {train_positives} samples")
        print(f"Train negatives: {train_negatives} samples")
        print(f"\nTest positives: {test_positives} samples")
        print(f"Test negatives: {test_negatives} samples")
    
    if args.sampler == 'custom':
        positive_ratio = train_positives / len(train_dataset)
        positive_ratio = round(math.ceil(positive_ratio / 0.05) * 0.05, 2)
        custom_sampler_train = CustomSampler(train_dataset, args.train.train_batch_size, train=True, positive_ratio=positive_ratio)
        dataloader_train = DataLoader(train_dataset, batch_sampler=custom_sampler_train)
    else:
        dataloader_train = DataLoader(train_dataset, batch_size=args.train.train_batch_size, shuffle=True, drop_last=False)
    
    dataloader_val = DataLoader(test_dataset, batch_size=args.train.val_batch_size, shuffle=False, drop_last=False)

    model = TRACE(hidden_size=args.model.hidden_size,
                       feature_metadata=feature_metadata,
                       num_indices=num_indices,
                       num_mode=args.model.num_mode,
                       num_labels=args.data.num_labels,
                       dropout_p=args.model.dropout,
                       cls_token=args.model.cls_token,
                       tran_layers=args.model.tran_layers,
                       heads=args.model.heads,
                       mlp_ratio=args.model.mlp_ratio,
                       use_num_norm=args.model.use_num_norm,
                       use_cat_norm=args.model.use_cat_norm,
                       checkbox_mode=args.model.checkbox_mode)
    
    if args.checkpoint and args.eval:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.checkpoint)
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=False)
    
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nNumber of trainable params: {n_parameters}')

    ## Calculate GFLOPS
    # num_features = len(feature_metadata['continuous']) + len(feature_metadata['categorical'])
    # dummy_input = torch.ones(2, num_features).to('cuda').long() # change 2nd axis to the number of features
    # model.eval()
    # flops = FlopCountAnalysis(model.to('cuda'), dummy_input)
    # print(f"Total Flops: {flops.total() / 2}")
    # model.train()

    if args.optim.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.optim.lr, momentum=0.9, weight_decay=1e-3)
    elif args.optim.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim.lr)
    elif args.optim.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim.lr)
    else:
        raise ValueError(f"Unsupported optimzer '{args.optim.loss}'. Supported optimizer algorithms: 'adam', 'rmsprop', 'adamw'.")
    
    if args.optim.loss == 'bce':
        if args.optim.bce.use_pos_weight:
            pos_weight = torch.tensor([args.optim.bce.pos_weight], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    elif args.optim.loss == 'focal':
        criterion = FocalLoss(alpha=args.optim.focal.alpha)
    elif args.optim.loss=="ce":
        criterion=torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function '{args.optim.loss}'. Supported loss functions: 'focal', 'bce'.")
    print(f"Criterion: {criterion}")

    loss_scaler = misc.NativeScaler()

    if args.optim.lr_scheduler == 'rop':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.9, patience=10)
    elif args.optim.lr_scheduler == 'ca':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.optim.epochs, eta_min=2e-6)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.eval:
        test_stats, cm = evaluate(dataloader_val, model, criterion, device, return_cm=True, features=feature_metadata)
        print(f"Accuracy of the network on the {len(test_dataset)} test samples: {test_stats['acc']:.1f}%")
        print(f"F1-Score (binary/macro): {test_stats['f1']:.3f}")
        print(f"F1-Score (weighted): {test_stats['f1_w']:.3f}")
        print(f'Confustion matrix: {cm}')
        if args.data.num_labels == 1:
            print(f"Balanced Accuracy of the network on the {len(test_dataset)} test samples: {test_stats['bal_acc']:.1f}%")
            print(f"Recall on positive class (Sensitivity): {test_stats['rec_pos']:.3f}")
            print(f"Recall on negative class (Specificity): {test_stats['rec_neg']:.3f}")
            print(f"Precision on positive class: {test_stats['prec_pos']:.3f}")
            print(f"Precision on negative class: {test_stats['prec_neg']:.3f}")
        exit(0)
    
    print(f"Start training for {args.optim.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0
    if args.data.num_labels==1:
        max_balanced_accuracy = 0.0
    for epoch in range(args.optim.epochs):
        train_stats = train_one_epoch(
            model, criterion, dataloader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None, args=args
        )
        
        if args.output_dir and (epoch+1) % 20 == 0:
            output_dir = Path(args.output_dir)
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % str(epoch))]
            for checkpoint_path in checkpoint_paths:
                misc.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats, cm = evaluate(dataloader_val, model, criterion, device, return_cm=True, features=feature_metadata)
        print(f"Accuracy of the network on the {len(test_dataset)} test samples: {test_stats['acc']:.1f}%")
        print(f'Confustion matrix: {cm}')
        max_accuracy = max(max_accuracy, test_stats["acc"])
        max_f1 = max(max_f1, test_stats["f1"])
        if args.data.num_labels==1:
            max_balanced_accuracy = max(max_balanced_accuracy, test_stats["bal_acc"])
        print(f'Max accuracy: {max_accuracy:.2f}%, Max F1-Score: {max_f1:.3f}')
        
        if args.optim.lr_scheduler == 'rop':
            lr_scheduler.step(test_stats['f1']) # TODO: choose a custom metric to monitor
        elif args.optim.lr_scheduler == 'ca':
            lr_scheduler.step()

        if (max_f1 == test_stats["f1"]):
            output_dir = Path(args.output_dir)
            checkpoint_paths = [output_dir / ('ckpt_best.pth')]
            for checkpoint_path in checkpoint_paths:
                misc.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args = setup(args)
    main(args)
