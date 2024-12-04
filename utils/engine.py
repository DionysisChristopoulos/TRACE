import math
import sys
from typing import Iterable
from utils import misc
from utils.losses import FocalLoss
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score

import torch
import torch.nn.functional as F

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), flush=True)
            sys.exit(1)

        optimizer.zero_grad()

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False)
        
        # model.num_mlp.enforce_positive_weights()
        # model.head.enforce_positive_weights()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate(data_loader, model, criterion, device, return_cm=False, features=None):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if return_cm:
        cumulative_cm = None

    with torch.no_grad():
        for sample, target in metric_logger.log_every(data_loader, 5, header):
            sample = sample.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output, attention = model(sample, return_attn=True)
                loss = criterion(output, target)

            if return_cm:
                acc, bal_acc, f1b, f1w, rec_pos, rec_neg, prec_pos, prec_neg, cm = compute_metrics(output, target, return_cm=return_cm)
                if cumulative_cm is None:
                    cumulative_cm = cm
                else:
                    for i in range(target.shape[-1]):
                        cumulative_cm[i] += cm[i]
            else:    
                acc, bal_acc, f1b, f1w, rec_pos, rec_neg, prec_pos, prec_neg = compute_metrics(output, target)

            batch_size = sample.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc'].update(acc, n=batch_size)
            metric_logger.meters['bal_acc'].update(bal_acc, n=batch_size)
            metric_logger.meters['f1_b'].update(f1b, n=batch_size)
            metric_logger.meters['f1_w'].update(f1w, n=batch_size)
            metric_logger.meters['rec_pos'].update(rec_pos, n=batch_size)
            metric_logger.meters['rec_neg'].update(rec_neg, n=batch_size)
            metric_logger.meters['prec_pos'].update(prec_pos, n=batch_size)
            metric_logger.meters['prec_neg'].update(prec_neg, n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Accuracy {acc.global_avg:.3f} F1 Score (binary) {f1_b.global_avg:.3f} loss: {losses.global_avg:.3f}'
          .format(acc=metric_logger.acc, f1_b=metric_logger.f1_b, losses=metric_logger.loss))

    if return_cm:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, cumulative_cm
    else:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_metrics(outputs, targets, return_cm=False):
    n_labels = targets.shape[-1]
    probas = F.sigmoid(outputs)
    preds = (probas > 0.5).float()

    accuracy = (accuracy_score(targets.cpu(), preds.cpu()) * 100.)
    balanced_accuracy = (balanced_accuracy_score(targets.cpu(), preds.cpu()) * 100.)
    if n_labels == 2:
        avg = 'samples'
    else:
        avg = 'binary'
    f1_binary = f1_score(targets.cpu(), preds.cpu(), average=avg)
    f1_weighted = f1_score(targets.cpu(), preds.cpu(), average='weighted')
    recall_positive = recall_score(targets.cpu(), preds.cpu(), pos_label=1, average=avg)
    recall_negative = recall_score(targets.cpu(), preds.cpu(), pos_label=0, average=avg)
    precision_positive = precision_score(targets.cpu(), preds.cpu(), pos_label=1, average=avg)
    precision_negative = precision_score(targets.cpu(), preds.cpu(), pos_label=0, average=avg)

    if return_cm:
        cm = [confusion_matrix(targets[:,i].cpu(), preds[:,i].cpu()) for i in range(n_labels)]
        return accuracy, balanced_accuracy, f1_binary, f1_weighted, recall_positive, recall_negative, precision_positive, precision_negative, cm
    else:
        return accuracy, balanced_accuracy, f1_binary, f1_weighted, recall_positive, recall_negative, precision_positive, precision_negative