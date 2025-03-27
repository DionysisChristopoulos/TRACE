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
        if model.num_labels >= 2:
            targets = targets.squeeze().long()

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), flush=True)
            sys.exit(1)

        optimizer.zero_grad()

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False)
        
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
            if model.num_labels >= 2:
                target = target.squeeze().long()

            # compute output
            with torch.cuda.amp.autocast():
                output, attention = model(sample, return_attn=True)
                loss = criterion(output, target)

            if return_cm:
                if model.num_labels == 1:
                    acc, bal_acc, f1, f1w, rec_pos, rec_neg, prec_pos, prec_neg, cm = compute_metrics(output, target, num_labels=model.num_labels, return_cm=return_cm)
                else:
                    acc, f1, f1w, cm = compute_metrics(output, target, num_labels=model.num_labels, return_cm=return_cm)
                
                if cumulative_cm is None:
                    cumulative_cm = cm
                else:
                    cumulative_cm += cm

            else:
                if model.num_labels == 1:
                    acc, bal_acc, f1, f1w, rec_pos, rec_neg, prec_pos, prec_neg = compute_metrics(output, target, num_labels=model.num_labels)
                else:
                    acc, f1, f1w = compute_metrics(output, target, num_labels=model.num_labels, return_cm=return_cm)

            batch_size = sample.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc'].update(acc, n=batch_size)
            metric_logger.meters['f1'].update(f1, n=batch_size)
            metric_logger.meters['f1_w'].update(f1w, n=batch_size)
            
            if model.num_labels == 1:
                metric_logger.meters['bal_acc'].update(bal_acc, n=batch_size)
                metric_logger.meters['rec_pos'].update(rec_pos, n=batch_size)
                metric_logger.meters['rec_neg'].update(rec_neg, n=batch_size)
                metric_logger.meters['prec_pos'].update(prec_pos, n=batch_size)
                metric_logger.meters['prec_neg'].update(prec_neg, n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Accuracy {acc.global_avg:.3f} F1 Score (binary/macro) {f1.global_avg:.3f} loss: {losses.global_avg:.3f}'
          .format(acc=metric_logger.acc, f1=metric_logger.f1, losses=metric_logger.loss))

    if return_cm:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, cumulative_cm
    else:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_metrics(outputs, targets, num_labels, return_cm=False):
    if num_labels >= 2:
        probas = F.softmax(outputs, dim=1)
        preds = torch.argmax(probas, dim=1)

        accuracy = accuracy_score(targets.cpu(), preds.cpu()) * 100.
        f1 = f1_score(targets.cpu(), preds.cpu(), average='macro')
        f1_weighted = f1_score(targets.cpu(), preds.cpu(), average='weighted')
        if return_cm:
            cm = confusion_matrix(targets.cpu(), preds.cpu())
            return accuracy, f1, f1_weighted, cm
        else:
            return accuracy, f1, f1_weighted

    else:
        probas = torch.sigmoid(outputs).squeeze()
        preds = (probas > 0.5).long()

        accuracy = accuracy_score(targets.cpu(), preds.cpu()) * 100.
        balanced_accuracy = balanced_accuracy_score(targets.cpu(), preds.cpu()) * 100.
        f1 = f1_score(targets.cpu(), preds.cpu(), average='binary')
        f1_weighted = f1_score(targets.cpu(), preds.cpu(), average='weighted')
        recall_positive = recall_score(targets.cpu(), preds.cpu(), pos_label=1, average='binary')
        recall_negative = recall_score(targets.cpu(), preds.cpu(), pos_label=0, average='binary')
        precision_positive = precision_score(targets.cpu(), preds.cpu(), pos_label=1, average='binary')
        precision_negative = precision_score(targets.cpu(), preds.cpu(), pos_label=0, average='binary')

        if return_cm:
            cm = confusion_matrix(targets.cpu(), preds.cpu())
            return accuracy, balanced_accuracy, f1, f1_weighted, recall_positive, recall_negative, precision_positive, precision_negative, cm
        else:
            return accuracy, balanced_accuracy, f1, f1_weighted, recall_positive, recall_negative, precision_positive, precision_negative