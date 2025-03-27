import copy, os, argparse, pickle, random
import numpy as np

from utils.datasets import read_data
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score, recall_score

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from utils.models import nnMLP
from utils.losses import FocalLoss
from utils.config import setup
from fvcore.nn import FlopCountAnalysis, flop_count_table

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./config/nnmlp_cdc22.yaml', help='config file path')
    parser.add_argument('--output_dir', default='./results', help='path to save the output model')
    parser.add_argument('--exp_name', default='nnMLP_cdc22_focal_0_9', help='experiment name')
    parser.add_argument('--save_best', help='Saves the best model in .pth format', action='store_true')

    # For evaluation only
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--checkpoint', default='./results/nnMLP_cdc22_focal_0_9.pth', help='Load model from checkpoint')
    return parser

def eval_train(model, dataloader_train, dataloader_val, optimizer, criterion, num_epochs=100, lr=2e-4, log_path='./models/cool/train_logs.txt'):

    dirname = os.path.dirname(log_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    # freeze the weights of W3 (ones) for additive scale on risk estimation
    # model.freeze_weights()

    with open(log_path, "w") as log_file:
        best_f1 = 0.0
        
        for epoch in range(num_epochs):
            print(f'Starting epoch {epoch+1}')

            current_loss = 0.0
            model.train()

            for i, data in enumerate(dataloader_train, 0):

                inputs, targets = data
                optimizer.zero_grad()

                outputs = model(inputs)
                
                if model.num_labels >= 2:
                    targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                model.enforce_positive_weights()
                
                current_loss += loss.item()

            # Model evaluation after every epoch
            model.eval()

            total_preds = []
            total_targets = []
            with torch.no_grad():
                for val_data in dataloader_val:
                    val_inputs, val_targets = val_data
                    
                    val_logits = model(val_inputs)
                    if model.num_labels >= 2:
                        probas = F.softmax(val_logits, dim=1)
                        preds = torch.argmax(probas, dim=1)
                    else:
                        probas = F.sigmoid(val_logits)
                        preds = (probas > 0.5).float()

                    total_preds.append(preds)
                    total_targets.append(val_targets)

            total_preds = torch.cat(total_preds, dim=0)
            total_targets = torch.cat(total_targets, dim=0)
            accuracy = (accuracy_score(total_targets.cpu(), total_preds.cpu()) * 100)
            if model.num_labels >= 2:
                avg = 'macro'
            else:
                avg = 'binary'
            f1 = f1_score(total_targets.cpu(), total_preds.cpu(), average=avg)
            precision = precision_score(total_targets.cpu(), total_preds.cpu(), average=avg)
            conf_matrix = confusion_matrix(total_targets[:,0].cpu(), total_preds.cpu())
            print(conf_matrix)

            if f1 >= best_f1:
                best_epoch = epoch
                best_f1 = f1
                best_params = copy.deepcopy(model.state_dict())
            
            log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss}, Accuracy: {accuracy}, Precision Score: {precision}, F1 Score: {f1}\n')
            print(f'Finished Epoch [{epoch+1}/{num_epochs}], Total Loss: {current_loss}, Accuracy: {accuracy}, Precision Score: {precision}, F1 Score: {f1}\n')
    
    model.load_state_dict(best_params)
    return model, best_f1, best_epoch

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device} device\n")

    # fix the seed for reproducibility
    set_seed(args.train.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Data preparation
    num_labels = args.data.num_labels
    X, y, features = read_data(data_path=args.data.data_dir,
                               meta_path=args.data.metadata_dir,
                               target_label=args.data.target_label,
                               encoded=args.data.encoding)
    
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, stratify=y, shuffle=True)
    # train, val = X_train.index, X_test.index
    
    X, y = torch.tensor(X.values, dtype=torch.float32, device=device), torch.tensor(y.values, dtype=torch.float32, device=device)
    input_size = X.shape[-1]

    if args.eval:
        base, ext = os.path.splitext(args.checkpoint)
        indices_path = base + '.pkl'
        with open(indices_path, 'rb') as f:
            test_indices = pickle.load(f)
    
        # Load the state dict
        model = torch.load(args.checkpoint, map_location=device)
        model.eval()

        X_test, y_test = X[test_indices], y[test_indices]
        if num_labels >= 2:
            test_logits=(model(X_test))
            test_probas = F.softmax(test_logits, dim=1)
            test_preds = torch.argmax(test_probas, dim=1)
        else:
            test_probas = F.sigmoid(model(X_test))
            test_preds = (test_probas > 0.5).float()
        test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
        test_conf_matrix = confusion_matrix(y_test[:, 0].cpu(), test_preds.cpu())
        
        avg = 'macro' if num_labels >= 2 else 'binary'
        
        f1 = f1_score(y_test.cpu(), test_preds.cpu(), average=avg)
        f1_weighted = f1_score(y_test.cpu(), test_preds.cpu(), average='weighted')
        
        print(f'Overall Test Accuracy: {(test_acc*100.):.2f}')
        print(f'F1-Score (binary/macro): {(f1):.4f}')
        print(f'F1-Score (weighted): {(f1_weighted):.4f}')
        if num_labels == 1:
            recall_positive = recall_score(y_test.cpu(), test_preds.cpu(), pos_label=1, average=avg)
            recall_negative = recall_score(y_test.cpu(), test_preds.cpu(), pos_label=0, average=avg)
            precision_positive = precision_score(y_test.cpu(), test_preds.cpu(), pos_label=1, average=avg)
            precision_negative = precision_score(y_test.cpu(), test_preds.cpu(), pos_label=0, average=avg)
            print(f'Recall on positive class (Sensitivity): {recall_positive:.4f}')
            print(f'Recall on negative class (Specificity): {recall_negative:.4f}')
            print(f'Precision on positive class: {precision_positive:.4f}')
            print(f'Precision on negative class: {precision_negative:.4f}')
        print(f'Confusion Matrix: {test_conf_matrix}')
        exit(0)

    # Implement KFold training and validation
    stratified_split = StratifiedShuffleSplit(n_splits=5, test_size=1-args.train.train_perc)
    
    max_accuracy = 0.0
    best_model = None
    best_split = {}

    for i, (train, val) in enumerate(stratified_split.split(X.cpu(), y[:,0].cpu())):
        # Initialize the MLP
        mlp = nnMLP(input_size=input_size,
                    hidden_size=args.model.hidden_size, 
                    num_labels=num_labels, 
                    fix_baseline_risk=-1).to(device)
        print(f"Model structure: {mlp}\n")

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Print the number of parameters
        print(f"The model has {count_parameters(mlp):,} trainable parameters\n")

        train_ds = TensorDataset(X[train], y[train])
        val_ds = TensorDataset(X[val], y[val])

        dataloader_train = DataLoader(train_ds, batch_size=args.train.train_batch_size, shuffle=True)
        dataloader_val = DataLoader(val_ds, batch_size=args.train.val_batch_size, shuffle=False)

        if args.optim.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(mlp.parameters(), lr=args.optim.lr, momentum=0.9, weight_decay=1e-3)
        elif args.optim.optimizer == 'adam':
            optimizer = torch.optim.Adam(mlp.parameters(), lr=args.optim.lr)
        print(f"Optimizer: {optimizer}")
        
        if args.optim.loss == 'bce':
            if args.optim.bce.use_pos_weight:
                pos_weight = torch.tensor([9.0], dtype=torch.float32, device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
        elif args.optim.loss == 'focal':
            criterion = FocalLoss(alpha=args.optim.focal.alpha)
        elif args.optim.loss=="ce":
            criterion=torch.nn.CrossEntropyLoss()
        print(f"Criterion: {criterion}")

        ## Calculate GFLOPS
        # dummy_input = torch.ones(2, 24).to('cuda').float()
        # mlp.eval()
        # flops = FlopCountAnalysis(mlp.to('cuda'), dummy_input)
        # print(f"Total Flops: {flops.total() / 2}")
        # mlp.train()

        model, accuracy, epoch = eval_train(model=mlp, 
                                            dataloader_train=dataloader_train, 
                                            dataloader_val=dataloader_val, 
                                            optimizer=optimizer, 
                                            criterion=criterion, 
                                            num_epochs=args.optim.epochs, 
                                            lr=args.optim.lr, 
                                            log_path=os.path.join(args.output_dir, f'{args.exp_name}_train_logs_{i+1}.txt'))
        
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_epoch = epoch
            best_split = {'train': train, 'val': val}
            best_model = model
            if args.save_best:
                print(os.path.join(args.output_dir, f'{args.exp_name}.pth'))
                torch.save(best_model, os.path.join(args.output_dir, f'{args.exp_name}.pth'))
                with open(os.path.join(args.output_dir, f'{args.exp_name}.pkl'), 'wb') as f:
                    pickle.dump(val, f)

    # for name, param in best_model.named_parameters():
    #             print(f"Layer: {name} | Size: {param.size()} | Values : {param.data} \n")
    
    # print(best_split)
    print(f'Training process has finished. Max F1 score found: {max_accuracy:.4f} in epoch {best_epoch}')

    # Test with the best model
    best_model.eval()
    X_test, y_test = X[best_split['val']], y[best_split['val']]
    if num_labels>2:
        test_logits=(best_model(X_test))
        test_probas = F.softmax(test_logits, dim=1)
        test_preds = torch.argmax(test_probas, dim=1)
    else:
        test_probas = F.sigmoid(best_model(X_test))
        test_preds = (test_probas > 0.5).float()
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    test_conf_matrix = confusion_matrix(y_test[:,0].cpu(), test_preds.cpu())

    print(f'\nOverall Test Accuracy: {(test_acc*100.):.2f}\nConfusion Matrix: {test_conf_matrix}')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args = setup(args)
    main(args)