# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Script for training concentric hyperspheres (ch) universe
"""

import argparse
import os
import torch

import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm

from classical_model import ClassicalModel
from soft_order_model import SoftOrderModel
from traveling_observer_model import TravelingObserverModel
from utils import compute_loss_and_accuracy
from utils import load_dataset
from utils import squared_hinge

N_TASKS = 90


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu')
parser.add_argument('--results_dir', default='results/ch_test')
parser.add_argument('--context_size', type=int, default=2)
parser.add_argument('--context_std', type=float, default=0.001)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--core_layers', type=int, default=3)
parser.add_argument('--decoder_layers', type=int, default=3)
parser.add_argument('--encoder_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--max_steps', type=int, default=10000000)
parser.add_argument('--steps_per_eval', type=int, default=10000)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--use_classic', action='store_true')
parser.add_argument('--use_soft_order', action='store_true')
parser.add_argument('--oracle_locations', action='store_true')
parser.add_argument('--datasets_per_step', type=int, default=1)
parser.add_argument('--single_task_index', type=int, default=None)
parser.add_argument('--save_contexts_over_time', action='store_true')
args = parser.parse_args()

# Set up logging
exp_dir = args.results_dir + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.makedirs(exp_dir)
results_path = exp_dir + '/results_per_task.csv'
metrics_path = exp_dir + '/metrics.csv'
with open(metrics_path, 'w') as my_file:
    my_file.write('step,train_acc,val_acc,best_val_acc,test_acc\n')
contexts_template = exp_dir + '/contexts_{}.pt'

# Load datasets
if args.single_task_index is not None:
    dataset_idxs_to_load = [args.single_task_index] # Single task
else:
    dataset_idxs_to_load = np.arange(N_TASKS) # All tasks
datasets = []
for dataset_idx in dataset_idxs_to_load:
    dataset_path = f'ch_tasks/task_{dataset_idx}.csv'
    dataset = load_dataset(dataset_path)
    datasets.append(dataset)

# Add contexts (variable embeddings) to datasets
for dataset in datasets:
    num_variables = dataset['train'].shape[1]
    if args.oracle_locations:
        # Place VEs at intuitively meaningful locations.
        context_tensor = torch.zeros((1, args.context_size, num_variables),
                                     device=args.device)
        for i, idx in enumerate(dataset['true_input_variable_indices']):
            context_tensor[0,:,idx] = dataset['origin'][i]
        for i, idx in enumerate(dataset['true_output_variable_indices']):
            context_tensor[0,:,idx] = (1. + i) * 0.1
    else:
        context_tensor = torch.normal(mean=0.,
                                      std=args.context_std,
                                      size=(1, args.context_size, num_variables),
                                      requires_grad=True,
                                      device=args.device)
    dataset['context_tensor'] = context_tensor
    for split in ['train', 'val', 'test']:
        data_set_tensor = dataset[split].to(args.device)
        dataset[split] = data_set_tensor

# Add task index for soft order
for d, dataset in enumerate(datasets):
    dataset['dataset_idx'] = d

# Build model
if args.use_classic:
    print("Model type: Classic")
    assert args.context_size == args.hidden_size # Required for classical model
    model = ClassicalModel(args.context_size,
                           args.hidden_size,
                           args.core_layers,
                           dropout=args.dropout)
elif args.use_soft_order:
    print("Model type: Soft Order")
    assert args.context_size == args.hidden_size # Required for soft order model
    model = SoftOrderModel(args.hidden_size,
                           args.core_layers,
                           len(datasets),
                           dropout=args.dropout)
else:
    print("Model type: TOM")
    model = TravelingObserverModel(args.context_size,
                                   args.hidden_size,
                                   args.encoder_layers,
                                   args.core_layers,
                                   args.decoder_layers,
                                   dropout=args.dropout)
model.to(args.device)

# Set up optimizer
context_parameters = [dataset['context_tensor'] for dataset in datasets]
optimizer_parameters = list(model.parameters()) + context_parameters
optimizer = torch.optim.Adam(optimizer_parameters,
                             lr=args.learning_rate,
                             weight_decay=args.weight_decay)

# Train
best_mean_val_acc = 0.
epochs_without_improvement = 0
best_val_accs = [0. for dataset in datasets]
test_accs = [0. for dataset in datasets]
for step in tqdm(range(args.max_steps)):

    # Perform training update
    for d in range(args.datasets_per_step):
        dataset = np.random.choice(datasets)
        context_tensor = dataset['context_tensor']
        data_set_tensor = dataset['train']
        num_samples = data_set_tensor.shape[0]
        batch_idxs = np.random.choice(np.arange(num_samples),
                                      size=num_samples,
                                      replace=True)
        batch = data_set_tensor[batch_idxs]

        input_var_indices = dataset['true_input_variable_indices']
        output_var_indices = dataset['true_output_variable_indices']
        batch_input = batch[:,input_var_indices]
        input_contexts = context_tensor[:,:,input_var_indices]
        output_contexts = context_tensor[:,:,output_var_indices]

        if args.use_soft_order:
            pred = model(batch_input, input_contexts, output_contexts, dataset['dataset_idx'])
        else:
            pred = model(batch_input, input_contexts, output_contexts)

        target = batch[:,output_var_indices]

        loss = squared_hinge(pred, target)

        loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    if step % args.steps_per_eval == 0:
        print("Evaluating...")
        model.eval()
        for i, dataset in tqdm(enumerate(datasets)):
            train_loss, train_acc = compute_loss_and_accuracy(model, dataset, 'train',
                                                              dataset['train'].shape[0],
                                                              args.use_soft_order)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_loss, val_acc = compute_loss_and_accuracy(model, dataset, 'val',
                                                          dataset['val'].shape[0],
                                                          args.use_soft_order)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            if val_acc >= best_val_accs[i]:
                best_val_accs[i] = val_acc
                test_loss, test_acc = compute_loss_and_accuracy(model, dataset, 'test',
                                                                dataset['test'].shape[0],
                                                                args.use_soft_order)
                test_accs[i] = test_acc
        mean_train_acc = np.mean(train_accs)
        mean_train_loss = np.mean(train_losses)
        mean_val_acc = np.mean(val_accs)
        mean_val_loss = np.mean(val_losses)
        mean_best_val_acc = np.mean(best_val_accs)
        mean_test_acc = np.mean(test_accs)

        df = pd.DataFrame({
            'test_acc': test_accs
        })
        df.to_csv(results_path)

        if mean_val_acc > best_mean_val_acc:
            epochs_without_improvement = 0
            best_mean_val_acc = mean_val_acc
        else:
            epochs_without_improvement += 1

        print('Step:', step)
        print('Mean Train Acc/Loss:', mean_train_acc, mean_train_loss)
        print('Mean Val Acc/Loss:', mean_val_acc, mean_val_loss)
        print('Best Mean Val Acc:', best_mean_val_acc)
        print('Mean Best Val Acc:', mean_best_val_acc)
        print('Mean Test Acc:', mean_test_acc)
        print('Epochs w/o Improvement', epochs_without_improvement)

        with open(metrics_path, 'a') as my_file:
            my_file.write(f'{step},{mean_train_acc},{mean_val_acc},{mean_best_val_acc},{mean_test_acc}\n')
        print("Wrote to", metrics_path)

        if args.save_contexts_over_time:
            contexts_path = contexts_template.format(step)
            all_contexts = [dataset['context_tensor'] for dataset in datasets]
            torch.save(all_contexts, contexts_path)
        model.train()

        if epochs_without_improvement > args.patience:
            break
print('Done. Thank You.')
