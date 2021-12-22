#
# Common utility functions for TOM experiments
#

import numpy as np
import pandas as pd
import torch


def load_dataset(path):
    """
    Load a csv dataset into the format TOM expects.
    """
    dataset_df = pd.read_csv(path)
    feature_columns = [c for c in dataset_df.columns if 'feature' in c]
    class_columns = [c for c in dataset_df.columns if 'class' in c]
    data_columns = feature_columns + class_columns
    dataset = {}
    for split in ['train', 'val', 'test']:
        split_df = dataset_df[dataset_df.split == split]
        data_df = split_df[data_columns]
        data_arr = data_df.values
        data_tensor = torch.from_numpy(data_arr).float()
        dataset[split] = data_tensor
    dataset['true_input_variable_indices'] = np.arange(len(feature_columns))
    dataset['true_output_variable_indices'] = np.arange(len(feature_columns),
                                                        len(data_columns))

    # Load origin so the oracle can use it. This is specific to the ch problem.
    origin_row = dataset_df[dataset_df.split == 'origin']
    origin = origin_row[feature_columns].iloc[0].values
    dataset['origin'] = origin

    return dataset


def squared_hinge(pred, target):
    """
    The squared hinge loss as an alternative to crossentropy in classification.
    """
    return torch.mean(torch.sum(torch.clamp(1 - (2 * target - 1) * pred, 0.)**2, dim=1))


def compute_loss_and_accuracy(model,
                              dataset,
                              split,
                              batch_size,
                              soft_model=False,
                              loss_fn=squared_hinge):
    with torch.no_grad():
        data_set_tensor = dataset[split]
        context_tensor = dataset['context_tensor']
        num_samples = data_set_tensor.shape[0]
        output_contexts = context_tensor
        input_contexts = context_tensor[:,:,dataset['true_input_variable_indices']]

        # Compute number of validation steps
        nsteps = int(np.ceil(data_set_tensor.shape[0] / batch_size))

        # Run validation step-by-step
        correct = 0
        total_loss = 0.
        for val_step in range(nsteps):
            start_idx = val_step * batch_size
            end_idx = start_idx + batch_size

            # Pull out batch
            batch_input = data_set_tensor[start_idx:end_idx]
            batch_input = batch_input[:,dataset['true_input_variable_indices']]

            # Forward pass to get prediction
            if soft_model:
                pred = model(batch_input, input_contexts, output_contexts, dataset['dataset_idx'])
            else:
                pred = model(batch_input, input_contexts, output_contexts)

            # Pull out target
            target = data_set_tensor[start_idx:end_idx]

            # Compute number correct
            class_pred = pred[:,dataset['true_output_variable_indices']]
            class_target = target[:,dataset['true_output_variable_indices']]
            pred_label = torch.argmax(class_pred, dim=1)
            target_label = torch.argmax(class_target, dim=1)
            correct += (pred_label == target_label).sum().item()

            # Compute loss
            loss = loss_fn(class_pred, class_target)
            total_loss += loss.item() * batch_size

        total = data_set_tensor.shape[0]
        accuracy = correct / float(total)
        mean_loss = total_loss / float(total)

    return mean_loss, accuracy
