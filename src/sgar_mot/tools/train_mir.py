"""Training script for the Mask Instance Resolver (MIR).

MIR is trained on samples extracted from MOTSynth: situations in which the initial
association of the tracker failed, each described by three mask embeddings and a
label saying whether the track should have been recovered.
"""

import argparse
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sgar_mot.data import MIRDatasetTest, MIRDatasetTrain
from sgar_mot.mir import MIR
from sgar_mot.utils.config import get_ram_usage, load_args_from_config, merge_args
from sgar_mot.utils.log_writer import LogWriter


def get_loaders(args):
    """Extraction of data loaders for training and validation subsets."""
    mem_usg_before = get_ram_usage() / (1024 * 1024 * 1024)
    train_dataset = MIRDatasetTrain(data_path=args.train_set, memory_limit=200, seq_split=args.n_splits > 1,
                                    seq_split_halves=args.n_splits, data_type=args.data_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.nb_worker, pin_memory=True, drop_last=True)
    mem_usg_after = get_ram_usage() / (1024 * 1024 * 1024)
    mem_consumption = mem_usg_after - mem_usg_before
    print('Loaded Train Dataset. Contains {} samples from {} sequences ({} IDs), occupying {:.2f} GB (roughly {:.2f} GB per sequence)'.format(
        len(train_dataset), train_dataset.num_seq, train_dataset.num_training_targets, mem_consumption, mem_consumption / len(train_dataset)), flush=True)

    mem_usg_before = get_ram_usage() / (1024 * 1024 * 1024)
    val_dataset = MIRDatasetTest(data_path=args.val_set, data_type=args.data_type)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.nb_worker, pin_memory=True, drop_last=True)
    mem_usg_after = get_ram_usage() / (1024 * 1024 * 1024)
    mem_consumption = mem_usg_after - mem_usg_before
    print('Loaded Val Dataset. Contains {} samples from {} sequences ({} IDs), occupying {:.2f} GB (roughly {:.2f} GB per sequence)'.format(
        len(val_dataset), val_dataset.num_seq, val_dataset.num_training_targets, mem_consumption, mem_consumption / len(val_dataset)), flush=True)

    return train_loader, val_loader


def set_all_seeds(seed):
    """Sets all seeds for reproducibility."""
    print("Setting functions of seed setting to {}, in order to achieve reproducibility".format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(args):
    """Extraction of the model"""
    # Model + memory consumption statistics:
    mem_usg_before = get_ram_usage() / (1024 * 1024 * 1024)
    model = MIR(args.transformer)
    mem_usg_after = get_ram_usage() / (1024 * 1024 * 1024)
    mem_consumption = mem_usg_after - mem_usg_before

    # Occupation computation:
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    mem_model = (mem_params + mem_bufs) / (1024 * 1024 * 1024)

    print('Model contains {:.2f}M parameters ({:.2f}M learnable) and occupies {:.2f} GB of GPU and {:.2f} GB of RAM'.format(
        model.num_params / 1e6, model.num_trainable_params / 1e6, mem_model, mem_consumption), flush=True)

    return model


def warmup_learning_rate(optimizer, nb_epo, warmup_epo, max_lr):
    """warmup learning rate"""
    lr = max_lr * nb_epo / warmup_epo
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    # Parameters:
    parser = argparse.ArgumentParser(description="MIR training", add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-set", type=str, default=None, help="training data file or directory")
    parser.add_argument("--val-set", type=str, default=None, help="validation data file or directory")
    parser.add_argument("--data-type", type=str, default=None, help="if not None, type of data that will be used in training")
    parser.add_argument("--config", type=str, default="configs/mir_vitbase.yml", help="path to file with MIR configurations")

    parser.add_argument("--out-dir", type=str, default="./outputs", help="output directory")
    parser.add_argument("--exp-name", type=str, default="mir_vitbase", help="name of the experiment, that will result on a file inside the output directory")
    parser.add_argument("--resume-ckpt-pth", type=str, default=None, help="path to, if required, a checkpoint file to resume training process")

    parser.add_argument("--n-splits", type=int, default=2, help="number of splits to be used when loading dataset")

    # Training specific parameters load:
    parser_args = parser.parse_args()
    trainer_args = load_args_from_config(parser_args.config).trainer

    args = merge_args(trainer_args, parser_args)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    args.transformer.device = args.device

    # Set all seeds for reproducibility:
    if args.seed is not None:
        set_all_seeds(args.seed)

    train_loader, val_loader = get_loaders(args)

    model = get_model(args)

    # Loss function -- binary crossentropy, as it is a binary classification problem:
    transformer_criterion = nn.BCELoss()

    # Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, momentum=0.9)

    # Number of iterations:
    args.total_iter = len(train_loader) * args.num_epochs

    # Save directory:
    save_dir = os.path.join(args.out_dir, args.exp_name)
    print("Results save directory: {}".format(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # We save the args in a file
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    # We also save the config file
    shutil.copyfile(parser_args.config, os.path.join(save_dir, 'config.yml'))

    # tensorboard utils
    tb_dir = os.path.join(save_dir, 'log')
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    writer.add_text('Args', str(args), global_step=0)

    log_writer = LogWriter(tb_writer=writer, print_iter=args.print_iter)

    warmup_iter = args.warmup_epochs * len(train_loader)

    # Resume from checkpoint instead from scratch:
    if args.resume_ckpt_pth:
        print('Resuming training from {}'.format(args.resume_ckpt_pth))
        model.load_pretrained(args.resume_ckpt_pth)

    loss_trace = []
    val_trace = []
    train_trace_success = []
    val_trace_success = []

    start_time = time.time()

    for epoch in tqdm(range(args.num_epochs)):
        if epoch != 0:
            # From first epoch onwards, we will update the data loaded:
            train_loader.dataset.alternate_load()
        print('', flush=True)  # So our log messages are printed nicely
        current_loss = 0
        current_success = 0
        current_lr = args.max_lr
        is_warmup = False
        model.train()
        for i, (frame_id, object_id, mask_embeddings, msk_bboxes, det_frame_id, t_frame_id, label) in enumerate(train_loader):

            if i % 100 == 0:
                print('Epoch: {} - Iter: {}/{}'.format(epoch, i, len(train_loader)), end='\r', flush=True)

            nb_iter = epoch * len(train_loader) + i

            # Convert to device:
            mask_embeddings = mask_embeddings.to(torch.float32).to(args.device)
            msk_bboxes = msk_bboxes.to(torch.long).to(args.device)
            label = label.to(torch.float32).to(args.device)

            if nb_iter < warmup_iter:
                is_warmup = True
                warmup_beta = nb_iter / warmup_iter  # This will be used only for warmup
            else:
                is_warmup = False

            outputs = model(mask_embeddings, msk_bboxes, det_frame_id, t_frame_id)

            # Apply BCE loss:
            transformer_loss = transformer_criterion(outputs, label)
            loss = transformer_loss

            if nb_iter < warmup_iter:
                loss = loss * warmup_beta * args.beta_z

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if warmup_iter > 0 and nb_iter < warmup_iter:
                current_lr = warmup_learning_rate(optimizer, nb_iter, warmup_iter, args.max_lr)
            else:
                current_lr = args.max_lr

            current_predictions = outputs  # Sigmoid is ran as the last step in the transformer.

            # Round number to get the predicted class (if higher than 0.5 to 1)
            pred_label = torch.Tensor([[1] if x >= 0.5 else [0] for x in current_predictions]).to(torch.float32).to(args.device)
            gt_label = label  # Labels from the ground truth.

            num_success = torch.sum(pred_label == gt_label).item()
            current_success += num_success

            num_samples = len(mask_embeddings)  # to take batch size basically

            log_extra_info = {'Loss/cls_loss': {'vals': [transformer_loss.item()] * num_samples, 'calc_exp': True},  # We will have as many (equal) samples as the batch size
                              'Softmax/inputs': {'vals': outputs.tolist(), 'type': 'histogram'},
                              }

            log_writer.step(epoch_num=epoch,
                            num_samples=num_samples,
                            epoch_loss=loss,
                            pred_probs=current_predictions,
                            gt_labels=label,
                            current_lr=current_lr,
                            is_warmup=is_warmup,
                            extra_info=log_extra_info,
                            mode='Train')

            current_loss += loss.item() * mask_embeddings.size(0)

        batch_loss = current_loss / len(train_loader.dataset)
        batch_success = current_success / len(train_loader.dataset)
        loss_trace.append(batch_loss)
        train_trace_success.append(batch_success)

        log_writer.flush(mode='Train')

        # VALIDATION
        current_loss = 0
        current_success = 0
        model.eval()
        with torch.no_grad():  # To turn off gradients computation:
            for i, (frame_id, object_id, mask_embeddings, msk_bboxes, det_frame_id, t_frame_id, label) in enumerate(val_loader):

                # Convert to device:
                mask_embeddings = mask_embeddings.to(torch.float32).to(args.device)
                msk_bboxes = msk_bboxes.to(torch.long).to(args.device)
                label = label.to(torch.float32).to(args.device)

                num_samples = len(mask_embeddings)

                outputs = model(mask_embeddings, msk_bboxes, det_frame_id, t_frame_id)

                # Apply BCE loss:
                transformer_loss = transformer_criterion(outputs, label)
                loss = transformer_loss

                current_predictions = outputs
                pred_label = torch.Tensor([[1] if x >= 0.5 else [0] for x in current_predictions]).to(torch.float32).to(args.device)
                gt_label = label  # Labels from the ground truth.
                num_success = torch.sum(pred_label == gt_label).item()
                current_success += num_success

                log_extra_info = {'Loss/cls_loss': {'vals': [transformer_loss.item()] * num_samples, 'calc_exp': True},
                                  'Softmax/inputs': {'vals': outputs.tolist(), 'type': 'histogram'},
                                  }

                log_writer.step(epoch_num=epoch,
                                num_samples=num_samples,
                                epoch_loss=loss,
                                pred_probs=current_predictions,
                                gt_labels=label,
                                current_lr=current_lr,
                                is_warmup=is_warmup,
                                extra_info=log_extra_info,
                                mode='Eval')

                current_loss += loss.item() * mask_embeddings.size(0)

        batch_loss = current_loss / len(val_loader.dataset)
        batch_success = current_success / len(val_loader.dataset)
        val_trace.append(batch_loss)
        val_trace_success.append(batch_success)

        log_writer.flush(mode='Eval')

        torch.save(model.state_dict(), os.path.join(save_dir, 'mytransformer_{:05d}.pth'.format(epoch)))

    torch.save(model.state_dict(), os.path.join(save_dir, 'mytransformer.pth'))

    min_val_epoch = val_trace.index(min(val_trace))
    min_val_loss = val_trace[min_val_epoch]
    min_val_perc = np.exp(-min_val_loss)

    min_val_train_loss = loss_trace[min_val_epoch]
    min_val_train_perc = np.exp(-min_val_train_loss)

    ####### PRINT SUMMARY #######
    with open(os.path.join(save_dir, 'SUMMARY.txt'), "w") as text_file:
        text_file.write("Experiment: {}\n".format(args.exp_name))
        text_file.write("train-filename': {}\n".format(args.train_set))
        text_file.write("val-filename': {}\n".format(args.val_set))
        text_file.write("dim-embedding: {}\n".format(args.transformer.dim_embedding))
        text_file.write("num-epochs: {}\n".format(args.num_epochs))
        text_file.write("dropout-p: {}\n".format(args.transformer.dropout_p))
        text_file.write("activation: {}\n".format(args.transformer.activation))
        text_file.write("nhead: {}\n".format(args.transformer.nhead))
        text_file.write("num-layer: {}\n".format(args.transformer.num_layer))
        text_file.write("trans-dim: {}\n".format(args.transformer.trans_dim))
        text_file.write("ff-size: {}\n".format(args.transformer.ff_size))

        text_file.write("min_val_epoch: {}\n".format(min_val_epoch))
        text_file.write("min_val_loss: {}\n".format(min_val_loss))
        text_file.write("min_val_perc: {:.0%}\n".format(min_val_perc))

        text_file.write("min_val_train_loss: {}\n".format(min_val_train_loss))
        text_file.write("min_val_train_perc: {:.0%}\n".format(min_val_train_perc))

        text_file.write("val_losses: {}\n".format(val_trace))
        text_file.write("train_losses: {}\n".format(loss_trace))

        text_file.write("val_trace_success: {}\n".format(val_trace_success))
        text_file.write("train_trace_success: {}\n".format(train_trace_success))

        text_file.write("\n")

    print("Total time: " + str(time.time() - start_time) + " sec.")

    # loss curve
    plt.plot(range(1, args.num_epochs + 1), loss_trace, 'r-')
    plt.plot(range(1, args.num_epochs + 1), val_trace, 'b-')
    plt.plot(range(1, args.num_epochs + 1), val_trace_success, 'g-')
    plt.plot(range(1, args.num_epochs + 1), train_trace_success, 'y-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax = plt.gca()
    ax.set_ylim([0.0, 1.0])
    plt.savefig(os.path.join(save_dir, 'transformers.png'))


if __name__ == '__main__':
    main()
