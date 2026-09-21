"""TensorBoard logging for MIR training."""

import numpy as np
from sklearn import metrics


class LogWriter(object):
    def __init__(self, tb_writer, print_iter=500):
        self.tb_writer = tb_writer
        self.print_iter = print_iter
        self.global_step = 0

        self.data = {}

    def _reset_data_field(self, name):
        if name not in self.data:
            self.data[name] = {}
            self.data[name]['epoch_num'] = 0  # We do not reset the epoch_num

        self.data[name]['num_iter'] = 0
        self.data[name]['loss'] = []
        self.data[name]['lr'] = []
        self.data[name]['pred_probs'] = []
        self.data[name]['gt_labels'] = []
        self.data[name]['num_samples'] = 0
        self.data[name]['is_warmup'] = -1

        self.data[name]['extra_info'] = {}

    def step(self, epoch_num, num_samples, epoch_loss, pred_probs, gt_labels, current_lr, is_warmup, mode='Train', extra_info=None):
        if mode not in self.data:
            self._reset_data_field(mode)

        self.global_step += 1  # Global variable so every plot is sync

        self.data[mode]['loss'].extend([epoch_loss.item()] * num_samples)  # We will have as many samples as the batch size
        self.data[mode]['lr'].extend([current_lr] * num_samples)
        self.data[mode]['pred_probs'].extend(pred_probs.cpu().detach().numpy().tolist())
        self.data[mode]['gt_labels'].extend(gt_labels.cpu().detach().numpy().tolist())

        self.data[mode]['num_iter'] += 1
        self.data[mode]['num_samples'] += num_samples
        self.data[mode]['epoch_num'] = epoch_num  # We directly update it. No average
        self.data[mode]['is_warmup'] = int(is_warmup)  # We directly update it. No average

        if extra_info is not None:
            for key, value in extra_info.items():
                if key not in self.data[mode]['extra_info']:
                    self.data[mode]['extra_info'][key] = {}
                    self.data[mode]['extra_info'][key]['vals'] = []

                self.data[mode]['extra_info'][key]['vals'].extend(value['vals'])
                for key_subfield in list(value):
                    if key_subfield != 'vals':
                        self.data[mode]['extra_info'][key][key_subfield] = value[key_subfield]

        if self.data[mode]['num_iter'] % self.print_iter == 0:
            self.flush(mode=mode)

    def flush(self, mode='Train'):
        if self.data[mode]['num_samples'] > 0:
            avg_loss = np.mean(self.data[mode]['loss'])
            avg_lr = np.mean(self.data[mode]['lr'])

            pred_probs = np.array(self.data[mode]['pred_probs'])
            pred_labels = np.array([[1] if x[0] >= 0.5 else [0] for x in self.data[mode]['pred_probs']])
            gt_labels = np.array(self.data[mode]['gt_labels'])
            acc_score = metrics.accuracy_score(y_true=gt_labels, y_pred=pred_labels, normalize=True)

            conf_hist = self._get_confusion_hist(pred_labels, gt_labels)
            self.tb_writer.add_histogram(mode + '/Confusion/G-P-_G-P+_G+P-_G+P+', conf_hist, global_step=self.global_step, bins=4)

            epoch_num = self.data[mode]['epoch_num']
            is_warmup = self.data[mode]['is_warmup']

            self.tb_writer.add_scalar(mode + '/Loss/loss', avg_loss, global_step=self.global_step)
            self.tb_writer.add_scalar(mode + '/Loss/acc', acc_score, global_step=self.global_step)
            self.tb_writer.add_scalar(mode + '/Lr', avg_lr, global_step=self.global_step)

            self.tb_writer.add_histogram(mode + '/Softmax/labels', pred_labels, global_step=self.global_step)
            self.tb_writer.add_histogram(mode + '/Softmax/outputs', pred_probs, global_step=self.global_step)

            self.tb_writer.add_scalar('Epoch', epoch_num, global_step=self.global_step)  # This is shared across all modes
            self.tb_writer.add_scalar('Warmup', is_warmup, global_step=self.global_step)  # This is shared across all modes

            msg = mode + " iter {:d} (epoch {:d}): lr {:.8f}  Loss: {:.3f} Acc: {:.3f}".format(self.global_step, epoch_num, avg_lr, avg_loss, acc_score)
            if is_warmup:
                msg = 'Warmup ' + msg

            if acc_score > 0.5:
                print(bcolors.OKGREEN + msg + bcolors.ENDC, flush=True)
            else:
                print(msg, flush=True)

            for key_name in list(self.data[mode]['extra_info'].keys()):
                key_data = self.data[mode]['extra_info'][key_name]

                key_vals = np.array(key_data['vals'])

                if 'type' in key_data and key_data['type'] == 'histogram':
                    self.tb_writer.add_histogram(mode + '/' + key_name, key_vals, global_step=self.global_step)
                else:
                    avg_key_vals = np.mean(key_vals)
                    self.tb_writer.add_scalar(mode + '/' + key_name, avg_key_vals, global_step=self.global_step)

                    if 'calc_exp' in key_data and key_data['calc_exp'] is True:
                        exp_val = np.exp(-avg_key_vals)

                        self.tb_writer.add_scalar(mode + '/' + key_name + '/exp', exp_val, global_step=self.global_step)

            self._reset_data_field(mode)

    @staticmethod
    def _get_confusion_hist(pred_labels, gt_labels):
        cm = metrics.confusion_matrix(gt_labels, pred_labels, labels=[0, 1])

        # Pred - GT
        neg_neg = cm[0, 0]
        neg_pos = cm[0, 1]
        pos_neg = cm[1, 0]
        pos_pos = cm[1, 1]

        # Now we put it in the format of the confusion "histogram"
        neg_neg = [-15] * neg_neg
        neg_pos = [-5] * neg_pos
        pos_neg = [5] * pos_neg
        pos_pos = [15] * pos_pos

        conf_hist = np.array(neg_neg + neg_pos + pos_neg + pos_pos)

        return conf_hist


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
