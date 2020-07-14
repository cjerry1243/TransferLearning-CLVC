
import os
import torch
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from .plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from .plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)
        self.logdir = logdir

    def log_training(self, reduced_loss, learning_rate, duration, iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        # self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        mel_outputs_before_postnet, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted_before_postnet",
            plot_spectrogram_to_numpy(mel_outputs_before_postnet[idx].data.cpu().numpy()), iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
                iteration, dataformats='HWC')
        ### write output mel to file
        # out = mel_outputs[idx]
        # out = torch.FloatTensor([out[:, f].tolist() for f in range(out.shape[1]) if not torch.all(out[:, f] == 0)]).t()
        # torch.save(out, os.path.join(self.logdir, "..", "mels", "mel_outputs_it{}.pt".format(iteration)))


class WaveglowLogger(SummaryWriter):
    def __init__(self, logdir):
        super(WaveglowLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
