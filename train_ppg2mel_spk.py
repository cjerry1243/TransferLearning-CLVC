
import os
import time
import math
import h5py
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from common.model import Tacotron2_multispeaker
from common.logger import Tacotron2Logger
from common.hparams_spk import create_hparams
from torch.utils.data import DataLoader
from common.loss_function import Tacotron2Loss


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        return iter(range(self.min_id, self.max_id))


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = torch.LongTensor([x.shape[0] for x in xx])
    y_lens = torch.LongTensor([y.shape[0] for y in yy])

    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0).transpose(1, 2) # (batch_size, feature_dim_1, num_frames_1)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0).transpose(1, 2) # (batch_size, feature_dim_2, num_frames_2)

    gate_pad = torch.zeros([yy_pad.shape[0], yy_pad.shape[2]])
    for i in range(yy_pad.shape[0]):
        gate_pad[i, y_lens[i]-1:] = 1.
    return xx_pad, x_lens, yy_pad, gate_pad, y_lens


class PPG2MEL_Dataset_spk(torch.utils.data.Dataset):
    def __init__(self, h5file, type=None, max_frames=600, random=False, ppg_dim=None):
        self.h5file = h5file
        self.h5 = None
        self.type = type
        self.max_frames = max_frames
        self.random = random
        self.ppg = "ppg_70" if ppg_dim == 72 else "ppg"
    def __getitem__(self, item):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5file, "r")
            if self.type is not None:
                self.h5 = self.h5[self.type]
        if self.random:
            self.max_frames = np.random.randint(400, 800)
        utterance_gp = self.h5[str(item)] # if item < 110 else self.h5
        num_frames = utterance_gp[self.ppg].shape[0]
        start = np.random.randint(0, num_frames - self.max_frames)
        end = start + self.max_frames
        ppg = torch.softmax(torch.FloatTensor(utterance_gp[self.ppg][start:end]), dim=-1)
        zcr = torch.log(torch.FloatTensor(utterance_gp["zcr"][start:end]) + 1e-8).unsqueeze(1)
        log_energy = torch.FloatTensor(utterance_gp["log_energy"][start:end]).unsqueeze(1)
        mel = torch.FloatTensor(utterance_gp["mel_24k"][start: end]) # 861 frames of mel
        dvec = torch.FloatTensor(utterance_gp["dvec"][np.random.randint(0, utterance_gp["dvec"].shape[0]), :])
        return torch.cat((ppg, zcr, log_energy), dim=-1), mel, dvec


def pad_collate_spk(batch):
    (xx, yy, spk) = zip(*batch)
    x_lens = torch.LongTensor([x.shape[0] for x in xx])
    y_lens = torch.LongTensor([y.shape[0] for y in yy])

    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0).transpose(1, 2) # (batch_size, feature_dim_1, num_frames_1)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0).transpose(1, 2) # (batch_size, feature_dim_2, num_frames_2)
    spk_pad = torch.zeros([xx_pad.shape[0], spk[0].shape[0], xx_pad.shape[2]])

    gate_pad = torch.zeros([yy_pad.shape[0], yy_pad.shape[2]])
    for i in range(yy_pad.shape[0]):
        gate_pad[i, y_lens[i]-1:] = 1.
        spk_pad[i, :, :x_lens[i]] = spk[i].unsqueeze(1)
    return (xx_pad, x_lens, yy_pad, gate_pad, y_lens), spk_pad


def prepare_dataloaders(hparams):
    dataset = PPG2MEL_Dataset_spk(hparams.h5_feature_path, max_frames=600, random=True, ppg_dim=hparams.n_symbols)
    val_dataset = PPG2MEL_Dataset_spk(hparams.h5_feature_path, max_frames=800, random=False, ppg_dim=hparams.n_symbols)

    train_loader = DataLoader(dataset, num_workers=min(hparams.batch_size//2, 8),
                              sampler=RandomSampler(0, 14),
                              batch_size=hparams.batch_size,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=pad_collate_spk)

    val_loader = DataLoader(val_dataset, num_workers=8,
                            sampler=RandomSampler(0, 14),
                            batch_size=16,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=pad_collate_spk)

    return train_loader, val_loader


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2_multispeaker(hparams).cuda()
    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, val_loader, iteration, batch_size, n_gpus, logger, distributed_run, rank, teacher_prob):
    model = model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            spk_emb = batch[1].cuda()
            x, y = model.parse_batch(batch[0])
            y_pred = model(x, spk_emb, teacher_prob)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            print("Iteration {} ValLoss {:.6f}  ".format(i+1, val_loss/(i+1)), end="\r")
            if i + 1 == 10:
                break
        val_loss = val_loss / (i + 1)

    if rank == 0:
        print("Validation Loss: {:9f}     ".format(val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)
    model = model.train()


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):

    os.makedirs(os.path.join(hparams.output_directory, "ckpt"), exist_ok=True)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    criterion = Tacotron2Loss(hparams.mel_weight, hparams.gate_weight)
    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    train_loader, val_loader = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path:
        if warm_start: # set to False
            model = warm_start_model(checkpoint_path, model)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = 0

    reduced_loss = 0.
    duration = 0.
    teacher_prob = 1.
    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            spk_emb = batch[1].cuda()
            x, y = model.parse_batch(batch[0])
            y_pred = model(x, spk_emb, teacher_prob)

            loss = criterion(y_pred, y)
            reduced_loss += loss.item()

            if hparams.fp16_run: # set to False
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()
            iters_from_last_save = (iteration - 1) % hparams.iters_per_checkpoint + 1
            running_loss = reduced_loss/iters_from_last_save

            if not math.isnan(reduced_loss) and rank == 0:
                duration += time.perf_counter() - start
                print("Iteration: {} Loss: {:.6f} Teacher: {:.8f} {:.2f}s/it              "
                      "".format(iteration, running_loss, teacher_prob, duration/iters_from_last_save), end="\r")
                logger.log_training(running_loss, learning_rate, duration/iters_from_last_save, iteration)

            if iteration % hparams.iters_per_checkpoint == 0:
                print()
                duration = 0.
                reduced_loss = 0.
                validate(model, criterion, val_loader, iteration,
                         hparams.batch_size, n_gpus, logger,
                         hparams.distributed_run, rank, teacher_prob)
                if rank == 0:
                    checkpoint_path = os.path.join(output_directory, "ckpt", "checkpoint_{}.pt".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    hparams = create_hparams()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train(hparams.output_directory, hparams.log_directory,
          hparams.checkpoint_path, hparams.warm_start, hparams.n_gpus,
          hparams.rank, hparams.group_name, hparams)
