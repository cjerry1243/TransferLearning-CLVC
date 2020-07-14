
import os
import json
import torch
import argparse

import numpy as np

from time import time
from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
from mel2samp import Mel2Samp


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config, filter_length=model.filter_length, hop_length=model.hop_length).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)
    def __len__(self):
        return self.max_id - self.min_id


def train(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    criterion = WaveGlowLoss(sigma)
    model = WaveGlow(**waveglow_config, filter_length=data_config["filter_length"], hop_length=data_config["hop_length"]).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)

    trainset = Mel2Samp(**data_config)
    train_loader = DataLoader(trainset, num_workers=6,
                              sampler=RandomSampler(0, 14),
                              batch_size=batch_size,
                              pin_memory=True,
                              drop_last=False)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()
    model = model.cuda()

    s = time()
    reduced_loss = 0
    for i, batch in enumerate(train_loader):
        model.zero_grad()

        mel, audio = batch
        mel = torch.autograd.Variable(mel.cuda())
        audio = torch.autograd.Variable(audio.cuda())
        outputs = model((mel, audio))

        loss = criterion(outputs)

        reduced_loss += loss.item()

        if fp16_run:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        denominator = i % iters_per_checkpoint + 1
        print("iteration:{}, loss:{:.4f}, time:{:.2f}            "
              "".format(iteration + 1, reduced_loss/denominator, (time() - s)/denominator), end="\r")

        if with_tensorboard and rank == 0:
            logger.add_scalar('training_loss', reduced_loss/denominator, iteration + 1)

        if (iteration + 1) % iters_per_checkpoint == 0:
            s = time()
            reduced_loss = 0
            if rank == 0:
                checkpoint_path = "{}/waveglow_it{}.pt".format(output_directory, iteration + 1)
                save_checkpoint(model, optimizer, learning_rate, iteration + 1, checkpoint_path)
        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)
