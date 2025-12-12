import numpy as np
import os
import torch
import torch.nn as nn

from tqdm import tqdm

from dataset import from_path, MAX_WAV_VALUE
from model import Grad

import scipy.io
from scipy.io import loadmat
from scipy.io.wavfile import write
import torch.nn.functional as F

import matplotlib.pylab as plt
import sys

from torchmetrics.functional.audio import \
    scale_invariant_signal_distortion_ratio as si_sdr

import params as params_all


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class GradLearner:
    def __init__(self, model_dir, model, train_dataset, valid_dataset, optimizer, params, fp16):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = optimizer
        self.params = params

        self.autocast = torch.cuda.amp.autocast(enabled=fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        self.step = 0
        self.epoch_train_sisdr = 0
        self.epoch_train_loss = 0
        self.train_sisdr = []
        self.train_loss = []
        self.val_sisdr = []
        self.max_val = -10000
        self.is_master = True

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        if getattr(params, "use_l2_loss", False):
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.L1Loss()

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'params': dict(self.params),
            'scaler': self.scaler.state_dict(),
            'train_sisdr': self.train_sisdr,
            'train_loss': self.train_loss,
            'val_sisdr': self.val_sisdr,
            'max_val': self.max_val
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        # self.step = state_dict['step']
        self.step = state_dict['step'] + 1
        self.train_sisdr = state_dict['train_sisdr']
        self.train_loss = state_dict['train_loss']
        self.val_sisdr = state_dict['val_sisdr']
        self.max_val = state_dict['max_val']

    def save_to_checkpoint(self, filename='weights', best=False):
        save_basename = f'{filename}-{self.step}.pt'
        save_best_basename = f'{filename}-best.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        best_save_name = f'{self.model_dir}/{save_best_basename}.pt'
        if best:
            torch.save(self.state_dict(), best_save_name)
        else:
            torch.save(self.state_dict(), save_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_steps=10000):
        log_file = open('output_log.txt', 'w')
        sys.stdout = log_file
        device = next(self.model.parameters()).device
        self.num_epochs = self.step // len(self.train_dataset)
        val_all_n_epochs = 1
        save_all_n_epochs = 10
        while self.step < max_steps:
            for batch in tqdm(self.train_dataset,
                              desc=f'Epoch {self.step // len(self.train_dataset)}') if self.is_master else self.train_dataset:
                batch = _nested_map(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_step(batch)
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')
                if self.is_master:

                    if self.step % (len(self.train_dataset) * save_all_n_epochs) == 0 and self.step != 0:
                        self.save_to_checkpoint()
                    if self.step % (
                            len(self.train_dataset) * val_all_n_epochs) == 0 and self.step != 0:  # save ckpt per 5 epoch
                        self.train_sisdr.append(self.epoch_train_sisdr / len(self.train_dataset))
                        self.train_loss.append(self.epoch_train_loss / len(self.train_dataset))
                        self.validation(fast_sampling=True)

                        self.num_epochs += val_all_n_epochs
                        epochs = range(0, self.num_epochs, val_all_n_epochs)
                        fig = plt.figure()
                        plt.plot(epochs, self.train_sisdr, label='Training SI-SDR', color='b', linewidth=2)
                        plt.plot(epochs, self.val_sisdr, label='Validation SI-SDR', color='r', linewidth=2)
                        plt.xlabel('Number of epochs', fontsize=10)
                        plt.ylabel('SI-SDR', fontsize=10)
                        plt.legend()
                        plt.savefig("training_si-sdr.png", dpi=300, bbox_inches="tight")
                        plt.show()

                        fig = plt.figure()
                        plt.plot(epochs, self.train_loss, label='Training loss', color='b', linewidth=2)
                        plt.xlabel('Number of epochs', fontsize=10)
                        plt.ylabel('L1 loss (diffusion noise prediction)', fontsize=10)
                        plt.legend()
                        plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
                        plt.show()

                        self.epoch_train_loss = 0
                        self.epoch_train_sisdr = 0

                    # if self.step % (
                    #         len(self.train_dataset) // 10) == 0:  # print loss of a specific batch 3 times an epoch
                    #     print("loss on step no. {:d} is {:2.5f}".format(self.step, loss))
                    #     log_file.flush()

                self.step += 1
        log_file.close()

    def train_step(self, batch):
        for param in self.model.parameters():
            param.grad = None

        audio_bsm_with_array_rot, audio_bsm_with_head_rot, filenames = batch

        audio_bsm_with_array_rot = audio_bsm_with_array_rot.unsqueeze(1)
        audio_bsm_with_head_rot = audio_bsm_with_head_rot.unsqueeze(1)

        N, channel, T = audio_bsm_with_array_rot.shape

        device = audio_bsm_with_array_rot.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio_bsm_with_array_rot.device)
            noise_scale = self.noise_level[t[:, None].repeat(1, channel)].unsqueeze(2)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(audio_bsm_with_array_rot)
            noisy_audio = noise_scale_sqrt * audio_bsm_with_array_rot + (1.0 - noise_scale) ** 0.5 * noise

            conditioner = audio_bsm_with_head_rot

            num_steps = 1
            total_loss, total_loss_sisdr = 0.0, 0.0
            for _ in range(num_steps):
                predicted, extra_output, _ = self.model(noisy_audio, t, conditioner=conditioner)

                pred_audio = (noisy_audio - (1.0 - noise_scale) ** 0.5 * predicted) / noise_scale_sqrt
                pred_audio = torch.clamp(pred_audio, -1.0, 1.0)  # Clamping to ensure values are within range

                step_loss = self.loss_fn(noise, predicted)

                t = torch.maximum(t - 1, torch.tensor(0, device=device))

                noise_scale = self.noise_level[t[:, None].repeat(1, channel)].unsqueeze(2)
                noise_scale_sqrt = noise_scale ** 0.5

                total_loss += step_loss

                total_loss_sisdr += si_sdr(preds=pred_audio, target=audio_bsm_with_array_rot).mean().item()

            loss = total_loss / num_steps
            loss_sisdr = total_loss_sisdr / num_steps

            self.epoch_train_sisdr += loss_sisdr
            self.epoch_train_loss += loss.item()

            # if self.step % (
            #         len(self.train_dataset) // 10) == 0:  # print loss of a specific batch 3 times an epoch
            #     print("SI-SDR loss on step no. {:d} is {:2.3f}".format(self.step, loss_sisdr))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def validation(self, fast_sampling=False):
        device = next(self.model.parameters()).device
        self.model.eval()

        val_sisdr_tot = 0
        with torch.no_grad():
            for j, batch in enumerate(self.valid_dataset):
                batch = _nested_map(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

                audio_bsm_with_array_rot, audio_bsm_with_head_rot, filenames = batch

                audio_bsm_with_array_rot = audio_bsm_with_array_rot.unsqueeze(1)
                audio_bsm_with_head_rot = audio_bsm_with_head_rot.unsqueeze(1)

                training_noise_schedule = np.array(self.model.params.noise_schedule)
                inference_noise_schedule = np.array(
                    self.model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

                talpha = 1 - training_noise_schedule
                talpha_cum = np.cumprod(talpha)

                beta = inference_noise_schedule
                alpha = 1 - beta
                alpha_cum = np.cumprod(alpha)

                T = []
                for s in range(len(inference_noise_schedule)):
                    for t in range(len(training_noise_schedule) - 1):
                        if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                            twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                                    talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                            T.append(t + twiddle)
                            break

                T = np.array(T, dtype=np.float32)

                audio = torch.randn(audio_bsm_with_array_rot.shape[0], audio_bsm_with_array_rot.shape[1],
                                    audio_bsm_with_array_rot.shape[2], device=device)

                conditioner = audio_bsm_with_head_rot

                for n in range(len(alpha) - 1, -1, -1):
                    c1 = 1 / alpha[n] ** 0.5
                    c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5

                    audio = c1 * (audio - c2 * self.model(audio, torch.tensor([T[n]], device=audio.device),
                                                          conditioner=conditioner)[0])
                    if n > 0:
                        noise = torch.randn_like(audio)
                        sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                        audio += sigma * noise
                    audio = torch.clamp(audio, -1.0, 1.0)

                val_sisdr_tot += si_sdr(preds=audio, target=audio_bsm_with_array_rot).mean().item()

                # # if j in  [0, 1, 2, 3]:
                # if j == 10:

                #     os.makedirs('cp/pred/', exist_ok=True)
                #     pred_y = audio[0][0]
                #     pred_y = pred_y * MAX_WAV_VALUE
                #     pred_y = pred_y.cpu().numpy().astype('int16')

                #     output_file = os.path.join('cp/pred/',
                #                                os.path.splitext(filenames[0])[0] + '_pred_{}.wav'.format(self.step))
                #     write(output_file, self.params.sample_rate, pred_y)

            val_sisdr = val_sisdr_tot / (j + 1)
            self.val_sisdr.append(val_sisdr)

            print('Steps : {:d}, Val SI-SDR: {:4.3f}'.format(self.step, val_sisdr))

            if val_sisdr > self.max_val:
                self.max_val = val_sisdr
                print("best val epoch so far at step no. {0}".format(self.step))
                self.save_to_checkpoint(best=True)

                os.makedirs('cp/pred/', exist_ok=True)
                pred_y = audio[0][0]
                pred_y = pred_y * MAX_WAV_VALUE
                pred_y = pred_y.cpu().numpy().astype('int16')

                output_file = os.path.join('cp/pred/',
                                           os.path.splitext(filenames[0])[0] + '_pred_{}.wav'.format(self.step))
                write(output_file, self.params.sample_rate, pred_y)

            self.model.train()


def _train_impl(replica_id, model, train_dataset, valid_dataset, args, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    learner = GradLearner(args["model_dir"], model, train_dataset, valid_dataset, opt, params,
                                  fp16=args["fp16"])
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint(args['weights_file'])
    learner.train(max_steps=args["max_steps"])


def train(args, params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # dor added
    print(device)

    if os.path.exists("train_valid_file_names.mat"):
        mat_data = scipy.io.loadmat('train_valid_file_names.mat')

        # Access the list of strings
        training_filelist = mat_data['training_filelist']
        validation_filelist = mat_data['validation_filelist']

        # Convert the numpy array to a Python list
        training_filelist = training_filelist.tolist()
        validation_filelist = validation_filelist.tolist()

        training_filelist = [file.replace(" ", "") for file in training_filelist]
        validation_filelist = [file.replace(" ", "") for file in validation_filelist]
        len_dataset = len(training_filelist) + len(validation_filelist)

    train_dataset = from_path(training_filelist, len_dataset, params)
    valid_dataset = from_path(validation_filelist, len_dataset, params, shuffle=False)
    model = Grad(params).to(device)

    num_params_generator = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters generator:", num_params_generator)
    print(model)

    _train_impl(0, model, train_dataset, valid_dataset, args, params)


if __name__ == '__main__':
    args = {}
    args['model_dir'] = 'cp'
    args['weights_file'] = 'weights-558001'
    args['max_steps'] = 10e6  # maximum number of training steps
    args['fp16'] = False  # use 16-bit floating point operations for training
    args['params'] = 'params'  # param set name

    params = getattr(params_all, args['params'])
    train(args, params)

