import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    # batch_size=16,
    batch_size=8,
    learning_rate=2e-4,
    max_grad_norm=None,
    use_mono=True,
    clip_length=16000,
    lambda_phase=0.01,
    # lambda_phase=0.1,
    loss_per_layer=3,
    use_l2_loss=True,
    use_mean_condition=True,


    # Data params
    # sample_rate=48000,
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,  # Probably an error in paper.

    # Model params
    residual_layers=30,
    residual_channels=128,
    dilation_cycle_length=10,
    # residual_layers=9,
    # residual_channels=128,
    # dilation_cycle_length=3,
    unconditional = False,
    noise_schedule=np.linspace(1e-4, 0.02, 200).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
)