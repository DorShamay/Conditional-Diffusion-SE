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
    batch_size=8,
    learning_rate=2e-4,
    max_grad_norm=None,
    loss_per_layer=3,
    use_l2_loss=False,

    # Data params
    sample_rate=16000,

    # Model params
    residual_layers=30,
    residual_channels=128,
    dilation_cycle_length=10,
    # residual_layers=9,
    # residual_channels=64,
    # dilation_cycle_length=3,
    unconditional = False,
    noise_schedule=np.linspace(1e-4, 0.02, 200).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
)