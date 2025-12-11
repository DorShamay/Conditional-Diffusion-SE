from torch.cuda import device_count

from learner import train
import params as params_all

def main(args):
    replica_count = device_count()
    params = getattr(params_all, args['params'])


    train(args, params)


if __name__ == '__main__':
    args = {}
    args['model_dir'] = 'cp'
    args['weights_file'] = 'weights-427350'
    args['max_steps'] = 10e6                                                        # maximum number of training steps
    args['fp16'] = False                                                                 # use 16-bit floating point operations for training
    args['params'] = 'params'                                                             # param set name
    main(args)

