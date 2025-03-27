import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    """
    parser = argparse.ArgumentParser(
        description='Helper script for training and evaluating models on the ECG dataset.'
    )

    # Model and dataset parameters
    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', nargs="?", type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--num_features', nargs="?", type=int, default=1080, help='Number of input features (columns in the dataset)')
    parser.add_argument('--num_classes', nargs="?", type=int, default=2, help='Number of output classes (e.g., N and A)')
    parser.add_argument('--dataset_path', nargs="?", type=str, default='MIT3Sec.csv', help='Path to the ECG dataset')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=0,
                        help='Weight decay to use for Adam')
    # Experiment parameters
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Epoch to resume training from')
    parser.add_argument('--actually_continue', nargs="?", type=str2bool, default=False, help='True if we are not starting a fine-tune')
    parser.add_argument('--seed', nargs="?", type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="ecg_experiment",
                        help='Name of the experiment (used for folder creation)')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True, help='Flag to use GPU if available')
    parser.add_argument('--transformer_heads', nargs="?", type=int, default=1, help='Number of transformer heads')
    parser.add_argument('--hidden_units', nargs="?", type=int, default=12, help='Number of hidden units in final layer')
    parser.add_argument('--is_titan',nargs="?", type=str2bool, default=False, help='True if we are using a titan model')
    parser.add_argument('--is_pretrain',nargs="?", type=str2bool, default=False, help='True if we are pretraining a model')

    #Hyper-parameters specific to our model
    parser.add_argument('--phi', nargs="?", type=float, default=1, help='Scale of CNN Backbone')
    parser.add_argument('--alpha', nargs="?", type=float, default=0.1, help='Alpha for neural memory')
    parser.add_argument('--nm_hu', nargs="?", type=int, default=128, help='Neural Memory hidden units')
    parser.add_argument('--nm_kqv_size', nargs="?", type=int, default=64, help='Size of query/key/value in neural memory')
    parser.add_argument('--pers_dim', nargs="?", type=int, default=32, help='Size of the persistent memory')
    parser.add_argument('--nu', nargs="?", type=float, default=0.9, help='Nu for neural memory')
    parser.add_argument('--theta', nargs="?", type=float, default=0.3, help='Theta for neural memory')

    args = parser.parse_args()
    print(args)
    return args