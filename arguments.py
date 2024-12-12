import argparse
from distutils.util import strtobool

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=276, help='seed of the experiment')
    parser.add_argument('--learning_rate', type=float, default=0.000587, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='number of minibatches to split the batch into')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train the network for')
    parser.add_argument('--model_name', type=str, default='Unet3D_simple', help='name of the model to use')
    parser.add_argument('--sample_size', type=int, default=512, help='number of classes')
    parser.add_argument('--project_name', type=str, default='Unet3D_Brightfield', help='name of the project')
    parser.add_argument('--step_size', type=int, default=7, help='number of classes')
    parser.add_argument('--gamma', type=int, default=0.1, help='number of classes')
    parser.add_argument('--weight_decay', type=int, default=0.0000464, help='number of classes')
    parser.add_argument('--momentum', type=int, default=0.9, help='number of classes')
    parser.add_argument('--optimiser', type=str, default='adam', help='number of classes')
    parser.add_argument('--patience', type=int, default=40, help='number of classes')
    parser.add_argument('--delta', type=int, default=0.1, help='number of classes')
    parser.add_argument('--lr_scheduler', type=bool, default=False, help='number of classes')
    parser.add_argument('--train_3d', type=bool, default=True, help='number of classes')
    
    parser.add_argument('-f')
    
    args = parser.parse_args()

    return args