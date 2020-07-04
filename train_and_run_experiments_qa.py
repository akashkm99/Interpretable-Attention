import argparse
parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['ortho_lstm','vanilla_lstm','diversity_lstm'], required=True)
parser.add_argument('--attention', type=str, choices=['tanh', 'dot', 'all'], required=True)
parser.add_argument("--diversity",type=float,default=0)


args, extras = parser.parse_known_args()
args.extras = extras

from Transparency.Trainers.DatasetQA import *
from Transparency.ExperimentsQA import *

dataset = datasets[args.dataset](args)
args.attention='tanh'

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

encoders = [args.encoder]

dataset.diversity = args.diversity

train_dataset_on_encoders(dataset, encoders)
generate_graphs_on_encoders(dataset, encoders)


