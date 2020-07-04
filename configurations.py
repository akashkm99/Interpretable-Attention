import os
from Transparency.common_code.common import *

def generate_basic_config(dataset, exp_name) :
    config = {
        'model' :{
            'encoder' : {
                'vocab_size' : dataset.vec.vocab_size,
                'embed_size' : dataset.vec.word_dim
            },
            'decoder' : {
                'attention' : {
                    'type' : 'tanh'
                },
                'output_size' : dataset.output_size
            },

            'generator':{}
        },
        'training' : {
            'bsize' : dataset.bsize if hasattr(dataset, 'bsize') else 32,
            'weight_decay' : 1e-5,
            'pos_weight' : dataset.pos_weight if hasattr(dataset, 'pos_weight') else None,
            'basepath' : dataset.basepath if hasattr(dataset, 'basepath') else 'outputs',
            'exp_dirname' : os.path.join(dataset.name, exp_name),
        }
    }
    return config

def generate_diversity_lstm_config(dataset) :
    
    config = generate_basic_config(dataset, exp_name='lstm+tanh')
    
    hidden_size = dataset.hidden_size if hasattr(dataset, 'hidden_size') else 128
    sparsity_lambda = dataset.sparsity_lambda if hasattr(dataset, 'sparsity_lambda') else 0.5


    config['model']['encoder'].update({'type': 'vanillalstm', 'hidden_size' : hidden_size})
    config['model']['generator'].update({'hidden_size' : 64,'sparsity_lambda':sparsity_lambda})

    diversity_weight = dataset.diversity
    config['training']['diversity_weight'] = diversity_weight
    config['training']['context_weight'] = 0
    config['training']['exp_dirname'] += "__diversity_weight_" + str(diversity_weight)
    return config

def generate_orthogonal_lstm_config(dataset) :
    
    config = generate_basic_config(dataset, exp_name='ortho_lstm+tanh')

    hidden_size = dataset.hidden_size if hasattr(dataset, 'hidden_size') else 128
    sparsity_lambda = dataset.sparsity_lambda if hasattr(dataset, 'sparsity_lambda') else 0.5
    config['model']['encoder'].update({'type': 'ortholstm', 'hidden_size' : hidden_size})
    config['model']['generator'].update({'hidden_size' : 64,'sparsity_lambda':sparsity_lambda})
    return config

def generate_lstm_config(dataset) :
    
    config = generate_basic_config(dataset, exp_name='lstm+tanh')
    
    hidden_size = dataset.hidden_size if hasattr(dataset, 'hidden_size') else 128
    sparsity_lambda = dataset.sparsity_lambda if hasattr(dataset, 'sparsity_lambda') else 0.2
    config['model']['encoder'].update({'type': 'vanillalstm', 'hidden_size' : hidden_size})
    config['model']['generator'].update({'hidden_size' : 256,'sparsity_lambda':sparsity_lambda})

    return config


configurations = {
    'vanilla_lstm': generate_lstm_config,
    'diversity_lstm': generate_diversity_lstm_config,
    'ortho_lstm': generate_orthogonal_lstm_config
}

def wrap_config_for_qa(func) :
    def new_func(dataset) :
        config = func(dataset)
        config['model']['decoder']['attention']['type'] += '_qa'
        return config

    return new_func

configurations_qa = { k:wrap_config_for_qa(v) for k, v in configurations.items()}
