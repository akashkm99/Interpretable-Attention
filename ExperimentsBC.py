from Transparency.common_code.common import *
from Transparency.Trainers.PlottingBC import generate_graphs
from Transparency.configurations import configurations
from Transparency.Trainers.TrainerBC import Trainer, Evaluator, RationaleTrainer

def train_dataset(dataset, config='lstm') :

    config = configurations[config](dataset)
    trainer = Trainer(dataset, config=config, _type=dataset.trainer_type)
    if hasattr(dataset,'n_iter'):
        n_iters = dataset.n_iter
    else:
        n_iters = 8
    
    trainer.train(dataset.train_data, dataset.dev_data, n_iters=n_iters, save_on_metric=dataset.save_on_metric)
    evaluator = Evaluator(dataset, trainer.model.dirname, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return trainer, evaluator

def train_dataset_on_encoders(dataset, encoders) :
    for e in encoders :
        train_dataset(dataset, e)
        run_experiments_on_latest_model(dataset, e)
        run_rationale_on_latest_model(dataset, e)
        
def generate_graphs_on_encoders(dataset, encoders) :
    for e in encoders :
        generate_graphs_on_latest_model(dataset, e)

def run_rationale_on_latest_model(dataset, config='lstm') :
    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    rationale_gen = RationaleTrainer(dataset, config, latest_model, _type=dataset.trainer_type)
    print ('Training the Rationale Generator ...')
    _ = rationale_gen.train(dataset.train_data,dataset.dev_data)
    print ('Running Exp to Compute Attention given to Rationales ...')
    rationale_gen.rationale_attn_experiment(dataset.test_data)
    return rationale_gen

def run_evaluator_on_latest_model(dataset, config='lstm') :
    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return evaluator

def run_experiments_on_latest_model(dataset, config='lstm', force_run=True) :
        evaluator = run_evaluator_on_latest_model(dataset, config)
        test_data = dataset.test_data
        evaluator.gradient_experiment(test_data, force_run=force_run)
        evaluator.quantitative_analysis_experiment(test_data, dataset, force_run=force_run)
        evaluator.importance_ranking_experiment(test_data, force_run=force_run)
        evaluator.conicity_analysis_experiment(test_data)
        evaluator.permutation_experiment(test_data, force_run=force_run)
        evaluator.integrated_gradient_experiment(dataset, force_run=force_run)

def generate_graphs_on_latest_model(dataset, config='lstm'):

    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=False)
    generate_graphs(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data)
