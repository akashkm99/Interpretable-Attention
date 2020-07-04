from Transparency.common_code.common import *
from Transparency.common_code.metrics import *
import Transparency.model.Binary_Classification as BC
import numpy as np

metrics_type = {
    'Single_Label' : calc_metrics_classification,
    'Multi_Label' : calc_metrics_multilabel
}

class Trainer() :
    def __init__(self, dataset, config, _type="Single_Label") :
        Model = BC.Model
        self.model = Model(config, pre_embed=dataset.vec.embeddings)
        self.metrics = metrics_type[_type]
        self.display_metrics = True

    def train(self, train_data, test_data, n_iters=8, save_on_metric='roc_auc') :
        best_metric = 0.0
        for i in (range(n_iters)) :

            print ('Starting Epoch: {}'.format(i))

            self.model.train(train_data.X, train_data.y,epoch=i)
            predictions, attentions, conicity_values = self.model.evaluate(test_data.X)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.y, predictions)

            if conicity_values is not None:
                test_metrics['conicity_mean'] = np.mean(conicity_values)
                test_metrics['conicity_std'] = np.std(conicity_values)

            if self.display_metrics :
                print_metrics(test_metrics)

            metric = test_metrics[save_on_metric]

            if(i == 0):
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)

            elif metric > best_metric and i > 0 :
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)
            else :
                save_model = False
                print("Model not saved on ", save_on_metric, metric)

            dirname = self.model.save_values(save_model=save_model)
            print(dirname)
            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()

class RationaleTrainer() :
    def __init__(self, dataset, config, dirname, _type="Single_Label") :
        Model = BC.Model
        
        self.dirname = dirname
        self.config = config
        self.dataset = dataset

        self.model = Model.init_from_config(self.dirname, config_update=self.config, load_gen=False)
        self.model.dirname = self.dirname
    
    def train(self, train_data, test_data, n_iters=40) :
        best_reward = float('-inf')

        for i in (range(n_iters)) :
            print ('Starting Epoch: {}'.format(i))

            _ = self.model.train_generator(train_data.X, train_data.y, epoch=i)

            train_reward, train_predictions = self.model.eval_generator(self.dataset, train_data.X, train_data.y,epoch=i, name='train')
            reward, test_predictions = self.model.eval_generator(self.dataset, test_data.X, test_data.y,epoch=i,name="dev")


            train_predictions = np.array(train_predictions)
            metrics = metrics_type["Single_Label"]
            train_acc = metrics(train_data.y, train_predictions)['accuracy']

            test_predictions = np.array(test_predictions)
            metrics = metrics_type["Single_Label"]
            test_acc = metrics(test_data.y, test_predictions)['accuracy']

            print ('Epoch: {}, Train Reward {}, Train Accuracy {}, Validation Reward {}, Validation Accuracy {}'.format(i,train_reward, train_acc,reward, test_acc))
            
            if reward > best_reward and i > 0 :
                best_reward = reward
                save_model = True
                print("Model Saved")
            else :
                save_model = False
                print("Model not saved")

            dirname = self.model.save_values_generator(save_model=save_model)
    
    def rationale_attn_experiment(self, test_data):
        
        Model = BC.Model
        self.model = Model.init_from_config(self.dirname, config_update=self.config, load_gen=True)
        self.model.dirname = self.dirname
        rationale_attn, predictions = self.model.rationale_attn(self.dataset, test_data.X, test_data.y,name="test")
        pdump(self.model, rationale_attn, 'rationale_attn')

        predictions = np.array(predictions)
        metrics = metrics_type["Single_Label"]
        test_metrics = metrics(test_data.y, predictions)
        f = open(self.model.dirname + '/rationale_evaluate.json', 'w')
        json.dump(test_metrics, f)
        f.close()

class Evaluator() :
    def __init__(self, dataset, dirname, _type='Single_Label') :
        Model = BC.Model
        self.model = Model.init_from_config(dirname,load_gen=False)
        self.model.dirname = dirname
        self.metrics = metrics_type[_type]
        self.display_metrics = True
        self.dataset = dataset

    def evaluate(self, test_data, save_results=False) :
        predictions, attentions, conicity_values = self.model.evaluate(test_data.X)
        predictions = np.array(predictions)

        test_metrics = self.metrics(test_data.y, predictions)

        if conicity_values is not None:
            test_metrics['conicity_mean'] = str(np.mean(conicity_values))
            test_metrics['conicity_std'] = str(np.std(conicity_values))

        if self.display_metrics :
            print_metrics(test_metrics)

        if save_results :
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        test_data.yt_hat = predictions
        test_data.attn_hat = attentions

        test_output = {'X': test_data.X,'y': test_data.y, 'yt_hat':test_data.yt_hat, 'attn_hat': test_data.attn_hat}
        pdump(self.model, test_output, 'test_output')

        return predictions, attentions

    # IG helpers
    def get_grads_from_custom_td(self, test_data):
        print("getting normal grads")
        grads = self.model.gradient_mem(test_data)
        return grads

    def evaluate_outputs_from_embeds(self, embds):
        predictions, attentions = self.model.evaluate(embds)
        return predictions, attentions

    def evaluate_outputs_from_custom_td(self, testdata):
        predictions, _ = self.model.evaluate(testdata)
        return predictions

    def permutation_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'permutations') :
            print('Running Permutation Expt ...')
            perms = self.model.permute_attn(test_data.X)
            print('Dumping Permutation Outputs')
            pdump(self.model, perms, 'permutations')

    def importance_ranking_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'importance_ranking') :
            print('Running Importance Ranking Expt ...')
            importance_ranking = self.model.importance_ranking(test_data.X)
            print('Dumping Importance Ranking Outputs')
            pdump(self.model, importance_ranking, 'importance_ranking')

    def conicity_analysis_experiment(self, test_data):
        self.model.conicity_analysis(test_data.X)

    def integrated_gradient_experiment(self, dataset, force_run=False):
        if force_run or not is_pdumped(self.model, 'integrated_gradients'):
            print('Running Integrated Gradients Expt ...')
            int_grads = self.model.integrated_gradient_mem(dataset)
            print('Dumping Integrated Gradients Outputs')
            pdump(self.model, int_grads, 'integrated_gradients')

    def quantitative_analysis_experiment(self, test_data, dataset, force_run=False) :
        if force_run or not is_pdumped(self.model, 'quant_analysis') :
            print('Running Analysis by Parts-of-speech Expt ...')
            quant_output = self.model.quantitative_analysis(test_data.X,test_data.y,dataset)
            print('Dumping Parts-of-speech Expt Outputs')
            pdump(self.model, quant_output, 'quant_analysis')

    def gradient_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'gradients'):
            print('Running Gradients Expt ...')
            grads = self.model.gradient_mem(test_data.X)[0]
            print('Dumping Gradients Outputs')
            pdump(self.model, grads, 'gradients')
