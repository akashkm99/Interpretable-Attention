from Transparency.common_code.common import *
from Transparency.common_code.metrics import *
import Transparency.model.Question_Answering as QA


class Trainer() :
    def __init__(self, dataset, config, _type='qa') :
        Model = QA.Model
        self.model = Model(config, pre_embed=dataset.vec.embeddings)
        self.metrics = calc_metrics_qa
        self.display_metrics = True
        self.dataset = dataset
    
    def train(self, train_data, test_data, n_iters=20, save_on_metric='accuracy') :
        best_metric = 0.0
        for i in (range(n_iters)) :
            train_loss, train_predictions = self.model.train(train_data,epoch=i)
            train_metrics = self.metrics(train_data.A, train_predictions)

            print ('End of Epoch: {}, Train Loss: {}, Train Accuracy: {}'.format(i,train_loss,train_metrics['accuracy']))

            predictions, attentions,conicity_values, entropy_values = self.model.evaluate(test_data)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.A, predictions)

            test_metrics['conicity_mean'] = str(np.mean(conicity_values))
            test_metrics['conicity_std'] = str(np.std(conicity_values))
            test_metrics['entropy_mean'] = str(np.mean(entropy_values))
            test_metrics['entropy_std'] = str(np.std(entropy_values))

            if self.display_metrics :
                print ('End of Epoch: {}, Test Accuracy: {}, Conicity: {}, Entropy: {}'.format(i,test_metrics['accuracy'],test_metrics['conicity_mean'],test_metrics['entropy_mean']))
                print_metrics(test_metrics)
            
            metric = test_metrics[save_on_metric]
            if metric > best_metric and i > 0 :
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)
            else :
                save_model = False
                print("Model not saved on ", save_on_metric, metric)
            
            dirname = self.model.save_values(save_model=save_model)
            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()


class Evaluator() :
    def __init__(self, dataset, dirname, _type='qa') :
        Model = QA.Model
        self.model = Model.init_from_config(dirname)
        self.model.dirname = dirname
        self.metrics = calc_metrics_qa
        self.display_metrics = True

    def evaluate(self, test_data, save_results=False, is_embds=False) :
        predictions, attentions,conicity_values, entropy_values  = self.model.evaluate(test_data, is_embds=is_embds)
        predictions = np.array(predictions)

        test_metrics = self.metrics(test_data.A, predictions)

        test_metrics['conicity_mean'] = str(np.mean(conicity_values))
        test_metrics['conicity_std'] = str(np.std(conicity_values))
        test_metrics['entropy_mean'] = str(np.mean(entropy_values))
        test_metrics['entropy_std'] = str(np.std(entropy_values))

        if self.display_metrics :
            print_metrics(test_metrics)

        if save_results :
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        test_data.yt_hat = predictions
        test_data.attn_hat = attentions

        test_output = {'P': test_data.P, 'Q': test_data.Q, 'y':test_data.A, 'yt_hat':test_data.yt_hat, 'attn_hat': test_data.attn_hat}
        pdump(self.model, test_output, 'test_output')

        return predictions, attentions

    def permutation_experiment(self, test_data, force_run=False):
        if force_run or not is_pdumped(self.model, 'permutations'):
            print('Running Permutation Expt ...')
            perms = self.model.permute_attn(test_data)
            print('Dumping Permutation Outputs')
            pdump(self.model, perms, 'permutations')

    def gradient_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'gradients') :
            print('Running Gradients Expt ...')
            grads = self.model.gradient_mem(test_data)[0]
            print('Dumping Gradients Outputs')
            pdump(self.model, grads, 'gradients')
   
    def integrated_gradient_experiment(self, test_data, force_run=False, no_of_instances=10) :
        if force_run or not is_pdumped(self.model, 'integrated_gradients'):
            print('Running Integrated Gradients Expt ...')
            grads = self.model.integrated_gradient_mem(test_data, no_of_instances=len(test_data.P))
            print('Dumping Integrated Gradients Outputs')
            pdump(self.model, grads, 'integrated_gradients')

    def importance_ranking_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'importance_ranking'):
            print('Running Importance Ranking Expt ...')
            importance_ranking = self.model.importance_ranking(test_data)
            print('Dumping Importance Ranking Outputs')
            pdump(self.model, importance_ranking, 'importance_ranking')
    
    def quantitative_analysis_experiment(self, test_data, dataset, force_run=False) :
        if force_run or not is_pdumped(self.model, 'quant_analysis'):
            print('Running Analysis by Parts-of-speech Expt ...')
            quant_output = self.model.quantitative_analysis(test_data,dataset)
            print('Dumping Parts-of-speech Expt Outputs')
            pdump(self.model, quant_output, 'quant_analysis')
