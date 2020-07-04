import json
import os
import shutil
from copy import deepcopy
from typing import Dict
    
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from allennlp.common import Params
from sklearn.utils import shuffle
from sklearn import preprocessing
from tqdm import tqdm
import sys
from helpers import *

from scipy.special import softmax
from scipy.special import expit as sigmoid
from tensorboardX import SummaryWriter

from Transparency.model.modules.Decoder import AttnDecoder
from Transparency.model.modules.Encoder import Encoder
from Transparency.model.modules.Attention import masked_softmax
from Transparency.model.modules.Rationale_Generator import RGenerator
from Transparency.common_code.metrics import calc_metrics_classification, calc_metrics_multilabel
from sklearn.metrics import accuracy_score
from Transparency.common_code.common import pload1
from Transparency.Trainers.PlottingBC import process_grads, process_int_grads
import copy

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
from .modelUtils import jsd as js_divergence
import pathlib
import nltk
from multiprocessing import Pool
import codecs

file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

metrics_type = {
    'Single_Label' : calc_metrics_classification,
    'Multi_Label' : calc_metrics_multilabel
}

class Model() :
    def __init__(self, configuration, pre_embed=None) :

        torch.manual_seed(0)

        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed
        print ("encoder params",configuration['model']['encoder'])
        sys.stdout.flush()
        self.encoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)

        configuration['model']['decoder']['hidden_size'] = self.encoder.output_size
        self.decoder = AttnDecoder.from_params(Params(configuration['model']['decoder'])).to(device)

        self.encoder_params = list(self.encoder.parameters())
        self.attn_params = list([v for k, v in self.decoder.named_parameters() if 'attention' in k])
        self.decoder_params = list([v for k, v in self.decoder.named_parameters() if 'attention' not in k])

        # print ('configuration', configuration, self.configuration)

        self.generator = RGenerator(vocab_size=self.configuration['model']['encoder']['vocab_size'], embed_size=self.configuration['model']['encoder']['embed_size'],hidden_size=self.configuration['model']['generator']['hidden_size'], pre_embed=pre_embed).to(device)
        self.generator_params = list(self.generator.parameters())

        self.bsize = configuration['training']['bsize']

        print ('config ',configuration)

        self.diversity_weight = self.configuration['training'].get('diversity_weight',0)
        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.encoder_optim = torch.optim.Adam(self.encoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.attn_optim = torch.optim.Adam(self.attn_params, lr=0.001, weight_decay=0, amsgrad=True)
        self.decoder_optim = torch.optim.Adam(self.decoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.generator_optim = torch.optim.Adam(self.generator_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)

        pos_weight = configuration['training'].get('pos_weight', [1.0]*self.decoder.output_size)
        self.pos_weight = torch.Tensor(pos_weight).to(device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)

        import time
        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)
        print ("Running on device:",device)

    @classmethod
    def init_from_config(cls, dirname, config_update=None, load_gen=False) :
        config = json.load(open(dirname + '/config.json', 'r'))
        if config_update is not None:
            config.update(config_update)
        obj = cls(config)
        obj.load_values(dirname)
        if load_gen:
            obj.load_values_generator(dirname)
        return obj

    def train(self, data_in, target_in, train=True,epoch=0) :
        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        total_iter = len(batches)
        batches = shuffle(batches)

        for idx,n in enumerate(batches):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)
            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.Tensor(batch_target).to(device)

            if len(batch_target.shape) == 1 : #(B, )
                batch_target = batch_target.unsqueeze(-1) #(B, 1)

            bce_loss = self.criterion(batch_data.predict, batch_target)
            weight = batch_target * self.pos_weight + (1 - batch_target)
            bce_loss = (bce_loss * weight).mean(1).mean()

            attn = batch_data.attn

            diverity_loss = self.conicity(batch_data).mean()
            loss = bce_loss + self.diversity_weight*(diverity_loss) 

            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train:
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                self.attn_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                self.attn_optim.step()
                print ("Epoch: {} Step: {} Total Loss: {:.3f}, BCE loss: {:.3f}, Diversity Loss: {:.3f} \
                    (Diversity_weight = {})".format(epoch, idx, loss, bce_loss.cpu().data, diverity_loss, self.diversity_weight))

                n_iters = total_iter*epoch + idx
                sys.stdout.flush()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def train_generator(self, data_in, target_in, train=True, epoch=0) :

        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        total_iter = len(batches)
        batches = shuffle(batches)
        predictions = []

        for idx,n in enumerate(batches):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            probs = self.generator(batch_data)
            m = Bernoulli(probs=probs)
            rationale = m.sample().squeeze(-1)
            batch_data.seq = batch_data.seq * rationale.long()  #(B,L)
            masks = batch_data.masks.float()

            with torch.no_grad():
                self.encoder(batch_data)
                self.decoder(batch_data)

                batch_target = target[n:n+bsize]
                batch_target = torch.Tensor(batch_target).to(device)

                if len(batch_target.shape) == 1 : #(B, )
                    batch_target = batch_target.unsqueeze(-1) #(B, 1)

                bce_loss = self.criterion(batch_data.predict, batch_target)
                weight = batch_target * self.pos_weight + (1 - batch_target)
                bce_loss = (bce_loss * weight).mean(1)

                predict = torch.sigmoid(batch_data.predict).cpu().data.numpy().tolist()
                predictions.append(predict)

            lengths = (batch_data.lengths-2)  #excl <s> and <eos>
            temp = (1-rationale)*(1-masks)
            sparsity_reward = temp.sum(1)/ (lengths.float())
            total_reward = -1*bce_loss +  self.configuration['model']['generator']['sparsity_lambda']*sparsity_reward

            log_probs = m.log_prob(rationale.unsqueeze(-1)).squeeze(-1)
            loss = -log_probs * total_reward.unsqueeze(-1)
            loss = loss.sum(1).mean(0)

            if train:
                self.generator_optim.zero_grad()
                loss.backward()
                self.generator_optim.step()
                print ("Epoch: {}, Step: {} Loss {}, Total Reward: {}, BCE loss: {} Sparsity Reward: {} (sparsity_lambda = {})".format(epoch, idx, loss, total_reward.mean(),
                                                                                     bce_loss.mean(), sparsity_reward.mean(), self.configuration['model']['generator']['sparsity_lambda']))
                n_iters = total_iter*epoch + idx
                sys.stdout.flush()
            loss_total += float(loss.data.cpu().item())
        
        predictions = [x for y in predictions for x in y]

        return loss_total*bsize/N, predictions

    def eval_generator(self, dataset, data, target, epoch, name="") :

        self.encoder.train()
        self.decoder.train()
        self.generator.eval()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        total_iter = len(batches)
        overall_reward = 0
        predictions = []

        for idx,n in enumerate(batches):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            with torch.no_grad():

                probs = self.generator(batch_data)
                m = Bernoulli(probs=probs)
                rationale = m.sample().squeeze(-1)

                input_seq = batch_data.seq.cpu().data.numpy()
                output_seq = input_seq* rationale.long().cpu().data.numpy() #(B,L)
                masks = batch_data.masks.float()

                batch_data.seq = batch_data.seq*rationale.long()
                self.encoder(batch_data)
                self.decoder(batch_data)

                batch_target = target[n:n+bsize]
                batch_target = torch.Tensor(batch_target).to(device)
                
                predict = torch.sigmoid(batch_data.predict).cpu().data.numpy().tolist()
                predictions.append(predict)

                if len(batch_target.shape) == 1 : #(B, )
                    batch_target = batch_target.unsqueeze(-1) #(B, 1)

                bce_loss = self.criterion(batch_data.predict, batch_target)
                weight = batch_target * self.pos_weight + (1 - batch_target)
                bce_loss = (bce_loss * weight).mean(1).cpu().data.numpy()

                lengths = (batch_data.lengths-2)  #excl <s> and <eos>

                sum = ((1 - rationale) * (1 - masks)).sum(1)

                sparsity_reward = (sum.float()/(lengths.float())).cpu().data.numpy() #TODO check again

                total_reward = -1*bce_loss +  self.configuration['model']['generator']['sparsity_lambda']*sparsity_reward

                label = batch_target.cpu().data.numpy()

                for i in range((batch_data.seq.shape[0])):
                    output_dict = {}
                    output_dict['input_sentence'] = dataset.vec.map2words(input_seq[i] )
                    output_dict['generated_rationale'] = dataset.vec.map2words(output_seq[i])
                    output_dict['sparsity_reward'] = sparsity_reward[i]
                    output_dict['bce_loss'] = bce_loss[i]
                    output_dict['total_reward'] = total_reward[i]
                    output_dict['predict'] = predict[i][0]
                    output_dict['label'] = label[i][0]

                    overall_reward +=  total_reward[i]

        overall_reward = overall_reward/N
        predictions = [x for y in predictions for x in y]

        return overall_reward,predictions

    def rationale_attn(self, dataset, data, target, name="") :
                
        self.encoder.train()
        self.decoder.train()
        self.generator.eval()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        total_iter = len(batches)
        overall_reward = 0

        f = codecs.open(self.dirname + '/rationale_' + name + '.txt', 'w',encoding='utf-8')

        fracs = []
        sum_attns = []
        losses = []
        rationales = []
        predictions = []

        for idx,n in enumerate(batches):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            with torch.no_grad():

                masks = batch_data.masks.float()

                probs = self.generator(batch_data)
                m = Bernoulli(probs=probs)
                rationale = m.sample().squeeze(-1)

                input_seq = batch_data.seq
                output_seq = input_seq* rationale.long() #(B,L)
                batch_data.seq = output_seq

                self.encoder(batch_data)
                self.decoder(batch_data)

                ######### Attn Comparision #############
                attn = batch_data.attn.cpu().data.numpy()  #(B,L)
                sum_attn = np.sum(attn*rationale.cpu().data.numpy(),axis=1).tolist()
                sum_attns.extend(sum_attn)

                ########  BCE Loss  ##################
                batch_target = target[n:n+bsize]
                batch_target = torch.Tensor(batch_target).to(device)
                predict = torch.sigmoid(batch_data.predict).cpu().data.numpy()

                predictions.append(predict)

                if len(batch_target.shape) == 1 : #(B, )
                    batch_target = batch_target.unsqueeze(-1) #(B, 1)

                bce_loss = self.criterion(batch_data.predict, batch_target)
                weight = batch_target * self.pos_weight + (1 - batch_target)
                bce_loss = (bce_loss * weight).mean(1).cpu().data.numpy()

                ######### Sparsity  #####################
                lengths = (batch_data.lengths-2)  #excl <s> and <eos>
                sparsity_reward = (torch.sum((1-rationale)*(1-masks),dim=1).float()/(lengths.float())).cpu().data.numpy()

                rationale_frac = torch.sum(rationale,dim=1).float()/lengths.float()
                rationale_frac = rationale_frac.cpu().data.numpy().tolist()
                loss = bce_loss.tolist()

                fracs.extend(rationale_frac)
                losses.extend(loss)

                total_reward = -1*bce_loss +  self.configuration['model']['generator']['sparsity_lambda']*sparsity_reward

                label = batch_target.cpu().data.numpy()

                for i in range((batch_data.seq.shape[0])):
                    output_dict = {}
                    output_dict['input_sentence'] = dataset.vec.map2words(input_seq[i].cpu().data.numpy() )
                    output_dict['generated_rationale'] = dataset.vec.map2words(output_seq[i].cpu().data.numpy())
                    output_dict['sparsity_reward'] = sparsity_reward[i]
                    output_dict['bce_loss'] = bce_loss[i]
                    output_dict['total_reward'] = total_reward[i]
                    output_dict['predict'] = predict[i][0]
                    output_dict['label'] = label[i][0]
                    f.write(str(output_dict) + '\n')

                    overall_reward +=  total_reward[i]

        overall_reward = overall_reward/N
        f.close()

        fracs = np.array(fracs)
        sum_attns = np.array(sum_attns)
        losses = np.array(losses)

        predictions = [x for y in predictions for x in y]

        result_summary = {'Fraction Length Average':np.mean(fracs),'Fraction Length STD':np.std(fracs),'Attn Sum Average':np.mean(sum_attns),'Attn Sum STD':np.std(sum_attns),'loss':np.mean(losses)}
        
        g = open(self.dirname + '/rationale_summary_' + name + '.txt','w')
        g.write(str(result_summary))
        g.close()

        print ("Summary on Test Dataset",result_summary)
        results = {'fraction_lengths':fracs,'Sum_Attentions':sum_attns, 'losses':losses}
        return results, predictions

    def evaluate(self, data) :

        # is_embed check fails when B is very large, embed inputs is not (B,L,E) ndarray
        if (len(np.array(data).shape) == 3):
            is_embed = True
        else:
            is_embed = False

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []
        conicity_values = []

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc, is_embds=is_embed)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict)
            if self.decoder.use_attention :
                attn = batch_data.attn

                attn = attn.cpu().data.numpy()
                attns.append(attn)

            conicity_values.append(self.conicity(batch_data).cpu().data.numpy())
            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        if self.decoder.use_attention :
            attns = [x for y in attns for x in y]
        
        conicity_values = np.concatenate(conicity_values,axis=0)
        return outputs, attns, conicity_values


    def conicity_analysis(self,data):

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []
        cosine_sim_values = []

        conicity = []

        for n in range(0,N,bsize):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)
                
            hidden_states = batch_data.hidden    # (B,L,H)
            masks = batch_data.masks 
            lengths = batch_data.lengths

            conicity.append(self._conicity(hidden_states, masks, lengths).cpu().data.numpy())
        
        conicity = np.concatenate(conicity,axis=0)
        conicity_dict = {'Conicity_hidden': np.mean(conicity)}

        print ("Conicity Analysis",conicity_dict)

        f = codecs.open(self.dirname + '/conicity_analysis.txt', 'w',encoding='utf-8')
        f.write(str(conicity_dict))
        f.close()
        return

    def quantitative_analysis(self, data, target, dataset) :

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []
                    
        word_attn_positive = {}
        word_attn_negative = {}
        pos_tag_dict = {}

        for key in dataset.vec.word2idx.keys():
            word_attn_positive[key] = [0,0]
            word_attn_negative[key] = [0,0]

        word_attn_positive['0.0'] = [0,0]
        word_attn_negative['0.0'] = [0,0]

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)
            batch_target = target[n:n+bsize]

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict)
            attn = batch_data.attn
            predict = batch_data.predict.cpu().data.numpy()

            for idx in range(len(batch_doc)):

                L = batch_data.lengths[idx].cpu().data
                label = batch_target[idx]
                seq = batch_data.seq[idx][1:L-1].cpu().data.numpy()
                words = dataset.vec.map2words(seq)
                words = [word if word != "" else "'<UNK>'" for word in words if word]
                words_pos = [word if word != "qqq" else "0.0" for word in words]

                tags = nltk.tag.pos_tag(words_pos)
                tags = [(word, nltk.tag.map_tag('en-ptb', 'universal', tag)) for word, tag in tags]
                attention = attn[idx,1:L-1].cpu().data.numpy()

                for i, (word,tag) in enumerate(tags):
                    if tag not in pos_tag_dict.keys():
                        pos_tag_dict[tag] = []
                        pos_tag_dict[tag].extend([1, attention[i]])
                    else:
                        pos_tag_dict[tag][0] += 1
                        pos_tag_dict[tag][1] += attention[i] 

                    if label == 0:
                        word_attn_negative[word][0] += 1
                        word_attn_negative[word][1] += attention[i]
                    else:
                        word_attn_positive[word][0] += 1
                        word_attn_positive[word][1] += attention[i]
        
        for keys,values in pos_tag_dict.items():

            if values[0] == 0:
                pos_tag_dict[keys].append(0)
            else:    
                pos_tag_dict[keys].append(values[1]/values[0])

        for keys,values in word_attn_positive.items():

            if values[0] == 0:
                word_attn_positive[keys].append(0)
            else:    
                word_attn_positive[keys].append(values[1]/values[0])

        for keys,values in word_attn_negative.items():

            if values[0] == 0:
                word_attn_negative[keys].append(0)
            else:
                word_attn_negative[keys].append(values[1]/values[0])

        pos_tag_sorted = (sorted(pos_tag_dict.items(), key=lambda kv: kv[1][1],reverse=True))
        attn_positive_sorted = (sorted(word_attn_positive.items(), key=lambda kv: kv[1][1],reverse=True))
        attn_negative_sorted = (sorted(word_attn_negative.items(), key=lambda kv: kv[1][1],reverse=True))

        outputs = {'pos_tags':pos_tag_sorted,'word_attn_positive':attn_positive_sorted,'word_attn_negative':attn_negative_sorted}
        return outputs

    def _conicity(self, hidden, masks, lengths):

        hidden_states = hidden    # (B,L,H)
        b,l,h = hidden_states.size()
        masks = masks.float() #(B,L)
        lengths = (lengths.float() - 2) ## (B)

        hidden_states = hidden_states* (1-masks.unsqueeze(2))
        mean_state = hidden_states.sum(1) / lengths.unsqueeze(1)
        mean_state = mean_state.unsqueeze(1) #.repeat(1,l,1) #(B,L,H)
        cosine_sim = torch.abs(torch.nn.functional.cosine_similarity(hidden_states, mean_state, dim=2, eps=1e-6))  #(B,L)
        cosine_sim = cosine_sim*(1-masks)

        conicity = cosine_sim.sum(1) / lengths  # (B)
        return conicity
    
    def conicity(self, data):
        conicity = self._conicity(data.hidden, data.masks, data.lengths)
        return conicity

    def gradient_mem(self, data) :

        if (len(np.array(data).shape) == 3):
            is_embed = True
        else:
            is_embed = False

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : [], 'X':[]}
        output_arr = []

        for n in (range(0, N, bsize)):
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]

            grads_xxe = []
            grads_xxex = []
            grads_H = []
            grads_x = []
            outputs = []


            for i in range(self.decoder.output_size) :

                batch_data = BatchHolder(batch_doc,is_embds=is_embed)
                batch_data.keep_grads = True
                batch_data.detach = True

                self.encoder(batch_data)
                self.decoder(batch_data)

                torch.sigmoid(batch_data.predict[:, i]).sum().backward()
                g = batch_data.embedding.grad
                em = batch_data.embedding
                g1 = (g * em).sum(-1)

                grads_x.append(g.cpu().data.numpy()) 

                grads_xxex.append(g1.cpu().data.numpy())

                g1 = (g * self.encoder.embedding.weight.sum(0)).sum(-1)
                grads_xxe.append(g1.cpu().data.numpy())

                outputs.append(torch.sigmoid(batch_data.predict[:, i]).cpu().data.numpy())

                g1 = (batch_data.hidden.grad * batch_data.hidden).sum(-1)
                grads_H.append(g1.cpu().data.numpy())

            grads_xxe = np.array(grads_xxe).swapaxes(0, 1)
        
            grads_xxex = np.array(grads_xxex).swapaxes(0, 1)  #(batch_size, 1 , L)
            grads_H = np.array(grads_H).swapaxes(0, 1)
            grads_x = np.array(grads_x).swapaxes(0,1).squeeze(1)  #(batch_size, L, hidden_size)
            
            outputs = np.array(outputs).swapaxes(0,1).squeeze(1)  #(batch_size, 1)

            grads['XxE'].append(grads_xxe)
            grads['XxE[X]'].append(grads_xxex)
            grads['H'].append(grads_H)
            grads['X'].append(grads_x)
            output_arr.append(outputs)

        for k in grads :
            grads[k] = [x for y in grads[k] for x in y] #(N * 1 * L)

        outputs = [x for y in output_arr for x in y]

        return grads, outputs

    def integrated_gradient_mem(self, data, grads_wrt='X', steps=50):

        no_of_instances = len(data.test_data.X)
        print("running Int grad for {} instances".format(no_of_instances))
        embd_dict = np.array(self.encoder.embedding.weight.cpu().data)
        int_grads = []

        print("calculating IG")

        for i in tqdm(range(no_of_instances)):
            
            one_sample = get_complete_testdata_embed_col(data, embd_dict, idx=i, steps=steps)
            grads,outputs = self.get_grads_from_custom_td(one_sample)
            int_grads.append(integrated_gradients(grads, outputs, one_sample , grads_wrt='X'))

        return int_grads

    def get_grads_from_custom_td(self, test_data):
        grads,outputs = self.gradient_mem(test_data)
        return grads,outputs

    def evaluate_outputs_from_embeds(self, embds):
        predictions, attentions, _, _ = self.evaluate(embds)
        return predictions, attentions

    def evaluate_outputs_from_custom_td(self, testdata):
        predictions, _, _, _, _ = self.evaluate(testdata)
        return predictions

    def permute_attn(self, data, num_perm=100) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        permutations = []

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_perms = np.zeros((batch_data.B, num_perm, self.decoder.output_size))
            for i in range(num_perm) :
                batch_data.permute = True
                self.decoder(batch_data)
                output = torch.sigmoid(batch_data.predict)
                batch_perms[:, i] = output.cpu().data.numpy()

            permutations.append(batch_perms)

        permutations = np.concatenate(permutations,axis=0)
        return permutations

    def minimum_length(self,batch_data, ranking):

        length = batch_data.lengths[0]
        logit = batch_data.logit[0]

        flip = 0
        for i in range(1,(length-2)+1):  #excluding <s> and <eos>

            batch_data.erase_attn = ranking[:,:i]
            self.decoder(batch_data)
            new_logit = batch_data.predict[0]
            if ((new_logit*logit) < 0):
                flip = 1
                fraction_length = (float(i)/float(length))
                return fraction_length

        return 1.0    

    def importance_ranking(self,data):

        self.encoder.train()
        self.decoder.train()
        bsize = 1
        N = len(data)

        erase_max = []
        erase_random = []

        attention_lengths = np.zeros(N)
        random_lengths = np.zeros(N)
        lengths = {}

        for n in (range(0, N, bsize)):

            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.logit = batch_data.predict[0]
            attn = batch_data.attn
            mask = batch_data.masks.float()
            batch_data.erase_given = True

            attention_ranking = attn.sort(dim=1,descending=True)[1]
            length = self.minimum_length(batch_data, attention_ranking)
            attention_lengths[n] = (length)

            random_ranking = (1+torch.randperm(batch_data.lengths[0]-2)).view(1,-1)  #excluding <start> and <eos>
            length = self.minimum_length(batch_data, random_ranking)
            random_lengths[n] = (length)

        lengths = {'attention':attention_lengths,'random':random_lengths}
        return lengths


    def save_values(self, use_dirname=None, save_model=True) :

        print ('saved config ',self.configuration)

        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.encoder.load_state_dict(torch.load(dirname + '/enc.th', map_location={'cuda:1': 'cuda:0'}))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th', map_location={'cuda:1': 'cuda:0'}))

    def save_values_generator(self, use_dirname=None, save_model=True) :

        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.generator.state_dict(), dirname + '/gen.th')
        return dirname

    def load_values_generator(self, dirname) :
        self.generator.load_state_dict(torch.load(dirname + '/gen.th', map_location={'cuda:1': 'cuda:0'}))
