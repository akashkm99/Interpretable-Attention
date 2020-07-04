import json
import os
import shutil
from copy import deepcopy

from helpers import *
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from tqdm import tqdm
import sys
from allennlp.common import Params

from scipy.special import softmax
from scipy.special import expit as sigmoid

from .modelUtils import isTrue, get_sorting_index_with_noise_from_lengths
from .modelUtils import BatchHolder, BatchMultiHolder
from torch.distributions.bernoulli import Bernoulli

from Transparency.model.modules.Decoder import AttnDecoderQA
from Transparency.model.modules.Encoder import Encoder
from Transparency.model.modules.Rationale_Generator import RGenerator_QA

from .modelUtils import jsd as js_divergence
import nltk


file_name = os.path.abspath(__file__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AdversaryMulti(nn.Module) :
    def __init__(self, decoder=None) :
        super().__init__()
        self.decoder = decoder
        self.K = 5

    def forward(self, data) :
        data.P.hidden_volatile = data.P.hidden.detach()
        data.Q.last_hidden_volatile = data.Q.last_hidden.detach()

        new_attn = torch.log(data.P.generate_uniform_attn()).unsqueeze(1).repeat(1, self.K, 1) #(B, 10, L)
        new_attn = new_attn + torch.randn(new_attn.size()).to(device)*3

        new_attn.requires_grad = True
        
        data.log_attn_volatile = new_attn 
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01, amsgrad=True)
        data.multiattention = True

        for _ in range(500) :
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.P.masks.unsqueeze(1), -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn) #(B, 10, L)
            self.decoder.get_output(data)
            
            predict_new = data.predict_volatile #(B, *, O)
            y_diff = self.output_diff(predict_new, data.predict.detach().unsqueeze(1))
            diff = nn.ReLU()(y_diff - 1e-2) #(B, *, 1)

            jsd = js_divergence(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, *, 1)

            cross_jsd = js_divergence(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))

            loss =  -(jsd**1) + 500 * diff #(B, *, 1)
            loss = loss.sum() - cross_jsd.sum(0).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.P.masks.unsqueeze(1), -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)

    def output_diff(self, p, q) :
        #p : (B, *, O)
        #q : (B, *, O)
        softmax = nn.Softmax(dim=-1)
        y_diff = torch.abs(softmax(p) - softmax(q)).sum(-1).unsqueeze(-1) #(B, *, 1)

        return y_diff

class Model() :
    def __init__(self, configuration, pre_embed=None) :
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed

        encoder_copy = deepcopy(configuration['model']['encoder'])
        self.Pencoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)
        self.Qencoder = Encoder.from_params(Params(encoder_copy)).to(device)

        configuration['model']['decoder']['hidden_size'] = self.Pencoder.output_size
        self.decoder = AttnDecoderQA.from_params(Params(configuration['model']['decoder'])).to(device)

        self.bsize = configuration['training']['bsize']

        self.adversary_multi = AdversaryMulti(self.decoder)
        self.diversity_weight = self.configuration['training'].get('diversity_weight',0)

        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.params, weight_decay=weight_decay, amsgrad=True)
        # self.optim = torch.optim.Adagrad(self.params, lr=0.05, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        import time
        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)

    @classmethod
    def init_from_config(cls, dirname, config_update=None, load_gen=False) :
        config = json.load(open(dirname + '/config.json', 'r'))
        print ('old config',config)
        if config_update is not None:
            config.update(config_update)
        print ('new config',config)
        obj = cls(config)
        obj.load_values(dirname)
        if load_gen:
            obj.load_values_generator(dirname)
        return obj

    def train(self, train_data, train=True,epoch=0) :
        docs_in = train_data.P
        question_in = train_data.Q
        entity_masks_in = train_data.E
        target_in = train_data.A

        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in docs_in], noise_frac=0.1)
        docs = [docs_in[i] for i in sorting_idx]
        questions = [question_in[i] for i in sorting_idx]
        entity_masks = [entity_masks_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]
        
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        outputs = []

        bsize = self.bsize
        N = len(questions)
        loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        for idx,n in enumerate((batches)) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Qencoder(batch_data.Q)
            self.Pencoder(batch_data.P)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.LongTensor(batch_target).to(device)

            ce_loss = self.criterion(batch_data.predict, batch_target)
            diverity_loss = self.conicity(batch_data.P).mean()

            loss = ce_loss + + self.diversity_weight*diverity_loss
            predict = torch.argmax(batch_data.predict, dim=-1).cpu().data.numpy()
            outputs.append(predict)

            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train :
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            loss_total += float(loss.data.cpu().item())
            if idx%100 == 0:
                print ("Epoch: {} Step: {} Total Loss: {}, CE loss: {}, Diversity Loss: {} (Diversity_weight = {})".format(epoch,idx,loss,ce_loss.cpu().data, diverity_loss, self.diversity_weight))
        
        outputs = [x for y in outputs for x in y]
        loss_total = loss_total*bsize/N
        
        return loss_total, outputs

    def evaluate(self, data, is_embds=False, is_int_grads=False) :
       
        if(is_int_grads):
            docs = data['P']
            questions = data['Q']
            entity_masks = data['E']
        else:
            docs = data.P
            questions = data.Q
            entity_masks = data.E
        
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        bsize = self.bsize
        N = len(questions)

        outputs = []
        attns = []
        conicity_values = []
        entropy_values = []

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc, is_embds=is_embds)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Qencoder(batch_data.Q)
            self.Pencoder(batch_data.P, is_embds=is_embds)
            self.decoder(batch_data)

            conicity_values.append(self.conicity(batch_data.P).cpu().data.numpy())
            entropy_values.append(self.entropy(batch_data).cpu().data.numpy())

            batch_data.predict = torch.argmax(batch_data.predict, dim=-1)
            if self.decoder.use_attention :
                attn = batch_data.attn
                attns.append(attn.cpu().data.numpy())

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        conicity_values = np.concatenate(conicity_values,axis=0)
        entropy_values = (np.concatenate(entropy_values,axis=0))

        
        return outputs, attns, conicity_values, entropy_values

    
    def entropy(self,data):
        
        attention = data.attn  #(B,L)
        lengths = (data.P.lengths.float() - 2) ## (B)
        masks = data.P.masks.float() #(B,L)

        entropy = -attention*torch.log(attention + 1e-6) *(1-masks)
        entropy = torch.sum(entropy,dim=1)/lengths
        return entropy


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
        self.generator.load_state_dict(torch.load(dirname + '/gen.th'))

    def save_values(self, use_dirname=None, save_model=True) :
        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.Pencoder.state_dict(), dirname + '/encP.th')
            torch.save(self.Qencoder.state_dict(), dirname + '/encQ.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.Pencoder.load_state_dict(torch.load(dirname + '/encP.th'))
        self.Qencoder.load_state_dict(torch.load(dirname + '/encQ.th'))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th'))

    def permute_attn(self, data, num_perm=100) :
        docs = data.P
        questions = data.Q
        entity_masks = data.E

        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        bsize = self.bsize
        N = len(questions)

        permutations_predict = []
        permutations_diff = []

        for n in (range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Qencoder(batch_data.Q)
            self.Pencoder(batch_data.P)
            self.decoder(batch_data)

            predict_true = batch_data.predict.clone().detach()

            batch_perms_predict = np.zeros((batch_data.P.B, num_perm))
            batch_perms_diff = np.zeros((batch_data.P.B, num_perm))

            for i in range(num_perm) :
                batch_data.permute = True
                self.decoder(batch_data)

                predict = torch.argmax(batch_data.predict, dim=-1)
                batch_perms_predict[:, i] = predict.cpu().data.numpy()
            
                predict_difference = self.adversary_multi.output_diff(batch_data.predict, predict_true)
                batch_perms_diff[:, i] = predict_difference.squeeze(-1).cpu().data.numpy()
                
            permutations_predict.append(batch_perms_predict)
            permutations_diff.append(batch_perms_diff)

        permutations_predict = [x for y in permutations_predict for x in y]
        permutations_diff = [x for y in permutations_diff for x in y]
        
        return permutations_predict, permutations_diff
    
    def quantitative_analysis(self, data, dataset) :

        docs = data.P
        questions = data.Q
        entity_masks = data.E
        target = data.A

        self.Pencoder.eval()
        self.Qencoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        N = len(questions)

        outputs = []
        attns = []
        pos_tag_dict = {}
        """
        word_attns = {}

        for i in range(dataset.output_size):
            word_attns[i] = {}
            for key in dataset.vec.word2idx.keys():
                word_attns[i][key] = [0,0]
        """
        for n in (range(0, N, bsize)) :

            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]
            batch_target = target[n:n+bsize]


            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)
            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Qencoder(batch_data.Q)
            self.Pencoder(batch_data.P)
            self.decoder(batch_data)
        
            pred = torch.argmax(batch_data.predict, dim=-1)[0]
            attn = batch_data.attn

        
            batch_data.predict = torch.sigmoid(batch_data.predict)
            attn = batch_data.attn.cpu().data.numpy()
            attns.append(attn)
            predict = batch_data.predict.cpu().data.numpy()

            for idx in range((attn.shape[0])):

                L = batch_data.P.lengths[idx].cpu().data
                label = batch_target[idx]
                seq = batch_data.P.seq[idx][1:L-1].cpu().data.numpy()
                attention = attn[idx][1:L-1]

                words = dataset.vec.map2words(seq)
                words = [word if word != "" else "'<UNK>'" for word in words if word]
                tags = nltk.tag.pos_tag(words)
                tags = [(word, nltk.tag.map_tag('en-ptb', 'universal', tag)) for word, tag in tags]

                for i, (word,tag) in enumerate(tags):

                    if tag not in pos_tag_dict.keys():
                        pos_tag_dict[tag] = []
                        pos_tag_dict[tag].extend([1, attention[i]])

                    else:
                        pos_tag_dict[tag][0] += 1
                        pos_tag_dict[tag][1] += attention[i] 
                    
                    """
                    word_attns[label][word][0] += 1
                    word_attns[label][word][1] += attention[i]
                    """
                    # print ("word, word_attn_negative",word, word_attn_negative[word])

        for keys,values in pos_tag_dict.items():
            
            if values[0] == 0:
                pos_tag_dict[keys].append(0)
            else:    
                pos_tag_dict[keys].append(values[1]/values[0])

        
        # for label in word_attns.keys():
        
            # for keys,values in word_attns[label].items():
                # if values[0] == 0:
                    # word_attns[label][keys].append(0)
                # else:    
                    # word_attns[label][keys].append(values[1]/values[0])
        
        pos_tag_sorted = sorted(pos_tag_dict.items(), key=lambda kv: kv[1][1],reverse=True)
        
        """
        for label in word_attns.keys():
            word_attns[label] = sorted(word_attns[label].items(), key=lambda kv: kv[1][1],reverse=True)
        """
        outputs = {'pos_tags':pos_tag_sorted,'word_attns':{}}

        print ("Pos_attn")
        print(pos_tag_sorted)
        """
        for label in word_attns.keys():
            print ("word_attns ",label)
            print(word_attns[label][:20])
        """
        return outputs

    def minimum_length(self,batch_data, ranking, pred):
        
        length = batch_data.P.lengths[0]
        
        flip = 0        
        for i in range(1,(length-2)+1):  #excluding <s> and <eos>
            
            batch_data.erase_attn = ranking[:,:i]
            self.decoder(batch_data)
            new_pred = torch.argmax(batch_data.predict, dim=-1)[0]
            new_attn = batch_data.attn.cpu().data.numpy()
            
            # print ('i new_logit new_attn',i, new_logit, new_attn)

            if (new_pred != pred):
                flip = 1
                fraction_length = (float(i)/float(length))
                return fraction_length
        
        return 1.0

    def importance_ranking(self, data) :
        docs = data.P
        questions = data.Q
        entity_masks = data.E

        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        attention_lengths = []
        random_lengths = []

        bsize = 1
        N = len(questions)

        for n in range(0, N, bsize) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            batch_data.P.keep_grads = True
            batch_data.detach = True

            self.Qencoder(batch_data.Q)
            self.Pencoder(batch_data.P)
            self.decoder(batch_data)
        
            pred = torch.argmax(batch_data.predict, dim=-1)[0]
            attn = batch_data.attn
            batch_data.erase_given = True

            attention_ranking = attn.sort(dim=1,descending=True)[1]
            length = self.minimum_length(batch_data, attention_ranking, pred)
            attention_lengths.append(length)

            random_ranking = (1+torch.randperm(batch_data.P.lengths[0]-2)).view(1,-1)  #excluding <start> and <eos>
            length = self.minimum_length(batch_data, random_ranking, pred)
            random_lengths.append(length)
            
        return [attention_lengths,random_lengths]

    def conicity(self,data):

        hidden_states = data.hidden    # (B,L,H)
        b,l,h = hidden_states.size()
        masks = data.masks.float() #(B,L)
        lengths = (data.lengths.float() - 2) ## (B)

        hidden_states = hidden_states* (1-masks.unsqueeze(2))
        # hidden_states.masked_fill_(masks.unsqueeze(2).bool(), 0.0)

        mean_state = hidden_states.sum(1) / lengths.unsqueeze(1)

        mean_state = mean_state.unsqueeze(1) #.repeat(1,l,1) #(B,L,H)
        #print (mean_state.size(), hidden_states.size())
        sys.stdout.flush()
        cosine_sim = torch.abs(torch.nn.functional.cosine_similarity(hidden_states, mean_state, dim=2, eps=1e-6))  #(B,L)
        # cosine_sim.masked_fill_(masks.bool(), 0.0)
        cosine_sim = cosine_sim*(1-masks)

        conicity = cosine_sim.sum(1) / lengths  # (B)
        return conicity

    def gradient_mem(self, data, is_embds=False, is_int_grads=False) :


        if(is_int_grads):
            docs = data['P']
            questions = data['Q']
            entity_masks = data['E']
            predicted = data['pred']
        else:
            docs = data.P
            questions = data.Q
            entity_masks = data.E
        
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        bsize = self.bsize
        N = len(questions)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : [], 'X':[]}

        grads_xxe = []
        grads_xxex = []
        grads_H = []
        grads_x = []

        output_arr = []
        conicity_list = []


        for n in range(0, N, bsize) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]
            batch_doc = BatchHolder(batch_doc, is_embds=is_embds)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            batch_data.P.keep_grads = True
            batch_data.detach = True

            self.Qencoder(batch_data.Q)
            self.Pencoder(batch_data.P, is_embds=is_embds)
            self.decoder(batch_data)
            conicity = self.conicity(batch_data.P).cpu().data.numpy()
            conicity_list.append(conicity)

            if is_int_grads:
                max_predict = predicted[n:n+bsize]
            
            else:
                max_predict = torch.argmax(batch_data.predict, dim=-1)
            
            prob_predict = nn.Softmax(dim=-1)(batch_data.predict)
            max_class_prob = torch.gather(prob_predict, -1, max_predict.unsqueeze(-1))
            max_class_prob.sum().backward()
            
            g = batch_data.P.embedding.grad
            em = batch_data.P.embedding
            g1 = (g * em).sum(-1)
            
            grads_x.append(g.cpu().data.numpy())
            grads_xxex.append(g1.cpu().data.numpy())
            
            g1 = (g * self.Pencoder.embedding.weight.sum(0)).sum(-1)
            grads_xxe.append(g1.cpu().data.numpy())
            output_arr.append(max_class_prob)
            # using max_class_prob doesnt make a big diff    
            g1 = (batch_data.P.hidden.grad * batch_data.P.hidden).sum(-1)
            grads_H.append(g1.cpu().data.numpy())
 
        grads['XxE'] = grads_xxe 
        grads['XxE[X]'] = grads_xxex
        grads['H'] = grads_H
        grads['X'] = grads_x
        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]

        conicity_array = np.concatenate(conicity_list,axis=0)
        outputs = [x for y in output_arr for x in y]
        grads['conicity'] = conicity_array        
        return grads, outputs        


    def integrated_gradient_mem(self, data, grads_wrt='X', no_of_instances=100, steps=129):
        
        # no_of_instances = len(data.test_data.P)
        print("Running Integrated gradients for {} examples".format(no_of_instances))

        preds,_,_,_ = self.evaluate(data)
        embd_dict = np.array(self.Pencoder.embedding.weight.cpu().data)
        int_grads = []

        for i in tqdm(range(no_of_instances)):
            new_data = dict()
            #(B,L,E), (B,L), (B,L)
            one_sample = get_complete_testdata_embed_col(data, embd_dict, idx=i, is_qa=True, steps=steps)
            new_data['P'] = one_sample # (B,L,E)
            new_data['Q'] = []
            new_data['Q'].append(data.Q[i])
            new_data['Q'] = new_data['Q'] * steps

            new_data['E'] = []
            new_data['E'].append(data.E[i])
            new_data['E'] = new_data['E'] * steps
            new_data['pred'] = torch.LongTensor([preds[i]]*steps).to(device)
     
            grads, outputs = self.gradient_mem(new_data, is_embds=True, is_int_grads=True)
            int_grads.append(integrated_gradients(grads, outputs, one_sample , grads_wrt='X'))

        return int_grads

