'''Define the tasks and code for loading them'''
import os
import glob
import codecs
import random
import logging as log
from abc import ABCMeta, abstractmethod
import nltk

from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Average


import spacy, re

nlp = spacy.load("en", disable=["parser", "tagger", "ner"])


def cleaner(text, spacy=True):
    text = re.sub(r"\s+", " ", text.strip())
    if spacy:
        text = [t.text.lower() for t in nlp(text)]
    else:
        text = [t.lower() for t in text.split()]
    text = ["qqq" if any(char.isdigit() for char in word) else word for word in text]
    return " ".join(text)

def load_tsv(data_file, max_seq_len, s1_idx=0, s2_idx=1, targ_idx=2, idx_idx=None,
             targ_map=None, targ_fn=None, skip_rows=0, delimiter='\t'):
    '''Load a tsv '''
    sent1s, sent2s, targs, idxs = [], [], [], []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            try:
                row = row.strip().split(delimiter)
                sent1 = cleaner(row[s1_idx])
                if (targ_idx is not None and not row[targ_idx]) or not len(sent1):
                    continue

                if targ_idx is not None:
                    if targ_map is not None:
                        targ = targ_map[row[targ_idx]]
                    elif targ_fn is not None:
                        targ = targ_fn(row[targ_idx])
                    else:
                        targ = int(row[targ_idx])
                else:
                    targ = 0


                if s2_idx is not None:
                    sent2 = cleaner(row[s2_idx])
                    if not len(sent2):
                        continue
                    sent2s.append(sent2)

                if idx_idx is not None:
                    idx = int(row[idx_idx])
                    idxs.append(idx)

                sent1s.append(sent1)
                targs.append(targ)

            except Exception as e:
                print(e, " file: %s, row: %d" % (data_file, row_idx))
                continue

    if idx_idx is not None:
        return sent1s, sent2s, targs, idxs
    else:
        return sent1s, sent2s, targs

def split_data(data, ratio, shuffle=1):
    '''Split dataset according to ratio, larger split is first return'''
    n_exs = len(data[0])
    split_pt = int(n_exs * ratio)
    splits = [[], []]
    for col in data:
        splits[0].append(col[:split_pt])
        splits[1].append(col[split_pt:])
    return tuple(splits[0]), tuple(splits[1])

class Task():
    '''Abstract class for a task

    Methods and attributes:
        - load_data: load dataset from a path and create splits
        - yield dataset for training
        - dataset size
        - validate and test

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes
        self.train_data_text, self.val_data_text, self.test_data_text = \
            None, None, None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pred_layer = None
        self.pair_input = 1
        self.categorical = 1 # most tasks are
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer1 = CategoricalAccuracy()
        self.scorer2 = None

    @abstractmethod
    def load_data(self, path, max_seq_len):
        '''
        Load data from path and create splits.
        '''
        raise NotImplementedError

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        return {'accuracy': acc}

class QQPTask(Task):
    '''
    Task class for Quora Question Pairs.
    '''

    def __init__(self, path, max_seq_len, name="quora"):
        ''' '''
        super(QQPTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.scorer2 = F1Measure(1)

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading QQP data.")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        prc, rcl, f1 = self.scorer2.get_metric(reset)
        return {'accuracy': (acc + f1) / 2, 'acc': acc, 'f1': f1,
                'precision': prc, 'recall': rcl}

class SNLITask(Task):
    ''' Task class for Stanford Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="snli"):
        ''' Args: '''
        super(SNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)
        self.scorer2 = None

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=7, s2_idx=8, targ_idx=-1, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=7, s2_idx=8, targ_idx=-1, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=7, s2_idx=8, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SNLI data.")

    def get_metrics(self, reset=False):
        ''' No F1 '''
        return {'accuracy': self.scorer1.get_metric(reset)}

class MultiNLITask(Task):
    ''' Task class for Multi-Genre Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(MultiNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)
        self.scorer2 = None

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at path.'''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=8, s2_idx=9, targ_idx=11, targ_map=targ_map, skip_rows=1)
        val_matched_data = load_tsv(os.path.join(path, 'dev_matched.tsv'), max_seq_len,
                                    s1_idx=8, s2_idx=9, targ_idx=15, targ_map=targ_map, skip_rows=1)
        val_mismatched_data = load_tsv(os.path.join(path, 'dev_mismatched.tsv'), max_seq_len,
                                       s1_idx=8, s2_idx=9, targ_idx=15, targ_map=targ_map,
                                       skip_rows=1)
        val_data = [m + mm for m, mm in zip(val_matched_data, val_mismatched_data)]
        val_data = tuple(val_data)

        te_matched_data = load_tsv(os.path.join(path, 'test_matched.tsv'), max_seq_len,
                                   s1_idx=8, s2_idx=9, targ_idx=None, idx_idx=0, skip_rows=1)
        te_mismatched_data = load_tsv(os.path.join(path, 'test_mismatched.tsv'), max_seq_len,
                                      s1_idx=8, s2_idx=9, targ_idx=None, idx_idx=0, skip_rows=1)
        # te_diagnostic_data = load_tsv(os.path.join('diagnostic/diagnostic.tsv'), max_seq_len,
                                    #   s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        # te_data = [m + mm + d for m, mm, d in \
                    # zip(te_matched_data, te_mismatched_data, te_diagnostic_data)]
        # te_data[3] = list(range(len(te_data[3])))

        self.train_data_text = tr_data
        self.val_data_text = val_data
        # self.test_data_text = te_data
        log.info("\tFinished loading MNLI data.")

    def get_metrics(self, reset=False):
        ''' No F1 '''
        return {'accuracy': self.scorer1.get_metric(reset)}

class MRPCTask(Task):
    ''' Task class for Microsoft Research Paraphase Task.  '''

    def __init__(self, path, max_seq_len, name="mrpc"):
        ''' '''
        super(MRPCTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.scorer2 = F1Measure(1)

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=0, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=4, targ_idx=0, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading MSRP data.")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        prc, rcl, f1 = self.scorer2.get_metric(reset)
        return {'accuracy': (acc + f1) / 2, 'acc': acc, 'f1': f1,
                'precision': prc, 'recall': rcl}

class STSBTask(Task):
    ''' Task class for Sentence Textual Similarity Benchmark.  '''
    def __init__(self, path, max_seq_len, name="sts_benchmark"):
        ''' '''
        super(STSBTask, self).__init__(name, 1)
        self.categorical = 0
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Average()
        self.scorer2 = Average()
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: float(x) / 5)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len, skip_rows=1,
                            s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: float(x) / 5)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=7, s2_idx=8, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading STS Benchmark data.")

    def get_metrics(self, reset=False):
        # NB: I think I call it accuracy b/c something weird in training
        return {'accuracy': self.scorer1.get_metric(reset),
                'spearmanr': self.scorer2.get_metric(reset)}

class SSTTask(Task):
    ''' Task class for Stanford Sentiment Treebank.  '''
    def __init__(self, path, max_seq_len, name="sst"):
        ''' '''
        super(SSTTask, self).__init__(name, 2)
        self.pair_input = 0
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1, skip_rows=1)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len,
                            s1_idx=0, s2_idx=None, targ_idx=1, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SST data.")

class RTETask(Task):
    ''' Task class for Recognizing Textual Entailment 1, 2, 3, 5 '''

    def __init__(self, path, max_seq_len, name="rte"):
        ''' '''
        super(RTETask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' Process the datasets located at path. '''
        targ_map = {"not_entailment": 0, "entailment": 1}
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading RTE{1,2,3}.")

class QNLITask(Task):
    '''Task class for SQuAD NLI'''
    def __init__(self, path, max_seq_len, name="squad"):
        super(QNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        targ_map = {'not_entailment': 0, 'entailment': 1}
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading QNLI.")

class QNLIv2Task(Task):
    '''Task class for SQuAD NLI'''
    def __init__(self, path, max_seq_len, name="qnliv2"):
        super(QNLIv2Task, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        targ_map = {'not_entailment': 0, 'entailment': 1}
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading QNLIv2.")

class CoLATask(Task):
    '''Class for Warstdadt acceptability task'''
    def __init__(self, path, max_seq_len, name="acceptability"):
        ''' '''
        super(CoLATask, self).__init__(name, 2)
        self.pair_input = 0
        self.load_data(path, max_seq_len)
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Average()
        self.scorer2 = CategoricalAccuracy()

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=None, targ_idx=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=None, targ_idx=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading CoLA.")

    def get_metrics(self, reset=False):
        # NB: I think I call it accuracy b/c something weird in training
        return {'accuracy': self.scorer1.get_metric(reset),
                'acc': self.scorer2.get_metric(reset)}

class WNLITask(Task):
    '''Class for Winograd NLI task'''
    def __init__(self, path, max_seq_len, name="winograd"):
        ''' '''
        super(WNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading Winograd.")

#######################################
# Non-benchmark tasks
#######################################

class DPRTask(Task):
    '''Definite pronoun resolution'''
    def __init__(self, path, max_seq_len, name="dpr"):
        super(DPRTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, data_file, max_seq_len):
        '''Load data'''
        with open(data_file) as data_fh:
            raw_data = data_fh.read()
        raw_data = [datum.split('\n') for datum in raw_data.split('\n\n')]

        targ_map = {'entailed': 1, 'not-entailed': 0}
        tr_data = [[], [], []]
        val_data = [[], [], []]
        te_data = [[], [], []]
        for raw_datum in raw_data:
            sent1 = process_sentence(raw_datum[2].split(':')[1], max_seq_len)
            sent2 = process_sentence(raw_datum[3].split(':')[1], max_seq_len)
            targ = targ_map[raw_datum[4].split(':')[1].strip()]
            split = raw_datum[5].split(':')[1].strip()
            if split == 'train':
                tr_data[0].append(sent1)
                tr_data[1].append(sent2)
                tr_data[2].append(targ)
            elif split == 'dev':
                val_data[0].append(sent1)
                val_data[1].append(sent2)
                val_data[2].append(targ)
            elif split == 'test':
                te_data[0].append(sent1)
                te_data[1].append(sent2)
                te_data[2].append(targ)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

class STS14Task(Task):
    '''
    Task class for Sentence Textual Similarity 14.
    Training data is STS12 and STS13 data, as provided in the dataset.
    '''
    def __init__(self, path, max_seq_len, name="sts14"):
        ''' '''
        super(STS14Task, self).__init__(name, 1)
        self.name = name
        self.pair_input = 1
        self.categorical = 0
        #self.val_metric = "%s_accuracy" % self.name
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer = Average()
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''

        def load_year_split(path):
            sents1, sents2, targs = [], [], []
            input_files = glob.glob('%s/STS.input.*.txt' % path)
            targ_files = glob.glob('%s/STS.gs.*.txt' % path)
            input_files.sort()
            targ_files.sort()
            for inp, targ in zip(input_files, targ_files):
                topic_sents1, topic_sents2, topic_targs = \
                        load_file(path, inp, targ)
                sents1 += topic_sents1
                sents2 += topic_sents2
                targs += topic_targs
            assert len(sents1) == len(sents2) == len(targs)
            return sents1, sents2, targs

        def load_file(path, inp, targ):
            sents1, sents2, targs = [], [], []
            with open(inp) as fh, open(targ) as gh:
                for raw_sents, raw_targ in zip(fh, gh):
                    raw_sents = raw_sents.split('\t')
                    sent1 = process_sentence(raw_sents[0], max_seq_len)
                    sent2 = process_sentence(raw_sents[1], max_seq_len)
                    if not sent1 or not sent2:
                        continue
                    sents1.append(sent1)
                    sents2.append(sent2)
                    targs.append(float(raw_targ) / 5) # rescale for cosine
            return sents1, sents2, targs

        sort_data = lambda s1, s2, t: \
            sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))
        unpack = lambda x: [l for l in map(list, zip(*x))]

        sts2topics = {
            12: ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', \
                    'surprise.SMTnews'],
            13: ['FNWN', 'headlines', 'OnWN'],
            14: ['deft-forum', 'deft-news', 'headlines', 'images', \
                    'OnWN', 'tweet-news']
            }

        sents1, sents2, targs = [], [], []
        train_dirs = ['STS2012-train', 'STS2012-test', 'STS2013-test']
        for train_dir in train_dirs:
            res = load_year_split(path + train_dir + '/')
            sents1 += res[0]
            sents2 += res[1]
            targs += res[2]
        data = [(s1, s2, t) for s1, s2, t in zip(sents1, sents2, targs)]
        random.shuffle(data)
        sents1, sents2, targs = unpack(data)
        split_pt = int(.8 * len(sents1))
        tr_data = sort_data(sents1[:split_pt], sents2[:split_pt],
                targs[:split_pt])
        val_data = sort_data(sents1[split_pt:], sents2[split_pt:],
                targs[split_pt:])
        te_data = sort_data(*load_year_split(path))

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished loading STS14 data.")

    def get_metrics(self, reset=False):
        return {'accuracy': self.scorer.get_metric(reset)}
