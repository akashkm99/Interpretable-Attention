import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
from itertools import zip_longest

def integrated_gradients(grads, outputs, testdata, grads_wrt='X'):
    
    grads_list = grads[grads_wrt]  # (steps,L,H)
    input = np.array(testdata)   # (H)

    x_dash = input[0]
    x = input[-1]
    diff = x - x_dash  # (H)
    
    ydiff = outputs[-1] - outputs[0] 

    grads_list = np.add(grads_list[:-1], grads_list[1:])/2.0
    integral = np.average(np.array(grads_list), axis=0) # (L,H)
    int_grads = np.multiply(integral, diff)  #(L,H) * (1,H) --> (L,H)
    int_grads = np.sum(int_grads,axis=1)   #(L)

    return int_grads

def normalise_grads(grads_list):
    cleaned = []

    for g in grads_list:
        sum = np.sum(g)
        c = [e / sum * 100 for e in g]
        cleaned.append(c)

    return cleaned

def make_single_attri_dict(txt, int_grads, norm_grads_unpruned):
    words = [e for e in txt.split(" ")]

    int_grads_dict = {}
    norm_grads_dict = {}
    norm_grads_pruned = (norm_grads_unpruned[0])[:len(int_grads[0])]

    assert len(int_grads[0]) == len(norm_grads_pruned)

    for i in range(len(words)):
        int_grads_dict[words[i]] = int_grads[0][i]
        norm_grads_dict[words[i]] = norm_grads_unpruned[0][i]

    return (int_grads_dict, norm_grads_dict)

def write_ig_to_file(int_grads, normal_grads_norm, preds, testdata_eng):
    print("Writing IG vs SG results to file")

    with open("./analysis/ig_vs_norm.txt", "a") as f:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f.write("\n\nCurrent Time = {}".format(current_time))

        for i in range(len(testdata_eng)):
            f.write("\nSentence:\n")
            f.write("prediction is: {}\n".format(preds[i]))
            f.write(testdata_eng[i] + "\n")
            i, n = make_single_attri_dict(testdata_eng[i], int_grads[i], normal_grads_norm[i])
            f.write("IG Says:\n")
            f.write(str(i) + "\n")
            f.write("Normal grad says\n")
            f.write(str(n))
            f.write("\n")

def get_sentence_from_testdata(vec, testdata):
    # testdata.X is a list of ndarrays
    reverse_dict = vec.idx2word
    txt = []

    for t in testdata:
        try:
            sent = []
            for ele in t:
                sent.append(reverse_dict[ele])
            sent = " ".join(sent)
            txt.append(sent)
        except:
            pass
    return (txt)

def load_int_grads(file='./pickles/int_grads.pickle'):
    print("loading int_grads from pickle")
    # load int_grads from pickle, wont affect because dataset random seed is fixed
    with open(file, 'rb') as handle:
        int_grads = pickle.load(handle)
    return int_grads

def swap_axis(test):
    # swap 0 and 1 axis of 3d list
    return [[i for i in element if i is not None] for element in list(zip_longest(*test))]

def get_collection_from_embeddings(embd_sent, steps=50):
    # takes test sentence embedding list [wc, 300] and converts into collection [wc, steps, 300]
    # embd_sent is a list of ndarrays

    embd_sent = np.array(embd_sent) # [wc, embed_size]
    zero_vector = np.zeros_like(embd_sent) # [wc, embed_size]

    inc = (embd_sent - zero_vector)/steps

    buffer = []
    buffer.append(list(zero_vector))

    for i in range(steps - 2):
        zero_vector = np.add(zero_vector, inc)
        buffer.append(list(zero_vector))

    buffer.append(list(embd_sent))

    embed_collection = np.array(buffer).swapaxes(0,1)
    return embed_collection

def get_complete_testdata_embed_col(dataset, embd_dict, idx=0, steps=50, is_qa=False):
    # returns tesdata of shape [No.of.instances, Steps, WC, hidden_size] for IG
    if(is_qa):
        embds = get_embeddings_for_testdata(dataset.P[idx], embd_dict)
    else:
        embds = get_embeddings_for_testdata(dataset.test_data.X[idx], embd_dict)
    embds_col = get_collection_from_embeddings(embds, steps=steps)
    embds_col_swapped = swap_axis(embds_col)
    return embds_col_swapped


def get_embeddings_for_testdata(test_data, embd_dict):
    # takes one instance of testdata of shape 1xWC and returns embds of instance of shape 1xWCx300
    # returns list of ndarrays
    embd_sentence = []
    for t in test_data:  # token wise
        embd_sentence.append(list(embd_dict[t]))

    return embd_sentence

def get_embeddings_for_testdata_full(test_data_full, embd_dict, testdata_count=50):
    # does the same thing as the above function but returns the entire collection of test_data

    embed_col = []
    for i in range(testdata_count):
        sent = test_data_full[i]
        buffer = []
        for word in sent:
            buffer.append(list(embd_dict[word]))

        embed_col.append(buffer)
    return embed_col
