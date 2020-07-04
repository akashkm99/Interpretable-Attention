from Transparency.common_code.common import *
from Transparency.common_code.plotting import *
from Transparency.common_code.kendall_top_k import kendall_top_k
from scipy.stats import kendalltau, pearsonr, spearmanr
from functools import partial
import matplotlib.pyplot as plt
import os
import logging
from matplotlib import cm
import numpy as np

import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)


def process_int_grads(grads,X):
    for i in range(len(grads)):
        L = len(X[i])
        grads[i] = np.abs(grads[i])
        grads[i] = grads[i] / grads[i][1:L-1].sum()
    return grads

###########################################################################################################################

def process_grads(grads, X) :

    for k in grads :
        if (k != "conicity") and (k != "X"):
            xxe = grads[k]
            for i in range(len(xxe)) :
                L = len(X[i])
                xxe[i] = np.abs(xxe[i])
                xxe[i] = xxe[i] / np.sum(xxe[i][1:L-1])

###########################################################################################################################

def plot_quant_results(quant_dict, dirname):

    quant_dict['pos_tags']

    pos_attn_scores = [ i[1][1] for i in quant_dict['pos_tags'] ]
    pos_count = [ i[1][0] for i in quant_dict['pos_tags'] ]
    pos_tags = [ i[0] for i in quant_dict['pos_tags'] ]
    print ('pos tags',pos_tags)

    a = np.random.random(40)
    cs = cm.Set1(np.arange(10)/10.)

    fig= plt.figure(figsize=(6,6))
    plt.pie(pos_count,labels=pos_tags,colors=cs, explode=[0.1]*len(pos_count))
    plt.savefig(os.path.join(dirname,"quant_pos_count.png")) 
    plt.show()

    fig= plt.figure(figsize=(6,6))
    plt.pie(pos_attn_scores,labels=pos_tags,colors=cs, explode=[0.1]*len(pos_attn_scores))
    plt.savefig(os.path.join(dirname,"quant_pos_attn.png")) 
    plt.show()


def plot_correlations(test_data, var1, var2, correlation_measure, correlation_measure_name, dirname='', name='New_Attn_Gradient_X', num_samples=None,xlim=(0,1)) :

    X, yhat= test_data.P, test_data.yt_hat

    fig, ax = init_gridspec(3, 3, 1)
    pval_tables = {}
    spcorrs_all = []

    if num_samples is None:
        num_samples = len(X)

    for i in range(num_samples) :
        L = len(X[i])
        if (L <= 3):
            spcorrs_all.append(0.0)    
            print ("Skipping sentence with length < 2")
            continue
 
        spcorr = correlation_measure(list(var1[i][1:L-1]), list(var2[i][1:L-1]))
        spcorrs_all.append(spcorr)

    axes = ax[0]
    pval_tables[name] = plot_measure_histogram_by_class(axes, spcorrs_all, yhat, bins=60)
    annotate(axes, xlim=xlim, ylabel="Density", xlabel=correlation_measure_name+" Correlation", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, name + '_Hist_'+correlation_measure_name)
    save_table_in_file(pval_tables[name], dirname, name + '_val_'+correlation_measure_name)
    show_gridspec()

##############################################################################################################################

def plot_permutations(test_data, permutations, dirname='') :
    X, attn, yhat = test_data.P, test_data.attn_hat, test_data.yt_hat
    ad_y, ad_diffs = permutations
    ad_diffs = 0.5*np.array(ad_diffs)

    med_diff = np.median(ad_diffs, 1)
    max_attn = calc_max_attn(X, attn)
    fig, ax = init_gridspec(3, 3, 1)

    plot_violin_by_class(ax[0], max_attn, med_diff, yhat, xlim=(0, 1.0))
    annotate(ax[0], xlim=(-0.05, 1.05), ylabel="Max attention", xlabel="Median Output Difference", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "Permutation_MAvDY")
    show_gridspec()

##########################################################################################################

def plot_importance_ranking(test_data, importance_ranking, dirname='') :

    fig, ax = init_gridspec(3, 3, 1)
    plot_boxplot(ax, importance_ranking, classes=['attention','random'])
    annotate(ax[0], ylim=(-0.05, 1.05), ylabel="Fraction of attention weights removed", xlabel="Ranking",legend=None)
    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "importance_ranking_MAvDY")
    # show_gridspec()

######################################################################################################################
def plot_conicity(conicity, dirname):
    
    fig= plt.figure(figsize=(6,3))
    plt.hist(conicity,normed=True,bins=50)
    plt.title('Histogram of Conicity')
    plt.xlabel("conicity")
    plt.ylabel("Frequency")
    plt.xlim([0,1])
    plt.savefig(os.path.join(dirname,"conicity_hist.png")) 
    np.savetxt(os.path.join(dirname,"conicity_values"), conicity, fmt="%0.3f")

######################################################################################################################################################

def generate_graphs(dataset, exp_name, model, test_data) :
    logging.info("Generating graph for %s", model.dirname)
    average_length = int(np.clip(test_data.get_stats('P')['mean_length'] * 0.1, 10, None))
    logging.info("Average Length of test set %d", average_length)
    kendall_top_k_dataset = partial(kendall_top_k, k=average_length)

    logging.info('generating part-of-speech plots')
    quant_dict = pload(model, 'quant_analysis')
    plot_quant_results(quant_dict, dirname=model.dirname)


    logging.info("Generating Gradients Graph ...")
    grads = pload(model, 'gradients')
    process_grads(grads, test_data.P)

    attn = test_data.attn_hat
    yhat = test_data.yt_hat

    int_grads = pload(model, 'integrated_gradients')
    int_grads = process_int_grads(int_grads,test_data.P)

    measure_dict = {'pearsonr':lambda x,y:pearsonr(x,y)[0], 'jsd': lambda x,y:jsd(x,y),'total deviation': lambda x,y: np.mean(np.abs(np.array(x) - np.array(y)))}

    for measure_name, measure in measure_dict.items():
         plot_correlations(test_data, grads['XxE[X]'], attn, measure, measure_name, dirname=model.dirname, name='Attn_Gradient_X')
         plot_correlations(test_data, int_grads, attn, measure, measure_name, dirname=model.dirname, name='Attn_Integrated_Gradient')


    plot_conicity(grads['conicity'],dirname=model.dirname)

    
    logging.info("Generating Permutations Graph ...")
    perms = pload(model, 'permutations')
    plot_permutations(test_data, perms, dirname=model.dirname)

    
    logging.info("Generating importance ranking Graph ...")
    importance_ranking = pload(model, 'importance_ranking')
    plot_importance_ranking(test_data, importance_ranking, dirname=model.dirname)

    print("="*300)
