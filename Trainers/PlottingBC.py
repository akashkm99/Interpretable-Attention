from scipy.stats import pearsonr, spearmanr
from Transparency.common_code.common import *
from Transparency.common_code.plotting import *
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import logging
from random import shuffle
import numpy as np
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

#######################################################################################################################

def process_grads(grads, X) :
    for k in grads :
        xxe = grads[k]
        for i in range(len(xxe)) :
            L = len(X[i])
            xxe[i] = np.abs(xxe[i]).sum(0)
            xxe[i] = xxe[i] / xxe[i][1:L-1].sum()

def process_int_grads(grads,X):
    for i in range(len(grads)):
        L = len(X[i])
        grads[i] = np.abs(grads[i])
        grads[i] = grads[i] / grads[i][1:L-1].sum()
    return grads

def plot_correlations(test_data, var1, var2, correlation_measure, correlation_measure_name, dirname='', name='New_Attn_Gradient_X', num_samples=None,xlim=(0,1)) :

    X, yhat= test_data.X, test_data.yt_hat

    fig, ax = init_gridspec(3, 3, 1)
    pval_tables = {}
    spcorrs_all = []

    if num_samples is None:
        num_samples = len(X)

    for i in range(num_samples) :
        L = len(X[i])
        spcorr = correlation_measure(list(var1[i][1:L-1]), list(var2[i][1:L-1]))
        spcorrs_all.append(spcorr)

    axes = ax[0]
    pval_tables[name] = plot_measure_histogram_by_class(axes, spcorrs_all, yhat, bins=60)
    annotate(axes, xlim=xlim, ylabel="Density", xlabel=correlation_measure_name+" Correlation", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, name + '_Hist_'+correlation_measure_name)
    save_table_in_file(pval_tables[name], dirname, name + '_val_'+correlation_measure_name)
    show_gridspec()

def plot_quant_results(quant_dict, dirname):

    pos_attn_scores = [ i[1][1] for i in quant_dict['pos_tags']]
    pos_count = [ i[1][0] for i in quant_dict['pos_tags']]
    pos_tags = [ i[0] for i in quant_dict['pos_tags']]
    print ('pos tags',pos_tags)

    words_positive = [i[0] for i in quant_dict['word_attn_positive']][:10]
    print ('words_positive',words_positive)

    word_attn_positive = [ i[1][1] for i in quant_dict['word_attn_positive']][:10]
    words_negative = [i[0] for i in quant_dict['word_attn_negative']][:10]
    print ('words_negative',words_negative)

    word_attn_negative = [ i[1][1] for i in quant_dict['word_attn_negative']][:10]
    a = np.random.random(40)
    cs = cm.Set1(np.arange(10)/10)

    fig= plt.figure(figsize=(6,6))
    plt.pie(pos_count,labels=pos_tags,colors=cs, explode=[0.1]*len(pos_count))
    plt.savefig(os.path.join(dirname,"quant_pos_count.png")) 
    plt.show()

    fig= plt.figure(figsize=(6,6))
    plt.pie(pos_attn_scores,labels=pos_tags,colors=cs, explode=[0.1]*len(pos_attn_scores))
    plt.savefig(os.path.join(dirname,"quant_pos_attn.png")) 
    plt.show()

    fig= plt.figure(figsize=(6,6))
    plt.pie(word_attn_positive,labels=words_positive,colors=cs, explode=[0.1]*len(word_attn_positive))
    plt.savefig(os.path.join(dirname,"quant_positive_attn.png")) 
    plt.show()

    fig= plt.figure(figsize=(6,6))
    plt.pie(word_attn_negative,labels=words_negative,colors=cs, explode=[0.1]*len(word_attn_negative))
    plt.savefig(os.path.join(dirname,"quant_negative_attn.png")) 
    plt.show()

def plot_rationale(test_data, rationale_attn_dict, dirname):

    X, yhat= test_data.X, test_data.yt_hat
    fracs = rationale_attn_dict['fraction_lengths']
    sum_attns = rationale_attn_dict['Sum_Attentions']

    fig, ax = init_gridspec(3, 3, 3)

    ###########################################
    axes = ax[0]
    rationale_length = plot_histogram_by_class(axes, fracs, yhat, bins=60)
    print ('rationale_length',rationale_length)
    annotate(axes, xlim=[0,1], ylabel="Density", xlabel="Rationale\'s length (in fraction)", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, axes, dirname, "rationale_lengths_hist")
    # save_table_in_file(rationale_length, dirname, "rationale_lengths")
    show_gridspec()

    ########################################
    
    axes = ax[1]
    rationale_attn = plot_histogram_by_class(axes, sum_attns, yhat, bins=60)
    annotate(axes, xlim=[0,1], ylabel="Density", xlabel="Attention given to Rationale", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, axes, dirname, "rationale_attn_hist")
    # save_table_in_file(rationale_attn, dirname, "rationale_attn")
    show_gridspec()
    
    ##########################################

    axes = ax[2]
    rationale_attn_length = plot_scatter_by_class(axes, fracs, sum_attns, yhat)
    annotate(axes, xlim=[0,1], ylim=[0,1], ylabel="Attention given to Rationale", xlabel="Rationale\'s length (in fraction)", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, axes, dirname, "rationale_attn_lengths_plot")
    show_gridspec()

    ##########################################

###########################################################################################################################

def average_correlation(test_data, var1, var2, correlation_measure, num_samples=None):

    corrs_all = []
    X = test_data.X

    if num_samples is None:
        num_samples = len(X)

    for i in range(num_samples) :
        L = len(X[i])
        corr = correlation_measure(list(var1[i][1:L-1]), list(var2[i][1:L-1]))
        corrs_all.append(corr)


    sprho = np.mean(np.array([x for x in corrs_all]))
    return sprho

#########################################################################################################################

def plot_permutations(test_data, permutations, dirname='') :

    X, yhat, attn = test_data.X, test_data.yt_hat, test_data.attn_hat
    med_diff = np.abs(np.array(permutations) - yhat[:, None, :]).mean(-1)
    med_diff = np.median(med_diff, 1)
    max_attn = calc_max_attn(X, attn)
    fig, ax = init_gridspec(3, 3, 1)

    plot_violin_by_class(ax[0], max_attn, med_diff, yhat, xlim=(0, 1.0))
    annotate(ax[0], xlim=(-0.05, 1.05), ylabel="Max attention", xlabel="Median Output Difference", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "Permutation")

    show_gridspec()

def plot_word_importance_ranking(test_data, importance_ranking, dirname='') :

    
    attn = np.array(importance_ranking['attention'])
    random = np.array(importance_ranking['random'])

    fig, ax = init_gridspec(2, 2, 1)
    values = [attn,random]
    
    plot_boxplot(ax, values, classes=['Attention','Random'])
    annotate(ax[0], ylim=(-0.05, 1.05), ylabel="Fraction of words removed", xlabel="Ranking", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "word_importance_ranking")

    # show_gridspec()

############################################################################################################

def plot_importance_ranking(test_data, importance_ranking, dirname='') :

    fig, ax = init_gridspec(3, 3, 1)
    
    values = [importance_ranking['attention'],importance_ranking['random']]
    
    plot_boxplot(ax, values, classes=['attention','random'])
    annotate(ax[0], ylim=(-0.05, 1.05), ylabel="Fraction of attention weights removed", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "importance_ranking_MAvDY")


    fig, ax = init_gridspec(2, 2, 1)
    
    key_val = importance_ranking.items()
    keys = [i[0] for i in key_val]
    values = [i[1] for i in key_val]
    
    plot_boxplot(ax, values, classes=keys)
    annotate(ax[0], ylim=(-0.05, 1.05), ylabel="Fraction of attention weights removed", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "importance_ranking_MAvDY_all")

##########################################################################################################

def generate_graphs(dataset, exp_name, model, test_data):

    logging.info("Generating graph for %s", model.dirname)
    average_length = int(np.clip(test_data.get_stats('X')['mean_length'] * 0.1, 10, None))
    logging.info("Average Length of test set %d", average_length)

    quant_dict = pload(model, 'quant_analysis')
    plot_quant_results(quant_dict, dirname=model.dirname)
    
    logging.info("Generating Gradients Graph ...")
    grads = pload(model, 'gradients')
    process_grads(grads,test_data.X)

    attn = test_data.attn_hat
    yhat = test_data.yt_hat

    int_grads = pload(model, 'integrated_gradients')
    int_grads = process_int_grads(int_grads,test_data.X)

    measure_dict = {'pearsonr':lambda x,y:pearsonr(x,y)[0],'jsd': lambda x,y:jsd(x,y), 'total deviation': lambda x,y: np.mean(np.abs(np.array(x) - np.array(y)))}
    
    for measure_name, measure in measure_dict.items():
        plot_correlations(test_data, grads['XxE[X]'], attn, measure, measure_name, dirname=model.dirname, name='Attn_Gradient_X')
        plot_correlations(test_data, int_grads, attn, measure, measure_name, dirname=model.dirname, name='Attn_Integrated_Gradient',num_samples=len(int_grads))        

    logging.info("Generating Permutations Graph ...")
    perms = pload(model, 'permutations')
    plot_permutations(test_data, perms, dirname=model.dirname)

    logging.info("Generating importance ranking Graph ...")
    importance_ranking = pload(model, 'importance_ranking')
    plot_importance_ranking(test_data, importance_ranking, dirname=model.dirname)

    rationale_attn_dict = pload(model, 'rationale_attn')
    plot_rationale(test_data,rationale_attn_dict,dirname=model.dirname)
    
print("="*300)
