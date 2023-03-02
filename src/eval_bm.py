import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette()
from sklearn.metrics.pairwise import euclidean_distances as euc_dist
from eval_ds import *
from embed_evals import syn_evals, get_NI

import argparse
parser = parser = argparse.ArgumentParser()
parser.add_argument("-w", "--wandb_csv", default=None, type=str)
parser.add_argument("-p", "--wandb_project", default=None, type=str)
parser.add_argument("-g", "--wandb_group", default=None, type=str)
parser.add_argument("-o", "--output_path", default=None, type=str)
parser.add_argument("-s", "--syn_embs_path", default=None, type=str)
parser.add_argument("-n", "--h2h_train_embs_path", default=None, type=str)
parser.add_argument("-u", "--update", default=None, type=str)
args = parser.parse_args()
print(args)

import pathlib
if args.wandb_csv is not None:
    wandb_csv = args.wandb_csv
    wandb_csv_path = pathlib.Path(wandb_csv)
    file_path = wandb_csv.split("/")[-1]
    project, group = file_path.split(".")[:2]
    results_path = str(wandb_csv_path.parents[1]) + '/'
elif args.wandb_project is not None and args.wandb_group is not None:
    project = args.wandb_project
    group = args.wandb_group
    results_path = 'results/'
    wandb_csv = results_path + 'wandb/' + '.'.join([project, group, 'csv'])
else: 
    print("Error: no argument passed!")
    exit()

if args.output_path:
    results_path = args.output_path + '/' if args.output_path[-1] != '/' else args.output_path

results_csv = results_path + '.'.join([project, group, 'csv'])
print(results_path, results_csv)

y_train, y_valid, y_test = pickle.load(open('../data/datasets/bm/labels.pkl', 'rb'))


def load_all_embs(path, models, dim, arch=None, seeds=None):
    emb, folds = {}, ['train', 'valid', 'test']
    for model in models:
        if model not in emb: emb[model] = {}
        model_path = '/'.join([path, model, arch]) if arch else '/'.join([path, model])
        if seeds is not None:
            for seed in seeds:
                if model not in emb[model]: emb[model][seed] = {}
                for fold in folds:
                    if fold not in emb[model][seed]: emb[model][seed][fold] = {}
                    emb[model][seed][fold] = pickle.load(
                        open(f'{model_path}_{fold}_emb{dim}_s{seed}.pkl', 'rb'))
        else:
            for fold in folds:
                if fold not in emb[model]: emb[model][fold] = {}
                emb[model][fold] = pickle.load(
                    open(f'{model_path}_{fold}_emb{dim}.pkl', 'rb'))
    return emb

def get_dst_from_embs(embs):
    train, test = embs['train'], embs['test']
    return euc_dist(test, train)

def get_dss_from_embs(embs):
    train, test = embs['test'], embs['test']
    return euc_dist(test, train)

df_wandb = pd.read_csv(wandb_csv)
df = df_wandb.copy()
df['model'] = df['Name'].map(lambda x: x.split('s')[0])
df['perf'] = df['test_clf_acc'] + df['test_triplet_acc']
df = df.loc[df.groupby('model')['perf'].idxmax()]
df_wandb_best = df.copy()
best = {k:int(v) for k, v in df.Name.str.split('s')}
agent = 's'.join(['MTL0', str(best['MTL0'])])
print("best seeds:", best)

embeds_path = '../data/embeds/bm/prolific/' + '/'.join([project, group])
models = [f'MTL{l}' for l in [0, 0.2, 0.5, 0.8, 1]]
dim = df_wandb.loc[0]['_content.embed_dim']
seeds = sorted(df_wandb['_content.seed'].unique())
print("all seeds:", seeds)
embs = load_all_embs(embeds_path, models, dim, 'MTL_han', seeds)
dsts = {}
for model in models:
    for seed in seeds:
        name = 's'.join([model, str(seed)])
        dsts[name] = get_dst_from_embs(embs[model][seed])
b_embs = {m: embs[m][best[m]] for m in embs}

if args.syn_embs_path:
    syn_path = args.syn_embs_path
    syn_models = [f'MTL{l}' for l in [0]]
    syn_embs = load_all_embs(syn_path, syn_models, 50, 'MTL_han', range(5))
    syn_dsts = {}
    for model in syn_models:
        for seed in range(5):
            name = 's'.join([model, str(seed)])
            syn_dsts[name] = get_dst_from_embs(syn_embs[model][seed])
    syns = {k: v for k, v in syn_dsts.items() if 'MTL0s' in k}
else:
    syns = {k: v for k, v in dsts.items() if 'MTL0s' in k}
# syns['lpips.alex'] = pickle.load(open('../data/embeds/lpips/bm/lpips.alex.dstt.pkl', 'rb'))
# syns['lpips.vgg'] = pickle.load(open('../data/embeds/lpips/bm/lpips.vgg.dstt.pkl', 'rb'))
# b_syns = {s: syns[s] for s in [agent, 'lpips.alex', 'lpips.vgg']}
print("syn agents:", syns.keys(), len(syns))

if args.h2h_train_embs_path:
    train_path = pathlib.Path(args.h2h_train_embs_path)
    test_path = train_path.with_name(train_path.name.replace('train', 'test'))
    z_train = pickle.load(open(train_path, 'rb'))
    z_test = pickle.load(open(test_path, 'rb'))
    r_dst = euc_dist(z_test, z_train)
    resn_nis = get_NI(r_dst, y_train, y_test)
    resn_nos = get_NI(r_dst, 1-y_train, y_test)
    resn_fos = get_NI(-r_dst, 1-y_train, y_test)
else:
    model, seed = 'MTL1', best[model]
    z_train, z_test = embs[model][seed]['train'], embs[model][seed]['test']
    r_dst = euc_dist(z_test, z_train)
    resn_nis = get_NI(r_dst, y_train, y_test)
    resn_nos = get_NI(r_dst, 1-y_train, y_test)
    resn_fos = get_NI(-r_dst, 1-y_train, y_test)

id_columns = ['agent', 'name', 'model', 'seed']
ts_columns = ['test_clf_acc', 'test_1nn_acc', 'test_triplet_acc']
ds_columns = ['NINO_ds_acc', 'NIFO_ds_acc'] # rNINO_ds_acc
kd = [] # [2, 3]
kd_columns = ['_'.join([str(k), ds]) for ds in ds_columns for k in kd]
# er_columns = ['NINO_ds_err', 'NIFO_ds_err', 'rNINO_ds_err']
ni_columns = ['NI_acc', 'NO_acc', 'FO_acc']

all_columns = id_columns + ts_columns + ds_columns + kd_columns + ni_columns
results = pd.DataFrame(columns=all_columns)
for syn in syns:
    print(f"Evaluating models with agent: {syn}")
    for model in models:
        for seed in seeds:
            evals = {}
            if np.any([col in id_columns for col in all_columns]):
                name = 's'.join([model, str(seed)])
                evals.update({k: v for k, v in zip(id_columns, [syn, name, model, seed])})
            if np.any([col in ts_columns for col in all_columns]):
                test_values = df_wandb[df_wandb['Name'] == name][ts_columns].values[0]
                evals.update({k: v for k, v in zip(ts_columns, test_values)})
            if np.any([col in (ds_columns + kd_columns + ni_columns) for col in all_columns]):
                z_train, z_test = embs[model][seed]['train'], embs[model][seed]['test']
                evals.update(syn_evals(z_train, y_train, z_test, y_test, None, None, None, None, dist=syns[syn]))
                for k in kd:
                    k_evals = syn_evals(z_train, y_train, z_test, y_test, None, None, None, None, dist=syns[syn], k=k)
                    evals.update({'_'.join([str(k), ds]): k_evals[ds] for ds in ds_columns})
                # NI
                nn_mat = np.hstack([np.arange(len(y_test)).reshape(-1, 1), evals['NIs'], resn_nis])
                sames = np.where(nn_mat[:, 1] == nn_mat[:, 2])[0]
                corr = (get_ds_choice(syns[syn], nn_mat) == 0).astype(float)
                corr[sames] = 0.5
                evals['NI_acc'] = corr.mean()
                # NO
                nos = get_NI(euc_dist(z_test, z_train), 1-y_train, y_test)
                nn_mat = np.hstack([np.arange(len(y_test)).reshape(-1, 1), nos, resn_nos])
                sames = np.where(nn_mat[:, 1] == nn_mat[:, 2])[0]
                corr = (get_ds_choice(syns[syn], nn_mat) == 0).astype(float)
                corr[sames] = 0.5
                evals['NO_acc'] = corr.mean()
                # FO
                fos = get_NI(-euc_dist(z_test, z_train), 1-y_train, y_test)
                nn_mat = np.hstack([np.arange(len(y_test)).reshape(-1, 1), fos, resn_fos])
                sames = np.where(nn_mat[:, 1] == nn_mat[:, 2])[0]
                corr = (get_ds_choice(syns[syn], nn_mat) == 0).astype(float)
                corr[sames] = 0.5
                evals['FO_acc'] = corr.mean()

            results.loc[len(results)] = [evals[k] for k in all_columns]

results.to_csv(results_csv, index=False)
print("Saved results at:", results_csv)