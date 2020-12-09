# Colab setup ------------------
import os, sys, subprocess
if "google.colab" in sys.modules:
    cmd = "pip install --upgrade iqplot colorcet watermark"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    data_path = "../data/"
# ------------------------------
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
import bebi103

import iqplot
import mle
import reps

import colorcet

import bokeh.io 
from bokeh.layouts import column
import holoviews as hv
from bokeh.plotting import figure, output_file, save

def _load_data(): 
    fname = os.path.join(data_path, "gardner_mt_catastrophe_only_tubulin.csv")
    df = pd.read_csv(fname, comment="#")
    return df
    
def clean_data(df): 
    df = pd.melt(df).dropna().reset_index(drop=True)
    df = df.rename(columns={"variable": "concentration", "value":"time to catastrophe (s)"})
    splitter = lambda s: int(s.split(" ")[0])
    df['ints'] = [splitter(x) for x in df["concentration"]]
    df = df.sort_values("ints")
    return df
    
def concentration(conc): 
    c = "{} uM".format(conc)
    return df.loc[df['concentration'] == c].values[:,1].astype('float64')
        
def cat_conc_ecdf(df): 
    p = iqplot.ecdf(
        data = df,
        q = 'time to catastrophe (s)',
        cats = ['concentration'],
        style = 'staircase',
        conf_int = True,
        ptiles = [2.5, 97.5],
        show_legend = True
    )     
    return p
    
def cat_conc_stripbox(df): 
    unlabeled1 = iqplot.stripbox(
            data = df, 
            q  = 'time to catastrophe (s)',
            cats = ['concentration'],
            show_legend = True,
            jitter=0.3,
    )
    unlabeled2 = iqplot.stripbox(
        data = df, 
        q  = 'time to catastrophe (s)',
        cats = ['concentration'],
        show_legend = True
    )
    return unlabeled1, unlabeled2
    
def mle_gamma(conc): 
    return mle.mle_iid_gamma(concentration(conc))
    
def mle_bespoke(conc):
    return mle.mle_iid_bespoke(concentration(conc))
    
def _clean_mle_data(conc): 
    gmle = mle_gamma(conc)
    bmle = mle_bespoke(conc)
    glog = mle.log_like_iid_gamma(gmle, concentration(conc))
    blog = mle.log_like_iid_bespoke(bmle, concentration(conc))
    mle_df = pd.DataFrame(index=['beta1', 'dbeta', 'beta', 'alpha', 'log_like_gamma', 
                             'log_like_bespoke', 'AIC_gamma','AIC_bespoke'],
                             data=np.array([bmle[0], bmle[1], gmle[0], gmle[1], glog, blog, 0, 0]))
    mle_df = mle_df.T 
    mle_df['AIC_gamma'] = -2 * (mle_df['log_like_gamma'] - 2)
    mle_df['AIC_bespoke'] = -2 * (mle_df['log_like_bespoke'] - 2)
    mle_df['AIC_gamma_sub'] = mle_df['AIC_gamma'] - mle_df["AIC_bespoke"]
    mle_df['AIC_bespoke_sub'] = mle_df['AIC_bespoke'] - mle_df['AIC_bespoke']
    numerator = np.exp(-(mle_df['AIC_bespoke_sub'].values)/2)
    denominator = numerator + np.exp(-(mle_df['AIC_gamma_sub'].values )/2)
    mle_df['weight_bespoke'] = numerator/denominator
    return mle_df.T 

def reps_and_conf(conc): 
    c = concentration(conc)
    bs_reps_parametric = reps.draw_parametric_bs_reps_mle(
        mle.mle_iid_gamma,
        reps.sp_gamma,
        c,
        args=(),
        size=1000,
        progress_bar=True,
    )
    conf_int = np.percentile(bs_reps_parametric, [2.5, 97.5], axis=0)
    return bs_reps_parametric, conf_int
    
def _clean_summaries(concentrations): 
    summaries_alpha = []
    summaries_beta = []
    treps = []
    for c in concentrations: 
        tmle = mle.mle_iid_gamma(concentration(c))
        tbs_reps, tconf_int = reps_and_conf(c)
        l = "{}uM".format(c)
        summaries_beta.append(dict(label=l, 
                         estimate = tmle[0]
                         ,conf_int= [tconf_int[0][0], tconf_int[1][0]]))
        summaries_alpha.append(dict(label =l, 
                                 estimate = tmle[1]
                                 ,conf_int = [tconf_int[0][1], tconf_int[1][1]]))
        treps.append(tbs_reps)
    return treps, summaries_alpha, summaries_beta
    
def show_beta_alpha(concentrations): 
    reps, summaries_alpha, summaries_beta = _clean_summaries(concentrations)
    p1 = bebi103.viz.confints(summaries_beta, x_axis_label='Beta', y_axis_label='Concentration')
    p2 = bebi103.viz.confints(summaries_alpha, x_axis_label='Alpha', y_axis_label='Concentration')
    return p1, p2
        
    

if __name__ == "__main__":
    df = clean_data(_load_data())
    a = cat_conc_ecdf(df)
    b, c = cat_conc_stripbox(df)
    print(clean_mle_data(12))
    d, e = show_beta_alpha([7, 9, 10, 12, 14])
    final = column(a, b, c, d, e)
    output_file("final.html")
    save(final)
    