
from typing import Union, Optional, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
from collections import defaultdict
import sys 
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from itertools import combinations
import statsmodels.api as sm
from sklearn.linear_model import RANSACRegressor

import os
script_dir = os.path.dirname( __file__ )
sys.path.append( script_dir )
from omniplot.utils import _calc_r2, _ci_pi, _draw_ci_pi, _save
import scipy.stats as stats

colormap_list=["nipy_spectral", "terrain","tab20b","gist_rainbow","CMRmap","coolwarm","gnuplot","gist_stern","brg","rainbow"]
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme()

def regression_single(df: pd.DataFrame, 
                      x: str,
                      y: str, 
                      method: str="ransac",
                      category: str="", 
                      figsize: List[int]=[5,5],
                      show=False, ransac_param={"max_trials":1000},
                      robust_param: dict={},
                      xunit: str="",
                      yunit: str="",
                      title: str="",
                      random_state: int=42,
                      ax: Optional[plt.Axes]=None,
                      save: str="") -> Dict:
    """
    Drawing a scatter plot with a single variable linear regression.  
    
    Parameters
    ----------
    df : pandas DataFrame
    
    x: str
        the column name of x axis. 
    y: str
        the column name of y axis. 

    method: str, optional (default: ransac) ["ransac", "robust","lasso","elastic_net"]
        Method name for regression. Default: ransac
    category: str, optional
    figsize: list[int]
        figure size
    show : bool
        Whether or not to show the figure.
    robust_param: dict, optional
        Hyper parameters for robust regression. Please see https://www.statsmodels.org/dev/generated/statsmodels.robust.robust_linear_model.RLM.html
    xunit: str, optional
        X axis unit
    yunit: str, optional
        Y axis unit
    title: str, optional
        Figure title

    random_state: int, optional (default=42)
        random state for RANSAC regression
    ax: plt.Axes, optional,
        pyplot axis
    save: str, optional
        The prefix of file names to save.
    Returns
    -------
    dict: dict {"axes":ax, "coefficient":coef,"intercept":intercept,"coefficient_pval":coef_p, "r2":r2, "fitted_model":fitted_model}
    
        fitted_model:
            this can be used like: y_predict=fitted_model.predict(_X)
    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """ 
    
    
    Y=df[y]
    _X=np.array(df[x]).reshape([-1,1])
    X=np.array(df[x])
    plotline_X = np.arange(X.min(), X.max()).reshape(-1, 1)
    n = X.shape[0]
    plt.rcParams.update({'font.size': 14})
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title)
    else:
        fig=None
        # plt.title(title)
    plt.subplots_adjust(left=0.15)
    if method=="ransac":
        _title="RANSAC regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}"
        fitted_model, coef, coef_p, intercept, r2, x_line, y_line, ci, pi,std_error, MSE, inlier_mask, outlier_mask=_ransac(X,Y,plotline_X,random_state, ransac_param)
        
        ############### Ploting

        _draw_ci_pi(ax, ci, pi,x_line, y_line)
        sns.scatterplot(x=X[inlier_mask], y=Y[inlier_mask], color="blue", label="Inliers")
        sns.scatterplot(x=X[outlier_mask], y=Y[outlier_mask], color="red", label="Outliers")
        plt.xlabel(x)
        plt.ylabel(y)
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title(_title.format(
            r2, MSE,coef,intercept,coef_p
            )
        )
        plt.plot(plotline_X.flatten(),y_line)
        
        _save(save, method)
        if len(category)!=0:
            fig, ax1=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax1, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, hue=category, ax=ax1)
            
            plt.xlabel(x)
            plt.ylabel(y)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title(_title.format(
                r2, MSE,coef,intercept,coef_p
                )
            )
            plt.plot(plotline_X.flatten(),y_line)
            _save(save, method+"_"+category)
    elif method=="robust":
        _title="Robust linear regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x , p-values: coefficient {:.2f}, \
        intercept {:.2f}"
        fitted_model, summary, coef, coef_p, intercept, intercept_p, r2, x_line, y_line, ci, pi,std_error, MSE=_robust_regression(X, Y, plotline_X, robust_param)

        _draw_ci_pi(ax, ci, pi,x_line, y_line)
        sns.scatterplot(data=df,x=x, y=y, color="blue")
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title(_title.format(
            r2, MSE,coef,intercept,coef_p,intercept_p
            )
        )
        plt.plot(plotline_X.flatten(),y_line)
        _save(save, method)
        if len(category)!=0:
            fig, ax1=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax1, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, hue=category, ax=ax1)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title(_title.format(
                r2, MSE,coef,intercept,coef_p,intercept_p
                )
            )
            plt.plot(plotline_X.flatten(),y_line)
            _save(save, method+"_"+category)
    elif method=="lasso" or method=="elastic_net" or method=="ols":
        _title="OLS ({}), r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}"

        fitted_model, coef, coef_p, intercept, r2, x_line, y_line, ci, pi,std_error, MSE=_ols(X, Y, plotline_X, method)


        _draw_ci_pi(ax, ci, pi,x_line, y_line)   
        sns.scatterplot(data=df,x=x, y=y, color="blue")
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title(_title.format(method,
            r2, MSE,coef,intercept,coef_p
            )
        )
        plt.plot(plotline_X.flatten(),y_line)
        _save(save, method)
        if len(category)!=0:
            fig, ax1=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax1, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, color="blue",hue=category)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title(_title.format(method,
                r2, MSE,coef,intercept,coef_p
                )
            )
            plt.plot(plotline_X.flatten(),y_line)
            _save(save, method+"_"+category)
    if yunit!="":
        ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
        ax1.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
    if xunit!="":
        ax.text(1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left",va="top")
        ax1.text(1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left",va="top")
    
    return {"axes":ax, "coefficient":coef,"intercept":intercept,"coefficient_pval":coef_p, "r2":r2, "fitted_model":fitted_model}


def _robust_regression(X, Y, plotline_X, robust_param):
    n = X.shape[0]
    rlm_model = sm.RLM(Y, sm.add_constant(X),
    M=sm.robust.norms.HuberT(),**robust_param)
    fitted_model = rlm_model.fit()
    summary=fitted_model.summary()
    coef=fitted_model.params[1]
    intercept=fitted_model.params[0]
    intercept_p=fitted_model.pvalues[0]
    coef_p=fitted_model.pvalues[1]
    y_model=fitted_model.predict(sm.add_constant(X))
    r2 = _calc_r2(X,Y)
    x_line = plotline_X.flatten()
    y_line = fitted_model.predict(sm.add_constant(x_line))
    
    ci, pi,std_error=_ci_pi(X,Y,plotline_X.flatten(),y_model)
    MSE = 1/n * np.sum( (Y - y_model)**2 )
    return fitted_model, summary, coef, coef_p, intercept, intercept_p, r2, x_line, y_line, ci, pi,std_error, MSE

def _ols(X, Y, plotline_X, method):
    n = X.shape[0]
    if method=="lasso":
        method="sqrt_lasso"
    rlm_model = sm.OLS(Y, sm.add_constant(X))
    if method=="ols":
        fitted_model = rlm_model.fit()
    else:
        fitted_model = rlm_model.fit_regularized(method)
    coef=fitted_model.params[1]
    intercept=fitted_model.params[0]
    y_model=fitted_model.predict(sm.add_constant(X))
    r2 = _calc_r2(X,Y)
    x_line = plotline_X.flatten()
    y_line = fitted_model.predict(sm.add_constant(x_line))
    ci, pi, std_error=_ci_pi(X,Y,plotline_X.flatten(),y_model)
    q=((X-X.mean()).transpose() @ (X-X.mean()))
    sigma=std_error*(q**-1)**(0.5)
    # print(sigma,coef )
    coef_p=stats.t.sf(abs(coef/sigma), df=X.shape[0]-2)
    MSE = 1/n * np.sum( (Y - y_model)**2 )
    return fitted_model, coef, coef_p, intercept, r2, x_line, y_line, ci, pi,std_error, MSE

def _ransac(X,Y,plotline_X,random_state, ransac_param):
    n = X.shape[0]
    _X=np.array(X).reshape([-1,1])
    fitted_model = RANSACRegressor(random_state=random_state,**ransac_param).fit(_X,Y)
    y_line= fitted_model.predict(plotline_X)
    coef = fitted_model.estimator_.coef_[0]
    intercept=fitted_model.estimator_.intercept_
    inlier_mask = fitted_model.inlier_mask_
    outlier_mask =np.invert(inlier_mask)
    
                            # number of samples
    y_model=fitted_model.predict(_X)

    r2 = _calc_r2(X,Y)
    # mean squared error
    MSE = 1/n * np.sum( (Y - y_model)**2 )
    
    # to plot the adjusted model
    x_line = plotline_X.flatten()
        
    ci, pi, std_error=_ci_pi(X,Y,plotline_X.flatten(),y_model)
    q=((X-X.mean()).transpose() @ (X-X.mean()))
    sigma=std_error*(q**-1)**(0.5)
    coef_p=stats.t.sf(abs(fitted_model.estimator_.coef_[0]/sigma), df=X.shape[0]-2)

    return fitted_model, coef, coef_p, intercept, r2, x_line, y_line, ci, pi,std_error, MSE, inlier_mask, outlier_mask