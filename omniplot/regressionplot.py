
from typing import Union, Optional, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
from collections import defaultdict
import matplotlib.colors
import sys 
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from itertools import combinations

import os
script_dir = os.path.dirname( __file__ )
sys.path.append( script_dir )
from utils import *
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

    method: str
        Method name for regression. Default: ransac
        Available methods: ["ransac", 
                            "robust",
                            "lasso","elastic_net"
                            ]
    figsize: list[int]
        figure size
    show : bool
        Whether or not to show the figure.
    
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
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    plt.subplots_adjust(left=0.15)
    if method=="ransac":
        from sklearn.linear_model import RANSACRegressor
        
        
        
        fit_df=pd.DataFrame()
        fitted_model = RANSACRegressor(random_state=random_state,**ransac_param).fit(_X,Y)
        fit_df["ransac_regression"] = fitted_model.predict(plotline_X)
        coef = fitted_model.estimator_.coef_[0]
        intercept=fitted_model.estimator_.intercept_
        inlier_mask = fitted_model.inlier_mask_
        outlier_mask = ~inlier_mask
        
                                # number of samples
        y_model=fitted_model.predict(_X)

        r2 = _calc_r2(X,Y)
        # mean squared error
        MSE = 1/n * np.sum( (Y - y_model)**2 )
        
        # to plot the adjusted model
        x_line = plotline_X.flatten()
        y_line = fit_df["ransac_regression"]
         
        ci, pi, std_error=_ci_pi(X,Y,plotline_X.flatten(),y_model)
        q=((X-X.mean()).transpose() @ (X-X.mean()))
        sigma=std_error*(q**-1)**(0.5)
        coef_p=stats.t.sf(abs(fitted_model.estimator_.coef_[0]/sigma), df=X.shape[0]-2)
        ############### Ploting

        _draw_ci_pi(ax, ci, pi,x_line, y_line)
        sns.scatterplot(x=X[inlier_mask], y=Y[inlier_mask], color="blue", label="Inliers")
        sns.scatterplot(x=X[outlier_mask], y=Y[outlier_mask], color="red", label="Outliers")
        plt.xlabel(x)
        plt.ylabel(y)
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title("RANSAC regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}".format(
            r2, MSE,coef,intercept,coef_p
            )
        )
        plt.plot(plotline_X.flatten(),fit_df["ransac_regression"])
        
        _save(save, "ransac")
        if len(category)!=0:
            fig, ax=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, hue=category)
            
            plt.xlabel(x)
            plt.ylabel(y)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title("RANSAC regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}".format(
                r2, MSE,coef,intercept,coef_p
                )
            )
            plt.plot(plotline_X.flatten(),fit_df["ransac_regression"])
            _save(save, "ransac_"+category)
    elif method=="robust":
        import statsmodels.api as sm
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

        _draw_ci_pi(ax, ci, pi,x_line, y_line)
        sns.scatterplot(data=df,x=x, y=y, color="blue")
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title("Robust linear regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x , p-values: coefficient {:.2f}, \
        intercept {:.2f}".format(
            r2, MSE,coef,intercept,coef_p,intercept_p
            )
        )
        plt.plot(plotline_X.flatten(),y_line)
        _save(save, "robust")
        if len(category)!=0:
            fig, ax=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, hue=category)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title("Robust linear regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x , p-values: coefficient {:.2f}, \
            intercept {:.2f}".format(
                r2, MSE,coef,intercept,coef_p,intercept_p
                )
            )
            plt.plot(plotline_X.flatten(),y_line)
            _save(save, "robust_"+category)
    elif method=="lasso" or method=="elastic_net" or method=="ols":
        if method=="lasso":
            method="sqrt_lasso"
        import statsmodels.api as sm
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
        print(sigma,coef )
        coef_p=stats.t.sf(abs(coef/sigma), df=X.shape[0]-2)
        MSE = 1/n * np.sum( (Y - y_model)**2 )

        _draw_ci_pi(ax, ci, pi,x_line, y_line)   
        sns.scatterplot(data=df,x=x, y=y, color="blue")
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title("OLS ({}), r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}".format(method,
            r2, MSE,coef,intercept,coef_p
            )
        )
        plt.plot(plotline_X.flatten(),y_line)
        _save(save, method)
        if len(category)!=0:
            fig, ax=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, color="blue",hue=category)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title("OLS ({}), r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}".format(method,
                r2, MSE,coef,intercept,coef_p
                )
            )
            plt.plot(plotline_X.flatten(),y_line)
            _save(save, method+"_"+category)
    return {"axes":ax, "coefficient":coef,"intercept":intercept,"coefficient_pval":coef_p, "r2":r2, "fitted_model":fitted_model}

def regression_single_polynomial(df: pd.DataFrame, 
                      x: str,
                      y: str, 
                      method: str="ransac",
                      category: str="", 
                      figsize: List[int]=[5,5],
                      show=False, ransac_param={"max_trials":1000},
                      robust_param={},
                      xunit: str="",
                      yunit: str="",
                      title: str="",
                      random_state: int=42,ax: Optional[plt.Axes]=None,
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

    method: str
        Method name for regression. Default: ransac
        Available methods: ["ransac", 
                            "robust",
                            "lasso","elastic_net"
                            ]
    figsize: list[int]
        figure size
    show : bool
        Whether or not to show the figure.
    
    Returns
    -------
    dict: dict {"axes":ax, "coefficient":coef,"intercept":intercept,"coefficient_pval":coef_p, "r2":r2, "fitted_model":fitted_model}
    
        fitted_model:
            this can be used to predict y. e.g.,) y_predict=fitted_model.predict(_X)

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
    pass


def regression_multivariable():
    pass

def regression_multivariable_nonlinear():
    pass

def regression_glm():
    pass


