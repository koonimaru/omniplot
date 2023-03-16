import numpy as np
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf

def _nb_regression(x, y):
    _x=np.stack([np.ones(x.shape[0]),x],axis=-1)
    poisson_training_results = sm.GLM(y, _x, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())
    pmu=poisson_training_results.mu
    #print(pmu)
    _y=((y-pmu)**2-pmu)/pmu
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
    train={"AUX_OLS_DEP":_y, "BB_LAMBDA":pmu}
    
    aux_olsr_results = smf.ols(ols_expr, train).fit()
    print(aux_olsr_results.params[0])
    nb2_training_results = sm.GLM(y, _x,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
    print(nb2_training_results.summary())
    print(nb2_training_results.params)
    return {"dispersion": aux_olsr_results.params[0], "beta": nb2_training_results.params}

def _main():
    import matplotlib.pyplot as plt
    import sys
    beta_0 =2
    beta_1 = 4
    
    N = 2000
    x = np.random.randint(0, 100, N)
    
    true_mu = beta_0 + beta_1 * x
    true_r = 0.5
    p =1- true_mu / (float(true_r) + true_mu)
    
    y = np.random.negative_binomial(n = true_r, p = p, size = N)
    _x=np.arange(100)
    
    
    
    res=_nb_regression(x, y)
    print(res)
    _y=res["beta"][0] + res["beta"][1] * _x
    sde=(_y+res["dispersion"]*_y**2)**0.5
    plt.plot(_x, _y)
    plt.fill_between(_x, _y+sde,np.where(_y-sde<0, 0,_y-sde), color="skyblue")
    plt.scatter(x, y, color='gray', s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
if __name__ == '__main__':
    _main()