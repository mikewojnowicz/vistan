"""
Goal: check into VISTAN's ability to do ADVI for our stuff
* Same conclusions on BLR as with my ADVI library?
* What happens with cateogrical models?
"""

# reference: https://github.com/abhiagwl/vistan

# import stan 
# pystan = stan 

import vistan 
import matplotlib.pyplot as plt
import numpy as np 
import scipy 
plt.style.use("ggplot")
code = """
data {
    int<lower=0> N;
    int<lower=0, upper=1> x[N];
}
parameters {
    real<lower=0, upper=1> p;
}
model {
    p ~ beta(1,1);
    x ~ bernoulli(p);
}
"""
data = {"N":5, "x":[0,1,0,0,0]}

algo=vistan.recipe("meanfield")
posterior=algo(code,data)
samples=posterior.sample(100000)
plt.hist(samples["p"], 200, density=True, histtype="step", label="meanfield", linewidth=1.5)

points=np.arange(0,1,.01)
plt.plot(points, scipy.stats.beta(2,5).pdf(points), label="True Posterior", linewidth=1.5)
plt.legend()
plt.show()