# A Decision Theoretic Approach for Model Interpretability in Bayesian Framework

This repository contains code and data implementing the methods and experiments described in

Afrabandpey, H., Peltola, T., Piironen, J., Vehtari, A., and Kaski, S. "**A Decision-Theoretic Approach for Model Interpretability in Bayesian Framework**." arXiv preprint arXiv:1910.09358 (2019)

The paper is available in https://arxiv.org/abs/1910.09358v2. 

## Overview

### Requirements

 * Python 3.7
    * numpy
    * scipy
    * scikit-learn
    * pandas
    * GPy
    * igraph
    * matplotlib
 * R
    * MASS
    * BART
    * BEST
    * ggplot2

## Basic Usage

The script [Optimization.py](Optimization.py) contains all the required functions to fitt a proxy model (in this case a decision tree) to a complex un-interpretable model (a.k.a. reference model in the paper). The script [main.R](main.R) has three parameters:

```R
% d_ind            index of the data set to be used for the experiment. possible options are 1: body fat,
                   2: baseball players' salary, and 3: auto risk
% ref_effect       if True, the results of Section Section 4.1.2 will be reproduced where the effect of different
                   reference models have been investigated. If False, results of Section 4.1.3 will be reproduced
                   where the proposed approach, i.e., fitting an interpretable proxy model to the complex reference
                   model, is compared with the alternative of fitting a-priori interpretable model to the data.
% num_run          the number of runs over which the results are averaged. results of the paper are produced with
                   num_run = 50.
```
To reproduce results of the illustrative example in Section 2.2.1, run the Python script in folder [illustrative_example](illustrative_example).

## Contact

 * Homayun Afrabandpey, homayun.afrabandpey@nokia.com
 * Tomi Peltola, tomi.peltola@tmpl.fi
 
 Work done in the [Probabilistic Machine Learning research group](https://research.cs.aalto.fi/pml/) at [Aalto University](https://www.aalto.fi/fi).
 
 ## Reference

 * Afrabandpey, H., Peltola, T., Piironen, J., Vehtari, A., and Kaski, S. "**A Decision-Theoretic Approach for Model Interpretability in Bayesian Framework**." arXiv preprint arXiv:1910.09358 (2019) https://arxiv.org/abs/1910.09358v2
