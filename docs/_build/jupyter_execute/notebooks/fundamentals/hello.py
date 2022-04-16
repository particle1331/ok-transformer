#!/usr/bin/env python
# coding: utf-8

# # Hello

# In[1]:


import plotly.io as pio
pio.renderers


# In[3]:


import plotly.express as px
fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
fig.show(renderer="svg")


# In[4]:


import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}


# In[8]:


fig = optuna.visualization.plot_optimization_history(study)
fig.show(renderer="svg")


# In[ ]:




