# machine-learning

Collection of notebooks on machine learning theory and engineering. The notebooks should run end-to-end after some initial setup, i.e. setting up the required directories and datasets. 


## Requirements

```
joblib==1.0.1
kaleido==0.2.1
matplotlib==3.4.2
mlxtend==0.19.0
numpy==1.21.1
optuna==2.9.1
pandas==1.3.1
scikit-learn==0.24.2
scipy==1.7.1
tqdm==4.62.0
xgboost==1.4.2
```

For plotly plots to show, we need to load `require.js` in `_config.yml` [[docs](https://github.com/executablebooks/jupyter-book/blob/master/docs/interactive/interactive.ipynb)]: 

```
sphinx:
  config:
    html_js_files:
    - https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js
```


## References 

```{bibliography}
```