# 𝗜𝗻𝗲𝗳𝗳𝗶𝗰𝗶𝗲𝗻𝘁 𝗡𝗲𝘁𝘄𝗼𝗿𝗸𝘀

<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/particle1331/steepest-ascent" data-color-scheme="no-preference: dark; light: light; dark: dark;" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star particle1331/steepest-ascent on GitHub">Star</a>
<!-- Place this tag in your head or just before your close body tag. -->
<script async defer src="https://buttons.github.io/buttons.js"></script>

This repo contains a collection of Jupyter notebooks that run end-to-end after some minimal initial setup (e.g. setting up the directory structure and downloading the required datasets). The name of this collection is inspired by the following study:

```
Brackbill D, Centola D (2020) Impact of network structure on collective 
learning: An experimental study in a data science competition. PLoS ONE 
15(9): e0237978. https://doi.org/10.1371/journal.pone.0237978
```

Below is one of the interesting results of this study:

```{figure} img/pone.0237978.g004.png
---
width: 40em
name: study
---
**Time series showing the tradeoff between diffusion and diversity.**
The dynamics of solution discovery in groups with efficient (blue dashed) and inefficient (orange solid) networks shows that inefficient groups performed better in terms of solution diversity (A and B) and solution quality (C and D). All panels plot the average values for each experimental condition over all eight trials. In groups with efficient networks, good solutions rapidly spread to other group members, whereas diffusion was slower in inefficient groups (A). Diffusion is measured by the fraction of individuals who adopted the best available solution in the group over time. Due to this slower rate of diffusion, groups with inefficient networks discovered more distinct solutions (B), and the quality of the best solutions in these groups was much higher, both in terms of the value of the best solution found (C) and the fraction of the population that adopted a solution that was better than the best available solution from the other network (D). (**Fig. 4** from the [[Brackbill & Centola (2020)]](https://doi.org/10.1371/journal.pone.0237978)  study.)
```



## Requirements

```
# Python 3.8.12
fastapi==0.70.1
httpie==2.6.0
joblib==1.0.1
jupyter-book==0.12.1
lightgbm==3.3.1
matplotlib==3.5.1
mlxtend==0.19.0
numpy==1.19.5
optuna==2.10.0
pandas==1.3.5
scikit-learn==1.0.1
scipy==1.7.3
statsmodels==0.13.2
tensorflow-datasets==4.4.0
tensorflow==2.7.0
torch==1.10.0
torchvision==0.2.2
tqdm==4.62.3
uvicorn==0.16.0
xgboost==1.5.1
```


## References 

```{bibliography}
```