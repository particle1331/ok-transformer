#!/usr/bin/env python
# coding: utf-8

# # End-to-End ML Project

# Main steps to an ML project:
# 
# 1. Look at the big picture.
# 2. Get the data.
# 3. Discover and visualize the data to gain insights.
# 4. Prepare the data for Machine Learning algorithms.
# 5. Select a model and train it.
# 6. Fine-tune your model.
# 7. Present your solution.
# 8. Launch, monitor, and maintain your system.
# 

# ## Looking at the big picture
# 
# Your first task is to use California census data to build a model of housing prices in the state. This data includes
# metrics such as the population, median income, and median housing price for each
# block group in California. Block groups are the smallest geographical unit for which
# the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). We will call them “districts” for short.
# Your model should learn from this data and be able to predict the **median housing price** in any district, given all the other metrics.
# 
# The first question to ask your boss is what exactly the business objective is. Building a
# model is probably not the end goal. How does the company expect to use and benefit
# from this model? Knowing the objective is important because it will determine how
# you frame the problem, which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it.
# 
# Your boss answers that your model’s output (a prediction of a district’s median housing price) will be fed to another Machine Learning system, along
# with many other signals. This downstream system will determine whether it is worth investing in a given area or not. Getting this right is critical, as it directly affects revenue.
# 

# ```{figure} ../img/ml_pipeline.png
# ---
# width: 35em
# name: ml_pipeline
# ---
# A Machine Learning pipeline for real estate investments. Situated within is the ML model currently being developed.
# ```
# 

# ## Getting the data

# In[ ]:




