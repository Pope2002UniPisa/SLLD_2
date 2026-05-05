# Statistical Learning and Large Data Project

## Overview

This repository contains the project developed for the course **Statistical Learning and Large Data** at the **Scuola Superiore Sant’Anna di Pisa**, under the supervision of **Prof.ssa Francesca Chiaromonte**.

The aim of the project is to apply statistical learning techniques to a high-dimensional dataset, combining exploratory analysis, dimensionality reduction, supervised classification, feature selection and regularized modelling. The workflow is designed to move progressively from data cleaning and preliminary exploration to more advanced predictive models, with particular attention to interpretability, multicollinearity and model generalization.

The project follows a structured pipeline composed of four main stages:

1. data preprocessing;
2. unsupervised analysis;
3. supervised statistical learning;
4. model comparison and possible methodological extensions.

---

## 1. Data Preprocessing

The first step of the project consists of preparing the dataset for statistical analysis. Since high-dimensional data often contain missing values, redundant variables and features measured on different scales, preprocessing is a crucial phase.

The preprocessing stage includes:

- removing rows or columns with an excessive amount of missing values;
- deciding whether and how to impute the remaining missing entries;
- inspecting the marginal distribution of each retained variable;
- identifying strongly skewed variables;
- applying suitable transformations, such as logarithmic transformations, when needed;
- standardizing the variables before applying methods that depend on scale, such as PCA, clustering and regularized models.

The motivation behind this phase is to obtain a clean, coherent and numerically stable dataset. A careful preprocessing strategy reduces the risk of fitting models on noisy, distorted or incomparable variables, and provides a more reliable basis for the following analyses.

---

## 2. Unsupervised Analysis

After preprocessing, the project proceeds with an unsupervised exploration of the data structure.

The first objective is to understand whether the observations naturally organize themselves into meaningful groups, independently of the known class labels. This is done through dimensionality reduction and clustering techniques.

The unsupervised analysis includes:

- applying **Principal Component Analysis** to represent the data in a lower-dimensional space;
- studying the amount of variance explained by the principal components;
- visualizing the observations along the first principal components;
- applying clustering methods to identify possible natural groups;
- comparing the obtained clusters with the available class labels.

This phase is exploratory. Its purpose is not primarily predictive, but diagnostic: it helps assess whether the data contain an intrinsic structure and whether such structure is coherent with the labels used in the supervised task.

If the clusters show a meaningful relationship with the known labels, this provides evidence that the feature space captures relevant information. If the clusters do not align well with the labels, this may indicate overlap between classes, weak separation, noisy features or the need for more sophisticated modelling strategies.

---

## 3. Supervised Analysis

The core of the project is a supervised multi-class classification problem. The goal is to predict one of three given labels using the available features.

The initial modelling framework is based on a **generalized linear model with multinomial link**, suitable for multi-class classification problems.

The supervised analysis is developed progressively through several steps.

---

### 3.1 Baseline Model

A first simple model is fitted as a baseline. This model serves as a reference point for the entire analysis.

The baseline allows us to answer a fundamental question:

> How well can the target classes be predicted before applying more advanced feature selection or regularization strategies?

This step is important because later models should not only be more complex, but also demonstrably better in terms of predictive performance, stability or interpretability.

---

### 3.2 Multicollinearity Assessment

High-dimensional datasets often contain strongly correlated predictors. This can create instability in coefficient estimates, make interpretation difficult and reduce the reliability of standard linear models.

To address this issue, the project includes an explicit analysis of multicollinearity through:

- correlation matrices;
- Variance Inflation Factors;
- partial \(R^2\) measures;
- dendrograms of the variables.

The purpose is to identify groups of highly redundant features and decide whether some of them should be removed before fitting the final models.

This step has both statistical and practical value. Statistically, it improves model stability. Practically, it helps reduce the dimensionality of the problem and makes the final model easier to interpret.

---

### 3.3 Screening of Variables

When the number of predictors is large compared to the number of observations, a preliminary screening step can be useful.

The project therefore includes a marginal utility-based screening procedure, whose goal is to reduce the number of terms before applying more computationally demanding models.

The idea is to retain only the most informative variables according to their marginal relationship with the target. This can reduce the feature space to a more manageable size, for example around 500 variables.

This step is not intended to produce the final model, but to create a better starting point for subsequent regularized methods.

---

### 3.4 Regularized Models

After the screening phase, the project applies regularized classification models.

The main regularization strategies considered are:

- **LASSO**, which promotes sparsity by forcing many coefficients to zero;
- **Elastic Net**, which combines L1 and L2 regularization.

LASSO is useful for feature selection because it identifies a smaller subset of relevant predictors. However, in the presence of strongly correlated variables, it may select only one variable from a group of correlated predictors in an unstable way.

Elastic Net addresses this limitation by combining sparsity with a ridge-like penalty, making the model more robust when groups of correlated variables are present.

The motivation for using these methods is twofold:

1. improve prediction in a high-dimensional setting;
2. obtain a more interpretable model by reducing the number of active predictors.

---

### 3.5 Sparsity and Feature Interpretation

Once regularized models are fitted, the project analyzes their sparsity structure.

This includes:

- counting how many coefficients are set to zero;
- identifying the variables retained by the model;
- comparing selected features across different regularization strategies;
- assessing whether the selected variables have a meaningful interpretation.

This phase connects predictive modelling with statistical interpretation. The objective is not only to obtain good classification accuracy, but also to understand which variables contribute most to the classification task.

---

### 3.6 Final Classifier

A final classifier is then trained using the selected subset of variables.

The final model is evaluated through standard classification metrics, such as:

- accuracy;
- confusion matrix;
- class-specific performance;
- possible macro-averaged and weighted metrics.

The goal is to assess whether the final modelling strategy improves generalization, reduces noise and provides a stable predictive framework.

---

## 4. Possible Extensions

The project also considers several possible extensions.

One possibility is to repeat the supervised analysis using the cluster labels obtained from the unsupervised phase. This would allow us to test whether the data-driven groups identified by clustering can themselves be predicted from the feature space.

Another possible direction is to move beyond generalized linear models and test alternative classifiers, such as:

- Support Vector Machines;
- k-Nearest Neighbors;
- Random Forests;
- small neural networks.

These models may capture nonlinear relationships that are not well represented by multinomial linear models. However, they may also reduce interpretability. For this reason, they are treated as possible extensions rather than as the main methodological focus.

---

## Expected Outcome

The expected outcome of the project is a complete statistical learning pipeline for high-dimensional classification.

The final analysis should provide:

- a cleaned and preprocessed dataset;
- an exploratory representation of the data structure;
- an assessment of natural clusters and their relationship with the known labels;
- a baseline supervised model;
- a study of multicollinearity;
- a reduced set of informative variables;
- regularized models for feature selection and prediction;
- a final classifier evaluated through appropriate metrics;
- a comparison between supervised and unsupervised evidence.

Overall, the project aims to show how statistical learning methods can be combined in a coherent workflow, moving from exploratory data analysis to interpretable and regularized predictive modelling.
