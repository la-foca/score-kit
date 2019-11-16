<b>score-kit</b>: package for scorecards development with banking-specific metrics and modules.
 
<b>Package is under development and far from completion.</b>
 
<b>What is it?</b>

score-kit is a Python package providing useful methods for data analysis, visualization, transformation and modeling, that can be used in banking-specific tasks. It is mainly used for feature binning and scorecards building (logistic regression models on WoE-transformed data), but there are also other features that can help you approximate income, calibrate complex models using several scores from different sources, split your data depending on feature fullness, etc.

<b>Main features</b>

- Visualization (factorplots, pairplots, gini in time, event rate in time/by values, score/gini distribution, roc curves etc.)
- Data analysis (missing values, feature fullness analysis)
- Clustering (using hdbscan, hierarchy and k-means based on feature fullness)
- Feature engineering (categorical feature encoding, missing values replacement, crosses by decision trees on several features)
- One-factor analysis (binning for categorical and interval features using decision trees, gini checker, business logic checker, WoE-stability checker, small bins processing, autofit)
- Stability analysis (PSI, ER/gini in time)
- Multifactor analysis (VIF, pairwise correlation)
- Modeling (logistic regression, feature selection, ordinal regression, model calibration)
- Reporting (report generation, scorecard forming, SAS-code export)

<b>Authors</b>

Anna Goreva & Yudochev Dmitry
