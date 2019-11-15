# -*- coding: utf-8 -*-

from ..data import Data, DataSamples
from ..cross import DecisionTree, Crosses
#from ..woe import WOE
import pandas as pd
#import math as m
import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, r2_score
from scipy.stats import chi2, chisquare, ks_2samp, ttest_ind
#import statsmodels.formula.api as sm
import warnings
from abc import ABCMeta, abstractmethod
#from sklearn.feature_selection import GenericUnivariateSelect, f_classif
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
import ast
import os
import xlsxwriter
from PIL import Image
import datetime
from dateutil.relativedelta import *
from scipy.optimize import minimize

import copy
import itertools
import calendar

warnings.simplefilter('ignore')

plt.rc('font', family='Verdana')
plt.style.use('seaborn-darkgrid')
pd.set_option('display.precision', 3)


class ScoringModel(metaclass = ABCMeta):
    '''
    Base class for binary scoring models
    '''

    @abstractmethod
    def __init__(self, model):
        self.model = model
        self.features = []



    @abstractmethod
    def fit(self, data):
        pass



    def predict(self, data, woe_transform=None):
        '''
        Predicts probability of target = 1

        Parameters
        -----------
        data: Data to use for prediction, Data type
        woe_transform: a WOE object to perform WoE-transformation before using model

        Returns
        -----------
        matrix with shape [(sample size) X (number of classes)]
        '''

        if woe_transform is not None:
            data=woe_transform.transform(data, keep_essential=True, original_values=True, calc_gini=False)

        if self.features == []:
            self.features = data.features
        return self.model.predict_proba(data.dataframe[self.features])



    def roc_curve(self, data, woe_transform=None, figsize=(10,7), filename = 'roc_curve', verbose = True):
        '''
        Displays ROC-curve and Gini coefficient for the model

        Parameters
        -----------
        data: a Data or DataSamples object
        woe_transform: a WOE object to perform WoE-transformation before using model
        figsize: a tuple for graph size
        filename: name of the picture with roc_curve
        verbose: show/not show roc_curve in output

        Returns
        ----------
        a list of gini values per input sample
        '''

        if woe_transform is not None:
            data=woe_transform.transform(data, keep_essential=True, original_values=True, calc_gini=False)

        tpr={}
        fpr={}
        roc_auc={}
        if type(data)==DataSamples:
            samples=[data.train, data.validate, data.test]
            sample_names=['Train', 'Validate', 'Test']
            for si in range(len(samples)):
                if samples[si] is not None:
                    preds = self.predict(samples[si])[:,1]
                    fpr[samples[si].name], tpr[samples[si].name], _ = roc_curve(samples[si].dataframe[samples[si].target], preds)
                    roc_auc[samples[si].name] = auc(fpr[samples[si].name], tpr[samples[si].name])
                else:
                    fpr[sample_names[si]]=None
                    tpr[sample_names[si]]=None
                    roc_auc[sample_names[si]]=None
        else:
            preds = self.predict(data)[:,1]
            fpr['Data'], tpr['Data'], _ = roc_curve(data.dataframe[data.target], preds)
            roc_auc['Data'] = auc(fpr['Data'], tpr['Data'])

        if verbose or (filename is not None):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            # Plot tpr vs 1-fpr
            for sample in roc_auc:
                if roc_auc[sample] is not None:
                    ax.plot(fpr[sample], tpr[sample], label=sample+' (AUC = %f)' % roc_auc[sample])
            ax.plot(tpr[list(tpr)[0]],tpr[list(tpr)[0]])

            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
        if filename is not None:
            plt.savefig(filename + ".png", dpi=100, bbox_inches='tight')
        if verbose:
            plt.show()
        if verbose or (filename is not None):
            plt.close()

        ginis=[]
        for sample in roc_auc:
            if roc_auc[sample] is not None:
                gini = round((roc_auc[sample]*2 - 1)*100, 2)
                ginis.append(gini)
                if verbose:
                    print ('Gini '+sample, gini)
        return ginis
#---------------------------------------------------------------




class DecisionTreeModel(ScoringModel):
    '''
    Decision tree classifier
    '''
    def __init__(self, **args):
        self.model = DecisionTreeClassifier(**args)
        self.features = []



    def fit(self, data):
        if data.weights != None:
            self.model.fit(data.dataframe[data.features], data.dataframe[data.target], sample_weight = np.array(data.dataframe[data.weights]))
        else:
            self.model.fit(data.dataframe[data.features], data.dataframe[data.target])
#---------------------------------------------------------------




class LogisticRegressionModel(ScoringModel):
    '''
    Logistic Regression for scoring.
    Contains LogisticRegressionClassifier, its coefficients and intercept, scores and scoring card.
    An object of this class can selected features, fit, edit coefficients, predict probabilities, calculate scores and transform a scorecard to SAS-code.
    '''

    def __init__(self, **args):
        self.model = LogisticRegression(**args)

        self.regularization = self.model.get_params()['penalty']
        self.regularization_value = self.model.get_params()['C']
        self.solver = self.model.get_params()['solver']


        # Checks the type of optimizator as it is important for models on weighted samples
        if self.model.solver != 'sag' and self.model.solver != 'newton-cg' and self.model.solver != 'lbfgs':
            print ('Warning: this model does not support sample weights! For weighted scoring please use solver sag, newton-cg or lbfgs')
        self.coefs = {}
        self.features = []
        self.scorecard = pd.DataFrame()
        self.selected = []



    #added 23.08.2018 by Yudochev Dmitry
    def inf_criterion(self, data, model=None, features=None, criterion='AIC', woe_transform=None):
        '''
        Calculation of information criterion (AIC/BIC) for given model on given data

        Parameters
        -----------
        data: data for calculation
        model: model with coefficients, that will be used to calculate information criterion
        features: features to be used for information criterion calculation (in case model
            was not fitted using its own selected features - model.selected)
        criterion: type of information criterion to calculate
        woe_transform: a woe object with binning information to perform WoE-transformation

        Returns
        -----------
        value of information criterion

        '''
        if features is None:
            features_initial=self.selected.copy()
        else:
            features_initial=features.copy()

        if model is None:
            model_to_check=self.model
        else:
            model_to_check=model

        if woe_transform is not None:
            data=woe_transform.transform(data, keep_essential=True, original_values=True, calc_gini=False)

        features_kept=[]
        weights_crit=[model_to_check.intercept_[0]]
        for i in range(len(features_initial)):
            if model_to_check.coef_[0][i]!=0:
                features_kept.append(features_initial[i])
                weights_crit.append(model_to_check.coef_[0][i])

        intercept_crit = np.ones((data.dataframe.shape[0], 1))
        features_crit = np.hstack((intercept_crit, data.dataframe[features_kept]))
        scores_crit = np.dot(features_crit, weights_crit)
        if data.weights is not None:
            ll = np.sum(data.dataframe[data.weights]*(data.dataframe[data.target]*scores_crit - np.log(np.exp(scores_crit) + 1)))
        else:
            ll = np.sum(data.dataframe[data.target]*scores_crit - np.log(np.exp(scores_crit) + 1))
        if criterion in ['aic', 'AIC']:
            return 2*len(weights_crit)-2*ll
        elif criterion in ['bic', 'BIC', 'sic', 'SIC', 'sbic', 'SBIC']:
            if data.weights is not None:
                return len(weights_crit)*np.log(data.dataframe[data.weights].sum())-2*ll
            else:
                return len(weights_crit)*np.log(data.dataframe.shape[0])-2*ll



    #added 23.08.2018 by Yudochev Dmitry
    def wald_test(self, data, model=None, features=None, woe_transform=None, out=None, sep=';'):
        '''
        Calculation of Standard Errors (sqrt from diagonal of covariance matrix),
        Wald Chi-Square (coefficient divided by SE and squared) and p-values for coefficicents
        of given model on given data

        Parameters
        -----------
        data: data for statistics calculation
        model: model with coefficients, that will be used to calculate statistics
        features: features to be used for statistics calculation (in case model
            was not fitted using its own selected features - model.selected)
        woe_transform: a woe object with binning information to perform WoE-transformation
        out: a path for csv/xlsx output file to export
        sep: the separator to be used in case of csv export

        Returns
       -----------
        a dataframe with standard errors, wald statistics and p-values for feature coefficients

        '''
        if features is not None:
            features_initial=features.copy()
        else:
            features_initial=self.features.copy()


        if model is None:
            model_to_check=self.model
        else:
            model_to_check=model


        features_to_check=[]
        coefs_list=[model_to_check.intercept_[0]]
        for i in range(len(features_initial)):
            if model_to_check.coef_[0][i]!=0:
                features_to_check.append(features_initial[i])
                coefs_list.append(model_to_check.coef_[0][i])

        if woe_transform is not None:
            data=woe_transform.transform(data, keep_essential=True, original_values=True, calc_gini=False)

        # Calculate matrix of predicted class probabilities.
        # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
        predProbs = np.matrix(model_to_check.predict_proba(data.dataframe[features_initial]))

        # Design matrix -- add column of 1's at the beginning of your X_train matrix
        X_design = np.hstack((np.ones(shape = (data.dataframe[features_to_check].shape[0],1)),
                              data.dataframe[features_to_check]))

        # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
        #not enough memory for big df
        #V = np.matrix(np.zeros(shape = (X_design.shape[0], X_design.shape[0])))
        #np.fill_diagonal(V, np.multiply(predProbs[:,0], predProbs[:,1]).A1)
        if data.weights is not None:
            V=np.multiply(np.matrix(data.dataframe[data.weights]).T, np.multiply(predProbs[:,0], predProbs[:,1])).A1
        else:
            V=np.multiply(predProbs[:,0], predProbs[:,1]).A1
        # Covariance matrix
        covLogit = np.linalg.inv(np.matrix(X_design.T * V) * X_design)

        # Output
        bse=np.sqrt(np.diag(covLogit))
        wald=(coefs_list / bse) ** 2
        pvalue=chi2.sf(wald, 1)

        features_test=pd.DataFrame({'feature':['intercept']+[x for x in features_initial],
                                    'coefficient':model_to_check.intercept_.tolist()+model_to_check.coef_[0].tolist()}).merge(pd.DataFrame({'feature':['intercept']+[x for x in features_to_check], 'se':bse,
                                                                                                                                            'wald':wald,
                                                                                                                                            'p-value':pvalue}),
                                                                                                                              on='feature',
                                                                                                                              how='left')

        if out is not None:
            if out[-4:]=='.csv':
                features_test[['feature', 'coefficient', 'se', 'wald', 'p-value']].to_csv(out, sep = sep, index=False)
            elif out[-4:]=='.xls' or out[-5:]=='.xlsx':
                features_test[['feature', 'coefficient', 'se', 'wald', 'p-value']].to_excel(out, sheet_name='Missing', index=False)
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')

        return features_test[['feature', 'coefficient', 'se', 'wald', 'p-value']]



    def regularized_feature_selection(self, data, regularization=None, regularization_value=None, features=None, solver = None,
                                      features_to_leave = None, scoring = 'roc_auc', threshold = .05):
        '''
        Feature selection based on regularization - model uses all available features, then the features with
        positive or insignificant coefficients are excluded (l1 regularization use is advised for more preliminary exclusion)

        Parameters
        -----------
        data: data for modeling (type Data)
        regularization: 'l1' for LASSO (less features to stay) or 'l2' for ridge (smaller coefficients) regression
        regularization_value: effect of regularization will be more prominent for lesser regularization value
        features: list of features to use for feature selection (if empty, then all features from data will be used)
        features_to_leave: features that must be included in the model
        scoring: type of score used to estimate the model quality
        threshold: threshold for p-value when removing a feature

        Returns
        -----------
        score for the model built on selected features and
        list of selected features

        '''

        if features_to_leave is None:
            features_to_leave=[]

        if features is None:
            features_to_check=data.features.copy()
        else:
            features_to_check=features.copy()

        if regularization is None:
            regularization=self.regularization
        if regularization_value is None:
            regularization_value=self.regularization_value
        if solver is None:
            solver = self.solver


        # correctness check
        for feature in features_to_leave:
            if feature not in data.features:
                print ('Feature is not available:', feature)
                return None

        if data.weights is None:
            lr=LogisticRegression(solver=solver, penalty=regularization, C=regularization_value)
        else:
            lr=LogisticRegression(solver='sag', penalty=regularization, C=regularization_value)

        if data.ginis is None or data.ginis == {}:
            ginis=data.calc_gini()
        else:
            ginis=data.ginis

        scores=[]

        to_refit=True
        while to_refit:
            to_refit=False
            if data.weights == None:
                lr.fit(data.dataframe[features_to_check], data.dataframe[data.target])
            else:
                lr.fit(data.dataframe[features_to_check], data.dataframe[data.target],
                       sample_weight = data.dataframe[data.weights])

            new_score = self.get_cv_score(Data(data.dataframe, target = data.target, features = features_to_check,
                                               weights = data.weights),
                                          scoring = scoring, selected_features=False)
            scores.append(new_score)

            positive_to_exclude=[x for x in np.asarray(features_to_check)[lr.coef_[0]>0] if x not in features_to_leave]

            if len(positive_to_exclude)>0:
                to_refit=True
                features_to_exclude={x:ginis[x] for x in positive_to_exclude}
                to_exclude=min(features_to_exclude, key=features_to_exclude.get)
                print('Dropping ', to_exclude, 'with positive coefficient and gini =', ginis[to_exclude])
                features_to_check.remove(to_exclude)
            else:
                wald=self.wald_test(data, model=lr, features=features_to_check)
                feature_to_exclude_array=wald[(wald['p-value']>threshold) & (wald['p-value']==wald['p-value'].max()) & (wald['feature'].isin(features_to_leave+['intercept'])==False)]['feature'].values

                if len(feature_to_exclude_array)>0:
                    to_refit=True
                    print('Dropping ', feature_to_exclude_array[0], 'with p-value =', wald[wald['feature']==feature_to_exclude_array[0]]['p-value'].values[0], 'and gini =', ginis[feature_to_exclude_array[0]])
                    features_to_check.remove(feature_to_exclude_array[0])

        result_features=[]
        for i in range(len(lr.coef_[0])):
            if lr.coef_[0][i]==0:
                print('Dropping ', features_to_check[i], 'with zero coefficient (gini =', ginis[features_to_check[i]], ')')
            else:
                result_features.append(features_to_check[i])

        plt.plot(np.arange(len(scores)), scores, 'bo-', linewidth=2.0)
        plt.xticks(np.arange(len(scores)), ['step ' + str(i) for i in np.arange(len(scores))], rotation = 'vertical')
        plt.ylabel(scoring)
        plt.title('Score changes')
        plt.show()
        self.selected = result_features
        return new_score, self.selected



    def stepwise_feature_selection(self, data, kind = 'mixed', features=None, features_initial=None, features_to_leave = None,
                                   eps = .0005, scoring = 'roc_auc', forward_threshold = .05, backward_threshold = .05,
                                   regularization=None, regularization_value=None):
        '''
        Stepwise feature selection can be of 3 types: forward, backward, mixed.
        Forward: on each step the feature is selected that increases the score most while the score changes
            are greater than epsilon.
        Backward: starts from all the possible features, on each step removes the feature with the least score
            decrease while the score changes are greater than epsilon (epsilon should be set to small negative value).
        Mixed: each step contains 2 stages. Stage 1: the algorithm selects from features-candidates a significant
            feature that increases the score most. Stage 2: from the features in models removes the feature with the
            least significance for the model.

        Parameters
        -----------
        data: data for modeling (type Data)
        kind: type of the algorithm, can be 'forward', 'backward' or 'mixed'
        features: list of features from which selection is working (if None, then data.features are used)
        features_initial: starting feature set for feature selection
        features_to_leave: features that must be included in the model
        eps: minimum significant score difference
        scoring: type of score used to estimate the model quality
        forward_threshold: threshold for p-value when adding a feature
        backward_threshold: threshold for p-value when removing a feature
        regularization: type of regularization to be used for wald test (l1 or l2)
        regularization_value: value of regularization parameter to be used for wald test

        Returns
        -----------
        score for the model built on selected features
        list of selected features

        '''

        if features_to_leave is None:
            features_to_leave=[]

        to_leave = features_to_leave.copy()
        #final_features = []
        candidates = []
        if features is None:
            features=data.features.copy()

        best_scores = []
        features_change = []

        # correctness check
        for feature in features_to_leave:
            if feature not in data.features:
                print ('Achtung bitte! Keine', feature)
                return None
            if features_initial is not None:
                if feature not in features_initial:
                    print ('No', feature, 'in initial feature list provided! ')
                    return None

        if regularization is None:
            regularization=self.regularization
        if regularization_value is None:
            regularization_value=self.regularization_value

        # Forward selection
        if kind == 'forward':
            print ('Forward feature selection started')
            if features_initial is None:
                features_initial=to_leave
            for feature in features:
                if feature not in features_initial:
                    candidates.append(feature)
            features=features_initial.copy()
            if len(features)>0:
                prev_score = self.get_cv_score(Data(data.dataframe, target = data.target,
                                                       features = features, weights = data.weights),
                                               scoring = scoring, selected_features=False)
                best_scores.append(prev_score)
                features_change.append('initial')
                print('Initial features:', features, '', scoring, 'score', prev_score)
            else:
                prev_score=-1000
            # maximum number of steps equals to the number of candidates
            for i in range(len(candidates)):
                # cross-validation scores for each of the remaining candidates of the step
                cvs = {}

                for feature in candidates:
                    tmp_features = features.copy()
                    # the feature is included in the model and the quality of the new model is calculated
                    tmp_features.append(feature)
                    try:
                        score = self.get_cv_score(Data(data.dataframe, target = data.target, features = tmp_features,
                                                       weights = data.weights),
                                                  scoring = scoring, selected_features=False)
                    except Exception:
                        pass
                    cvs[feature] = score
                # looking for the best new feature
                for f, s in cvs.items():
                    # warning: the metric is maximized
                    if s == max(cvs.values()):
                        # if the difference between the old and the new scores is greater than eps, the feature is added and the next step follows
                        if s - prev_score > eps:
                            print ('To add:', f, '', scoring, 'score:', cvs[f])
                            prev_score = s
                            features.append(f)
                            candidates.remove(f)
                            best_scores.append(s)
                            features_change.append(f)
                        # if the difference between the old and the new scoresis smaller than eps, the score dynamics are plotted and exit follows
                        else:
                            self.features = tmp_features
                            plt.plot(np.arange(len(features_change)), best_scores, 'bo-', linewidth=2.0)
                            plt.xticks(np.arange(len(features_change)), features_change, rotation = 'vertical')
                            plt.xlabel('Feature addition')
                            plt.ylabel(scoring)
                            plt.title('Stepwise score changes')
                            plt.show()
                            return prev_score, features
            # if no features added
            print('No features are available to add')
            self.features = features
            return prev_score, features

        # Backward selection
        elif kind == 'backward':
            if features_initial is not None:
                features=features_initial.copy()
            for feature in features:
                if feature not in to_leave:
                    candidates.append(feature)
            print ('Backward selection started')
            if len(features)>0:
                prev_score = self.get_cv_score(Data(data.dataframe, target = data.target, features = features,
                                                weights = data.weights), scoring = scoring, selected_features=False)
                best_scores.append(prev_score)
                features_change.append('initial')
                print('Initial features:', features, '', scoring, 'score', prev_score)
            else:
                prev_score=-1000

            #print('prev_score', prev_score, 'features', features, 'candidates', candidates)
             # maximum number of steps equals to the number of candidates
            for i in range(len(candidates)):
                cvs = {}
                if len(features)>1 and len(candidates)>0:
                    for feature in candidates:
                        tmp_features = features.copy()
                        # feature is removed and the cross-validation score is calculated
                        tmp_features.remove(feature)
                        cvs[feature] = self.get_cv_score(Data(data.dataframe, target = data.target, features = tmp_features,
                                                              weights = data.weights), scoring = scoring, selected_features=False)
                else:
                    print('No features are available to exclude (at least 1 feature should remain)')
                # searching for the feature that increases the quality most
                features_=features.copy()
                for f, s in cvs.items():
                    # if the difference between the old and the new scores is greater than eps, the feature is removed and the next step follows
                    if s == max(cvs.values()):
                        if s - prev_score > eps:
                            print ('To drop:', f, '', scoring, 'score:', cvs[f])
                            prev_score = s
                            candidates.remove(f)
                            features.remove(f)
                            best_scores.append(s)
                            features_change.append(f)
                        # if the quality increase is less than eps, exit
                if features==features_ or len(candidates)==0:
                    if len(features)>1 and len(candidates):
                        print('All features exclusion cause too significant score decrease')
                    self.features = candidates + to_leave
                    plt.plot(np.arange(len(features_change)), best_scores, 'bo-', linewidth=2.0)
                    plt.xticks(np.arange(len(features_change)), features_change, rotation = 'vertical')
                    plt.xlabel('Features removed')
                    plt.ylabel(scoring)
                    plt.title('Stepwise score changes')
                    plt.show()
                    return prev_score, self.features
            # if no feature was removed
            return prev_score, features

        # Mixed
        elif kind == 'mixed':
            print ('Mixed selection started')
            if features_initial is None:
                features_initial=to_leave
            for feature in features:
                if feature not in to_leave:
                    candidates.append(feature)
            if data.weights is None:
                lr=LogisticRegression(solver='saga', penalty=regularization, C=regularization_value)
            else:
                lr=LogisticRegression(solver='sag', penalty=regularization, C=regularization_value)

            prev_score = -1000

            result_features = features_initial.copy()

            scores = []
            feature_sets = []

            if len(result_features)>0:
                new_score = self.get_cv_score(Data(data.dataframe, target = data.target,
                                                       features = result_features, weights = data.weights),
                                              scoring = scoring, selected_features=False)
                scores.append(new_score)
                feature_sets.append(set(result_features))
            else:
                new_score = 0

            to_continue=True
            while to_continue and len(candidates)> 0:
                to_continue=False
                prev_score = new_score
                pvalues = {}
                cvs = {}
                for candidate in [x for x in candidates if (x in result_features)==False]:
                    # new feature addition and the model quality estimation
                    if data.weights == None:
                        lr.fit(data.dataframe[result_features + [candidate]], data.dataframe[data.target])

                    else:
                        lr.fit(data.dataframe[result_features + [candidate]], data.dataframe[data.target],
                               sample_weight = data.dataframe[data.weights])

                    new_score = self.get_cv_score(Data(data.dataframe, target = data.target,
                                                       features = result_features + [candidate], weights = data.weights),
                                                  scoring = scoring, selected_features=False)
                    wald=self.wald_test(data, model=lr, features=result_features + [candidate])

                    pvalues[candidate] = wald[wald['feature']==candidate]['p-value'].values[0]
                    cvs[candidate] = new_score
                # searching for a significant feature that gives the greatest score increase
                result_features_=result_features.copy()
                for feature in sorted(cvs, key = cvs.get, reverse = True):
                    if pvalues[feature] < forward_threshold and feature != 'intercept':
                        print ('To add:', feature, '', scoring, 'score:', cvs[feature], ' p-value', pvalues[feature])
                        result_features.append(feature)
                        break

                if result_features==result_features_:
                    print('No significant features to add were found')
                else:
                    if set(result_features) in feature_sets:
                        print('Feature selection entered loop: terminating feature selection')
                        break
                    elif cvs[feature]-prev_score>eps:
                        to_continue=True
                        scores.append(cvs[feature])
                        feature_sets.append(set(result_features))

                #print('result_features', result_features)
                # the least significant feature is removed
                # if it is Step1 then no removal
                if len(result_features)>1:
                    if data.weights == None:
                        lr.fit(data.dataframe[result_features], data.dataframe[data.target])
                    else:
                        lr.fit(data.dataframe[result_features], data.dataframe[data.target],
                               sample_weight = data.dataframe[data.weights])

                    wald=self.wald_test(data, model=lr, features=result_features)
                    wald_to_check=wald[wald['feature'].isin(to_leave+['intercept'])==False]
                    #display(wald_to_check)
                    if max(wald_to_check['p-value']) > backward_threshold:
                        to_delete = wald_to_check[wald_to_check['p-value']==wald_to_check['p-value'].max()]['feature'].values[0]

                        """if feature == to_delete:
                            candidates.remove(feature)
                            prev_score = prev_score-eps-0.05"""
                        result_features.remove(to_delete)
                        new_score = self.get_cv_score(Data(data.dataframe, target = data.target,
                                                           features = result_features, weights = data.weights),
                                                      scoring = scoring, selected_features=False)
                        print ('To drop:', to_delete, '', scoring, 'score', new_score, 'p-value', wald_to_check[wald_to_check['feature']==to_delete]['p-value'].values[0])

                        if set(result_features) in feature_sets:
                            print('Feature selection entered loop: terminating feature selection')
                            break
                        else:
                            to_continue=True
                            scores.append(new_score)
                            feature_sets.append(set(result_features))

                    elif wald_to_check[wald_to_check['coefficient']==0].shape[0] > 0:
                        to_delete = wald_to_check[wald_to_check['coefficient']==0]['feature'].tolist()
                        """if feature == to_delete:
                            candidates.remove(feature)
                            prev_score = prev_score-eps-0.05"""
                        print ('To drop:', to_delete, ' with zero coefficients (no score changes)')
                        result_features=[x for x in result_features if x not in to_delete]

                        if set(result_features) in feature_sets:
                            print('Feature selection entered loop: terminating feature selection')
                            break
                        else:
                            to_continue=True
                            new_score = self.get_cv_score(Data(data.dataframe, target = data.target,
                                                           features = result_features, weights = data.weights),
                                                      scoring = scoring, selected_features=False)
                            scores.append(new_score)
                            feature_sets.append(set(result_features))

            plt.plot(np.arange(len(scores)), scores, 'bo-', linewidth=2.0)
            plt.xticks(np.arange(len(scores)), ['step ' + str(i) for i in np.arange(len(scores))], rotation = 'vertical')
            plt.ylabel(scoring)
            plt.title('Stepwise score changes')
            plt.show()
            self.selected = sorted(list(feature_sets[-1]))
            return new_score, self.selected


        else:
            print ('Incorrect kind of selection. Please use backward, forward or mixed. Good luck.')
            return None



    #edited 22.08.2018 by Yudochev Dmitry - selected_features=True
    def fit(self, data, selected_features = True):
        '''
        Fits the model to the data given on the selected features or on all.

        Parameters
        -----------
        data: data (type Data) for fitting
        selected_features: whether to fit on the features selected previously or not,
            True - use selected features, False - use all features
        '''

        self.coefs = {}
        if selected_features:
            print('Using selected features: '+str(self.selected))
            self.features = self.selected
        else:
            print('Using all available features: '+str(data.features))
            self.features = data.features


        if self.features == None:
            print ('No features, how can that happen? :(')
            return None
        try:
            if data.weights is None:
                self.model.fit(data.dataframe[self.features], data.dataframe[data.target])
            else:
                self.model.fit(data.dataframe[self.features], data.dataframe[data.target],
                               sample_weight = data.dataframe[data.weights])
        except Exception:
            print('Fit failed! Maybe there are missings in data?...')
            return None
        for i in range(len(self.features)):
            self.coefs[self.features[i]] = self.model.coef_[0][i]



    def final_exclude(self, input_data, excluded=None, apply_changes=False):
        '''
        Checks the effect of one feature exclusion (after exclusion all features from 'excluded' list) for each of
        available features and prints initial gini values and difference after each feature exclusion. After exclusion
        list is decided this method can be used to exclude decided features and fit current model, using the rest of them.

        Parameters
        -----------
        input_data: A Data or DataSamples object for fitting and gini calculation
        excluded: a list of features to exclude before exclusion cycle
        apply_changes: if True then all features from 'excluded' list will be excluded and the model will be fitted
            using the rest of the features
        '''
        if len(self.selected)==0:
            print('No selected features to try exclusion. Abort.')
            return

        if excluded is None:
            excluded=[]

        if type(input_data)==DataSamples:
            retrain_data=input_data.train
        else:
            retrain_data=input_data

        new_selected=[x for x in self.selected if x not in excluded]

        if apply_changes:
            self.selected=new_selected
            self.fit(retrain_data, selected_features=True)
            self.draw_coefs()
            return self.roc_curve(input_data, figsize=(10, 7))
        else:
            try_model=LogisticRegressionModel(random_state = 42, penalty = self.regularization, C = self.regularization_value, solver = self.solver)
            try_model.selected=new_selected
            try_model.fit(retrain_data, selected_features=True)
            #try_model.draw_coefs()
            ginis_excl={}
            ginis_excl['initial']=try_model.roc_curve(input_data, verbose=False)
            for excl in new_selected:
                #try_model = LogisticRegressionModel(random_state = 42, penalty = self.regularization, C = self.regularization_value, solver = self.solver)
                try_model.selected=[x for x in new_selected if x!=excl]
                try_model.fit(retrain_data, selected_features=True)
                #try_model.draw_coefs()
                new_ginis=try_model.roc_curve(input_data, verbose=False)
                ginis_excl[excl]=[new_ginis[0]-ginis_excl['initial'][0], new_ginis[1]-ginis_excl['initial'][1], new_ginis[2]-ginis_excl['initial'][2]]

            ginis_excl_df=pd.DataFrame(ginis_excl).T
            if type(input_data)==DataSamples:
                cols=['Train']
                if input_data.validate is not None:
                    cols.append('Validate')
                if input_data.test is not None:
                    cols.append('Test')
                ginis_excl_df.columns=cols
                ginis_excl_df.sort_values('Test' if 'Test' in cols else 'Validate' if 'Validate' in cols else 'Train', ascending=False, inplace=True)
            else:
                ginis_excl_df.columns=['Data']
                ginis_excl_df.sort_values('Data', ascending=False, inplace=True)

            return ginis_excl_df



    def bootstrap_gini(self, bs_base, samples, bootstrap_part=0.75, bootstrap_number=10, stratify=True, replace=True, seed=0,
                       woe_transform=None, crosses_transform=None, figsize=(15,10), bins=None):
        '''
        Calculates Gini in bootstrap samples (either provided or generated) and plots their distribution with
        gini values from provided samples (Data, DataSamples or list of Data)

        Parameters
        -----------
        bs_base: a DataSamples object with bootstrap_base and bootstrap or a Data object to generate boostrap samples from
        samples: a DataSamples object with train/validate/test samples, a Data object or a list of Data objects to mark gini values on plot
        bootstrap_part: the size of each bootstrap sample is defined as part of input data sample
        bootstrap_number: number of generated bootstrap samples
        stratify: should bootstraping be stratified by data target
        replace: is it acceptable to repeat rows from train dataframe for bootstrap samples
        seed: value of random_state for dataframe.sample (each random_state is calculated as seed + number in bootstrap)
        woe_transform: a WOE object to perform WoE-transformation before using model
        bins: number of bins for the distribution plot (if None - use Freedman-Diaconis rule)
        crosses_transform: a Crosses object to perform cross-transformation before using model
        '''
        if isinstance(bs_base, DataSamples):
            if bs_base.bootstrap_base is None:
                print('No bootstrap data provided in the input DataSamples object. Return none')
                return None
            else:
                print('Using bootstrap data provided in the input DataSamples object..')
                bootstrap=bs_base.bootstrap
                bootstrap_base=bs_base.bootstrap_base
        elif isinstance(bs_base, Data):
            print('Generating bootstrap data from the input Data object..')
            DS_gini=DataSamples()
            DS_gini.bootstrap_split(bs_base, bootstrap_part=bootstrap_part, bootstrap_number=bootstrap_number, stratify=stratify, replace=replace, seed=seed)
            bootstrap=DS_gini.bootstrap
            bootstrap_base=DS_gini.bootstrap_base
        else:
            print('No bootstrap data was provided in the input. Return none')
            return None

        if isinstance(samples, DataSamples):
            check_samples=[]
            for sample in [samples.train, samples.validate, samples.test]:
                if sample is not None:
                    check_samples.append(sample)
        elif isinstance(samples, list):
            check_samples=samples.copy()
        elif isinstance(samples, Data):
            check_samples=[samples]
        else:
            print('No samples data was provided in the input')
            check_samples=[]

        samples_gini={}
        for i in range(len(check_samples)):
            if check_samples[i].name is None:
                current_sample=str(i)
            else:
                current_sample=check_samples[i].name
            print('Calculating gini for', current_sample,'sample..')
            if self.selected!=[x for x in self.selected if x in check_samples[i].dataframe]:
                print('Not all features from the current model were found in the',current_sample,'sample..')
                if woe_transform is None and crosses_transform is None:
                    print('No WOE or Crosses object were found. Return None.')
                    return None
                else:
                    if woe_transform is not None:
                        print('Starting woe-transformation..')
                        to_calc_gini=woe_transform.transform(check_samples[i],
                                                             features=[x[:-4] for x in self.selected if x[:-4] in woe_transform.feature_woes],
                                                             keep_essential=False if crosses_transform is not None else True, calc_gini=False)
                    if crosses_transform is not None:
                        print('Starting crosses-transformation..')
                        to_calc_gini=crosses_transform.transform(to_calc_gini if woe_transform is not None else check_samples[i],
                                                                 keep_essential=True, calc_gini=False)
            else:
                to_calc_gini=Data(check_samples[i].dataframe[self.selected+[check_samples[i].target]], check_samples[i].target, features=self.selected)
            preds = self.predict(to_calc_gini)[:,1]
            fpr, tpr, _ = roc_curve(to_calc_gini.dataframe[to_calc_gini.target], preds)
            samples_gini[current_sample] = (2*auc(fpr, tpr)-1)*100

        if self.selected!=[x for x in self.selected if x in bootstrap_base.dataframe]:
            print('Not all features from the current model were found in the bootstrap data..')
            if woe_transform is None and crosses_transform is None:
                print('No WOE or Crosses object were found. Return None.')
                return None
            else:
                if woe_transform is not None:
                    print('Starting woe-transformation..')
                    bootstrap_base=woe_transform.transform(bootstrap_base,
                                                         features=[x[:-4] for x in self.selected if x[:-4] in woe_transform.feature_woes],
                                                         keep_essential=False if crosses_transform is not None else True, calc_gini=False)
                if crosses_transform is not None:
                    print('Starting crosses-transformation..')
                    bootstrap_base=crosses_transform.transform(bootstrap_base, keep_essential=True, calc_gini=False)
                #bootstrap_base=woe_transform.transform(bootstrap_base, features=[x[:-4] for x in self.selected], keep_essential=True, calc_gini=False)

        bootstrap_gini=[]
        print('Calculating gini for bootstrap samples..')
        for i in range(len(bootstrap)):
            preds = self.predict(Data(bootstrap_base.dataframe.iloc[bootstrap[i]][self.selected], bootstrap_base.target, features=self.selected))[:,1]
            fpr, tpr, _ = roc_curve(bootstrap_base.dataframe.iloc[bootstrap[i]][bootstrap_base.target], preds)
            bootstrap_gini.append((2*auc(fpr, tpr)-1)*100)

        plt.figure(figsize=figsize)
        sns.distplot(bootstrap_gini, bins=bins)
        palette = itertools.cycle(sns.color_palette())
        for s in samples_gini:
            plt.axvline(x=samples_gini[s], linestyle='--', color=next(palette), label=s)

        plt.axvline(x=np.mean(bootstrap_gini)-2*np.std(bootstrap_gini), linestyle='-', color='red', alpha=0.5)
        plt.text(np.mean(bootstrap_gini)-2*np.std(bootstrap_gini), 0, '   mean-2*std = '+str(round(np.mean(bootstrap_gini)-2*np.std(bootstrap_gini),4)),
                     horizontalalignment='right', verticalalignment='bottom', rotation=90, fontsize=12)
        plt.axvline(x=np.mean(bootstrap_gini)+2*np.std(bootstrap_gini), linestyle='-', color='red', alpha=0.5)
        plt.text(np.mean(bootstrap_gini)+2*np.std(bootstrap_gini), 0, '   mean+2*std = '+str(round(np.mean(bootstrap_gini)+2*np.std(bootstrap_gini),4)),
                     horizontalalignment='right', verticalalignment='bottom', rotation=90, fontsize=12)
        plt.xlabel('Gini values in bootstrap')
        plt.ylabel('Distribution')
        plt.legend()
        #plt.title(feature.feature, fontsize = 16)
        #if out:
        #    plt.savefig(out_images+feature.feature+".png", dpi=100, bbox_inches='tight')
        plt.show()
        return samples_gini, bootstrap_gini


    def drop_features(self, to_drop = None):
        '''
        deletes features from the model

        Parameters
        -----------
        to_drop:  a feature or a list of features that should be excluded
        '''

        if to_drop is None:
            print ('Please enter the features you want to exclude. Use parameter features_to_drop and restart this method.')
            return None
        elif isinstance(to_drop, list):
            print ('The features will be removed from the "selected features" list.')
            for feature in to_drop:
                if feature in self.selected:
                    self.selected.remove(feature)
                    print (feature, 'removed')
        else:
            print ('The feature will be removed from the "selected features" list.')
            if to_drop in self.selected:
                self.selected.remove(feature)
                print (to_drop, 'removed')


    #edited 22.08.2018 by Yudochev Dmitry - selected_features=True
    def get_cv_score(self, data, cv = 5, scoring = 'roc_auc', selected_features = True):
        '''
        Calculates the model quality with cross-validation

        Parameters
        -----------
        data: data for cross-validation score calculation
        cv: number of folds
        scoring: metric of quality
        selected_features: whether to use selected features or not, True - use selected features, False - use all features

        Returns
        -----------
        cross-validation score
        '''

        if selected_features:
            features = self.selected
        else:
            features = data.features

        if features == None:
            print ('No features, how can that happen? :(')
            return None

        if data.weights == None:
            return cross_val_score(self.model, data.dataframe[features],
                                   data.dataframe[data.target], cv = cv, scoring = scoring).mean()
        else:
            return cross_val_score(self.model, data.dataframe[features], data.dataframe[data.target], cv = cv,
                                   scoring = scoring, fit_params = {'sample_weight' : data.dataframe[data.weights]}).mean()



    def form_scorecard(self, woe=None, crosses=None, out = None, sep=';', score_value=444, score_odds=10, double_odds=69):
        '''
        Makes a scorecard and exports it to a file.

        Parameters
        -----------
        woe: a WOE object for scoring card
        crosses: a Crosses object for scoring card
        out: file to export the scorecard in csv/xlsx format
        sep: the separator to be used in case of csv export
        score_value: score value, used for scaling
        score_odds: odds of score value, used for scaling
        double_odds:  score value increament, that halfes the odds, used for scaling

        Returns:
        ----------
        A scorecard (pandas.DataFrame)

        '''

        # if no WOE used then onle regression coefficients are included
        if woe is None and crosses is None:
            print ('Achung bitte: keine WOE')
            scorecard = pd.DataFrame(columns = ['feature', 'coefficient'])
            for feature in self.features:
                tmp = pd.DataFrame([[feature, self.coefs[feature]]], columns = ['feature', 'coefficient'])
                scorecard = scorecard.append(tmp, ignore_index=True)
            scorecard = scorecard.append(pd.DataFrame([['intercept', self.model.intercept_[0]]], columns = ['feature',
                                                                                                            'coefficient']),
                                         ignore_index=True)
            #scorecard.to_csv(fname, sep = ';')
            #return scorecard

        else:
            scorecard = pd.DataFrame(columns = ['feature', 'categorical', 'group', 'values', 'missing', 'woe', 'coefficient',
                                                'sample_part', 'ER'])
            for feature in self.features:
                if woe is not None and feature[:-4] in woe.feature_woes:
                    woes = woe.feature_woes[feature[:-4]].woes
                    missing_group=woe.feature_woes[feature[:-4]].missing_group
                    groups = woe.feature_woes[feature[:-4]].groups
                    categorical=woe.feature_woes[feature[:-4]].categorical
                    d=woe.feature_woes[feature[:-4]].data
                    if d.weights is None:
                        all_obs=d.dataframe.shape[0]
                    else:
                        all_obs=d.dataframe[d.weights].sum()

                    # searching for WOE for each interval of values
                    for group in [x for x in woes if woes[x] is not None]:
                        if d.weights is None:
                            obs=d.dataframe[d.dataframe[feature]==woes[group]].shape[0]
                            bad=d.dataframe[d.dataframe[feature]==woes[group]][d.target].sum()
                        else:
                            obs=d.dataframe[d.dataframe[feature]==woes[group]][d.weights].sum()
                            bad=d.dataframe[(d.dataframe[feature]==woes[group]) & (d.dataframe[d.target]==1)][d.weights].sum()

                        missing_in=(group==missing_group)*1
                        tmp = pd.DataFrame([[feature[:-4], categorical, group, groups[group], missing_in, woes[group], self.coefs[feature],
                                             obs/all_obs, bad/obs]],
                                           columns = ['feature', 'categorical', 'group', 'values', 'missing', 'woe', 'coefficient',
                                                      'sample_part', 'ER'])
                        scorecard = scorecard.append(tmp, ignore_index=True)
                elif crosses is not None and int(feature[len(crosses.prefix):-4]) in crosses.decision_trees:
                    tree = crosses.decision_trees[int(feature[len(crosses.prefix):-4])].tree.dropna(how='all', axis=1)
                    leaves = tree[tree['leaf']]
                    for group in sorted(leaves['group'].unique().tolist()):
                        current_group=leaves[leaves['group']==group]
                        used_features=list(leaves.columns[:leaves.columns.get_loc('node')])
                        current_woe=current_group['group_woe'].unique()[0]
                        current_er=current_group['group_target'].unique()[0]/current_group['group_amount'].unique()[0]
                        current_sample_part=current_group['group_amount'].unique()[0]/leaves[['group', 'group_amount']].drop_duplicates()['group_amount'].sum()
                        current_values=[]
                        for _, row in current_group.iterrows():
                            used_features=[]
                            parent_node=row['parent_node']
                            while parent_node is not None:
                                used_features=[tree[tree['node']==parent_node]['split_feature'].values[0]]+used_features
                                parent_node=tree[tree['node']==parent_node]['parent_node'].values[0]
                            current_values.append({x:row[x] for x in used_features})
                        #current_values=[{x:row[x] for x in used_features if row[x] is not None} for _, row in current_group.iterrows()]
                        scorecard = scorecard.append({'feature':feature[:-4], 'categorical':np.nan, 'group':group,
                                                      'values': current_values, 'missing':0, 'woe':current_woe,
                                                      'coefficient':self.coefs[feature], 'sample_part':current_sample_part,
                                                      'ER':current_er}
                                                     , ignore_index=True)
                else:
                    print ('Achung bitte: keine feature',feature,'. Skipping')

            scorecard = scorecard.sort_values(by = ['feature', 'group'])
            # bias addition
            scorecard_intercept = pd.DataFrame([['intercept', np.nan, np.nan, np.nan, np.nan, np.nan, self.model.intercept_[0], np.nan, np.nan]],
                                               columns = ['feature', 'categorical', 'group', 'values', 'missing', 'woe',
                                                          'coefficient', 'sample_part', 'ER'])

            multiplier=double_odds/np.log(2)
            if double_odds>0:
                scorecard=scorecard.merge(scorecard[['feature', 'woe']].groupby('feature', as_index=False).min().rename(index=str, columns={"woe": "woe_shift"}), on='feature',how='left')
            else:
                scorecard=scorecard.merge(scorecard[['feature', 'woe']].groupby('feature', as_index=False).max().rename(index=str, columns={"woe": "woe_shift"}), on='feature',how='left')
            scorecard['woe_shifted']=scorecard['woe']-scorecard['woe_shift']
            scorecard['score']=-(scorecard['woe_shifted']*scorecard['coefficient']*multiplier)
            for_intercept=scorecard[['coefficient', 'woe_shift']].drop_duplicates().copy()
            for_intercept['woe_on_coef']=-for_intercept['coefficient']*for_intercept['woe_shift']*multiplier
            scorecard_intercept['score']=-((scorecard_intercept['coefficient']+np.log(score_odds))*multiplier)+score_value+for_intercept['woe_on_coef'].sum()
            scorecard_intercept.index=[-1]
            scorecard=scorecard.append(scorecard_intercept).sort_index().reset_index(drop=True)[['feature', 'categorical', 'group',
                                                                                                 'values', 'missing', 'woe',
                                                                                                 'coefficient', 'score',
                                                                                                 'sample_part', 'ER']]
            #display(scorecard)
            scorecard['score']=round(scorecard['score']).astype('int64')
            scorecard['values']=scorecard['values'].astype(str)

        # export to a file
        if out is not None:
            if out[-4:]=='.csv':
                scorecard.to_csv(out, sep = sep, index=False)
            elif out[-4:]=='.xls' or out[-5:]=='.xlsx':
                scorecard.to_excel(out, sheet_name='Missing', index=False)
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')

        self.scorecard = scorecard
        return scorecard



    #edited 28.08.2018 by Yudochev Dmitry
    def score(self, data, features_to_leave=None, include_scores_in_features=False, unknown_score=0, verbose=True):
        '''
        Performs data scoring

        Parameters
        -----------
        data: data of type Data
        features_to_leave: list of fields to include in output dataframe
        include_scores_in_features: should all scores be treated as features in output Data object (otherwise new features will be empty)

        Returns
        -----------
        Data object, containing dataframe with initial features (+ features_to_leave), their scores and overall score
        '''

        if verbose:
            print ('Scores calculation...')

        if self.scorecard is None or self.scorecard.shape[0] == 0:
            print ('No scorecard! Where is it?')
            return None

        if 'score' not in self.scorecard.columns:
            print ('Please set scores: scorecard[score]')
            return None

        scorecard=self.scorecard.copy()
        scorecard['values']=scorecard['values'].astype(str)

        features_to_leave=[] if features_to_leave is None else features_to_leave.copy()
        features_to_leave+=([data.target] if data.target is not None else [])+([data.weights] if data.weights is not None else [])
        features_to_leave=list(set(features_to_leave))

        trees_for_score=scorecard[scorecard.apply(lambda row: pd.isnull(row['categorical']) and row['feature']!='intercept', axis=1)]['feature'].unique().tolist()
        features_for_score=[x for x in scorecard.feature.unique() if x!='intercept' and x not in trees_for_score]
        all_features=features_for_score.copy()
        scorecard.loc[scorecard.feature.isin(trees_for_score)==False, 'values']=\
            scorecard.loc[scorecard.feature.isin(trees_for_score)==False, 'values'].apply(lambda x:
                np.nan if x=='nan' else \
                    eval(x.replace('[nan]', '[np.nan]').replace('[nan,','[np.nan,').replace(', nan]',', np.nan]')\
                          .replace(', inf]',', np.inf]').replace('[-inf,','[-np.inf,')))
        if len(trees_for_score)>0:
            scorecard.loc[scorecard.feature.isin(trees_for_score), 'values']=\
                scorecard.loc[scorecard.feature.isin(trees_for_score), 'values'].apply(lambda x:
                    eval(x.replace(': nan,',': np.nan,').replace(': nan}',': np.nan}')\
                          .replace('), nan)','), np.nan)').replace(', nan,',', np.nan,')\
                          .replace('[nan,','[np.nan,').replace(', nan]',', np.nan]').replace('[nan]', '[np.nan]')\
                          .replace(', inf)',', np.inf)').replace('(-inf,','(-np.inf,')))
            all_features+=list(set([f for values in scorecard[scorecard.feature.isin(trees_for_score)]['values'] for node in values for f in node]))
        all_features=list(set(all_features))
        all_features=sorted(all_features)

        try:
            data_with_scores = data.dataframe[list(set(all_features+features_to_leave))].copy()
            for feature in features_for_score:

                if verbose:
                    print (feature)

                bounds = list(scorecard[scorecard.feature == feature]['values'])
                scores = list(scorecard[scorecard.feature == feature].score)
                missing = list(scorecard[scorecard.feature == feature].missing)
                categorical = list(scorecard[scorecard.feature == feature].categorical)[0]

                if categorical==False:

                    bs = {}
                    missing_score=0
                    for i in range(len(bounds)):
                        if missing[i]==1:
                            missing_score=scores[i]
                        if isinstance(bounds[i],list):
                            bs[scores[i]]=bounds[i][0]

                    bs[np.inf]=np.inf
                    bs={x:bs[x] for x in sorted(bs, key=bs.get)}
                    data_with_scores[feature+'_scr']=data_with_scores[feature].apply(
                                                        lambda x: missing_score if pd.isnull(x) \
                                                                  else list(bs.keys())[np.argmax([bs[list(bs.keys())[i]] <= x and bs[list(bs.keys())[i+1]] > x for i in range(len(bs.keys())-1)])])
                else:
                    bs = {}
                    missing_score=0
                    for i in range(len(bounds)):
                        bs[scores[i]]=bounds[i]
                        for b in bs[scores[i]]:
                            if pd.isnull(b) or b=='':
                                missing_score=scores[i]
                    data_with_scores[feature+'_scr']=data_with_scores[feature].apply(
                                                        lambda x: missing_score if pd.isnull(x) \
                                                                  else unknown_score if x not in [v for s in bs for v in bs[s]] \
                                                                  else list(bs.keys())[[(x in bs[s]) for s in bs].index(True)])
            for tree in trees_for_score:
                if verbose:
                    print (tree)
                current_tree=scorecard[scorecard.feature==tree]
                conditions=[]
                for group in current_tree.group:
                    current_conditions=current_tree[current_tree.group==group]['values'].values[0]
                    for condition in current_conditions:
                        final_condition={x:None for x in all_features}
                        final_condition.update(condition)
                        final_condition.update({'node':0, 'score':current_tree[current_tree.group==group]['score'].values[0]})
                        conditions.append(final_condition)
                pseudo_tree=pd.DataFrame(conditions)[all_features+['node', 'score']].dropna(how='all', axis=1)
                pseudo_tree['leaf']=True
                #display(pseudo_tree)
                data_with_scores[tree+'_scr']=DecisionTree().transform(data_with_scores, pseudo_tree, ret_values=['score'])
        except Exception:
            print('No raw data provided. Scoring by WoE values..')
            features_for_score = [x for x in scorecard.feature.unique() if x!='intercept']
            trees_for_score = []
            data_with_scores = data.dataframe[list(set([x+'_WOE' for x in features_for_score]+features_to_leave))].copy()
            for feature in features_for_score:

                if verbose:
                    print (feature+'_WOE')

                woes = list(scorecard[scorecard.feature == feature]['woe'])
                scores = list(scorecard[scorecard.feature == feature].score)  

                ws = {}
                for i in range(len(woes)):
                    ws[round(woes[i],5)]=scores[i]

                data_with_scores[feature+'_scr']=data_with_scores[feature+'_WOE'].apply(
                                                        lambda x: unknown_score if round(x,5) not in ws \
                                                                  else ws[round(x,5)])
            
        data_with_scores['score']=data_with_scores[[x+'_scr' for x in features_for_score+trees_for_score]].sum(axis=1)+\
                                    scorecard[scorecard.feature.str.lower() == 'intercept']['score'].values[0]
        if include_scores_in_features:
            return Data(data_with_scores, data.target, features=['score']+[x+'_scr' for x in features_for_score+trees_for_score], weights=data.weights, name=data.name)
        else:
            return Data(data_with_scores, data.target, weights=data.weights, name=data.name)



    def score_distribution(self, base, samples=None, bins=20, figsize=(15,10), width=0.2, proportion=False, draw_base=False):
        '''
        Calculates score bins on base sample and draws samples distribution by these bins

        Parameters
        -----------
        base: a Data object to calculates score bins (if there is no 'score' field in base.dataframe, then score calculation is performed)
        samples: a Data\DataSamples object or a list of Data objects to draw distribution by base-defined bins
        bins: number of bins to generate
        figsize: a tuple for graph size
        width: width of bars for graph (width*number of samples to draw should be less then 1)
        proportion: True, if proportions should be drawn (if False, then amounts are drawn)
        draw_base: if True, then base sample is also drawn
        '''
        if self.scorecard.shape[0]==0:
            print('No scorecard detected. Please, run form_scorecard method. Return None')
            return None

        if 'score' not in base.dataframe:
            print('Scoring base sample..')
            to_cut=self.score(base, verbose=False).dataframe['score']
        else:
            to_cut=base.dataframe['score']
        _, cuts=pd.cut(to_cut, bins=bins, right=False, precision=10, retbins=True)
        cuts[0]=-np.inf
        cuts[-1]=np.inf
        train_stats=pd.cut(to_cut, bins=cuts, right=False, precision=10).value_counts().sort_index()
        if proportion:
            train_stats=train_stats/train_stats.sum()

        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        pos=np.array(range(train_stats.shape[0]))
        if draw_base:
            if base.name is None:
                label='Base'
            else:
                label=base.name
            plt.bar(pos,train_stats, width=width, label=label)

        if isinstance(samples, DataSamples):
            check_samples=[]
            for sample in [samples.train, samples.validate, samples.test]:
                if sample is not None:
                    check_samples.append(sample)
        elif isinstance(samples, list):
            check_samples=samples.copy()
        elif isinstance(samples, Data):
            check_samples=[samples]
        else:
            print('No samples data provided in the input')
            check_samples=[]

        for i in range(len(check_samples)):
            if check_samples[i] is not None:
                if 'score' not in check_samples[i].dataframe:
                    print('Scoring',i,'sample..')
                    to_cut=self.score(check_samples[i], verbose=False).dataframe['score']
                else:
                    to_cut=check_samples[i].dataframe['score']
                stats=pd.cut(to_cut, bins=cuts, right=False, precision=10).value_counts().sort_index()
                if proportion:
                    stats=stats/stats.sum()
                if check_samples[i].name is None:
                    label=str(i)
                else:
                    label=check_samples[i].name
                num=i+1 if draw_base else i
                plt.bar(pos+num*width,stats, width=width, label=label)
        plt.xticks(pos+len(check_samples)*width/2,train_stats.index.astype(str))
        fig.autofmt_xdate()
        if proportion:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
        plt.legend()
        plt.show()



    def draw_coefs(self, figsize = (5,5), filename = 'model_coefficients'):
        '''
        Plots coefficients as a barplot.

        Parameters
        -----------
        figsize: figsize parameter for matplotlib
        filename: name of file for picture with model coefficients

        Returns
        -----------
        A barplot for coefficients
        '''

        with plt.style.context(('seaborn-deep')):
            plt.figure(figsize = figsize)
            feature_list = [f for f in self.coefs]
            coefs_list = [self.coefs[f] for f in self.coefs]
            plt.barh(range(len(coefs_list)), [coefs_list[i] for i in np.argsort(coefs_list)])
            plt.yticks(range(len(coefs_list)), [feature_list[i] for i in np.argsort(coefs_list)])
            if filename is not None:
                plt.savefig(filename + ".png", dpi=100, bbox_inches='tight')
            plt.show()



    def change_coefs(self, change_dict):
        '''
        Enables a user to change regression coefficients by hand.

        Parameters
        -----------
        change_dict: a dictionary {feature: new_coef} where feature id the name of a feature,
            new_coef is a value of the new coefficient for the feature

        '''
        for feature in change_dict:
            self.coefs[feature] = change_dict[feature]
        print ('New coefficients:', self.coefs)



    # Nov-13-2018 updated by Anna Goreva
    # Oct-10-2018 updated by Anna Goreva
    def to_sas(self, sasfname, missing_dict = None, prefix = 'SCR_', result_score_name = 'SCORE', variables = None):
        '''
        Transforms the model to SAS-code and exports to a file.
        Unknown values for categorical features gain as many scores as missings.

        Parameters
        -----------
        sasfname: name of the file with SAS-code of the model
        missing_dict: a dictionary {f: how_to_fill_NA}, where f is the feature name, how_to_fill_NA is a value to replace missings (NaNs);
                        default score for missing (if not in scorecard) is 0
        prefix: prefix for variables with scores for each feature
        result_score_name: variable for resulting score
        variables: {feature: feature_name_in_sas} - possibility for a variable to  be renamed in resulting code

        Returns
        ---------
        a file with SAS-code

        '''

        if variables is None:
            variables={}
        if missing_dict is None:
            missing_dict={}
        # sas-names of variables
        # for checking the legth of names and avoiding duplications
        name_list = []


        try:
            #костыль - преобразуем списки в строки, чтобы потом обратно распарсить, надо исправить
            scorecard=self.scorecard.copy()
            scorecard['values']=scorecard['values'].astype(str)
            trees=scorecard[scorecard.apply(lambda row: pd.isnull(row['categorical']) and row['feature']!='intercept', axis=1)]['feature'].unique().tolist()
            scorecard.loc[scorecard.feature.isin(trees), 'values']=\
                scorecard.loc[scorecard.feature.isin(trees), 'values'].apply(lambda x:
                    eval(x.replace(': nan,',': np.nan,').replace(': nan}',': np.nan}')\
                          .replace('), nan)','), np.nan)').replace(', nan,',', np.nan,')\
                          .replace('[nan,','[np.nan,').replace(', nan]',', np.nan]').replace('[nan]', '[np.nan]')\
                          .replace('(inf,','(np.inf,').replace(', inf)',', np.inf)')\
                          .replace('(-inf,','(-np.inf,').replace(', -inf)',', -np.inf)')))
            trees_features = {x:sorted(list(set([f for group in scorecard[scorecard['feature']==x]['values'] for cond in group for f in cond]))) for x in trees}
            categorical = list(scorecard[(scorecard.categorical.fillna(False) == True)&(scorecard.feature.isin(trees)==False)].feature.drop_duplicates())
            numerical = list(scorecard[scorecard.categorical.fillna(True) == False].dropna().feature.drop_duplicates())

        except Exception:
            print ('Aсhtung bitte! Keine Wertungskarte!')
            return None
        #debug
        print ('Categorical:', categorical)
        print ('Numerical:', numerical)
        print ('Trees:', trees)

        for feature in categorical + numerical + trees:
            if feature not in variables:
                variables[feature] = feature

        with open (sasfname, 'w') as f:
            f.write('SCORECARD_POINTS = ' + str(int(scorecard[scorecard.feature == 'intercept'].score)) + ';\n')
            #debug
            print ('SCORECARD_POINTS = '+ str(int(scorecard[scorecard.feature == 'intercept'].score)) + ';\n')

            for feature in variables:#scorecard.dropna().feature.drop_duplicates():
                #debug
                #print ('feature', feature)
                f.write('\n*------------------------------------------------------------*;\n* Variable: '+ feature
                        + (';\n* Features: '+str(trees_features[feature])[1:-1].replace("'",'') if feature in trees else '')
                        + ';\n*------------------------------------------------------------*;\n')

                # check if variable's name is not too long
                scr_feature = (prefix + variables[feature])
                if len(scr_feature) > 32:
                    print ('Warning: Name', prefix + variables[feature], 'is too long for SAS. Renaming the variable...')
                    postfix = 1
                    scr_feature = scr_feature[:30] + '_' + str(postfix)
                    while scr_feature in name_list:
                        postfix = postfix + 1
                        scr_feature = scr_feature[:-1] + str(postfix)
                    name_list.append(scr_feature)

                if (scorecard[scorecard.feature == feature].missing).sum() > 0:
                    missing_score = scorecard[(scorecard.feature == feature)&(scorecard.missing)].score.values[0]
                elif feature in missing_dict:
                    missing_score = missing_dict[feature]
                else:
                    missing_score = scorecard[scorecard.feature == feature].score.min()

                # Missings processing
                if feature not in trees:
                    f.write('\nif MISSING(' + variables[feature] + ') \n\t then ' + scr_feature + '= ' + str(missing_score) + ';\n')
                    #debug
                    print ('if MISSING(' + variables[feature] + ') \n\t then ' + scr_feature + '= ' + str(missing_score) + ';')

                tmp = scorecard[scorecard.feature == feature].copy()
                if feature in categorical:
                    # Categorical features processing
                    tmp = tmp[tmp['values'] != '[nan]']
                    tmp['values']=tmp['values'].apply(
                                    lambda x: eval(x.replace(', nan,',', ')\
                                                    .replace('[nan,','[').replace(', nan]',']').replace('[nan]', '[np.nan]')))

                    for _, row in tmp.iterrows():
                        f.write('else if NOT MISSING(' + variables[feature] + ') and ' + variables[feature] + ' in '
                                + str(row['values']).replace("[", "(").replace("]", ")")
                                + '\n\t then ' + scr_feature + ' = ' + str(int(row.score)) + ';\n')
                        print ('else if NOT MISSING(' + variables[feature] + ') and ' + variables[feature] + ' in '
                                + str(row['values']).replace("[", "(").replace("]", ")")
                                + '\n\t then ' + scr_feature + ' = ' + str(int(row.score)) + ';')

                    f.write('else ' + scr_feature + ' = ' + str(missing_score) + ';\n\n')
                    print ('else ' + scr_feature + ' = ' + str(missing_score) + ';\n')

                if feature in numerical:
                    # Numerical features processing
                    tmp = tmp[tmp['values'] != 'nan']
                    tmp['values']=tmp['values'].apply(lambda x:
                                    eval(x.replace('[-inf,','[-np.inf,').replace(', inf]',', np.inf]')))

                    for _, row in tmp.iterrows():
                        f.write('else if NOT MISSING(' + variables[feature] + ') AND '
                                + ((str(row['values'][0]) + ' <= ') if row['values'][0]!=-np.inf else '')
                                + variables[feature]
                                + (' < ' + str(row['values'][1]) if row['values'][1]!=np.inf else '')
                                + ' \n\t then ' + scr_feature + ' = ' + str(int(row['score'])) + ';\n')
                        print ('else if NOT MISSING(' + variables[feature] + ') AND '
                                + ((str(row['values'][0]) + ' <= ') if row['values'][0]!=-np.inf else '')
                                + variables[feature]
                                + (' < ' + str(row['values'][1]) if row['values'][1]!=np.inf else '')
                                + ' \n\t then ' + scr_feature + ' = ' + str(int(row['score'])) + ';')

                if feature in trees:
                    # Trees processing
                    '''
                    tmp['values']=tmp['values'].apply(
                                    lambda x: np.nan if x=='nan' else \
                                        eval(x.replace('), nan)','), np.nan)').replace(', nan,',', np.nan,')\
                                              .replace('[nan,','[np.nan,').replace(', nan]',', np.nan]')\
                                              .replace('(-inf,','(-np.inf,').replace(', inf)',', np.inf)')))
                    '''
                    first_row=True
                    for _, row in tmp.iterrows():
                        f.write('\nif \n(\n' if first_row else 'else if \n(\n')
                        print('\nif \n(' if first_row else 'else if \n(')
                        first_row=False
                        for ci in range(len(row['values'])):
                            for var in row['values'][ci]:
                                if isinstance(row['values'][ci][var], list):
                                    if [np.nan] == row['values'][ci][var]:
                                        f.write('\t(MISSING(' + var + '))')
                                        print('\t(MISSING(' + var + '))')
                                    else:
                                        if np.nan in row['values'][ci][var]:
                                            f.write('\t(MISSING(' + var + ') or ')
                                            print('\t(MISSING(' + var + ') or')
                                        else:
                                            f.write('\t(NOT MISSING(' + var + ') and ')
                                            print('\t(NOT MISSING(' + var + ') and')

                                        f.write(var + ' in ' + str([x for x in row['values'][ci][var] if pd.isnull(x)==False])\
                                                                                  .replace("[", "(").replace("]", ")") + ')')
                                        print(var + ' in ' + str([x for x in row['values'][ci][var] if pd.isnull(x)==False])\
                                                                                .replace("[", "(").replace("]", ")") + ')')
                                else:
                                    if pd.isnull(row['values'][ci][var]):
                                        f.write('\t(MISSING(' + var + '))')
                                        print('\t(MISSING(' + var + '))')
                                    elif row['values'][ci][var]==(-np.inf, np.inf):
                                        f.write('\t(NOT MISSING(' + var + '))')
                                        print('\t(NOT MISSING(' + var + '))')
                                    else:
                                        if pd.isnull(row['values'][ci][var][1]):
                                            f.write('\t(MISSING(' + var + ') or ')
                                            print('\t(MISSING(' + var + ') or')
                                        else:
                                            f.write('\t(NOT MISSING(' + var + ') and ')
                                            print('\t(NOT MISSING(' + var + ') and')

                                        if isinstance(row['values'][ci][var], tuple):
                                            cleared = row['values'][ci][var][0] if pd.isnull(row['values'][ci][var][1]) else row['values'][ci][var]

                                            f.write(((str(cleared[0]) + ' <= ') if cleared[0]!=-np.inf else '')
                                                    + var
                                                    + (' < ' + str(cleared[1]) if cleared[1]!=np.inf else '') + ')')
                                            print (((str(cleared[0]) + ' <= ') if cleared[0]!=-np.inf else '')
                                                    + var
                                                    + (' < ' + str(cleared[1]) if cleared[1]!=np.inf else '') + ')')
                                f.write(' and \n' if list(row['values'][ci]).index(var)!=len(row['values'][ci])-1 else '\n')
                                print(' and ' if list(row['values'][ci]).index(var)!=len(row['values'][ci])-1 else '')

                            f.write(') or \n(\n' if ci!=len(row['values'])-1 else ')\n')
                            print(') or \n(' if ci!=len(row['values'])-1 else ')')

                        f.write('\t then ' + scr_feature + ' = ' + str(int(row.score)) + ';\n')
                        print('\t then ' + scr_feature + ' = ' + str(int(row.score)) + ';')

                    f.write('else ' + scr_feature + ' = ' + str(missing_score) + ';\n\n')
                    print ('else ' + scr_feature + ' = ' + str(missing_score) + ';\n')


                f.write('SCORECARD_POINTS = SCORECARD_POINTS + ' + scr_feature + ';\n')
                print ('SCORECARD_POINTS = SCORECARD_POINTS + ' + scr_feature + ';\n')

            f.write(result_score_name + ' = SCORECARD_POINTS;\n')
            print(result_score_name + ' = SCORECARD_POINTS;\n')
#------------------------------------------------------------------------------------------------------------------




class OrdinalRegressionModel:
    '''
    Ordinal logistic regression for acceptable probability models (mainly for income prediction). It predicts a probability
    that real value is less or equal then the specified value or a value that would be less or equal then the real value with
    the specified probability.

    There is no ordinal regression in sklearn, so this class doesn't inherit sklearn's interface, but it has a similar one.

    An object of this class can:
    1) transform input continuous dependent variable into ordinal by defining classes,
    2) fit separate logistic regressions for analysing the possibility of using ordinal regression (all coefficients by the same
        variables must be "close" to each other) and guessing of initital parameters for ordinal regression,
    3) fit ordinal regression, using scipy.optimize.minimize with SLSQP method and constraints, that class estimates should
        be monotonically increasing, minimizing negative log-likelihood of separate logistic regressions,
    4) fit linear regression for estimating class estimates by the natural logarithm of predicted value, using
        scipy.optimize.minimize with the specified method, minimizing negative R-squared of this linear regression,
    5) predict a probability that real value is less or equal then the specified value,
    6) predict a value that would be less or equal then the real value with the specified probability,
    7) create SAS code for calculating income, using fitted ordinal logistic and linear regressions.
    '''

    def __init__(self, X, y, classes=None, coefficients=None, intercepts=None, bias=None,
                 log_intercept=None, log_coefficient=None, alpha=None):
        '''
        Parameters
        -----------
        X: pandas.Series containing the predictor variables
        y: pandas.Series containing the dependent variable
        classes: an array-like, containing the upper edges of classes of dependent variable
        coefficients: the coefficients by the predictor variables
        intercepts: an array-like with classes estimates
        bias: a shift of the dependent variable used for more accurate prediction of classes estimates
        log_intercept: an intercept for linear regression of classes estimates by classes upper edges
        log_coefficient: a coefficient for linear regression of classes estimates by classes upper edges
        alpha: a probability value used for predicting dependent variable lower edge
        '''
        if len(X.shape)==1:
            X=pd.DataFrame(X)
        if classes is None:
            classes=[x*10000 for x in range(1,11)]

        self.X=X
        self.y=y
        self.classes=classes
        self.coefficients=coefficients
        self.intercepts=intercepts
        self.bias=bias
        self.log_intercept=log_intercept
        self.log_coefficient=log_coefficient
        self.alpha=alpha


    def y_to_classes(self, y=None, classes=None, replace=True):
        '''
        Transform continuous dependent variable to ordinal variable

        Parameters
        -----------
        y: pandas.Series containing the dependent variable
        classes: an array-like, containing the upper edges of classes of dependent variable
        replace: should new ordinal variable replace initial continuous one in self

        Returns
        -----------
        pandas.Series of transformed ordinal dependent variable
        '''
        if y is None:
            y=self.y
        if classes is None:
            classes=self.classes

        bins = pd.IntervalIndex.from_tuples([(-np.inf if i==0 else classes[i-1], np.inf if i==len(classes) else classes[i]) \
                                             for i in range(len(classes)+1)], closed ='left')
        #display(bins)
        new_y=pd.cut(y, bins=bins).apply(lambda x: x.right).astype(float)
        if replace:
            self.y=new_y
        return new_y


    def fit(self, X=None, y=None, params=None, verbose=True):
        '''
        Fit ordinal logistic regression, using scipy.optimize.minimize with SLSQP method and constraints
        of inequality defined as differences between the estimates of the next and current classes (these
        differences will stay non-negative for SLSQP method)

        Parameters
        -----------
        X: pandas.Series containing the predictor variables
        y: pandas.Series containing the dependent variable
        params: an array-like with predictor's coefficient and classes estimates
        verbose: should detailed information be printed
        '''

        def get_guess(X, y, verbose=True):
            '''
            TECH

            Find initial coefficients and classes estimates values by fitting separate logistic regressions

            Parameters
            -----------
            X: pandas.Series containing the predictor variables
            y: pandas.Series containing the dependent variable
            verbose: should detailed information be printed

            Returns
            -----------
            an array-like with predictors' coefficients and classes estimates initial values
            '''
            lr_coefs=pd.DataFrame(columns=list(X.columns)+['intercept', 'amount', 'gini'])
            for target in self.classes:
                check_class=(y<=target)
                lr=LogisticRegression(C=100000, random_state=40)
                lr.fit(X, check_class)
                fpr, tpr, _ = roc_curve(check_class, lr.predict_proba(X)[:,1])

                lr_coefs.loc[target]=list(lr.coef_[0])+[lr.intercept_[0], check_class.sum(), 2*auc(fpr, tpr)-1]
            if verbose:
                for pred in X.columns:
                    plt.figure(figsize = (10,5))
                    plt.barh(range(lr_coefs.shape[0]), lr_coefs[pred].tolist())
                    for i in range(lr_coefs.shape[0]):
                        plt.text(lr_coefs[pred].tolist()[i], i, str(round(lr_coefs[pred].tolist()[i],5)),
                                 horizontalalignment='left', verticalalignment='center', fontweight='bold', fontsize=10)
                    plt.yticks(range(lr_coefs.shape[0]), lr_coefs.index.tolist())
                    plt.suptitle('Coefficients of separate logistic regressions')
                    plt.xlabel('Coefficients for predictor '+pred)
                    plt.ylabel('Predicted value caps (classes)')
                    plt.margins(x=0.1)
                    plt.show()
                plt.figure(figsize = (10,5))
                plt.bar(range(lr_coefs.shape[0]), lr_coefs.gini.tolist())
                for i in range(lr_coefs.shape[0]):
                    plt.text(i, lr_coefs.gini.tolist()[i], str(round(lr_coefs.gini.tolist()[i],4)),
                             horizontalalignment='center', verticalalignment='bottom', fontweight='bold', fontsize=10)
                plt.xticks(range(lr_coefs.shape[0]), lr_coefs.index.tolist())
                plt.suptitle('Gini values of separate logistic regressions')
                plt.ylabel('Gini value')
                plt.xlabel('Predicted value caps (classes)')
                plt.margins(y=0.1)
                plt.show()
            return np.array(lr_coefs[list(X.columns)].mean().tolist()+lr_coefs.intercept.tolist())

        if X is None:
            X=self.X
        if y is None:
            y=self.y

        if params is None:
            if verbose:
                print('No initial parameters specified. Guessing approximate parameters by fitting separate logistic regressions..')
            guess=get_guess(X, y, verbose=verbose)
            if verbose:
                print('Initial guess:')
                print('\tCoefficients (<predictor> = <coefficient>):')
                for i in range(len(X.columns)):
                    print('\t',X.columns[i],'=',guess[i])
                print('\tClass estimates (<class> = <estimate>):')
                for i in range(len(self.classes)):
                    print('\t',self.classes[i],'=',guess[i+len(X.columns)])
                print()
        else:
            guess=params.copy()


        cons=tuple({'type': 'ineq', 'fun': lambda x:  x[i+1] - x[i]} for i in range(len(X.columns),len(X.columns)+len(self.classes)-1))

        results = minimize(self.OrdinalRegressionLL, guess, args=(X, y),
                           method = 'SLSQP', options={'disp': verbose}, constraints=cons)

        self.coefficients=results['x'][:len(X.columns)]
        self.intercepts=list(results['x'][len(X.columns):])
        if verbose:
            print()
            print('Fitted values:')
            print('\tCoefficients (<predictor> = <coefficient>):')
            for i in range(len(X.columns)):
                print('\t',X.columns[i],'=',self.coefficients[i])
            print('\tClass estimates (<class> = <estimate>):')
            for i in range(len(self.classes)):
                print('\t',self.classes[i],'=',self.intercepts[i])


    def fit_class_estimates(self, bias=None, verbose=True, method='Nelder-Mead'):
        '''
        Fit linear regression, using scipy.optimize.minimize with the specified method. Nelder-Mead showed
        the best results during testing.

        Parameters
        -----------
        bias: a shift of the dependent variable used for more accurate prediction of classes estimates
        verbose: should detailed information be printed
        method: optimization method for finding the best bias value
        '''
        to_log=pd.DataFrame([self.classes, self.intercepts], index=['value', 'estimate']).T

        if bias is None:
            if verbose:
                print('Searching for an optimal bias value..')
            results = minimize(self.LinearRegressionR2, [0], args=(to_log.value, to_log.estimate),
                               method=method, options={'disp': verbose})
            self.bias=results['x'][0]
        else:
            self.bias=bias

        biased_X=np.log(to_log.value+self.bias)
        lir=LinearRegression()
        lir.fit(biased_X.reshape(-1,1), to_log.estimate)
        self.log_intercept=lir.intercept_
        self.log_coefficient=lir.coef_[0]

        r2=r2_score(to_log.estimate, lir.predict(biased_X.reshape(-1,1)))

        if verbose:
            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111)
            ax.scatter(y=to_log.estimate, x=to_log.value, label='Actual estimates')
            ax.plot(to_log.value, lir.predict(biased_X.reshape(-1,1)), 'r-', label='Predicted estimates')
            ax.text(0.95, 0.1,
                    'y = '+str(round(self.log_coefficient,5))+'*ln(x+'+str(round(self.bias,5))+') '+\
                        ('+ ' if self.log_intercept>=0 else '')+str(round(self.log_intercept,5)),
                    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='red', alpha=0.7))
            ax.set_xlabel('Predicted value caps (classes)')
            ax.set_ylabel('Classes estimates')
            ax.legend()
            plt.show()
            print('Bias =', self.bias, 'R2_score =', r2)


    def predict(self, alpha=None, X=None):
        '''
        Predict a value that would be less or equal then the real value with the specified probability

        Parameters
        -----------
        X: pandas.Series containing the predictor variable
        alpha: a probability value used for predicting dependent variable lower edge

        Returns
        -----------
        pandas.Series of predicted values
        '''
        if X is None:
            X=self.X
        if alpha is not None:
            self.alpha=alpha
        elif alpha is None:
            alpha=self.alpha
        return np.exp((np.log((1-alpha)/alpha) - (self.coefficients*X).sum(axis=1) - self.log_intercept)/self.log_coefficient) - self.bias


    def predict_proba(self, predicted_value, X=None):
        '''
        Predict a probability that real value is less or equal then the specified value

        Parameters
        -----------
        predicted_value: a value to be used as the upper edge of predicted values in probability calculation
        X: pandas.Series containing the predictor variable

        Returns
        -----------
        pandas.Series of predicted probabilities
        '''
        if X is None:
            X=self.X

        if self.log_intercept is None or self.log_coefficient is None:
            print('Class estimates were not fitted, only initial class values are available for prediction.')
            class_num=np.argmin(np.abs(np.array(self.classes)-predicted_value))
            predicted_value=self.classes[class_num]
            intercept=self.intercepts[class_num]
            print('Using', predicted_value,'instead of input value.')
            return 1/(1+np.exp(-((self.coefficients*X).sum(axis=1)+intercept)))
        else:
            intercept=self.log_coefficient*np.log(predicted_value+self.bias)+self.log_intercept
            return 1/(1+np.exp(-((self.coefficients*X).sum(axis=1)+intercept)))


    def to_sas(self, alpha=None):
        '''
        Print SAS code with income calculation formula

        Parameters
        -----------
        alpha: a probability value used for predicting dependent variable lower edge
        '''
        if alpha is None:
            alpha=self.alpha
        print('BIAS =', self.bias,';')
        print('INCOME_INTERCEPT =', self.log_intercept,';')
        print('INCOME_COEF =', self.log_coefficient,';')
        for i in range(len(self.X.columns)):
            print(self.X.columns[i]+'_COEF =', self.coefficients[i],';')
        print('ALPHA =', alpha,';')
        print()
        final_formula='INCOME_FORECAST = exp((log((1-ALPHA)/ALPHA)'
        for i in range(len(self.X.columns)):
            final_formula+=' - '+self.X.columns[i]+'_COEF*'+self.X.columns[i]
        final_formula+=' - INCOME_INTERCEPT)/INCOME_COEF) - BIAS;'
        print(final_formula)


    def OrdinalRegressionLL(self, params, X=None, y=None):
        '''
        Returns negative summed up log-likelihood of separate logistic regressions for MLE

        Parameters
        -----------
        params: an array-like with predictors' coefficients and classes estimates
        X: pandas.Series containing the predictor variables
        y: pandas.Series containing the dependent variable

        Returns
        -----------
        negative summed up log-likelihood
        '''
        if X is None:
            X=self.X
        if y is None:
            y=self.y

        negLL = 0
        #calculate and sum up log-likelyhood for each class as if it is a separate model
        for i in range(len(self.classes)):
            #taking class intercept and the common coefficients
            weights=[params[i+len(self.X.columns)]]
            for p in range(len(self.X.columns)):
                weights.append(params[p])

            intercept = np.ones((X.shape[0], 1))
            features = np.hstack((intercept, X))
            scores = np.dot(features, weights)
            negLL -= np.sum((y<=self.classes[i])*scores - np.log(np.exp(scores) + 1))
        # return negative LL
        return negLL

    def LinearRegressionR2(self, params, X, y):
        '''
        Returns negative R-squared of linear regression for minimizing

        Parameters
        -----------
        params: an array-like with the bias value
        X: pandas.Series containing the predictor variable
        y: pandas.Series containing the dependent variable

        Returns
        -----------
        negative R-squared
        '''
        biased_X=np.log(X+params[0])

        lir=LinearRegression()
        lir.fit(biased_X.reshape(-1,1), y)

        return - r2_score(y, lir.predict(biased_X.values.reshape(-1,1)))