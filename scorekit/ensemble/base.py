# -*- coding: utf-8 -*-

import numpy as np
from ..data import Data, DataSamples
from ..model import LogisticRegressionModel
from .._utils import color_background
import matplotlib.pyplot as plt
import pandas as pd
from abc import abstractmethod
import copy
import gc
from sklearn.linear_model import LinearRegression, LogisticRegression
from itertools import combinations


plt.rc('font', family='Verdana')
plt.style.use('seaborn-darkgrid')
pd.set_option('display.precision', 3)

gc.enable()

class WeightedScore:
    '''
    Interface class for weighing several scores or features.
    
    '''
    def __init__(self):
        self.stats = pd.DataFrame()
        self.weights = {}
        self.reg = None
        
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def to_sas(self, filename):
        pass
    
    
    
class BotWeightedScore(WeightedScore):
    '''
    For different scores weighting, including scores of outer vendors and data sources.
    Implements Yury Botnikov's algorithm for scores weighting with scores for missing values and protection against errors of integrations with outer production services.
    '''
    def __init__(self, root_score, outer_scores, target):
        '''
        Initialization
        
        Parameters
        ------------
        root_score: score considered the main, 'internal'
        outer_scores: list of scores to weight (except root_score)
        target: target for binary classification
        '''
        self.root_score = root_score
        if not isinstance(self.root_score, str):
            raise TypeError('Root_score should be a string! Only one root_score is allowed. Good luck.')
        self.target = target 
        self.outer_scores = outer_scores
        self.scores_na = {}
        self.means = {}
        self.stds = {}
        self.coefs = {}
        self.stats = pd.DataFrame({'scores' : [self.root_score] + self.outer_scores})
        
    
    
    def na_scores(self, dataset, random_state = 42):
        '''
        TECH
        
        Calculates scores for missings in outer_scores
        
        Parameters
        -----------
        dataset: a train dataset, pandas.DataFrame
        random_state: parameter for LogisticRegression
        '''
        reg = LogisticRegression(random_state = random_state, C = 10000)
        reg.fit(dataset[[self.root_score] + self.outer_scores + [score + '_no_hit' for score in self.outer_scores]].fillna(0), dataset[self.target])
        
        n = len(self.outer_scores)
        self.stats['coefs_na_scores'] = reg.coef_[0][:n + 1]
        self.stats['coefs_na_no_hit'] = [np.nan] + list(reg.coef_[0][n + 1:])
        self.stats['scores_na'] = self.stats['coefs_na_scores']/self.stats['coefs_na_no_hit']
        
        for i, score in enumerate(self.outer_scores):
            self.scores_na[score] = reg.coef_[0][i + n + 1]/reg.coef_[0][i + 1]
        
    
    
    def score_hist(self, score, dataset, round_val = 1):
        '''
        Plots a histogram of score values wiht scores for missings.
        
        Parameters
        -----------
        score: str, name of score to plot
        dataset: pandas.DataFrame with score
        round_val: int, rounding parameter
        '''
        fig, ax = plt.subplots()
        fig.set_figwidth(12)
        n, bins, patches = ax.hist(list(dataset[score].dropna()), bins = 40)
        if score in self.scores_na:
            patches[max(np.where(bins < self.scores_na[score])[0])].set_fc('r')
            patches[max(np.where(bins < self.scores_na[score])[0])].set_label('Score for missing value')
            ax.legend()
        ax.set_title(score)
        plt.xticks(bins, np.around(bins, round_val), rotation = 17)
        plt.show()
        
        
        
    def set_scores_na(self, scores_na, verbose = True):
        '''
        Setter for scores_na
        '''
        self.scores_na = scores_na
        self.stats['scores_na'] = [v for k, v in self.scores_na.items()]

    
    
    
    def calc_coefs(self, dataset, random_state = 42):
        '''
        TECH
        
        Calculates normalized regression coefficients for root score and outer scores.
        
        Parameters
        ------------
        dataset: pandas.DataFrame to process
        '''
        reg = LogisticRegression(random_state = random_state, C = 10000)
        reg.fit(dataset[[self.root_score] + self.outer_scores], dataset[self.target])
        self.stats['coefs'] = reg.coef_[0]
        self.coefs = {s: reg.coef_[0][i]/reg.coef_[0][0] for i, s in enumerate([self.root_score] + self.outer_scores)}
        self.stats['coefs_norm'] = list(self.coefs.values())
        
    
    
    def calc_moments(self, dataset):
        '''
        TECH
        
        Calculates mean and std for root score and outer scores
        
        Parameters
        -----------
        dataset: pandas.DataFrame with data
        '''
        for score in [self.root_score] + self.outer_scores:
            self.means[score] = dataset[score].mean()
            self.stds[score] = dataset[score].std()
        self.stats['means'] = list(self.means.values())
        self.stats['stds'] = list(self.stds.values())
    
    
    
    def fit(self, dataset, verbose = True, calc_na = True, random_state = 42):
        '''
        Calculates weights and "scores for missings"
        
        Parameters
        ------------
        dataset: Data or pandas.DataFrame, dataset with scores, including root_score an outer_scores
        random_state: parameter for LogisticRegression
        verbose: boolean,if True - the fitting process is visualized
        calc_na: boolean, if True - values for missings are calculated automatically
        '''
        
        train = dataset.dataframe[[self.target, self.root_score] + self.outer_scores].copy() if isinstance(dataset, Data) else dataset[[self.target, self.root_score] + self.outer_scores].copy()
        if verbose:
            print ('Calculation started...')
        
        # Step 1
        if calc_na:
            # Scores for missings
            if train.isnull().sum().sum() > 0:
                for score in self.outer_scores:
                    train[score + '_no_hit'] = train[score].isnull()*1
                self.na_scores(train, random_state)
            else:
                self.scores_na = {score : train[score].median() for score in self.outer_scores}
        elif len(self.scores_na) < len(self.outer_scores):
            raise ValueError('All the outer scores should present in scores_na')
            
        if verbose:
            print ('Scores for missings:', self.scores_na)
            for score in self.outer_scores:
                self.score_hist(score, train)
        
        
        # Step 2
        for score, val in self.scores_na.items():
            train[score + '_hit'] = 1*(train[score].isnull() == False)
            train[score].fillna(val, inplace = True)
        self.calc_coefs(train, random_state)
        if verbose:
            print ('Normalized coefficients:', self.coefs)
        
        # Step 3
        self.calc_moments(train)
        if verbose:
            print ('Mean values of scores:', self.means)
            print ('Standard deviations of scores:', self.stds)
        
        
        
    def predict(self, dataset, score_name = 'weighted_score', floor = False, errors = False):
        '''
        Calculates final weighted score from root_score and outer_scores.
        
        Parameters:
        -------------
        dataset: pandas.DataFrame or Data, dataset for processing
        score: str, final score's name
        floor: boolean, if True - np.floor is applied to the final score
        errors: dict of names of columns with errors in outer scores, e.g. if an integration with an outer source is broken or a score is invalid; columns with errors should
        contain 2 values: 1 - no error, 0 - error.
            Example:
                {'score_beeline_t' : 'score_beeeline_valid', 'score_MFM' : 'score_MFM_valid'}
        '''
        if errors:
            print ('Attention! {} should be 1 if the value is correct, and 0 if the value is damaged'.format(list(errors.values())))
        
        to_data = isinstance(dataset, Data)
        if isinstance(dataset, pd.DataFrame):
            df = copy.deepcopy(dataset)
            df0 = copy.deepcopy(dataset)
        elif to_data:
            df = copy.deepcopy(dataset.dataframe)
            df0 = copy.deepcopy(dataset.dataframe)
        else:
            raise TypeError('Wrong type of dataset! Bye-bye.')
        
        for score, val in self.scores_na.items():
            df[score].fillna(val, inplace = True)
        
        nom = df[self.root_score].copy()
        denom = pd.Series(np.ones(df.shape[0]))
        for score in self.outer_scores:
            nom += (df[score] - self.means[score])*(self.coefs[score] if not errors else self.coefs[score]*dataset[errors[score]])
            denom += self.coefs[score]**2*self.stds[score]**2/self.stds[self.root_score]**2*(pd.Series(np.ones(df.shape[0])) if not errors else self.coefs[score]*dataset[errors[score]])
        
        denom = np.sqrt(denom)
        
        df0[score_name] = (nom - self.means[self.root_score])/denom + self.means[self.root_score]
        if floor:
            df0[score_name] = np.floor(df0[score_name])
        
        if to_data:
            return Data(df0, target = dataset.target, features = dataset.features, weights = dataset.weights)
        else:
            return df0
            
    
    def to_sas(self, filename, round_digits = 3, result_score_name = 'weighted_score', score_prefix = 'SCR_', flag_prefix = 'Flag_', root_score_name = None, 
               outer_scores_names = None, verbose = True):
        '''
        SAS-code generation
       
        parameters
        ----------
        filename: str, name of the file to write into
        round_digits: int, parameter of rounding
        result_score_name: str, name of the resulting score in the SAS-code
        score_prefix: str, prefix for edited scores
        flag_prefix: str, prefix for flags of valid scores
        root_score_name: str, name of the root score in the script
        outer_scores_names: dict, {outer_score : outer_score_name} if scores' names differ from outer_scores 
        '''
        if root_score_name is None:
            root_score_name = self.root_score
         
        if outer_scores_names is None:
            outer_scores_names = {n : n for n in self.outer_scores}
         
        with open(filename, 'w') as f:
            for i, score in enumerate(self.outer_scores):
                row = 'if missing(' + outer_scores_names[score] + ') then ' + score_prefix + outer_scores_names[score] + ' = ' + str(round(self.scores_na[score] - self.means[score], round_digits)) + ';\n'
                row += score_prefix + outer_scores_names[score] + ' = ' + outer_scores_names[score] + ' - ' + str(round(self.means[score], round_digits)) + ';\n'
                f.write(row)
                if verbose:
                    print (row)
            f.write('/***************************************************************/\n')
            f.write('/*********************** Final score ***************************/\n')
            f.write('/***************************************************************/\n')
            row = 'nom = ' + root_score_name 
            for score in self.outer_scores:
                row += ' + ' + flag_prefix + outer_scores_names[score] + ' * ' + score_prefix + outer_scores_names[score] + ' * ' + str(round(self.coefs[score], round_digits))
            row += ';\n'
            f.write(row)
            if verbose:
                print (row)
            
            row = 'denom = sqrt(1' 
            for score in self.outer_scores:
                row += ' + ' + flag_prefix + outer_scores_names[score] + '*' + str(round(self.coefs[score]**2*self.stds[score]**2/self.stds[self.root_score]**2, round_digits))
            row += ');\n'
                
            f.write(row)
            if verbose:
                print (row)
            
            row = result_score_name + ' = floor((nom - ' + str(round(self.means[self.root_score], round_digits)) + ') / denom + '  + str(round(self.means[self.root_score], round_digits)) + ');\n'
            f.write(row)
            if verbose:
                print (row)
            
            f.close()
        
        
        
            
                
        
        
                    
        
                
        
    

class VanillaWeightedScore(WeightedScore):
    '''
    Implements plain weightening: firstly, each score is divided into 2 zones: zone with missings and zone without missings. After that, for each combination of the scores'
    zones separate weigths are calculated (for non-missing scores).
    E.g. (score1, score2, score3) ->
        -> f(score1, score2, score3) = (score1 is missing)*((score2 is missing)*f1(score3) 
                                                            + (score3 is missing)*f2(score2)
                                                            + (score2 is not missing and score3 is not missing)*f3(score2, score3)
                                                            )
                                      + (score2 is missing)*((score3 is missing)*f4(score1)
                                                            + (score1 is not missing and score3 is not missing)*f5(score1, score3)
                                                            )
                                      + (score3 is missing)*(score1 is not missing and score2 is not missing)*f6(score1, score2)
    '''
    def fit(self, dataset, scores = None, target = None, task_type = 'binary', random_state = 42):
        '''
        Calculates weights for scores
        
        Parameters
        -----------
        dataset: pandas.DataFrame or data.Data object with scores and binary target
        scores: (optional) list of scored to weighing
        target: (optional) name of target
        task_type: (optional) 'binary' for binary classification task or 'linear' for linear regression task
        random_state: random_state
        '''
        if isinstance(dataset, Data):
            if scores is None:
                scores = dataset.features
            if target is None:
                target = dataset.target
            df = copy.deepcopy(dataset.dataframe)
        elif isinstance(dataset, pd.DataFrame):
            df = copy.deepcopy(dataset)
        else:
            raise TypeError('Found {} where pandas.DataFrame or scorekit.data.Data was expected! Good luck!'.format(type(dataset)))
        
        if scores is None or target is None or target not in dataset.columns or sum([i in dataset.columns for i in scores]) < len(scores):
            raise ValueError('Please set correct scores and target! Good luck!')
         
        df = copy.deepcopy(df[[target] + scores])
        if task_type == 'binary':
            self.reg = LogisticRegression(random_state = random_state, C = 10000)
        elif task_type == 'linear':
            self.reg = LinearRegression()
        else:
            raise ValueError('Wrong task_type! Bye-bye.')
        
        # find scores with missings
        scores_with_nan = list(df.isnull().sum()[df.isnull().sum() > 0].index)
        
        # make combinations of scores
        ranges = []
        for i in range(1, len(scores_with_nan) + 1):
            for subset in combinations(scores_with_nan, i):
                ranges.append(subset)
        
        self.zones = dict((ind, list(v)) for ind, v in enumerate(ranges))
        del ranges
        
        for zone, zone_scores in self.zones.items():
            self.weights[zone] = self.fit_zone(zone_scores, df, scores, target)
            
    
    
    def fit_zone(self, zone_scores, df, scores, target):
        '''
        TECH
        
        Finds coefs for 1 zone 
        
        Parameters
        -----------
        zone_scores: scores that define zone (from self.zones)
        df: pd.DataFrame to fit on
        scores: list of all the scores used 
        target: target name
        
        Returns
        -----------
        result: dict with coefficients for each score and intercept
        '''
        df_fit = df[df[zone_scores].isnull().sum(axis = 1) > 0].drop(zone_scores, 1).dropna()
        if df_fit.shape[0] > 0:
            try:
                self.reg.fit(df_fit.drop(target, 1), df_fit[target])
                result = {'intercept': self.reg.intercept_[0]}
                for i, f in enumerate(df_fit.drop(target, 1).columns):
                    result[f] = self.reg.coef_[0][i]
            except Exception:
                result = {'intercept': df_fit[target].mean()}
                for i, f in enumerate(df_fit.drop(target, 1).columns):
                    result[f] = 0
            return result
        
        
        
    
    def predict(self, dataset, weighted_name = 'score_weighted'):
        '''
        Calculates weighted score for the dataset
        
        Parameters
        -----------
        dataset: a scorekit.data.Data or pandas.DataFrame object with scores to weight
        weighted_name: name for column with weighted score
        
        Returns
        ----------
        an object ot type(dataset) with weighted score
        '''
        try:
            if isinstance(dataset, Data):
                df = copy.deepcopy(dataset.dataframe)
            elif isinstance(dataset, pd.DataFrame):
                df = copy.deepcopy(dataset)
            else:
                raise TypeError('Wrong type of dataset! Try DataFrame or Data. Good luck >:-]')
        except Exception:
            print ('Initialization failed!')
            

    

    

            
        
                

