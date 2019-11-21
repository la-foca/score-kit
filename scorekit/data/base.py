# -*- coding: utf-8 -*-

import pandas as pd
#import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
#rom scipy.stats import chi2, chisquare, ks_2samp, ttest_ind
#import statsmodels.formula.api as sm
import warnings
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
import ast
import os
import xlsxwriter
from PIL import Image
import datetime
from dateutil.relativedelta import *
import gc
#import weakref
import copy
import itertools
import calendar
#from joblib import Parallel, delayed


gc.enable()

warnings.simplefilter('ignore')

plt.rc('font', family='Verdana')
plt.style.use('seaborn-darkgrid')
pd.set_option('display.precision', 3)



#----------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------Classes-------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
class Data:
    '''
    Stores dataset and its parameters like features, weights, etc.

    Parameters:
    -----------
    dataframe: an object of pandas.DataFrame(), the data itself
    target: the name of a column which is the target feature
    features: list of names of columns that will be used for target prediction
    weights: name of a column with sample weights (if applicable)
    ginis: dictionary of features and their ginis (can be calculated by .calc_gini())
    name: name of the Data Object
    compress: should the input data be compressed
    convert_to_numeric: try to convert object columns to numeric, so there won't be errors for example due to comparison of float and decimal
    '''

    def __init__(self, dataframe, target = None, features = None, weights = None, name = None, compress=False, convert_to_numeric=True):

        if convert_to_numeric:
            for col in dataframe:
                if dataframe[col].dtype.kind=='O':
                    try:
                        dataframe[col]=pd.to_numeric(dataframe[col])
                        print('Attention! Converting',col,'values to numeric dtype!')
                    except Exception:
                        pass  

        if compress:
            self.dataframe = self.reduce_mem_usage(dataframe)
        else:
            self.dataframe = dataframe.copy()
        if features == None:
            self.features = list(dataframe.columns)
        else:
            self.features = list(features)
        self.target = target

        if self.target in self.features:
            self.features.remove(self.target)
        self.weights = weights
        self.ginis = {}
        self.name = name


    def reduce_mem_usage(self, input_df, verbose=True):
        df=input_df.copy()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df


    def show_missings(self, features = None, out=None, sep=';'):
        '''
        Finds missing values (NaN) in itself

        Parameters:
        -----------
        features: list of features to check for having missing values
        out: a path for csv/xlsx output file to export
        sep: the separator to be used in case of csv export

        Returns:
        -----------
        The list of features having missings in their values and number of missing values per feature
        '''

        if features == None:
            missing_amount=self.dataframe[self.features].isnull().sum()[self.dataframe[self.features].isnull().sum() > 0]
        else:
            missing_amount=self.dataframe[features].isnull().sum()[self.dataframe[features].isnull().sum() > 0]

        if out is not None:
            if out[-4:]=='.csv':
                pd.DataFrame(missing_amount).rename({0:'Missing'},axis=1).to_csv(out, sep = sep)
            elif out[-4:]=='.xls' or out[-5:]=='.xlsx':
                pd.DataFrame(missing_amount).rename({0:'Missing'},axis=1).to_excel(out, sheet_name='Missing')
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')

        return missing_amount



    def calc_gini(self, features=None, inplace = False, out=None, sep=';', 
                  time_column = None, time_format = "%d.%m.%Y", time_func = (lambda x: 1000*x.year + x.month),
                  reversed_connection=True):
        '''
        Calculates gini for all features in the Data object

        Parameters:
        -----------
        features: list of features to calculate gini for
        inplace: whether to update self.ginis or not
        out: a path for csv/xlsx output file to export
        sep: the separator to be used in case of csv export
        time_column: name of a column with time values to calculate time periods
        time_format: format of time values in time_column
        time_func: function for time_column parsing (changes date to some value, representing time period) or
                a period type for dt.to_period() function
        reversed_connection: if increase in value leades to decrease of target rate

        Returns:
        -----------
        ginis: dictionary of {feature:gini} (if inplace = False)
        '''
        if features is None:
            cycle_features=list(self.features)
        else:
            cycle_features=list(features)

        not_in_features=[x for x in cycle_features if x not in self.features]
        if len(not_in_features)>0:
            print('No', not_in_features, 'in self.features. Abort.')
            return None

        if time_column is not None:
            for_gini_in_time=self.dataframe[cycle_features+[time_column, self.target]].copy()
            for_gini_in_time[time_column] = pd.to_datetime(for_gini_in_time[time_column], format=time_format, errors='coerce')
            if callable(time_func):
                for_gini_in_time['_period'] = for_gini_in_time[time_column].map(time_func)
            elif isinstance(time_func, str):
                try:
                    for_gini_in_time['_period'] = for_gini_in_time[time_column].dt.to_period(time_func).astype(str)
                except Exception:
                    print('No function or correct period code was provided. Return None.')
                    return None
            else:
                print('No function or correct period code was provided. Return None.')
                return None

        ginis={}
        for feature in cycle_features:
            try:
                fpr, tpr, _ = roc_curve(self.dataframe[self.target], (-1 if reversed_connection else 1)*self.dataframe[feature])
            except Exception:
                print('There was an error during fpr and tpr calculation for',feature,'. Possibly not all features were transformed. Skipping..')
                continue
            ginis[feature] = (2*auc(fpr, tpr)-1)*100

            if time_column is not None:
                if self.weights is not None:
                    feature_gini_in_time=for_gini_in_time[['_period', self.target, self.weights]]
                    feature_gini_in_time[self.target]=feature_gini_in_time[self.target]*feature_gini_in_time[self.weights]
                    feature_gini_in_time=feature_gini_in_time.groupby('_period', as_index=False).sum().rename({self.target:'target', self.weights:'amount'}, axis=1)
                else:
                    feature_gini_in_time=for_gini_in_time[['_period', self.target]].groupby('_period').agg(['sum','size']).T.reset_index(level=0).T.drop('level_0').reset_index().rename({'sum':'target', 'size':'amount'}, axis=1)
                for period in sorted(feature_gini_in_time['_period']):
                    fpr, tpr, _ = roc_curve(for_gini_in_time[for_gini_in_time['_period']==period][self.target], (-1 if reversed_connection else 1)*for_gini_in_time[for_gini_in_time['_period']==period][feature])
                    feature_gini_in_time.loc[feature_gini_in_time['_period']==period, 'gini'] = (2*auc(fpr, tpr)-1)*100
                #display(feature_gini_in_time)

                fig = plt.figure(figsize=(15,5))
                ax = fig.add_subplot(111)
                ax.set_ylabel('Observations')
                plt.xticks(range(feature_gini_in_time.shape[0]), feature_gini_in_time['_period'].astype(str), rotation=45, ha="right")
                ax.bar(range(feature_gini_in_time.shape[0]), feature_gini_in_time['amount'], zorder=0, alpha=0.3)

                ax.grid(False)
                ax.grid(axis='y', zorder=1, alpha=0.6)
                ax2 = ax.twinx()
                ax2.set_ylabel('Gini')
                ax2.grid(False)
                ax2.annotate('Targets:', xy=(0, 1), xycoords=('axes fraction', 'axes fraction'), xytext=(-42, 5), textcoords='offset pixels', color='red', size=11)
                for i in range(feature_gini_in_time.shape[0]):
                    ax2.annotate(str(feature_gini_in_time['target'][i]),
                                 xy=(i, 1),
                                 xycoords=('data', 'axes fraction'),
                                 xytext=(0, 5),
                                 textcoords='offset pixels',
                                 #rotation=60,
                                 ha='center',
                                 #va='bottom',
                                 color='red',
                                 size=11
                                )

                # red is for the target rate values
                ax2.plot(range(feature_gini_in_time.shape[0]), feature_gini_in_time['gini'], 'bo-', linewidth=2.0, zorder=4)
                ax2.set_ylim(bottom=min(feature_gini_in_time['gini'].min(),0)-1)
                plt.suptitle(feature)
                plt.show()

        if out is not None:
            if out[-4:]=='.csv':
                pd.DataFrame(ginis, index=[0]).T.rename({0:'Gini'},axis=1).to_csv(out, sep = sep)
            elif out[-4:]=='.xls' or out[-5:]=='.xlsx':
                pd.DataFrame(ginis, index=[0]).T.rename({0:'Gini'},axis=1).to_excel(out, sheet_name='Gini')
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')

        if inplace:
            self.ginis = ginis
        else:
            return ginis



    def append(self, data, ignore_index=False, unite_features=False):
        '''
        Appends dataframe from input Data object to self

        Parameters:
        -----------
        data: input Data object to append to self
        ignore_index: should indexes from self.dataframe and data.dataframe be ignored or kept
        unite_features: should features' lists be united or self.features be kept

        Returns:
        -----------
        a Data object, made from appending data object to self
        '''
        if type(data)==Data:
            new_data = Data(self.dataframe.append(data.dataframe, ignore_index=ignore_index),  self.target, features=self.features, weights = self.weights)
            if unite_features:
                new_data.features=sorted(list(set(self.features+data.features)))
            return new_data
        else:
            print('Expecting a Data object, getting', str(type(data)))
            return None



    def features_exclude(self, to_exclude, verbose=True):
        '''
        Excludes a feature or a list of features from self.features

        Parameters
        ------------
        to_exclude: a feature name or a list of feature names
        verbose: should details be printed out

        '''
        if verbose:
            print('Excluding', to_exclude)
        if isinstance(to_exclude, list):
            self.features = [x for x in self.features if x not in to_exclude]
            self.ginis = {x:self.ginis[x] for x in self.ginis if x not in to_exclude}
        elif isinstance(to_exclude, list)==False:
            self.features = [x for x in self.features if x != to_exclude]
            self.ginis = {x:self.ginis[x] for x in self.ginis if x != to_exclude}



    def keep(self, features, keep_target_weights=True):
        '''
        Generates Data object, containing only chosen features and target+weights (if keep_target_weights=True)

        Parameters
        ------------
        features: a feature or a list of features to keep
        keep_target_weights: should target and weights be kept or not

        '''
        if isinstance(features, list):
            to_keep=features.copy()
        else:
            to_keep=[features]

        if keep_target_weights:
            if self.target is not None:
                to_keep.append(self.target)
            if self.weights is not None:
                to_keep.append(self.weights)
            to_keep=list(set(to_keep))

        return Data(self.dataframe[to_keep].copy(), self.target,
                    features=[x for x in to_keep if x not in [self.target, self.weights]],
                    weights=self.weights, name=self.name)
#----------------------------------------------------------------------------------------------------------------------




class DataSamples:
    '''
    Stores all Data type objects related to model process

    Parameters:
    -----------
    train: Data type object used for model training
    validate: Data type object used for model validating (variable stability checks and so on)
    test: Data type object used for model testing
    bootstrap_base: Data type object for use in bootstrap tests
    bootstrap: list of index objects acquired by Bootstrapper.work (for stability, significance and other checks)
    '''
    def __init__(self, train=None, validate=None, test=None, bootstrap_base=None, bootstrap=None):

        if bootstrap is None:
            bootstrap=[]

        self.train = train
        self.validate=validate
        self.test = test
        self.bootstrap_base = bootstrap_base
        self.bootstrap = bootstrap



    def stats(self, date_column=None, out=None, sep=';'):
        '''
        Calculates samples stats for train, validate and test

        Parameters:
        -----------
        date_column: the name of column with dates to calculate samples time period
        out: a path for csv/xlsx output file to export
        sep: the separator to be used in case of csv export

        Returns:
        -----------
        the dataframe with samples stats
        '''
        stats = pd.DataFrame(columns=['period', 'amount', 'target', 'non_target', 'target_rate']).T
        samples=[self.train, self.validate, self.test]
        for si in range(len(samples)):
            if samples[si] is not None:
                if date_column is not None:
                    dates=pd.to_datetime(samples[si].dataframe[date_column])
                    period=dates.min().strftime("%d.%m.%Y")+'-'+dates.max().strftime("%d.%m.%Y")
                else:
                    period=''
                if samples[si].weights is not None:
                    amount=samples[si].dataframe[samples[si].weights].sum()
                    target=samples[si].dataframe[samples[si].dataframe[samples[si].target]==1][samples[si].weights].sum()
                else:
                    amount=samples[si].dataframe.shape[0]
                    target=samples[si].dataframe[samples[si].target].sum()
                non_target=amount-target
                target_rate=target/amount
                stats=stats.join(pd.DataFrame(dict(period=period, amount=amount, target=target,
                                                   non_target=non_target, target_rate=target_rate), index=[samples[si].name]).T)

        if out is not None:
            if out[-4:]=='.csv':
                stats.rename({'period':'Временной период', 'amount':'Количество договоров', 'target':'Количество дефолтов', 'non_target':'Количество не-дефолтов',
                              'target_rate':'Уровень дефолтов'}).rename({'Train':'Обучающая выборка',
                                                                         'Validate':'Выборка для оценки стабильности',
                                                                         'Test':'Тестовая выборка'},axis=1).to_csv(out, sep = sep)
            elif out[-4:]=='.xls' or out[-5:]=='.xlsx':
                writer = pd.ExcelWriter(out, engine='xlsxwriter')
                stats.rename({'period':'Временной период', 'amount':'Количество договоров', 'target':'Количество дефолтов', 'non_target':'Количество не-дефолтов',
                              'target_rate':'Уровень дефолтов'}).rename({'Train':'Обучающая выборка',
                                                                         'Validate':'Выборка для оценки стабильности',
                                                                         'Test':'Тестовая выборка'}, axis=1).to_excel(writer, sheet_name='Samples stats')

                # Get the xlsxwriter objects from the dataframe writer object.
                workbook  = writer.book
                worksheet = writer.sheets['Samples stats']

                format1 = workbook.add_format({'num_format': '0.00%'})
                worksheet.set_row(5, None, format1)
                worksheet.set_column('A:C', 30)
                writer.save()
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')

        return stats



    def features_exclude(self, to_exclude, verbose=True):
        '''
        Excludes a feature or a list of features from existing samples' features

        Parameters
        ------------
        to_exclude: a feature name or a list of feature names
        verbose: should details be printed out

        '''

        #print('Excluding', to_exclude)
        samples=[self.train, self.validate, self.test, self.bootstrap_base]
        for i in range(len(samples)):
            if samples[i] is not None:
                if verbose:
                    print(samples[i].name+':')
                samples[i].features_exclude(to_exclude, verbose=verbose)



# edited by Anna Goreva on Nov-07-2018
    def samples_split(self, data, id_group = None, test_size = .3, with_validate=False, validate_size=.3, split_type = 'oos', seed=42,
             time_column = None, time_format = "%d.%m.%Y", time_func = (lambda x: 1000*x.year + x.month), stratify=True, check_missings = False):
        '''
        Splits data into train, test and validate parts, supports 'out-of-time' and 'out-of-sample' variants.
        New data objects are stored in self.

        Parameters
        -----------
        data: an object of Data type that should be processed
        id_group: string or None, name of column with ids; for each id, all the samples should belong only for one of train/test/validate
        test_size: parameter for sklearn.cross_validation.train_test_split meaning size of the test sample
        with_validate: should validate sample be formed or not
        validate_size: parameter for sklearn.cross_validation.train_test_split meaning size of the validate sample
        split_type: 'oos' = 'out-of-sample', 'oot' = 'out-of-time'
        seed: value of random_state for train_test_split
        time_column: name of the column that contains time data for out-of-time splitting
        time_format: format used for time feature parsing
        time_func: function used for time parsing
        stratify: should split be stratified by target if split_type='oos'
        check_missings: if True then checks whether all the features with missing values in validate and test have missings in train
        '''
        data = copy.deepcopy(data)

        if id_group is not None:
            # if wrong column name
            if id_group not in data.dataframe.columns:
                print ('Achtung bitte! Keine', id_group, 'Spalte!')
                return None
            # if oot
            elif split_type == 'oot':
                print ('Sorry, out-of-time splitting is not available for id-based way. Please set id_groups = None or split_type = "oos". Good luck.')
                return None

            # rename target classes for cases of domination of the max-numbered class
            old_indeces = list(data.dataframe[data.target].value_counts().index).copy()
            new_indeces = sorted(list(data.dataframe[data.target].value_counts().index), reverse=True).copy()
            data.dataframe[data.target + '_reverse'] = data.dataframe[data.target].apply(lambda x: new_indeces[old_indeces.index(x)])

            # grouping for stratification
            for_split = data.dataframe.groupby(by = id_group)[data.target + '_reverse'].max().copy()
        else:
            for_split = data.dataframe[data.target].copy()

        if split_type == 'oos':
            if with_validate:
                train, validate_test = train_test_split(for_split, test_size = test_size+validate_size, random_state = seed, stratify=for_split if stratify else None)
                validate, test = train_test_split(validate_test, test_size = test_size/(test_size+validate_size), random_state = seed, stratify=validate_test if stratify else None)
            else:
                train, test = train_test_split(for_split, test_size = test_size, random_state = seed, stratify=for_split if stratify else None)
            if id_group is not None:
                train_dataframe = data.dataframe[data.dataframe[id_group].apply(lambda x: x in train.index)].drop([data.target + '_reverse'], 1)
                test_dataframe = data.dataframe[data.dataframe[id_group].apply(lambda x: x in test.index)].drop([data.target + '_reverse'], 1)
                if with_validate:
                    validate_dataframe = data.dataframe[data.dataframe[id_group].apply(lambda x: x in validate.index)].drop([data.target + '_reverse'], 1)
            else:
                train_dataframe = data.dataframe.loc[list(train.index)]
                test_dataframe = data.dataframe.loc[list(test.index)]
                if with_validate:
                    validate_dataframe = data.dataframe.loc[list(validate.index)]

        elif split_type == 'oot':
            print ('Validation for oot is unavailable. Sorry. :(')
            if time_column == None:
                print ('Wich column contains time data? Please pay attention to time_column parameter. Bye.')
                return None
            else:
                tmp_dataset = copy.deepcopy(data.dataframe)
                tmp_dataset[time_column + '_t'] = pd.to_datetime(tmp_dataset[time_column], format=time_format, errors='coerce')
                tmp_dataset['tt'] = tmp_dataset[time_column + '_t'].map(time_func)
                tmp_dataset = tmp_dataset.sort_values(by='tt')
                test_threshold = list(tmp_dataset.tt.drop_duplicates())[int(round((1 - test_size)*len(tmp_dataset.tt.drop_duplicates()), 0))]
                test_dataframe = copy.deepcopy(tmp_dataset[tmp_dataset.tt >= test_threshold])
                train_dataframe = copy.deepcopy(tmp_dataset[tmp_dataset.tt < test_threshold])
                train_dataframe.drop([time_column + '_t', 'tt'], 1, inplace = True)
                test_dataframe.drop([time_column + '_t', 'tt'], 1, inplace = True)
        else:
            print ('Wrong split type. Please use oot or oos.')
            return None

        self.train=Data(train_dataframe.copy(), data.target, data.features, data.weights, name='Train')
        self.test=Data(test_dataframe.copy(), data.target, data.features, data.weights, name='Test')
        if with_validate and split_type == 'oos':
            self.validate=Data(validate_dataframe.copy(), data.target, data.features, data.weights, name='Validate')

        gc.collect()

        print ('Actual parts of samples:')
        if with_validate:
            print ('Train:', round(self.train.dataframe.shape[0]/data.dataframe.shape[0],4), '   Validation:', round(self.validate.dataframe.shape[0]/data.dataframe.shape[0],4),'   Test:', round(self.test.dataframe.shape[0]/data.dataframe.shape[0],4))
        else:
            print ('Train:', round(self.train.dataframe.shape[0]/data.dataframe.shape[0],4), '   Test:', round(self.test.dataframe.shape[0]/data.dataframe.shape[0],4))



    def bootstrap_split(self, data, bootstrap_part=0.75, bootstrap_number=10, stratify=True, replace=True, seed=0):
        '''
        Splits data into stratified/not stratified by target bootstrap samples

        Parameters
        -----------
        data: an object of Data type that should be processed
        bootstrap_part: the size of each bootstrap sample is defined as part of input data sample
        bootstrap_number: number of generated bootstrap samples
        stratify: should bootstraping be stratified by data target
        replace: is it acceptable to repeat rows from train dataframe for bootstrap samples
        seed: value of random_state for dataframe.sample (each random_state is calculated as seed + number in bootstrap)
        '''

        self.bootstrap_base=Data(data.dataframe.copy(), data.target, features=data.features, weights=data.weights, name='Bootstrap Base')
        bootstrap_samples=[]

        for bi in range(bootstrap_number):
            if stratify:
                index_1=self.bootstrap_base.dataframe[self.bootstrap_base.dataframe[self.bootstrap_base.target]==1].sample(frac=bootstrap_part,
                                       replace=replace, random_state=seed+bi).index
                index_0=self.bootstrap_base.dataframe[self.bootstrap_base.dataframe[self.bootstrap_base.target]==0].sample(frac=bootstrap_part,
                                       replace=replace, random_state=seed+bi).index
                bootstrap_current=[self.bootstrap_base.dataframe.index.get_loc(idx) for idx in index_1.append(index_0)]
            else:
                bootstrap_current=[self.bootstrap_base.dataframe.index.get_loc(idx) for idx in self.bootstrap_base.dataframe.sample(frac=bootstrap_part, replace=replace, random_state=seed+bi).index]
            bootstrap_samples.append(bootstrap_current)

        self.bootstrap=bootstrap_samples



    def get_bootstrap_sample(self, number):
        '''
        Tech
        Returns Data object, generated from the chosen bootstrap sample

        Parameters
        -----------
        number: a number (by order in DataSamples.bootstrap) of bootstrap sample to generate

        Returns
        -----------
        a Data object, generated from the chosen bootstrap sample

        '''
        return Data(self.bootstrap_base.dataframe.iloc[self.bootstrap[number]], target=self.bootstrap_base.target,
                    features=self.bootstrap_base.features, weights=self.bootstrap_base.weights, name='Bootstrap'+str(number))







