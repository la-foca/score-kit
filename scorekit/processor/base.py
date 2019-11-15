# -*- coding: utf-8 -*-


from .._utils import color_digits, color_background
from ..data import Data, DataSamples
#from ..woe import WOE
import pandas as pd
#import math as m
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
#rom scipy.stats import chi2, chisquare, ks_2samp, ttest_ind
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
import gc
#import weakref
import copy
import itertools
import calendar
#from ..cross import DecisionTree, Crosses
import networkx as nx
from operator import itemgetter

import matplotlib.ticker as mtick
try:
    import fastcluster
except Exception:
    print('For fullness analysis using hierarchical clustering please install fastcluster package.')
from scipy.cluster.hierarchy import fcluster
try:
    import hdbscan
except Exception:
    print('For fullness analysis using HDBSCAN clustering please install hdbscan package.')
from sklearn.cluster import KMeans
from sklearn.tree import export_graphviz
from os import system
from IPython.display import Image as Display_Image

#from joblib import Parallel, delayed

# Created by Anna Goreva and Dmitry Yudochev


warnings.simplefilter('ignore')

plt.rc('font', family='Verdana')
plt.style.use('seaborn-darkgrid')
pd.set_option('display.precision', 3)



class Processor(metaclass = ABCMeta):
    """
    Base class for processing objects of Data class
    """
    @abstractmethod
    def __init__(self):
        '''
        self.stats is a DataFrame with statistics about self.work()
        '''
        self.stats = pd.DataFrame()



    @abstractmethod
    def work(self, data, parameters):
        pass


    def param_dict_to_stats(self, data, d):
        '''
        TECH

        Transforms a dict of parameters to self.stats

        Parameters
        -----------
        data: Data object being processed
        d: dictionary {action : list_of_features} where action is a string with action description and list_of_features is a list of features' names to apply the action to
        '''
        col1 = []
        col2 = []
        for (how, features) in d.items():
            col1 = col1 + [how + ' (' + str(round(data.dataframe[features[i]].mean(), 3)) + ')' if how == 'mean' else how for i in range(len(features))]
            col2 = col2 + features
        self.stats = pd.DataFrame({'action' : col1, 'features': col2})
#---------------------------------------------------------------




class MissingProcessor(Processor):
    '''
    Class for missing values processing
    '''

    def __init__(self):
        self.stats = pd.DataFrame()

    def work(self, data, parameters, quantiles=100, precision=4):
        '''
        Deals with missing values

        Parameters:
        -----------
        data: an object of Data type that should be processed
        inplace: whether to change the data or to create a new Data object
        parameters: {how_to_process : features_to_process}
            how_to_process takes:
                'delete' - to delete samples where the value of any feature from features_to_process is missing
                'mean' - for each feature from features_to_process to fill missings with the mean value
                'distribution' - for each feature from features_to_process to fill missings according to non-missing distribution
                a value - for each feature from features_to_process to fill missings with this value
            features_to_process takes list of features from data
        quantiles: number of quantiles for 'distribution' type of missing process - all values are divided into quantiles,
            then missing values are filled with average values of quantiles. If number of unique values is less then number of quantiles
            or field type is not int, float, etc, then no quantiles are calculated - missings are filled with existing values according
            to their frequency
        precision: precision for quantile edges and average quantile values

        Returns:
        ----------
        A copy of data with missings processed for features mentioned in parameters

        '''
        for how in parameters:
            if isinstance(parameters[how], str):
                parameters[how] = [parameters[how]]


        result = data.dataframe.copy()
        for how in parameters:
            if how == 'delete':
                for feature in parameters[how]:
                    result = result[result[feature].isnull() == False]
                    if data.features != None and feature in data.features:
                        data.features.remove(feature)
            elif how == 'mean':
                for feature in parameters[how]:
                    result[feature].fillna(result[feature].mean(), inplace = True)
            elif how == 'distribution':
                for feature in parameters[how]:
                    if data.dataframe[feature].dtype not in (float, np.float32, np.float64, int, np.int32, np.int64) or data.dataframe[feature].unique().shape[0]<quantiles:
                        summarized=data.dataframe[[feature]].dropna().groupby(feature).size()
                        summarized=summarized.reset_index().rename({feature:'mean', 0:'size'}, axis=1)
                    else:
                        summarized=data.dataframe[[feature]].rename({feature:'value'}, axis=1).join(pd.qcut(data.dataframe[feature].dropna(), q=quantiles, precision=4, duplicates='drop')).groupby(feature).agg(['mean', 'size'])
                        summarized.columns=summarized.columns.droplevel()
                        summarized=summarized.reset_index(drop=True)
                    #summarized=summarized.reset_index()
                    summarized['p']=summarized['size']/summarized['size'].sum()
                    result[feature]=result[feature].apply(lambda x: np.random.choice(summarized['mean'].round(precision), p=summarized['p']) if pd.isnull(x) else x)
            else:
                result[parameters[how]] = result[parameters[how]].fillna(how)

        # statistics added on Dec-04-2018
        self.param_dict_to_stats(data, parameters)

        return Data(result, data.target, data.features, data.weights, data.name)
#---------------------------------------------------------------




class StabilityAnalyzer(Processor):
    '''
    For stability analysis
    '''

    def __init__(self):
        self.stats = pd.DataFrame({'sample_name' : [], 'parameter' : [], 'meaning': []})


    def work(self, data, time_column, sample_name = None, psi = None, event_rate=None, normalized=True, date_format = "%d.%m.%Y", time_func = (lambda x: 100*x.year + x.month),
             yellow_zone = 0.1, red_zone = 0.25, figsize = None, out = True, out_images = 'StabilityAnalyzer/', sep=';', base_period_index=0):
        '''
        Calculates the dynamic of feature (or groups of values) changes over time so it should be used only for discrete or WOE-transformed
        features. There are 2 types of analysis:

            PSI. Represents a heatmap (Stability Table) of features stability that contains 3 main zones: green (the feature is
            stable), yellow (the feature is not very stable) and red (the feature is unstable). StabilityIndex (PSI) is calculated for each
            time period relatively to the first period.
            Stability index algorithm:
                For each feature value and time period number of samples is calculated: e.g., N[i, t] is number of samples for value i and time period t.
                StabilityIndex[t] = (N[i, t]/sum_i(N[i, t]) - (N[i, 0]/sum_i(N[i, 0])))* log(N[i, t]/sum_i(N[i, t])/(N[i, 0]/sum_i(N[i, 0])))

            ER (event rate). Calculates average event rate and number of observations for each feature's value over time.

        After calculation displays the Stability Table (a heatmap with stability indexes for each feature value and time period)
        and Event rate graphs

        Parameters:
        -----------
        data: data to analyze (type Data)
        time_column: name of a column with time values to calculate time periods
        sample_name: name of sample for report
        psi: list of features for PSI analysis (if None then all features from the input Data object will be used)
        event_rate: list of features for event rate and distribution in time analysis  (if None then all features from the input Data object will be used)
        date_format: format of time values in time_column. Codes for format:
            %a	Weekday as locale’s abbreviated name.	                               Sun, Mon, …, Sat (en_US)
            %A	Weekday as locale’s full name.	                                      Sunday, Monday, …, Saturday (en_US)
            %w	Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.    0, 1, …, 6
            %d	Day of the month as a zero-padded decimal number. 	                 01, 02, …, 31
            %b	Month as locale’s abbreviated name.	                               Jan, Feb, …, Dec (en_US)
            %B	Month as locale’s full name.	                                      January, February, …, December (en_US)
            %m	Month as a zero-padded decimal number. 	                            01, 02, …, 12
            %y	Year without century as a zero-padded decimal number. 	              00, 01, …, 99
            %Y	Year with century as a decimal number. 	                            1970, 1988, 2001, 2013
            %H	Hour (24-hour clock) as a zero-padded decimal number. 	              00, 01, …, 23
            %I	Hour (12-hour clock) as a zero-padded decimal number. 	              01, 02, …, 12
            %p	Locale’s equivalent of either AM or PM.	                            AM, PM (en_US)
            %M	Minute as a zero-padded decimal number.	                            00, 01, …, 59
            %S	Second as a zero-padded decimal number. 	                            00, 01, …, 59
            %f	Microsecond as a decimal number, zero-padded on the left.    	       000000, 000001, …, 999999
            %z	UTC offset in the form +HHMM or -HHMM (empty string if the the
                 object is naive). 	                                                (empty), +0000, -0400, +1030
            %Z	Time zone name (empty string if the object is naive). 	              (empty), UTC, EST, CST
            %j	Day of the year as a zero-padded decimal number. 	                 001, 002, …, 366
            %U	Week number of the year (Sunday as the first day of the week)
                 as a zero padded decimal number. All days in a new year preceding
                 the first Sunday are considered to be in week 0. 	                 00, 01, …, 53 (6)
            %W	Week number of the year (Monday as the first day of the week) as
                 a decimal number. All days in a new year preceding the first
                 Monday are considered to be in week 0. 	                            00, 01, …, 53 (6)
            %c	Locale’s appropriate date and time representation.	                  Tue Aug 16 21:30:00 1988 (en_US)
            %x	Locale’s appropriate date representation.	                            08/16/88 (None); 08/16/1988 (en_US)
            %X	Locale’s appropriate time representation.	                            21:30:00 (en_US)
        time_func: function for time_column parsing (changes date to some value, representing time period) or
                a period type for dt.to_period() function. Codes for available periods:
            B       business day frequency
            C       custom business day frequency (experimental)
            D       calendar day frequency
            W       weekly frequency
            M       month end frequency
            BM      business month end frequency
            CBM     custom business month end frequency
            MS      month start frequency
            BMS     business month start frequency
            CBMS    custom business month start frequency
            Q       quarter end frequency
            BQ      business quarter endfrequency
            QS      quarter start frequency
            BQS     business quarter start frequency
            A       year end frequency
            BA      business year end frequency
            AS      year start frequency
            BAS     business year start frequency
            BH      business hour frequency
            H       hourly frequency
            T, min  minutely frequency
            S       secondly frequency
            L, ms   milliseconds
            U, us   microseconds
            N       nanoseconds
        yellow_zone: the lower border for the yellow stability zone ('not very stable') in percents of derivation
        red_zone: the lower border for the red stability zone ('unstable') in percents of derivation
        figsize: matplotlib figsize of the Stability Table
        out: a boolean for image output or a path for xlsx output file to export the Stability Tables
        out_images: a path for image output (default - StabilityAnalyzer/)
        sep: the separator to be used in case of csv export
        base_period_index: index of period (starting from 0) for other periods to compare with (0 for the first, -1 for the last)
        '''

        print('Warning: only for discrete features!!!')

        if sample_name is None:
            if pd.isnull(data.name):
                sample_name = 'sample'
            else:
                sample_name = data.name

        out_images = out_images + sample_name + '/'
        self.stats = self.stats.append(pd.DataFrame({'sample_name' : [sample_name], 'parameter' : ['out'], 'meaning' : [out]}))
        self.stats = self.stats.append(pd.DataFrame({'sample_name' : [sample_name], 'parameter' : ['out_images'], 'meaning' : [out_images]}))



        psi = data.features.copy() if psi is None else [x for x in psi if x in data.features]
        event_rate = data.features.copy() if event_rate is None else [x for x in event_rate if x in data.features]

        all_features=list(set(psi+event_rate))

        if figsize is None:
            figsize=(12, max(1,round(len(psi)/2,0)))

        if out==True or isinstance(out, str):
            directory = os.path.dirname(out_images)
            if not os.path.exists(directory):
                os.makedirs(directory)

        if isinstance(out, str):
            writer = pd.ExcelWriter(out, engine='openpyxl')

        tmp_dataset = data.dataframe[all_features + [time_column, data.target] + ([] if data.weights is None else [data.weights])].copy()
        tmp_dataset[time_column] = pd.to_datetime(tmp_dataset[time_column], format=date_format, errors='coerce')
        if callable(time_func):
            tmp_dataset['tt'] = tmp_dataset[time_column].map(time_func)
        elif isinstance(time_func, str):
            try:
                tmp_dataset['tt'] = tmp_dataset[time_column].dt.to_period(time_func).astype(str)
            except Exception:
                print('No function or correct period code was provided. Return None.')
                return None
        c = 0
        for feature in sorted(all_features):
            print (feature)
            if data.weights is not None:
                feature_stats=tmp_dataset[[feature, 'tt', data.target, data.weights]]
                feature_stats['---weight---']=feature_stats[data.weights]
            else:
                feature_stats=tmp_dataset[[feature, 'tt', data.target]]
                feature_stats['---weight---']=1
            feature_stats[data.target]=feature_stats[data.target]*feature_stats['---weight---']
                                
            feature_stats=feature_stats[[feature, 'tt', data.target, '---weight---']].groupby([feature, 'tt'], as_index=False).\
                                agg({'---weight---':'size', data.target:'mean'}).rename({feature:'value', '---weight---':'size', data.target:'mean'}, axis=1)
            feature_stats['feature']=feature
            if c == 0:
                all_stats = feature_stats
                c = c+1
            else:
                all_stats = all_stats.append(feature_stats, ignore_index=True)

        all_stats['size']=all_stats['size'].astype(float)
        all_stats['mean']=all_stats['mean'].astype(float)

        if len(psi)>0:
            stability1=all_stats[all_stats.feature.isin(psi)][['feature', 'value', 'tt', 'size']].pivot_table(values='size', columns='tt', index=['feature', 'value']).reset_index().fillna(0)
            stability1.columns.name=None
            #display(stability1)
            dates = stability1.drop(['feature', 'value'], 1).columns.copy()
            stability2 = stability1[['feature', 'value']].copy()
            for date in dates:
                stability2[date] = list(stability1[date]/list(stability1.drop(['value'], 1).groupby(by = 'feature').sum()[date][:1])[0])
            #display(stability2)

            start_date = dates[base_period_index]
            stability3 = stability2[['feature', 'value']]
            for date in dates:
                stability3[date] = round(((stability2[date]-stability2[start_date])*np.log(stability2[date]/stability2[start_date])).fillna(0), 2).replace([])
            #display(stability3)

            stability4 = stability3.drop(['value'], 1).groupby(by = 'feature').sum()
            #display(stability4)
            fig, ax = plt.subplots(figsize = figsize)
            ax.set_facecolor("red")
            sns.heatmap(stability4, ax=ax, yticklabels=stability4.index, annot = True, cmap = 'RdYlGn_r', center = yellow_zone, vmax = red_zone, linewidths = .05, xticklabels = True)
            if out==True or isinstance(out, str):
                plt.savefig(out_images+"stability.png", dpi=100, bbox_inches='tight')
            plt.show()
            if isinstance(out, str):
                if out[-4:]=='.xls' or out[-5:]=='.xlsx':
                    stability4.style.apply(color_background,
                                           mn=0, mx=red_zone, cntr=yellow_zone).to_excel(writer, engine='openpyxl', sheet_name='PSI')
                    worksheet = writer.sheets['PSI']
                    for x in worksheet.columns:
                        if x[0].column=='A':
                            worksheet.column_dimensions[x[0].column].width = 40
                        else:
                            worksheet.column_dimensions[x[0].column].width = 12
                    worksheet.freeze_panes = worksheet['B2']
                else:
                    print('Unknown or unacceptable format for export several tables. Use .xlsx. Skipping export.')

        if len(event_rate)>0:
            for_event_rate=all_stats[all_stats['feature'].isin(event_rate)]
            date_base=pd.DataFrame(all_stats['tt'].unique(), columns=['tt']).sort_values('tt')

            for feature in sorted(for_event_rate['feature'].unique()):
                cur_feature_data=for_event_rate[for_event_rate['feature']==feature].copy()
                #display(cur_feature_data)
                if normalized:
                    for tt in sorted(cur_feature_data['tt'].unique(), reverse=True):
                        cur_feature_data.loc[cur_feature_data['tt']==tt, 'percent']=cur_feature_data[cur_feature_data['tt']==tt]['size']/cur_feature_data[cur_feature_data['tt']==tt]['size'].sum()
                #display(cur_feature_data)
                fig, ax = plt.subplots(1,1, figsize=(15, 5))
                ax2 = ax.twinx()
                ax.grid(False)
                ax2.grid(False)
                
                sorted_values=sorted(cur_feature_data['value'].unique(), reverse=True)
                for value in sorted_values:
                    to_visualize='percent' if normalized else 'size'
                    value_filter = (cur_feature_data['value']==value)
                    er=date_base.merge(cur_feature_data[value_filter], on='tt', how='left')['mean']
                    height=date_base.merge(cur_feature_data[value_filter], on='tt', how='left')[to_visualize].fillna(0)
                    bottom=date_base.merge(cur_feature_data[['tt',to_visualize]][cur_feature_data['value']>value].groupby('tt', as_index=False).sum(), on='tt', how='left')[to_visualize].fillna(0)

                    ax.bar(range(date_base.shape[0]), height, bottom=bottom if value!=sorted_values[0] else None, edgecolor='white', alpha=0.3)
                    ax2.plot(range(date_base.shape[0]), er, label=str(round(value,3)), linewidth=2)
                plt.xticks(range(date_base.shape[0]), date_base['tt'])
                fig.autofmt_xdate()

                ax2.set_ylabel('Event Rate')
                ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
                if normalized:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
                    ax2.annotate('Obs:', xy=(0, 1), xycoords=('axes fraction', 'axes fraction'), xytext=(-25, 5), textcoords='offset pixels', color='blue', size=11)
                    for i in range(date_base.shape[0]):
                        ax2.annotate(str(int(cur_feature_data[cur_feature_data['tt']==date_base['tt'][i]]['size'].sum())),
                                 xy=(i, 1),
                                 xycoords=('data', 'axes fraction'),
                                 xytext=(0, 5),
                                 textcoords='offset pixels',
                                 #rotation=60,
                                 ha='center',
                                 #va='bottom',
                                 color='blue',
                                 size=11)
                ax.set_ylabel('Total obs')
                plt.xlabel(time_column)
                plt.suptitle(feature + ' event rate in time' if callable(time_func) else feature + ' event rate in time, period = '+time_func)

                handles, labels = ax2.get_legend_handles_labels()
                ax2.legend(handles[::-1], labels[::-1], loc=0, fancybox=True, framealpha=0.3)
                if out==True or isinstance(out, str):
                    plt.savefig(out_images+feature+".png", dpi=100, bbox_inches='tight')
                plt.show()

            if isinstance(out, str):
                if out[-4:]=='.xls' or out[-5:]=='.xlsx':
                    event_rate_df=all_stats[['feature', 'value', 'tt', 'mean']].pivot_table(values='mean', columns='tt', index=['feature', 'value']).reset_index().fillna(0)
                    event_rate_df.columns.name=None
                    event_rate_df.style.apply(color_background,
                                              mn=0, mx=all_stats['mean'].mean()+2*all_stats['mean'].std(), cntr=None,
                                              cmap=matplotlib.cm.RdYlGn_r, subset=pd.IndexSlice[:, [x for x in event_rate_df.columns if x not in ['value','feature']]]).to_excel(writer, engine='openpyxl', sheet_name='Event Rate', index=False)
                    worksheet = writer.sheets['Event Rate']
                    for x in worksheet.columns:
                        if x[0].column=='A':
                            worksheet.column_dimensions[x[0].column].width = 40
                        else:
                            worksheet.column_dimensions[x[0].column].width = 12
                            if x[0].column!='B':
                                for cell in worksheet[x[0].column]:
                                    if cell.row!=1:
                                        cell.number_format = '0.000%'
                    worksheet.freeze_panes = worksheet['C2']

                    size_df=all_stats[['feature', 'value', 'tt', 'size']].pivot_table(values='size', columns='tt', index=['feature', 'value']).reset_index().fillna(0)
                    size_df.columns.name=None
                    size_df.style.apply(color_background,
                                              mn=0, mx=all_stats['size'].mean()+2*all_stats['size'].std(), cntr=None,
                                              cmap=matplotlib.cm.RdYlGn, subset=pd.IndexSlice[:, [x for x in size_df.columns if x not in ['value','feature']]]).to_excel(writer, engine='openpyxl', sheet_name='Observations', index=False)
                    worksheet = writer.sheets['Observations']
                    for x in worksheet.columns:
                        if x[0].column=='A':
                            worksheet.column_dimensions[x[0].column].width = 40
                        else:
                            worksheet.column_dimensions[x[0].column].width = 12
                    worksheet.freeze_panes = worksheet['C2']
                else:
                    print('Unknown or unacceptable format for export several tables. Use .xlsx. Skipping export.')
        if isinstance(out, str):
            writer.close()

#---------------------------------------------------------------




class DataVisualizer(Processor):
    '''
    Supports different types of data visualization
    '''
    def __init__(self):
        self.stats = pd.DataFrame()

    def work(self, data, distribution = True, factorplot = True, factorplot_separate = False, pairplot = None,
             out=False, out_images='DataVisualizer/', plot_cells=20, categorical=None):
        '''
        Produces distribution plot, factorplot, pairplot

        Parameters:
        -----------
        data: data to visualize
        distribution: parameter for a distribution plot,
            if True - plot for data.features, if list - plot for features from the list, if False - do not use distribution plot
        factorplot: parameter for a factorplot,
            if True - plot for data.features, if list - plot for features from the list, if False - do not use factorplot
        factorplot_separate: if True then separate plots for each target value
        pairplot: list of features to make a pairplot for
        out: a boolean for images output or a path for xlsx output file
        out_images: a path for images output (default - DataVisualizer/)
        plot_cells: how many cells would plots get in output excel
        categorical: a list of features to be treated as categorical (countplots will be produced instead of distplots)
        '''

        if pairplot is None:
            pairplot=[]

        if categorical is None:
            categorical=[]

        dataframe_t = data.dataframe[data.features + [data.target]].copy()
        data = Data(dataframe_t, features = data.features, target = data.target)

        if out is not None:
            if out==True or isinstance(out, str):
                directory = os.path.dirname(out_images)
                if not os.path.exists(directory):
                    os.makedirs(directory)
            if isinstance(out, str):
                # Create an new Excel file and add a worksheet.
                workbook = xlsxwriter.Workbook(out)
                worksheet = workbook.add_worksheet('Data Visualization')

                # Widen the first column to make the text clearer.
                worksheet.set_column('A:A', 100)

        current_plot_number=0

        if distribution:
            print ('Distributions of features: ')
            if type(distribution) == type([1, 1]):
                features = distribution
            else:
                if data.features == None:
                    print ('No features claimed. Please set data.features = ')
                    return None
                features = data.features

            for feature in features:
                current_plot_number=current_plot_number+1
                if data.dataframe[feature].dtype==object or feature in categorical:
                    f, axes = plt.subplots()
                    sns.countplot(data.dataframe[feature].dropna())
                    f.autofmt_xdate()
                else:
                    sns.distplot(data.dataframe[feature].dropna())
                if data.dataframe[feature].isnull().any():
                    plt.title(feature+' (miss = ' + str(round(data.dataframe[feature].isnull().value_counts()[True]/data.dataframe.shape[0],3))+')')
                else:
                    plt.title(feature+' (miss = 0)')
                if out==True or isinstance(out, str):
                    plt.savefig(out_images+feature+"_d.png", dpi=100, bbox_inches='tight')
                    if isinstance(out, str):
                        scale=(20*plot_cells)/Image.open(out_images+feature+"_d.png").size[1]
                        worksheet.write((current_plot_number-1)*(plot_cells+1), 0, 'Distribution plot for '+feature+":")
                        worksheet.insert_image((current_plot_number-1)*(plot_cells+1)+1, 0, out_images+feature+"_d.png",
                                               {'x_scale': scale, 'y_scale': scale})
                plt.show()
            print ('---------------------------------------\n')


        if factorplot:
            print ('Factorplot: ')
            if type(factorplot) == type([1, 1]):
                features = factorplot
            else:
                if data.features == None:
                    print ('No features claimed. Please set data.features = ')
                    return None
                features = data.features

            if factorplot_separate:
                for feature in features:
                    current_plot_number=current_plot_number+1
                    # edited 21-Jun-2018 by Anna Goreva
                    f, axes = plt.subplots(data.dataframe[data.target].drop_duplicates().shape[0], 1, figsize=(4, 4), sharex=True)
                    f.autofmt_xdate()
                    #for target_v in data.dataframe[data.target].drop_duplicates():
                    targets = list(data.dataframe[data.target].drop_duplicates())
                    for target_i in range(len(targets)):
                        if data.dataframe[data.dataframe[data.target]==targets[target_i]][feature].isnull().any():
                            x_label=feature + ': ' + data.target + ' = ' + str(targets[target_i]) + ', miss = ' + str(round(data.dataframe[data.dataframe[data.target]==targets[target_i]][feature].isnull().value_counts()[True]/data.dataframe[data.dataframe[data.target]==targets[target_i]].shape[0],3))
                        else:
                            x_label=feature + ': ' + data.target + ' = ' + str(targets[target_i]) + ', miss = 0'
                        if data.dataframe[feature].dtype==object or feature in categorical:
                            ax=sns.countplot(data.dataframe[data.dataframe[data.target] == targets[target_i]][feature].dropna(),
                                         ax=axes[target_i], color = 'm')
                            ax.set(xlabel=x_label)
                        else:
                            sns.distplot(data.dataframe[data.dataframe[data.target] == targets[target_i]][feature].dropna(),
                                         ax=axes[target_i],
                                         axlabel=x_label, color = 'm')
                    if out==True or isinstance(out, str):
                        plt.savefig(out_images+feature+"_f.png", dpi=100, bbox_inches='tight')
                        if isinstance(out, str):
                            scale=(20*plot_cells)/Image.open(out_images+feature+"_f.png").size[1]
                            worksheet.write((current_plot_number-1)*(plot_cells+1), 0, 'Factor plot for '+feature+":")
                            worksheet.insert_image((current_plot_number-1)*(plot_cells+1)+1, 0, out_images+feature+"_f.png",
                                                   {'x_scale': scale, 'y_scale': scale})
                    plt.show()
            else:
                for feature in features:
                    current_plot_number=current_plot_number+1
                    sns.factorplot(x=feature, hue = data.target, data = data.dataframe, kind='count', palette = 'Set1')
                    if out==True or isinstance(out, str):
                        plt.savefig(out_images+feature+"_f.png", dpi=100, bbox_inches='tight')
                        if isinstance(out, str):
                            scale=(20*plot_cells)/Image.open(out_images+feature+"_f.png").size[1]
                            worksheet.write((current_plot_number-1)*(plot_cells+1), 0, 'Factor plot for '+feature+":")
                            worksheet.insert_image((current_plot_number-1)*(plot_cells+1)+1, 0, out_images+feature+"_f.png",
                                                   {'x_scale': scale, 'y_scale': scale})
                    plt.show()
            print ('---------------------------------------\n')


        if pairplot != []:
            current_plot_number=current_plot_number+1
            print ('Pairplot')
            sns.pairplot(data.dataframe[pairplot].dropna())
            if out==True or isinstance(out, str):
                plt.savefig(out_images+"pairplot.png", dpi=100, bbox_inches='tight')
                if isinstance(out, str):
                    worksheet.write((current_plot_number-1)*(plot_cells+1), 0, 'Pair plot for '+str(pairplot)+":")
                    worksheet.insert_image((current_plot_number-1)*(plot_cells+1)+1, 0, out_images+"pairplot.png")
            plt.show()

        if isinstance(out, str):
            workbook.close()
#---------------------------------------------------------------




class TargetTrendVisualizer(Processor):
    '''
    Supports target trend visualization
    '''
    def __init__(self):
        self.stats = pd.DataFrame()

    def work(self, data, features=None, quantiles=100, magnify_trend=False, magnify_std_number=2, hide_every_even_tick_from=50,
             min_size=10, out=False, out_images='TargetTrendVisualizer/', plot_cells=20):
        '''
        Calculates specified quantiles/takes categories, calculates target rates and sizes, then draws target trends

        Parameters:
        -----------
        data: an object of Data type
        features: the list of features to visualize, can be omitted
        quantiles: number of quantiles to cut feature values on
        magnify_trend: if True, then axis scale for target rate will be corrected to exclude outliers
        magnify_std_number: how many standard deviations should be included in magnified scale
        hide_every_even_tick_from: if there is too many quantiles then every second tick on x axis will be hidden
        out: a boolean for images output or a path for xlsx output file
        out_images: a path for images output (default - TargetTrendVisualizer/)
        plot_cells: how many cells would plots get in output excel
        '''
        if features is None:
            cycle_features=data.features.copy()
        else:
            cycle_features=features.copy()

        if out is not None:
            if out==True or isinstance(out, str):
                directory = os.path.dirname(out_images)
                if not os.path.exists(directory):
                    os.makedirs(directory)
            if isinstance(out, str):
                # Create an new Excel file and add a worksheet.
                workbook = xlsxwriter.Workbook(out)
                worksheet = workbook.add_worksheet('Target Trend Visualization')

                # Widen the first column to make the text clearer.
                worksheet.set_column('A:A', 100)

        current_feature_number=0
        for f in cycle_features:
            if f not in data.dataframe:
                print('Feature', f, 'not in input dataframe. Skipping..')
            else:
                print('Processing', f,'..')
                current_feature_number=current_feature_number+1

                if data.dataframe[f].dtype not in (float, np.float32, np.float64, int, np.int32, np.int64) or data.dataframe[f].unique().shape[0]<quantiles:
                    summarized=data.dataframe[[f, data.target]].groupby([f]).agg(['mean', 'size'])
                else:
                    if data.dataframe[f].dropna().shape[0]<min_size*quantiles:
                        current_quantiles=int(data.dataframe[f].dropna().shape[0]/min_size)
                        if current_quantiles==0:
                            print('The number of non-missing observations is less then', min_size,'. No trend to visualize.')
                            if isinstance(out, str):
                                worksheet.write((current_feature_number-1)*(plot_cells+1), 0, 'Target trend for '+f+":")
                                worksheet.write((current_feature_number-1)*(plot_cells+1)+1, 0, 'The number of non-missing observations is less then '+str(min_size)+'. No trend to visualize.')
                            continue
                        else:
                            print('Too few non-missing observations for', quantiles, 'quantiles. Calculating', current_quantiles, 'quantiles..')
                    else:
                        current_quantiles=quantiles
                    summarized=data.dataframe[[data.target]].join(pd.qcut(data.dataframe[f], q=current_quantiles, precision=4, duplicates='drop')).groupby([f]).agg(['mean', 'size'])
                    small_quantiles=summarized[data.target][summarized[data.target]['size']<min_size]['size']
                    #display(small_quantiles)
                    if small_quantiles.shape[0]>0:
                        current_quantiles=int(small_quantiles.sum()/min_size)+summarized[data.target][summarized[data.target]['size']>=min_size].shape[0]
                        print('There are quantiles with size less then', min_size,'. Attempting', current_quantiles, 'quantiles..')
                        summarized=data.dataframe[[data.target]].join(pd.qcut(data.dataframe[f], q=current_quantiles, precision=4, duplicates='drop')).groupby([f]).agg(['mean', 'size'])

                summarized.columns=summarized.columns.droplevel()
                summarized=summarized.reset_index()
                if pd.isnull(data.dataframe[f]).any():
                    with_na=data.dataframe[[f,data.target]][pd.isnull(data.dataframe[f])]
                    summarized.loc[-1]=[np.nan, with_na[data.target].mean(), with_na.shape[0]]
                    summarized=summarized.sort_index().reset_index(drop=True)
                if summarized.shape[0]==1:
                    print('Too many observations in one value, so only 1 quantile was created. Increasing quantile number is recommended. No trend to visualize.')
                    if isinstance(out, str):
                        worksheet.write((current_feature_number-1)*(plot_cells+1), 0, 'Target trend for '+f+":")
                        worksheet.write((current_feature_number-1)*(plot_cells+1)+1, 0, 'Too many observations in one value, so only 1 quantile was created. Increasing quantile number is recommended. No trend to visualize.')
                    continue

                fig = plt.figure(figsize=(15,5))
                ax = fig.add_subplot(111)
                ax.set_ylabel('Observations')
                # blue is for the distribution
                if summarized.shape[0]>hide_every_even_tick_from:
                    plt.xticks(range(summarized.shape[0]), summarized[f].astype(str), rotation=60, ha="right")
                    xticks = ax.xaxis.get_major_ticks()
                    for i in range(len(xticks)):
                        if i%2==0:
                            xticks[i].label1.set_visible(False)
                else:
                    plt.xticks(range(summarized.shape[0]), summarized[f].astype(str), rotation=45, ha="right")

                ax.bar(range(summarized.shape[0]), summarized['size'], zorder=0, alpha=0.3)
                ax.grid(False)
                ax.grid(axis='y', zorder=1, alpha=0.6)
                ax2 = ax.twinx()
                ax2.set_ylabel('Target Rate')
                ax2.grid(False)
                #display(summarized)

                if magnify_trend:
                    ax2.set_ylim([0, np.average(summarized['mean'], weights=summarized['size'])+magnify_std_number*np.sqrt(np.cov(summarized['mean'], aweights=summarized['size']))])
                    for i in range(len(summarized['mean'])):
                        if summarized['mean'][i]>np.average(summarized['mean'], weights=summarized['size'])+magnify_std_number*np.sqrt(np.cov(summarized['mean'], aweights=summarized['size'])):
                            ax2.annotate(str(round(summarized['mean'][i],4)),
                                         xy=(i, np.average(summarized['mean'], weights=summarized['size'])+magnify_std_number*np.sqrt(np.cov(summarized['mean'], aweights=summarized['size']))),
                                         xytext=(i, np.average(summarized['mean'], weights=summarized['size'])+(magnify_std_number+0.05)*np.sqrt(np.cov(summarized['mean'], aweights=summarized['size']))),
                                         rotation=60,
                                         ha='left',
                                         va='bottom',
                                         color='red',
                                         size=8.5
                                        )
                # red is for the target rate values
                ax2.plot(range(summarized.shape[0]), summarized['mean'], 'ro-', linewidth=2.0, zorder=4)
                if out==True or isinstance(out, str):
                    plt.savefig(out_images+f+".png", dpi=100, bbox_inches='tight')
                    if isinstance(out, str):
                        scale=(20*plot_cells)/Image.open(out_images+f+".png").size[1]
                        worksheet.write((current_feature_number-1)*(plot_cells+1), 0, 'Target trend for '+f+":")
                        worksheet.insert_image((current_feature_number-1)*(plot_cells+1)+1, 0, out_images+f+".png",
                                               {'x_scale': scale, 'y_scale': scale})
                plt.show()

        if isinstance(out, str):
            workbook.close()



class CorrelationAnalyzer(Processor):
    '''
    Produces correlation analysis
    '''
    def __init__(self):
        self.stats = pd.DataFrame()


    def work(self, data, drop_features = True, features = None, features_to_leave = None, threshold=0.6, method = 'spearman',
             drop_with_most_correlations=True, verbose=False, out_before=None, out_after=None, sep=';', cdict = None):
        '''
        Calculates the covariance matrix and correlation coefficients for each pair of features.
        For each highly correlated pair the algorithm chooses the less significant feature and adds it to the delete list.

        Parameters
        -----------
        data: a Data or DataSamples object to check (in case of DataSamples, train sample will be checked)
        drop_features: permission to delete correlated features and return a Data object without them
        features: a list of features to analyze; by default - all the features
        features_to_leave: a list of features that must not be deleted from the feature list
        threshold: the lowest value of a correlation coefficient for two features to be considered correlated
        method: method for correlation calculation
        drop_with_most_correlations: should the features with the highest number of correlations be excluded first (otherwise just with any number of correlations and the lowest gini)
        verbose: flag for detailed output
        out_before: file name for export of correlation table before feature exclusion (.csv and .xlsx types are supported)
        out_after: file name for export of correlation table after feature exclusion (.csv and .xlsx types are supported)
        sep: the separator in case of .csv export

        Returns
        --------
        Resulting Data or DataSamples object and the correlation table
        '''

        if features is None:
            features=[]
        if features_to_leave is None:
            features_to_leave=[]

        self.stats = pd.DataFrame({'drop_features' : [drop_features], 'threshold' : [threshold], 'method' : [method], 'out_before' : out_before, 'out_after' : out_after})

        if type(data)==DataSamples:
            sample=data.train
        else:
            sample=data

        if len(sample.ginis)==0:
            print('No calculated ginis in datasamples.train/data object. Set calc_gini=True while using WOE.transform or use Data.calc_gini. Return None')
            return None

        if features == [] or features is None:
            candidates = sample.features.copy()
        else:
            candidates = features.copy()

        features_to_drop = []

        correlations = sample.dataframe[candidates].corr(method = method)
        cor_out=correlations.copy()
        if cdict is None:
            cdict = {'red' : ((0.0, 0.9, 0.9),
                              (0.5, 0.05, 0.05),
                              (1.0, 0.9, 0.9)),
                    'green': ((0.0, 0.0, 0.0),
                              (0.5, 0.8, 0.8),
                              (1.0, 0.0, 0.0)),
                     'blue' : ((0.0, 0.1, 0.1),
                          (0.5, 0.1, 0.1),
                          (1.0, 0.1, 0.1))}



        #edited 21.08.2018 by Yudochev Dmitry - added verbose variant, optimized feature dropping
        # edited on Dec-06-18 by Anna Goreva: added png
        draw_corr=correlations.copy()
        draw_corr.index=[x+' (%i)' % i for i,x in enumerate(draw_corr.index)]
        draw_corr.columns=range(len(draw_corr.columns))


        if out_before is not None:
            out_before_png = 'corr_before.png'
            if out_before[-4:]=='.csv':
                draw_corr.round(2).to_csv(out_before, sep = sep)
                out_before_png = out_before[:-4] + '.png'
            elif out_before[-5:]=='.xlsx' or out_before[-4:]=='.xls':
                draw_corr.round(2).style.applymap(color_digits, threshold_red=threshold, threshold_yellow=threshold**2).to_excel(out_before, engine='openpyxl', sheet_name='Correlation (before)')
                out_before_png = out_before[:-5]  + '.png' if out_before[-5:]=='.xlsx' else out_before[:-4] + '.png'
            elif out_before[-4:]=='.png':
                out_before_png = out_before
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')

            fig_before = sns.heatmap(draw_corr.round(2), annot = True, cmap = LinearSegmentedColormap('mycmap', cdict), cbar = False, center = 0, yticklabels = True,  xticklabels = True).figure
            fig_before.set_size_inches(draw_corr.shape[0]/2, draw_corr.shape[0]/2)
            fig_before.savefig(out_before_png, bbox_inches='tight')

            plt.close()
            self.stats['out_before'] = out_before_png
        if verbose:
            display(draw_corr.round(2).style.applymap(color_digits, threshold_red=threshold, threshold_yellow=threshold**2))


        to_check_correlation=True

        while to_check_correlation:
            to_check_correlation=False
            corr_number={}
            significantly_correlated={}
            for var in correlations:
                var_corr=correlations[var].apply(lambda x: abs(x))
                var_corr=var_corr[(var_corr.index!=var) & (var_corr>threshold)].sort_values(ascending=False).copy()
                corr_number[var]=var_corr.shape[0]
                significantly_correlated[var]=str(var_corr.index.tolist())
            if drop_with_most_correlations:
                with_correlation={x:sample.ginis[x] for x in corr_number if corr_number[x]==max({x:corr_number[x] for x in corr_number if x not in features_to_leave}.values()) and corr_number[x]>0 and x not in features_to_leave}
            else:
                with_correlation={x:sample.ginis[x] for x in corr_number if corr_number[x]>0 and x not in features_to_leave}
            if len(with_correlation)>0:
                feature_to_drop=min(with_correlation, key=with_correlation.get)
                features_to_drop.append(feature_to_drop)
                if verbose:
                    print('Dropping %(v)s because of high correlation with features: %(f)s (Gini=%(g)0.2f)' % {'v':feature_to_drop, 'f':significantly_correlated[feature_to_drop], 'g':with_correlation[feature_to_drop]})
                correlations=correlations.drop(feature_to_drop,axis=1).drop(feature_to_drop,axis=0).copy()
                to_check_correlation=True

        draw_corr=correlations.copy()
        draw_corr.index=[x+' (%i)' % i for i,x in enumerate(draw_corr.index)]
        draw_corr.columns=range(len(draw_corr.columns))
        out_after_png = 'corr_after.png'


        if out_after is not None:
            if out_after[-4:]=='.csv':
                draw_corr.round(2).to_csv(out_after, sep = sep)
                out_after_png = out_after[:-4] + '.png'
            elif out_after[-5:]=='.xlsx' or out_after[-4:]=='.xls':
                draw_corr.round(2).style.applymap(color_digits, threshold_red=threshold, threshold_yellow=threshold**2).to_excel(out_after, engine='openpyxl', sheet_name='Correlation (after)')
                out_after_png = out_after[:-5] + '.png' if out_after[-5:]=='.xlsx' else out_after[:-4] + '.png'
            elif out_after[-4:]=='.png':
                out_after_png = out_after
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')
        #sns.heatmap(draw_corr.round(2), annot = True, cmap = 'RdBu_r', cbar = False, center = 0).figure.savefig(out_after_png, bbox_inches='tight')
        fig_after = sns.heatmap(draw_corr.round(2), annot = True, cmap = LinearSegmentedColormap('mycmap', cdict), cbar = False, center = 0, yticklabels = True,  xticklabels = True).figure
        fig_after.set_size_inches(draw_corr.shape[0]/2, draw_corr.shape[0]/2)
        fig_after.savefig(out_after_png, bbox_inches='tight')

        plt.close()
        if verbose:
            display(draw_corr.round(2).style.applymap(color_digits, threshold_red=threshold, threshold_yellow=threshold**2))

        self.stats['out_after'] = out_after_png
        result_data = copy.deepcopy(data)
        if drop_features:
            result_data.features_exclude(features_to_drop, verbose=False)

        if verbose:
            print('Dropped (if drop_features=True):', features_to_drop)

        return result_data, cor_out


    def find_correlated_groups(self, data, features = None, features_to_leave = None, threshold=0.6, method = 'spearman',
             verbose=False, figsize=(12,12), corr_graph_type='connected'):
        '''
        Calculates the covariance matrix and correlation coefficients for each pair of features and
        returns groups of significantly correlated features

        Parameters
        -----------
        data: a Data or DataSamples object to check (in case of DataSamples it's train sample will be checked)
        features: a list of features to analyze; by default - all the features
        features_to_leave: a list of features that must not be included in analysis
        threshold: the lowest value of a correlation coefficient for two features to be considered correlated
        method: method for correlation calculation
        verbose: flag for detailed output
        figsize: the size of correlation connections graph (printed if verbose)
        corr_graph_type: type of connectivity to persue in finding groups of correlated features
            'connected' - groups are formed from features directly or indirectly connected by singnificant correlation
            'complete' - groups are formed from features that are directly connected to each other by significant
                correlation (each pair of features from a group will have a significant connection)

        Returns
        --------
        a list of lists representing correlated group
        '''

        if features is None:
            features=[]
        if features_to_leave is None:
            features_to_leave=[]

        if type(data)==DataSamples:
            sample=data.train
        else:
            sample=data

        if features == [] or features is None:
            candidates = [x for x in sample.features if x not in features_to_leave]
        else:
            candidates = [x for x in features if x not in features_to_leave]

        correlations = sample.dataframe[candidates].corr(method = method)

        if verbose:
            draw_corr=correlations.copy()
            draw_corr.index=[x+' (%i)' % i for i,x in enumerate(draw_corr.index)]
            draw_corr.columns=range(len(draw_corr.columns))
            display(draw_corr.round(2).style.applymap(color_digits,threshold_red=threshold))

        G=nx.Graph()
        for i in range(correlations.shape[0]):
            for j in range(i+1, correlations.shape[0]):
                if correlations.loc[correlations.columns[i], correlations.columns[j]]>threshold:
                    G.add_nodes_from([correlations.columns[i], correlations.columns[j]])
                    G.add_edge(correlations.columns[i], correlations.columns[j], label=str(round(correlations.loc[correlations.columns[i], correlations.columns[j]],3)))

        if verbose:
            plt.figure(figsize=(figsize[0]*1.2, figsize[1]))
            pos = nx.spring_layout(G, k=100)
            edge_labels = nx.get_edge_attributes(G,'label')
            nx.draw(G, pos, with_labels=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
            plt.margins(x=0.2)
            plt.show()

        correlated_groups=[]
        if corr_graph_type=='connected':
            for x in nx.connected_components(G):
                correlated_groups.append(sorted(list(x)))
        elif corr_graph_type=='complete':
            for x in nx.find_cliques(G):
                correlated_groups.append(sorted(x))
        else:
            print('Unknown correlation graph type. Please use "connected" or "complete". Return None.')
            return None

        return correlated_groups

#---------------------------------------------------------------



class VIF(Processor):
    '''
    Calculates variance inflation factor for each feature
    '''
    def __init__(self):
        self.stats = pd.DataFrame()


    def work(self, data, drop_features = False, features=None, features_to_leave=None, threshold = 5,
             drop_with_highest_VIF=True, verbose=True, out=None, sep=';'):
        '''
        Parameters
        -----------
        data: a Data or DataSamples object to check VIF on (in case of DataSamples it's train sample will be checked)
        drop_features: permition to delete excluded features and return a Data object without them
        features: a list of features to analyze; by default - all the features
        features_to_leave: a list of features that must not be deleted from the feature list
        threshold: the lowest value of VIF for feature to be excluded
        drop_with_highest_VIF: should the features with the highest VIF be excluded first (otherwise just with the lowest gini)
        verbose: flag for detailed output
        out: file name for export of VIF values (.csv and .xlsx types are supported)
        sep: the separator in case of .csv export

        Returns
        ---------
        Data or DataSamples object without excluded features
        A pandas DataFrame with VIF values on different iterations
        '''

        if features_to_leave is None:
            features_to_leave=[]

        self.stats = pd.DataFrame({'drop_features' : [drop_features], 'threshold' : [threshold], 'out' : [out]})

        if type(data)==DataSamples:
            sample=data.train
        else:
            sample=data

        if len(sample.ginis)==0:
            print('No calculated ginis in datasamples.train/data object. Set calc_gini=True while using WOE.transform or use Data.calc_gini. Return None')
            return None

        if features is None:
            features = sample.features.copy()

        features_to_drop = []
        to_check_VIF = True

        vifs_df=pd.DataFrame(index=features)
        iteration=-1

        while to_check_VIF:
            to_check_VIF = False
            iteration=iteration+1
            s = sample.target + ' ~ '
            for f in features:
                s = s + f + '+'
            s = s[:-1]

            # Break into left and right hand side; y and X
            y_, X_ = dmatrices(formula_like=s, data=sample.dataframe, return_type="dataframe")

            # For each Xi, calculate VIF
            vifs = {features[i-1]:variance_inflation_factor(X_.values, i) for i in range(1, X_.shape[1])}
            vifs_df=vifs_df.join(pd.DataFrame(vifs, index=[iteration]).T)

            if drop_with_highest_VIF:
                with_high_vif={x:sample.ginis[x] for x in vifs if vifs[x]==max({x:vifs[x] for x in vifs if x not in features_to_leave}.values()) and vifs[x]>threshold and x not in features_to_leave}
            else:
                with_high_vif={x:sample.ginis[x] for x in vifs if vifs[x]>threshold and x not in features_to_leave}

            if len(with_high_vif)>0:
                feature_to_drop=min(with_high_vif, key=with_high_vif.get)
                features_to_drop.append(feature_to_drop)
                if verbose:
                    print('Dropping %(v)s because of high VIF (VIF=%(vi)0.2f, Gini=%(g)0.2f)' % {'v':feature_to_drop, 'vi':vifs[feature_to_drop], 'g':with_high_vif[feature_to_drop]})
                features.remove(feature_to_drop)
                to_check_VIF=True

        result_data = copy.deepcopy(data)
        if drop_features:
            result_data.features_exclude(features_to_drop, verbose=False)

        out_png = 'VIF.png'
        if out is not None:
            if out[-4:]=='.csv':
                vifs_df.round(2).to_csv(out, sep = sep)
                out_png = out[:-4] + '.png'
            elif out[-5:]=='.xlsx' or out[-4:]=='.xls':
                vifs_df.round(2).style.applymap(color_digits, threshold_red=threshold).to_excel(out, engine='openpyxl', sheet_name='Variance Inflation Factor')
                out_png = out[:-5] + '.png' if out[-5:]=='.xlsx' else out[:-4] + '.png'
            elif out[-4:] == '.png':
                out_png = out
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')


        vif_fig = sns.heatmap(vifs_df.round(2).sort_values(0, ascending = False),  xticklabels = False, annot = True,
                              cmap = 'RdYlBu_r',
                              cbar = False, vmax = 5, yticklabels = True).figure
        vif_fig.set_size_inches(vifs_df.shape[0]/4, vifs_df.shape[0]/2)
        vif_fig.savefig(out_png, bbox_inches='tight')
        plt.close()

        self.stats['out'] = out_png
        if verbose:
            display(vifs_df.round(2).style.applymap(color_digits, threshold_red=threshold))
            print('Dropped (if drop_features=True):', features_to_drop)

        return result_data, vifs_df

#---------------------------------------------------------------


class FeatureEncoder(Processor):
    '''
    For processing non-numeric features
    '''
    def __init__(self):
        self.stats = pd.DataFrame()


    def work(self, data, how_to_code, inplace = False):
        '''
        Parameters
        -----------
        data: data to process, Data type
        how_to_code: a dictionary {how: features_list} where 'how' can be 'one_hot' or 'seq'(means 'sequential') and 'features_list' is a list of columns in data to process
        inplace: whether to change the data or to create a new Data object

        Returns
        ---------
        Data with additional features and dictionary for sequantial encoding
        '''
        result = data.dataframe.copy()
        feature_list = data.features.copy()

        d = {}
        for how in how_to_code:
            if how == 'one_hot':
                for feature in how_to_code[how]:
                    one_hot = pd.get_dummies(result[feature])
                    one_hot.columns = [feature + '_' + str(c) for c in one_hot.columns]
                    feature_list = feature_list + list(one_hot.columns)
                    result = result.join(one_hot)
            elif how == 'seq':
                for feature in how_to_code[how]:
                    for (i, j) in enumerate(result[feature].drop_duplicates()):
                        d[j] = i
                    result[feature + '_code'] = result[feature].apply(lambda x: d[x])
                    feature_list = feature_list + [feature + '_code']
            else:
                print ('Do not understand your command. Please use "one_hot" or "seq" for how_to_code. Good luck.')
                return None

        self.param_dict_to_stats(data, how_to_code)
        # for sequential, saves actual encoding
        self.stats.loc[self.stats.action == 'seq', 'action'] = str(d)

        if inplace:
            data = Data(result, features = feature_list, target = data.target, weights = data.weights)
            return d
        else:
            return Data(result, features = feature_list, target = data.target, weights = data.weights), d

#---------------------------------------------------------------




# Author - Dmitry Yudochev
class GiniChecker(Processor):
    '''
    Class for gini checking
    '''
    def __init__(self):
        self.stats = pd.DataFrame()

    def work(self, feature, datasamples, gini_threshold=5, gini_decrease_threshold=0.2, gini_increase_restrict=True, verbose=False, with_test=False,
             out=False, out_images='GiniChecker/'):
        '''
        Checks if gini of the feature is significant and stable enough

        Parameters
        -----------
        feature: an object of FeatureWOE type that should be checked
        datasamples: an object of DataSamples type containing the samples to check input feature on
        gini_threshold: gini on train and validate/95% bootstrap should be greater then this
        gini_decrease_threshold: gini decrease from train to validate/95% bootstrap deviation from mean to mean should be greater then this
        gini_increase_restrict: if gini increase should also be restricted
        verbose: if comments and graphs should be printed
        with_test: add checking of gini values on test (calculation is always on)
        out: a boolean for image output or a path for csv/xlsx output file to export gini values
        out_images: a path for image output (default - GiniChecker/)

        Returns
        ----------
        Boolean - whether the check was successful
        and if isinstance(out,str) then dictionary of gini values for all available samples
        '''
        if out:
            directory = os.path.dirname(out_images)
            if not os.path.exists(directory):
                os.makedirs(directory)

        if verbose:
            print('Checking', feature.feature)

        gini_correct=True

        d=feature.transform(datasamples.train, original_values=True)
        fpr, tpr, _ = roc_curve(d.dataframe[d.target], -d.dataframe[feature.feature+'_WOE'])
        gini_train= (2*auc(fpr, tpr)-1)*100
        if verbose:
            print('Train gini = '+str(round(gini_train,2)))
        if gini_train<gini_threshold:
            gini_correct=False
            if verbose:
                print('Train gini is less then threshold '+str(gini_threshold))

        samples=[datasamples.validate, datasamples.test]
        sample_names=['Validate', 'Test']
        gini_values={'Train':gini_train}
        for si in range(len(samples)):
            if samples[si] is not None:
                d=feature.transform(samples[si], original_values=True)
                fpr, tpr, _ = roc_curve(d.dataframe[d.target], -d.dataframe[feature.feature+'_WOE'])
                gini = (2*auc(fpr, tpr)-1)*100
                gini_values[samples[si].name]=gini
                if verbose:
                    print(samples[si].name+' gini = '+str(round(gini,2)))
                if with_test or samples[si].name!='Test':
                    if gini<gini_threshold:
                        gini_correct=False
                        if verbose:
                            print(samples[si].name+' gini is less then threshold '+str(gini_threshold))
                    decrease=1-gini/gini_train
                    if decrease>gini_decrease_threshold:
                        gini_correct=False
                        if verbose:
                            print('Gini decrease from Train to '+samples[si].name+' is greater then threshold: '+str(round(decrease,5))+' > '+str(gini_decrease_threshold))
                    if gini_increase_restrict and -decrease>gini_decrease_threshold:
                        gini_correct=False
                        if verbose:
                            print('Gini increase from Train to '+samples[si].name+' is greater then threshold: '+str(round(-decrease,5))+' > '+str(gini_decrease_threshold))
            else:
                gini_values[sample_names[si]]=None

        gini_list=[]
        if datasamples.bootstrap_base is not None:
            db=feature.transform(datasamples.bootstrap_base.keep(feature.feature), original_values=True)
            for bn in range(len(datasamples.bootstrap)):
                d=db.dataframe.iloc[datasamples.bootstrap[bn]]
                fpr, tpr, _ = roc_curve(d[db.target], -d[feature.feature+'_WOE'])
                roc_auc = auc(fpr, tpr)
                gini_list.append(round((roc_auc*2 - 1)*100, 2))
            mean=np.mean(gini_list)
            std=np.std(gini_list)
            if verbose:
                sns.distplot(gini_list)
                plt.axvline(x=mean, linestyle='--', alpha=0.5)
                plt.text(mean, 0, '  Mean = '+str(round(mean,2))+', std = '+str(round(std,2)),
                         horizontalalignment='right', verticalalignment='bottom', rotation=90)
                plt.xlabel('Gini values in bootstrap')
                plt.ylabel('Distribution')
                plt.title(feature.feature, fontsize = 16)
                if out:
                    plt.savefig(out_images+feature.feature+".png", dpi=100, bbox_inches='tight')
                plt.show()
            if mean-1.96*std<gini_threshold:
                gini_correct=False
                if verbose:
                    print('Less then 95% of gini distribution is greater then threshold: (mean-1.96*std) '+str(round(mean-1.96*std,5))+' < '+str(gini_threshold))
            val_decrease=1.96*std/mean
            if val_decrease>gini_decrease_threshold:
                gini_correct=False
                if verbose:
                    print('Gini deviation from mean for 95% of distribution is greater then threshold: (1.96*std/mean) '+str(round(val_decrease,5))+' > '+str(gini_decrease_threshold))

        if isinstance(out, str):
            gini_values.update({'Bootstrap'+str(i):gini_list[i] for i in range(len(gini_list))})
            return gini_correct, gini_values
        else:
            return gini_correct



    #added 13.08.2018 by Yudochev Dmitry
    def work_all(self, woe, features=None, drop_features=False, gini_threshold=5, gini_decrease_threshold=0.2,
                 gini_increase_restrict=True, verbose=False, with_test=False, out=False, out_images='GiniChecker/', sep=';'):
        '''
        Checks if gini of all features from WOE object is significant and stable enough

        Parameters
        -----------
        woe: an object of WOE type that should be checked
        drop_features: should the features be dropped from WOE.feature_woes list in case of failed checks
        gini_threshold: gini on train and validate/95% bootstrap should be greater then this
        gini_decrease_threshold: gini decrease from train to validate/95% bootstrap deviation from mean to mean should be greater then this
        gini_increase_restrict: if gini increase should also be restricted
        verbose: if comments and graphs should be printed
        with_test: add checking of gini values on test (calculation is always on)
        out: a boolean for image output or a path for csv/xlsx output file to export gini values
        out_images: a path for image output (default - GiniChecker/)
        sep: the separator to be used in case of csv export

        Returns
        ----------
        Dictionary with results of check for all features from input WOE object
        '''
        if features is None:
            cycle_features=list(woe.feature_woes)
        else:
            cycle_features=list(features)

        not_in_features_woe=[x for x in cycle_features if x not in woe.feature_woes]
        if len(not_in_features_woe)>0:
            print('No', not_in_features_woe, 'in WOE.feature_woes. Abort.')
            return None

        if out:
            directory = os.path.dirname(out_images)
            if not os.path.exists(directory):
                os.makedirs(directory)

        gini_correct={}
        if isinstance(out, str):
            gini_df=pd.DataFrame(columns=['Train', 'Validate', 'Test']+['Bootstrap'+str(i) for i in range(len(woe.datasamples.bootstrap))])
        for feature in cycle_features:
            if isinstance(out, str):
                gini_correct[feature], gini_values=self.work(woe.feature_woes[feature], datasamples=woe.datasamples, gini_threshold=gini_threshold,
                                                             gini_decrease_threshold=gini_decrease_threshold,
                                                             gini_increase_restrict=gini_increase_restrict, verbose=verbose, with_test=with_test,
                                                             out=out, out_images=out_images)
                #print(feature, gini_values)
                gini_df=gini_df.append(pd.DataFrame(gini_values, index=[feature]))
            else:
                gini_correct[feature]=self.work(woe.feature_woes[feature], datasamples=woe.datasamples, gini_threshold=gini_threshold,
                                                gini_decrease_threshold=gini_decrease_threshold,
                                                gini_increase_restrict=gini_increase_restrict, verbose=verbose, with_test=with_test,
                                                out=out, out_images=out_images)

        if isinstance(out, str):
            gini_df=gini_df[['Train', 'Validate', 'Test']+['Bootstrap'+str(i) for i in range(len(woe.datasamples.bootstrap))]].dropna(axis=1)
            if out[-4:]=='.csv':
                gini_df.to_csv(out, sep = sep)
            elif out[-4:]=='.xls' or out[-5:]=='.xlsx':
                writer = pd.ExcelWriter(out, engine='openpyxl')
                gini_df.style.apply(color_background,
                                    mn=gini_df.min().min(), mx=gini_df.max().max(), cmap='RdYlGn').to_excel(writer, sheet_name='Gini by Samples')
                 # Get the openpyxl objects from the dataframe writer object.
                worksheet = writer.sheets['Gini by Samples']
                for x in worksheet.columns:
                    worksheet.column_dimensions[x[0].column].width = 40 if x[0].column=='A' else 12
                writer.save()
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')
        if drop_features:
            woe.excluded_feature_woes.update({x:woe.feature_woes[x] for x in woe.feature_woes if gini_correct[x]==False})
            woe.feature_woes={x:woe.feature_woes[x] for x in woe.feature_woes if gini_correct[x]}
        return gini_correct


    def work_tree(self, dtree, input_df=None, gini_threshold=5, gini_decrease_threshold=0.2, gini_increase_restrict=True,
                   verbose=False, with_test=False, out=False):
        '''
        Checks if gini of the tree is significant and stable enough

        Parameters
        -----------
        dtree: a cross.DecisionTree object
        input_df: a DataFrame, containing tree description
        datasamples: an object of DataSamples type containing the samples to check input tree on
        gini_threshold: gini on train and validate/95% bootstrap should be greater then this
        gini_decrease_threshold: gini decrease from train to validate/95% bootstrap deviation from mean to mean should be greater then this
        gini_increase_restrict: if gini increase should also be restricted
        verbose: if comments and graphs should be printed
        with_test: add checking of gini values on test (calculation is always on)
        out: a boolean flag for gini values output

        Returns
        ----------
        Boolean - whether the check was successful
        and if out==True then dictionary of gini values for all available samples
        '''

        if input_df is None:
            tree_df=dtree.tree.copy()
        else:
            tree_df=input_df.copy()

        datasamples=dtree.datasamples
        features=[x for x in dtree.features if x in tree_df]
        #[x for x in tree_df.columns[:tree_df.columns.get_loc('node')] if tree_df[x].dropna().shape[0]>0]
        if verbose:
            print('Checking tree on', str(features))

        gini_correct=True

        samples=[datasamples.train, datasamples.validate, datasamples.test]
        sample_names=['Train', 'Validate', 'Test']
        gini_values={}
        for si in range(len(samples)):
            if samples[si] is not None:
                to_check=samples[si].keep(features=features).dataframe
                to_check['woe']=dtree.transform(to_check, tree_df, ret_values=['woe'])
                fpr, tpr, _ = roc_curve(to_check[samples[si].target], -to_check['woe'])
                gini = (2*auc(fpr, tpr)-1)*100
                gini_values[samples[si].name]=gini
                if verbose:
                    print(samples[si].name+' gini = '+str(round(gini,2)))
                if with_test or samples[si].name!='Test':
                    if gini<gini_threshold:
                        gini_correct=False
                        if verbose:
                            print(samples[si].name+' gini is less then threshold '+str(gini_threshold))
                    if samples[si].name!='Train':
                        decrease=1-gini/gini_values['Train']
                        if decrease>gini_decrease_threshold:
                            gini_correct=False
                            if verbose:
                                print('Gini decrease from Train to '+samples[si].name+' is greater then threshold: '+str(round(decrease,5))+' > '+str(gini_decrease_threshold))
                        if gini_increase_restrict and -decrease>gini_decrease_threshold:
                            gini_correct=False
                            if verbose:
                                print('Gini increase from Train to '+samples[si].name+' is greater then threshold: '+str(round(-decrease,5))+' > '+str(gini_decrease_threshold))
            else:
                gini_values[sample_names[si]]=None

        gini_list=[]
        if datasamples.bootstrap_base is not None:
            base_with_woe=datasamples.bootstrap_base.keep(features=features).dataframe
            base_with_woe['woe']=dtree.transform(base_with_woe, tree_df, ret_values=['woe'])

            for bn in range(len(datasamples.bootstrap)):
                to_check=base_with_woe.iloc[datasamples.bootstrap[bn]]
                fpr, tpr, _ = roc_curve(to_check[datasamples.bootstrap_base.target], -to_check['woe'])
                roc_auc = auc(fpr, tpr)
                gini_list.append(round((roc_auc*2 - 1)*100, 2))

            mean=np.mean(gini_list)
            std=np.std(gini_list)
            if verbose>True:
                sns.distplot(gini_list)
                plt.axvline(x=mean, linestyle='--', alpha=0.5)
                plt.text(mean, 0, '  Mean = '+str(round(mean,2))+', std = '+str(round(std,2)),
                         horizontalalignment='right', verticalalignment='bottom', rotation=90)
                plt.xlabel('Gini values in bootstrap')
                plt.ylabel('Distribution')
                plt.title('Tree on '+str(features), fontsize = 16)
                plt.show()
            elif verbose:
                print('Bootstrap: mean = '+str(round(mean,2))+', std = '+str(round(std,2)))
            if mean-1.96*std<gini_threshold:
                gini_correct=False
                if verbose:
                    print('Less then 95% of gini distribution is greater then threshold: (mean-1.96*std) '+str(round(mean-1.96*std,5))+' < '+str(gini_threshold))
            val_decrease=1.96*std/mean
            if val_decrease>gini_decrease_threshold:
                gini_correct=False
                if verbose:
                    print('Gini deviation from mean for 95% of distribution is greater then threshold: (1.96*std/mean) '+str(round(val_decrease,5))+' > '+str(gini_decrease_threshold))

        if out:
            gini_values.update({'Bootstrap'+str(i):gini_list[i] for i in range(len(gini_list))})
            return gini_correct, gini_values
        else:
            return gini_correct


#---------------------------------------------------------------




# Author - Dmitry Yudochev
class BusinessLogicChecker(Processor):
    '''
    Class for business logic checking
    '''
    def __init__(self):
        self.stats = pd.DataFrame()


    def work(self, feature, conditions='', verbose=False, out=None):
        '''
        Checks if the business logic condition is True

        Parameters
        -----------
        feature: an object of FeatureWOE type that should be checked
        conditions: a string with business logic conditions
            for feature.categorical==True: 'cond_1;cond_2;...;cond_n', where cond_i
                is 'A sign B', where A and B
                    are comma-separated lists of values (or nothing, but not both at the same time)
                and where sign
                     is one of the following: <, >, =, <=, >=

            each condition compares risk of bins with values from A to risk of bins with values from B (if B is omitted,
            then risk of bins with values from A is compared to risk of bins with values not in A);
            > means that risk of the second values group is smaller then the risk of the first values group (and values from
            different groups cannot be in one bin), < means the opposite (again, values from different groups cannot be in one
            bin), adding = allows values from different groups to be in one bin;

            ALL of the conditions should be True or conditions should be empty for the input feature to pass the check
            -----------------------------------------------------------------------------------------------------------
            for feature.categorical==False:'cond_1;cond_2;....;cond_n (excl_1;...;excl_n)', where cond_i
                is 'sign_1 value_2 sign_2 value_3 sign_3 ... value_n sign_n', where sign_i
                    is one of the following: <, >
                and where value_i
                    is a float/int and can be omitted
                and where excl_i
                    is a float/int and can be omitted (if there is not excl_i at all, then parentheses can be omitted too)

            each condition describes how should risk be changing when feature values are increasing;
            > means that risk will be monotonicaly decreasing with increase of values, < means the opposite, >< means that
            risk decrease and then increase, adding value between signs tells, that in the bin with this value should be
            the local risk extremum (>N< means that the bin with N in it should have the least risk);
            adding values in () will result in exclusion of bins with these values before risk trend checking (and so bins
            with these values are ignored);
            each condition should start with a sign and end with a sign, one sign is permitter, values between signs
            can be omitted;

            ANY one of the conditions should be True for the input feature to pass the check
            in case of conditions==None or conditions=='' checker wil return True is risk trend is monotonicaly
            increasing/decresing (the same check will be processed if only values to exclude are provided)
        verbose: if comments and graphs should be printed
        out: a path for csv/xlsx output file to export business logic check results

        Returns
        ----------
        Boolean - whether the check was successful
        and if out is not None then dataframe of check log
        '''
        if out is not None:
            out_df=pd.DataFrame(columns=['feature', 'categorical', 'condition', 'fact', 'condition_result'])

        if feature.categorical == False:
            woes_dropna={feature.groups[x][0]:feature.woes[x] for x in feature.woes if isinstance(feature.groups[x],list)}
            groups_info=pd.DataFrame(woes_dropna, index=['woe']).transpose().reset_index().rename({'index':'lower'}, axis=1)
            groups_info['upper']=groups_info['lower'].shift(-1).fillna(np.inf)
            if groups_info.shape[0]==1:
                if verbose:
                    print('Only 1 group with non-missing values is present. Skipping trend check..')
                all_cond_correct=True
            else:
                all_cond_correct=False
                for c in conditions.split(';'):
                    #find all floats/ints between > and < - minimal risk
                    #first there should be >, then + or - or nothing, then at least one digit, then . or , or nothing, then zero or more digits and < after that
                    min_risk = re.findall('(?<=>)[-+]?\d+[.,]?\d*(?=<)', c)
                    #find all floats/ints between < and > - maximal risk
                    max_risk = re.findall('(?<=<)[-+]?\d+[.,]?\d*(?=>)', c)
                    #find all floats/ints between ( and ), ( and ; or ; and ) - values to exclude (without risk checking)
                    excl_risk = re.findall('(?<=[(;])[-+]?\d+[.,]?\d*(?=[;)])', c)
                    clear_condition=''.join(x for x in c if x in '<>')

                    gi_check=groups_info.dropna(how='all', subset=['lower','upper'])[['woe','lower','upper']].copy()
                    for excl in excl_risk:
                        gi_check=gi_check[((gi_check['lower']<=float(excl)) & (gi_check['upper']>float(excl)))==False]
                    gi_check['risk_trend']=np.sign((gi_check['woe']-gi_check['woe'].shift(1)).dropna()).apply(lambda x: '+' if (x<0) else '-' if (x>0) else '0')
                    trend=gi_check['risk_trend'].str.cat()

                    reg_exp=r''
                    for s in clear_condition:
                        if s=='>':
                            reg_exp=reg_exp+r'-+'
                        if s=='<':
                            reg_exp=reg_exp+r'\++'
                    if len(reg_exp)==0:
                        reg_exp='-*|\+*'
                    if re.fullmatch(reg_exp, trend):
                        trend_correct=True
                        if verbose:
                            print('Risk trend in data is consistent with input trend: input ', clear_condition, ', data ', trend)
                    else:
                        trend_correct=False
                        if verbose:
                            print('Risk trend in data is not consistent with input trend: input ', clear_condition, ', data ', trend)


                    #local risk minimums
                    min_risk_data=gi_check[(gi_check['risk_trend']=='-') & (gi_check['risk_trend'].shift(-1)=='+')].reset_index(drop=True)
                    min_risk_correct=True
                    for mr in range(len(min_risk)):
                        if mr+1<=min_risk_data.shape[0]:
                            if verbose:
                                print(feature.feature+': checking min risk in', min_risk[mr], '(between ', min_risk_data['lower'].loc[mr], ' and ', min_risk_data['upper'].loc[mr], ')')
                            min_risk_correct=min_risk_correct and (float(min_risk[mr])>=min_risk_data['lower'].loc[mr] and float(min_risk[mr])<min_risk_data['upper'].loc[mr])
                        else:
                            if verbose:
                                print(feature.feature+': not enough minimums in data to check', min_risk[mr])
                            min_risk_correct=False
                    #local risk maximums
                    max_risk_data=gi_check[(gi_check['risk_trend']=='+') & (gi_check['risk_trend'].shift(-1)=='-')].reset_index(drop=True)
                    max_risk_correct=True
                    for mr in range(len(max_risk)):
                        if mr+1<=max_risk_data.shape[0]:
                            if verbose:
                                print(feature.feature+': checking max risk in', max_risk[mr], '(between ', max_risk_data['lower'].loc[mr], ' and ', max_risk_data['upper'].loc[mr], ')')
                            max_risk_correct=max_risk_correct and (float(max_risk[mr])>=max_risk_data['lower'].loc[mr] and float(max_risk[mr])<max_risk_data['upper'].loc[mr])
                        else:
                            if verbose:
                                print(feature.feature+': not enough maximums in data to check', max_risk[mr])
                            min_risk_correct=False
                    all_cond_correct=all_cond_correct or (trend_correct and min_risk_correct and max_risk_correct)

                    if out is not None:
                        out_df=out_df.append(dict(feature=feature.feature, categorical=feature.categorical, condition=c, fact=trend, condition_result=trend_correct and min_risk_correct and max_risk_correct), ignore_index=True)

            if verbose:
                if all_cond_correct:
                    print(feature.feature+': business logic check succeeded.')
                else:
                    fig=plt.figure(figsize=(5,0.5))
                    plt.plot(range(len(groups_info.dropna(how='all', subset=['lower','upper'])['lower'])),
                             groups_info.dropna(how='all', subset=['lower','upper'])['woe'], color='red')
                    plt.xticks(range(len(groups_info.dropna(how='all', subset=['lower','upper'])['lower'])),
                               round(groups_info.dropna(how='all', subset=['lower','upper'])['lower'],3))
                    plt.ylabel('WoE')
                    fig.autofmt_xdate()
                    plt.show()
                    print(feature.feature+': business logic check failed.')

            if out is not None:
                return all_cond_correct, out_df[['feature', 'categorical', 'condition', 'fact', 'condition_result']]
            else:
                return all_cond_correct
        else:
            all_cond_correct=True
            if conditions!='':
                w={}
                for x in feature.groups:
                    for y in feature.groups[x]:
                        w[y]=feature.woes[x]
                groups_info=pd.DataFrame(w, index=['woe']).transpose().reset_index().rename({'index':'categories'}, axis=1)
                groups_info=groups_info[groups_info['categories']!=-np.inf].reset_index(drop=True).copy()
                cond_types2=['>=','=>','<=','=<']
                cond_types1=['>','<','=']
                for c in conditions.split(';'):
                    c0=[]
                    c1=[]
                    cond_type=[x for x in cond_types2 if x in c]
                    if len(cond_type)==0:
                        cond_type=[x for x in cond_types1 if x in c]
                    cond_type=cond_type[0]
                    if cond_type in ['>=', '=>', '>']:
                        c0=ast.literal_eval('['+c[:c.find(cond_type)]+']')
                        c1=ast.literal_eval('['+c[c.find(cond_type)+len(cond_type):]+']')
                    elif cond_type in ['<=', '=<', '<']:
                        c0=ast.literal_eval('['+c[c.find(cond_type)+len(cond_type):]+']')
                        c1=ast.literal_eval('['+c[:c.find(cond_type)]+']')
                    elif cond_type=='=':
                        c0=ast.literal_eval('['+c[c.find(cond_type)+len(cond_type):]+']')
                        c1=ast.literal_eval('['+c[:c.find(cond_type)]+']')
                    can_be_equal=('=' in cond_type)
                    groups_info['risk_group']=groups_info['categories'].apply(lambda x: 0 if (x in c0 or (len(c0)==0 and x not in c1)) else 1 if (x in c1 or (len(c1)==0 and x not in c0)) else np.nan)
                    cond_correct = (cond_type!='=' and groups_info[groups_info['risk_group']==0]['woe'].max()<groups_info[groups_info['risk_group']==1]['woe'].min()) or (can_be_equal and (groups_info[groups_info['risk_group']==0]['woe'].max()==groups_info[groups_info['risk_group']==1]['woe'].min() or c0==c1))
                    all_cond_correct=all_cond_correct and cond_correct
                    if verbose:
                        print(feature.feature+': checking condition '+ c + ' => ' + str(cond_correct))

                    if out is not None:
                        out_df=out_df.append(dict(feature=feature.feature, categorical=feature.categorical, condition=c, fact='', condition_result=cond_correct), ignore_index=True)

                if verbose:
                    print(feature.feature+': conditions ' + conditions + ' => ' + str(all_cond_correct))
            else:
                if verbose:
                    print(feature.feature+': no conditions were specified, business logic check succeeded.')
            if out is not None:
                return all_cond_correct, out_df[['feature', 'categorical', 'condition', 'fact', 'condition_result']]
            else:
                return all_cond_correct



    #added 13.08.2018 by Yudochev Dmitry
    def work_all(self, woe, features=None, input_conditions=None, drop_features=False, verbose=False, out=None, sep=';'):
        '''
        Checks if business logic conditions for all features from the WOE object are True

        Parameters
        -----------
        woe: an object of FeatureWOE type that should be checked
        input_conditions: adress for excel-file with business logic conditions (columns 'variable' and 'condition' are mandatory)
        drop_features: should the features be dropped from WOE.feature_woes list in case of failed checks
        verbose: if comments and graphs should be printed
        out: a path for csv/xlsx output file to export business logic check results
        sep: the separator to be used in case of csv export

        Returns
        ----------
        Dictionary with results of check for all features from input WOE object
        '''
        if out is not None:
            out_df=pd.DataFrame(columns=['feature', 'categorical', 'condition', 'fact', 'condition_result', 'overall_result'])

        if features is None:
            cycle_features=list(woe.feature_woes)
        else:
            cycle_features=list(features)

        not_in_features_woe=[x for x in cycle_features if x not in woe.feature_woes]
        if len(not_in_features_woe)>0:
            print('No', not_in_features_woe, 'in self.feature_woes. Abort.')
            return None

        business_logic_correct={}
        '''
        if conditions_dict is not None:
            if isinstance(conditions_dict, dict):
                conditions_dict=pd.DataFrame(conditions_dict, index=['conditions']).T
            elif isinstance(conditions_dict, str) and (conditions_dict[-5:]=='.xlsx' or conditions_dict[-4:]=='.xls'):
                try:
                    conditions=pd.read_excel(conditions_dict).set_index('variable')
                    conditions['conditions']=conditions['conditions'].apply(lambda x: '' if (pd.isnull(x)) else x)
                except Exception:
                    print('No conditions dictionary was found / no "variable" or "conditions" fields were found. Abort.')
                    return None
            elif isinstance(conditions_dict, str):
                conditions_dict=pd.DataFrame({x:conditions_dict for x in cycle_features},
                                             index=['conditions']).T
        else:
            conditions=pd.DataFrame()
        '''

        if input_conditions is None:
            conditions_dict=pd.DataFrame(columns=['feature', 'conditions'])
        elif isinstance(input_conditions, dict) or isinstance(input_conditions, pd.DataFrame):
            conditions_dict=input_conditions.copy()
        elif isinstance(input_conditions, str):
            if input_conditions[-4:]=='.csv':
                conditions_dict=pd.read_csv(input_conditions, sep = sep)
            elif input_conditions[-4:]=='.xls' or input_conditions[-5:]=='.xlsx':
                conditions_dict=pd.read_excel(input_conditions)
            else:
                print('Unknown format for path to conditions dictionary file. Return None.')
        elif isinstance(input_conditions, tuple):
            conditions_dict={x:input_conditions[0] if x not in woe.categorical else input_conditions[1] for x in cycle_features}
        else:
            print('Unknown format for conditions dictionary file. Return None')
            return None

        if isinstance(conditions_dict, pd.DataFrame):
            for v in ['feature', 'variable', 'var']:
                if v in conditions_dict:
                    break
            try:
                conditions_dict=dict(conditions_dict.fillna('').set_index(v)['conditions'])
            except Exception:
                print("No 'feature' ,'variable', 'var' or 'conditions' field in input pandas.DataFrame. Return None.")
                return None

        for feature in cycle_features:
            if feature not in conditions_dict:
                current_conditions=''
            else:
                current_conditions=conditions_dict[feature]
            if out is not None:
                business_logic_correct[feature], out_feature_df=self.work(woe.feature_woes[feature], conditions=current_conditions, verbose=verbose, out=out)
                out_feature_df['overall_result']=business_logic_correct[feature]
                out_df=out_df.append(out_feature_df, ignore_index=True)
            else:
                business_logic_correct[feature]=self.work(woe.feature_woes[feature], conditions=current_conditions, verbose=verbose, out=out)
        if drop_features:
            woe.excluded_feature_woes.update({x:woe.feature_woes[x] for x in woe.feature_woes if business_logic_correct[x]==False})
            woe.feature_woes={x:woe.feature_woes[x] for x in woe.feature_woes if business_logic_correct[x]}

        if out is not None:
            out_df=out_df[['feature', 'categorical', 'condition', 'fact', 'condition_result', 'overall_result']]
            #display(out_df)
            if out[-4:]=='.csv':
                out_df.to_csv(out, sep = sep)
            elif out[-4:]=='.xls' or out[-5:]=='.xlsx':
                writer = pd.ExcelWriter(out, engine='openpyxl')
                out_df.style.apply(self.color_result, subset=pd.IndexSlice[:,['condition_result', 'overall_result']]).to_excel(writer, sheet_name='Business Logic', index=False)
                 # Get the openpyxl objects from the dataframe writer object.
                worksheet = writer.sheets['Business Logic']
                for x in worksheet.columns:
                    worksheet.column_dimensions[x[0].column].width = 40 if x[0].column=='A' else 20
                writer.save()
            else:
                print('Unknown format for export file. Use .csv or .xlsx. Skipping export.')
        return business_logic_correct



    def work_tree(self, dtree, input_df=None, input_conditions=None, max_corrections=None, sep=';', to_correct=False, verbose=False):
        '''
        Checks if the business logic conditions are True in every node of the input tree and corrects the tree for it to pass the check

        Parameters
        -----------
        dtree: a cross.DecisionTree object to check
        input_df: a DataFrame, containing tree description
        input_conditions: a DataFrame, a dictionary or a string with a path to conditions dictionary (in case of DataFrame or string
            the field with features' names should be called 'feature', 'variable' or 'var')
                for categorical features: 'cond_1;cond_2;...;cond_n', where cond_i
                    is 'A sign B', where A and B
                        are comma-separated lists of values (or nothing, but not both at the same time)
                    and where sign
                         is one of the following: <, >, =, <=, >=

                each condition compares risk of bins with values from A to risk of bins with values from B (if B is omitted,
                then risk of bins with values from A is compared to risk of bins with values not in A);
                > means that risk of the second values group is smaller then the risk of the first values group (and values from
                different groups cannot be in one bin), < means the opposite (again, values from different groups cannot be in one
                bin), adding = allows values from different groups to be in one bin;

                ALL of the conditions should be True or conditions should be empty for the input feature to pass the check
                -----------------------------------------------------------------------------------------------------------
                for interval features:'cond_1;cond_2;....;cond_n (excl_1;...;excl_n)', where cond_i
                    is 'sign_1 sign_2 sign_3 ... sign_n', where sign_i
                        is one of the following: <, >
                    and where excl_i
                        is a float/int and can be omitted (if there is not excl_i at all, then parentheses can be omitted too)

                each condition describes how should risk be changing when feature values are increasing;
                > means that risk will be monotonicaly decreasing with increase of values, < means the opposite, >< means that
                risk decrease and then increase, values between signs will be ignored because for most of nodes entire sample won't be
                available for division and extremum values' absense or the presence of new local extremums should not be prohibited;
                adding values in () will result in exclusion of bins with these values before risk trend checking (and so bins
                with these values are ignored);
                each condition should start with a sign and end with a sign, one sign is permitted;

                ANY one of the conditions should be True for the input feature to pass the check
                in case of conditions==None or conditions=='' checker wil return True is risk trend is monotonicaly
                increasing/decresing (the same check will be processed if only values to exclude are provided)
        max_corrections: maximal number of corrections in attempt to change the tree so it will pass the check
        sep: a separator in case of csv import for conditions dictionary
        to_correct: should there be attempts to correct tree by uniting nodes or not
        verbose: if comments and graphs should be printed

        Returns
        ----------
        if to_correct:
            True and a DataFrame with tree description - corrected or initial
        else:
            result of the input tree check and the input tree itself
        '''

        #-----------------------------------------------Subsidiary functions--------------------------------------------------
        def bl_check_categorical(df, conditions, verbose=False, missing_group_is_correct=True):
            '''
            TECH
            Check correctness of conditions for a categorical feature
            Parameters
            -----------
            df: a DataFrame, containing lists of categories and WoE values
            conditions: a string, containing business logic conditions for a feature
            verbose: if comments should be printed
            missing_group_is_correct: should missing of any value from condition in input data be considered as
                successful check or not

            Returns
            ----------
            boolean flag of successful check
            '''
            all_cond_correct=True
            if conditions!='':
                tree_df=df.copy()
                #display(tree_df)
                cat_woes=[]
                for i in tree_df.index:
                    categories, n, w = tree_df.loc[i]
                    #display(tree_df.loc[i])
                    #display(categories)
                    for c in categories:
                        cat_woes.append([c, n, w])
                groups_info=pd.DataFrame(cat_woes, columns=['categories', 'nodes', 'woe'])
                #display(groups_info)

                cond_types2=['>=','=>','<=','=<']
                cond_types1=['>','<','=']
                for c in conditions.split(';'):
                    c0=[]
                    c1=[]
                    cond_type=[x for x in cond_types2 if x in c]
                    if len(cond_type)==0:
                        cond_type=[x for x in cond_types1 if x in c]
                    cond_type=cond_type[0]
                    if cond_type in ['>=', '=>', '>']:
                        c0=ast.literal_eval('['+c[:c.find(cond_type)]+']')
                        c1=ast.literal_eval('['+c[c.find(cond_type)+len(cond_type):]+']')
                    elif cond_type in ['<=', '=<', '<']:
                        c0=ast.literal_eval('['+c[c.find(cond_type)+len(cond_type):]+']')
                        c1=ast.literal_eval('['+c[:c.find(cond_type)]+']')
                    elif cond_type=='=':
                        c0=ast.literal_eval('['+c[c.find(cond_type)+len(cond_type):]+']')
                        c1=ast.literal_eval('['+c[:c.find(cond_type)]+']')
                    can_be_equal=('=' in cond_type)
                    groups_info['risk_group']=groups_info['categories'].apply(lambda x: 0 if (x in c0 or (len(c0)==0 and x not in c1)) else 1 if (x in c1 or (len(c1)==0 and x not in c0)) else np.nan)
                    cond_correct = (cond_type!='=' and groups_info[groups_info['risk_group']==0]['woe'].max()<groups_info[groups_info['risk_group']==1]['woe'].min()) or \
                                   (can_be_equal and (groups_info[groups_info['risk_group']==0]['woe'].max()==groups_info[groups_info['risk_group']==1]['woe'].min() or c0==c1)) or \
                                   (missing_group_is_correct and len(groups_info['risk_group'].dropna().unique())<2)
                    all_cond_correct=all_cond_correct and cond_correct
                    if verbose:
                        print('\tChecking condition '+ c + ' => ' + str(cond_correct))
                if verbose:
                    print('\tConditions ' + conditions + ' => ' + str(all_cond_correct))

            elif verbose:
                print('\tNo conditions were specified, business logic check succeeded.')
            return all_cond_correct

        def bl_check_interval(df, conditions, verbose=False):
            '''
            TECH
            Check correctness of conditions for an interval feature

            Parameters
            -----------
            df: a DataFrame, containing intervals' descriptions and WoE values
            conditions: a string, containing business logic conditions for a feature
            verbose: if comments should be printed

            Returns
            ----------
            boolean flag of successful check
            '''
            tree_df=df.copy()
            split_feature=tree_df.columns[0]
            groups_info=tree_df[pd.isnull(tree_df[split_feature])==False]
            groups_info['upper']=groups_info[split_feature].apply(lambda x: x[0][1] if pd.isnull(x[1]) else x[1])
            groups_info['lower']=groups_info[split_feature].apply(lambda x: x[0][0] if pd.isnull(x[1]) else x[0])
            #display(groups_info)

            if groups_info.shape[0]==1:
                if verbose:
                    print('\tOnly 1 group with non-missing values is present. Skipping trend check..')
                all_cond_correct=True
            else:
                all_cond_correct=False
                for c in conditions.split(';'):
                    #find all floats/ints between > and < - minimal risk
                    #first there should be >, then + or - or nothing, then at least one digit, then . or , or nothing, then zero or more digits and < after that
                    #min_risk = re.findall('(?<=>)[-+]?\d+[.,]?\d*(?=<)', c)
                    #find all floats/ints between < and > - maximal risk
                    #max_risk = re.findall('(?<=<)[-+]?\d+[.,]?\d*(?=>)', c)
                    #find all floats/ints between ( and ), ( and ; or ; and ) - values to exclude (without risk checking)
                    excl_risk = re.findall('(?<=[(;])[-+]?\d+[.,]?\d*(?=[;)])', c)
                    clear_condition=''.join(x for x in c if x in '<>')

                    gi_check=groups_info.dropna(how='all', subset=['lower','upper'])[['woe','lower','upper']].copy()
                    for excl in excl_risk:
                        gi_check=gi_check[((gi_check['lower']<=float(excl)) & (gi_check['upper']>float(excl)))==False]
                    gi_check['risk_trend']=np.sign((gi_check['woe']-gi_check['woe'].shift(1)).dropna()).apply(lambda x: '+' if (x<0) else '-' if (x>0) else '0')
                    trend=gi_check['risk_trend'].str.cat()

                    reg_exp=r''
                    for s in clear_condition:
                        if s=='>':
                            reg_exp=reg_exp+r'-+'
                        if s=='<':
                            reg_exp=reg_exp+r'\++'
                    if len(reg_exp)==0:
                        reg_exp='-*|\+*'
                    if re.fullmatch(reg_exp, trend):
                        trend_correct=True
                        if verbose:
                            print('\tRisk trend in data is consistent with input trend: input ', clear_condition, ', data ', trend)
                    else:
                        trend_correct=False
                        if verbose:
                            print('\tRisk trend in data is not consistent with input trend: input ', clear_condition, ', data ', trend)


                    '''#local risk minimums
                    min_risk_data=gi_check[(gi_check['risk_trend']=='-') & (gi_check['risk_trend'].shift(-1)=='+')].reset_index(drop=True)
                    min_risk_correct=True
                    for mr in range(len(min_risk)):
                        if mr+1<=min_risk_data.shape[0]:
                            if verbose:
                                print('\tChecking min risk in', min_risk[mr], '(between ', min_risk_data['lower'].loc[mr], ' and ', min_risk_data['upper'].loc[mr], ')')
                            min_risk_correct=min_risk_correct and (float(min_risk[mr])>=min_risk_data['lower'].loc[mr] and float(min_risk[mr])<min_risk_data['upper'].loc[mr])
                        else:
                            if verbose:
                                print('\tNot enough minimums in data to check', min_risk[mr])
                            min_risk_correct=False
                    #local risk maximums
                    max_risk_data=gi_check[(gi_check['risk_trend']=='+') & (gi_check['risk_trend'].shift(-1)=='-')].reset_index(drop=True)
                    max_risk_correct=True
                    for mr in range(len(max_risk)):
                        if mr+1<=max_risk_data.shape[0]:
                            if verbose:
                                print('\tChecking max risk in', max_risk[mr], '(between ', max_risk_data['lower'].loc[mr], ' and ', max_risk_data['upper'].loc[mr], ')')
                            max_risk_correct=max_risk_correct and (float(max_risk[mr])>=max_risk_data['lower'].loc[mr] and float(max_risk[mr])<max_risk_data['upper'].loc[mr])
                        else:
                            if verbose:
                                print('\tNot enough maximums in data to check', max_risk[mr])
                            min_risk_correct=False
                    all_cond_correct=all_cond_correct or (trend_correct and min_risk_correct and max_risk_correct)'''
                    all_cond_correct=all_cond_correct or trend_correct

            if verbose:
                if all_cond_correct:
                    print('\tBusiness logic check succeeded.')
                else:
                    fig=plt.figure(figsize=(5,0.5))
                    plt.plot(range(len(groups_info.dropna(how='all', subset=['lower','upper'])['lower'])),
                             groups_info.dropna(how='all', subset=['lower','upper'])['woe'], color='red')
                    plt.xticks(range(len(groups_info.dropna(how='all', subset=['lower','upper'])['lower'])),
                               round(groups_info.dropna(how='all', subset=['lower','upper'])['lower'],3))
                    plt.ylabel('WoE')
                    fig.autofmt_xdate()
                    plt.show()
                    print('\tBusiness logic check failed.')
            return all_cond_correct

        def bl_recursive_correct(tree_df, node, allowed_corrections=1, corrections=None, conditions='', max_corrections=1,
                               verbose=False):
            '''
            TECH
            Recursive search of corrections needed for tree to pass business logic checks

            Parameters
            -----------
            tree_df: a DataFrame, containing tree description
            node: a node number, whose children are corrected and checked
            allowed_corrections: a number of remaining corrections, that are allowed
            max_corrections: maximal number of corrections in attempt to change the tree so it will pass the check
            corrections: the list of current corrections
            conditions: a string, containing business logic conditions for a feature, by which current node was split
            verbose: if comments and graphs should be printed

            Returns
            ----------
            boolean flag of corrected tree passing the check and
            the list of corrections, that were made
            '''

            if corrections is None:
                corrections=[]

            split_feature=tree_df[(tree_df['node']==node)]['split_feature'].values[0]
            if allowed_corrections>0:
                possible_nodes_to_correct=sorted(tree_df[(tree_df['parent_node']==node)]['node'].tolist())
                combinations=[]

                for n1 in range(len(possible_nodes_to_correct)):
                    for n2 in range(len(possible_nodes_to_correct[n1+1:])):
                        if dtree.check_unitability(tree_df, [possible_nodes_to_correct[n1], possible_nodes_to_correct[n1+1:][n2]]):
                            first_condition=tree_df[(tree_df['node']==possible_nodes_to_correct[n1])][split_feature].values[0]
                            if not(isinstance(first_condition, list) or isinstance(first_condition, tuple)):
                                nodes_combination=[possible_nodes_to_correct[n1+1:][n2], possible_nodes_to_correct[n1]]
                            else:
                                nodes_combination=[possible_nodes_to_correct[n1], possible_nodes_to_correct[n1+1:][n2]]
                            combinations.append([nodes_combination,
                                                abs(tree_df[tree_df['node']==possible_nodes_to_correct[n1]]['woe'].values[0]- \
                                                    tree_df[tree_df['node']==possible_nodes_to_correct[n1+1:][n2]]['woe'].values[0])])
                combinations.sort(key=itemgetter(1))

                for nodes_to_unite, woe in combinations:
                    if verbose:
                        print('Checking (',(max_corrections-allowed_corrections+1),'): for node', node, 'uniting children', str(nodes_to_unite), 'with woe difference =', woe)
                    tree_df_corrected=dtree.unite_nodes(tree_df, nodes_to_unite)
                    #display(tree_df_corrected)
                    if tree_df_corrected.shape[0]!=tree_df.shape[0]:
                        correct, final_corrections=bl_recursive_correct(tree_df_corrected, node, allowed_corrections-1, corrections+[nodes_to_unite],
                                                                      conditions, max_corrections=max_corrections, verbose=verbose)
                    else:
                        correct=False
                    if correct:
                        return correct, final_corrections
                else:
                    return False, corrections
            else:
                df_to_check=tree_df[(tree_df['parent_node']==node)][[split_feature, 'node', 'woe']]
                categorical=sum([isinstance(x, list) for x in df_to_check[split_feature]])>0
                if verbose:
                    print('Node', node, split_feature, (': Checking categorical business logic..' if categorical \
                                                        else ': Checking interval business logic..'))
                correct=bl_check_categorical(df_to_check, conditions, verbose=verbose) if categorical \
                            else bl_check_interval(df_to_check, conditions, verbose=verbose)
                return correct, corrections
        #---------------------------------------------------------------------------------------------------------------------

        if input_df is None:
            tree_df=dtree.tree.copy()
        else:
            tree_df=input_df.copy()

        features=[x for x in dtree.features if x in tree_df]

        if input_conditions is None:
            conditions_dict=pd.DataFrame(columns=['feature', 'conditions'])
        elif isinstance(input_conditions, dict) or isinstance(input_conditions, pd.DataFrame):
            conditions_dict=input_conditions.copy()
        elif isinstance(input_conditions, str):
            if input_conditions[-4:]=='.csv':
                conditions_dict=pd.read_csv(input_conditions, sep = sep)
            elif input_conditions[-4:]=='.xls' or input_conditions[-5:]=='.xlsx':
                conditions_dict=pd.read_excel(input_conditions)
            else:
                print('Unknown format for path to conditions dictionary file. Return None.')
        elif isinstance(input_conditions, tuple):
            conditions_dict={x:input_conditions[0] if x not in dtree.categorical else input_conditions[1] for x in features}
        else:
            print('Unknown format for conditions dictionary file. Return None')
            return None

        if isinstance(conditions_dict, pd.DataFrame):
            for v in ['feature', 'variable', 'var']:
                if v in conditions_dict:
                    break
            try:
                conditions_dict=dict(conditions_dict.fillna('').set_index(v)['conditions'])
            except Exception:
                print("No 'feature' ,'variable', 'var' or 'conditions' field in input pandas.DataFrame. Return None.")
                return None

            #tree_df['split_feature'].dropna().unique().tolist()
        categorical={}
        for f in features:
            if f not in conditions_dict:
                conditions_dict[f]=''
            categorical[f]=sum([isinstance(x,list) for x in tree_df[f]])>0

        nodes_to_check=tree_df[tree_df['leaf']==False].sort_values(['depth', 'node'])['node'].tolist()
        current_node_index=0
        to_check=True
        correct_all=True

        while to_check:
            node=nodes_to_check[current_node_index]
            to_check=False
            split_feature=tree_df.loc[tree_df['node']==node, 'split_feature'].values[0]
            conditions=conditions_dict[split_feature]
            if conditions is None:
                if verbose:
                    print('Node', node, split_feature, ': <None> conditions specified, skipping..')
                correct=True
            else:
                df_to_check=tree_df[(tree_df['parent_node']==node)][[split_feature, 'node', 'woe']]
                if verbose:
                    print('Node', node, split_feature, (': Checking categorical business logic..' if categorical[split_feature] \
                                                        else ': Checking interval business logic..'))
                correct=bl_check_categorical(df_to_check, conditions, verbose=verbose) if categorical[split_feature] \
                            else bl_check_interval(df_to_check, conditions, verbose=verbose)
            correct_all=correct_all and correct

            if correct==False and to_correct:
                new_correct=False
                if len(df_to_check['node'].unique())>2:
                    nodes_to_correct=sorted(df_to_check['node'].unique().tolist())
                    if max_corrections is None:
                        allowed_corrections=len(nodes_to_correct)-1
                    else:
                        allowed_corrections=min(len(nodes_to_correct)-1, max_corrections)
                    #print('correct', nodes_to_correct)

                    for cur_allowed_corrections in range(1,allowed_corrections):
                        new_correct, corrections=bl_recursive_correct(tree_df, node, allowed_corrections=cur_allowed_corrections, conditions=conditions,
                                                                      max_corrections=allowed_corrections, verbose=verbose)
                        if new_correct:
                            break
                    if new_correct:
                        if verbose:
                            print('Successful corrections:', str(corrections))
                        for correction in corrections:
                            tree_df=dtree.unite_nodes(tree_df, correction)
                if new_correct==False:
                    if verbose:
                        print('No successful corrections were found. Pruning node', node)
                    tree_df=dtree.prune(tree_df, node)
                nodes_to_check=tree_df[tree_df['leaf']==False].sort_values(['depth', 'node'])['node'].tolist()

            if current_node_index+1<len(nodes_to_check):
                current_node_index+=1
                to_check=True

        if to_correct:
            return True, tree_df
        else:
            return correct_all, tree_df





    def color_result(self, x):
        '''
        TECH

        Defines result cell color for excel export

        Parameters
        -----------
        x: input values

        Returns
        --------
        color description for style.apply()
        '''
        colors=[]
        for e in x:
            if e:
                colors.append('background-color: green')
            else:
                colors.append('background-color: red')
        return colors
#---------------------------------------------------------------




#added 13.08.2018 by Yudochev Dmitry
class WOEOrderChecker(Processor):
    '''
    Class for WoE order checking
    '''
    def __init__(self):
        self.stats = pd.DataFrame()


    def work(self, feature, datasamples, dr_threshold=0.01, correct_threshold=0.85, woe_adjust=0.5, miss_is_incorrect=True,
             verbose=False, out=False, out_images='WOEOrderChecker/'):
        '''
        Checks if WoE order of the feature remains stable in bootstrap

        Parameters
        -----------
        feature: an object of FeatureWOE type that should be checked
        datasamples: an object of DataSamples type containing the samples to check input feature on
        dr_threshold: if WoE order is not correct, then default rate difference between swaped bins is checked
        correct_threshold: what part of checks on bootstrap should be correct for feature to pass the check
        woe_adjust: woe adjustment factor (for Default_Rate_i formula)
        miss_is_incorrect: is there is no data for a bin on bootstrap sample, should it be treated as error or not
        verbose: if comments and graphs should be printed
        out: a boolean for image output or a path for csv/xlsx output file to export woe and er values
        out_images: a path for image output (default - WOEOrderChecker/)

        Returns
        ----------
        Boolean - whether the check was successful
        and if isinstance(out, str) then dataframes with WoE and ER values for groups per existing sample
        '''

        if out:
            directory = os.path.dirname(out_images)
            if not os.path.exists(directory):
                os.makedirs(directory)

        w={x:feature.woes[x] for x in feature.woes if feature.woes[x] is not None}
        woes_df=pd.DataFrame(w, index=['Train']).transpose().reset_index().rename({'index':'group'},axis=1).sort_values('group')
        if isinstance(out, str):
            out_woes=woes_df.copy()
            if feature.data.weights is None:
                out_er=woes_df.drop('Train', axis=1).merge(feature.data.dataframe[['group', feature.data.target]].groupby('group', as_index=False).mean(),
                                     on='group').rename({feature.data.target:'Train'}, axis=1)
            else:
                for_er=feature.data.dataframe[['group', feature.data.target, feature.data.weights]]
                for_er[feature.data.target]=for_er[feature.data.target]*for_er[feature.data.weights]
                out_er=woes_df.drop('Train', axis=1).merge(for_er[['group', feature.data.target]].groupby('group', as_index=False).mean(),
                                     on='group').rename({feature.data.target:'Train'}, axis=1)

            cur_sample_woe=pd.DataFrame(columns=['group', 'woe', 'event_rate'])
            samples=[datasamples.validate, datasamples.test]
            sample_names=['Validate', 'Test']
            for si in range(len(samples)):
                if samples[si] is not None:
                    to_keep=[feature.feature, samples[si].target]
                    if samples[si].weights is not None:
                        to_keep.append(samples[si].weights)
                    cur_sample=samples[si].dataframe[to_keep]
                    cur_sample['group']=feature.set_groups(woes=feature.woes, original_values=True, data=cur_sample[feature.feature])

                    #cur_sample=cur_sample.sort_values('group')

                    if samples[si].weights is None:
                        N_b = cur_sample[samples[si].target].sum()
                        N_g = (1-cur_sample[samples[si].target]).sum()
                    else:
                        N_b = cur_sample[cur_sample[samples[si].target] == 1][samples[si].weights].sum()
                        N_g = cur_sample[cur_sample[samples[si].target] == 0][samples[si].weights].sum()
                    DR = N_b*1.0/N_g
                    index=-1
                    # for each interval
                    for gr_i in sorted(cur_sample['group'].unique()):
                        index=index+1
                        if samples[si].weights is None:
                            N_b_i = cur_sample[cur_sample['group']==gr_i][samples[si].target].sum()
                            N_g_i = cur_sample[cur_sample['group']==gr_i].shape[0] - N_b_i
                        else:
                            N_b_i = cur_sample[(cur_sample['group']==gr_i)&(cur_sample[samples[si].target] == 1)][samples[si].weights].sum()
                            N_g_i = cur_sample[(cur_sample['group']==gr_i)&(cur_sample[samples[si].target] == 0)][samples[si].weights].sum()
                        if not(N_b_i==0 and N_g_i==0):
                            DR_i = (N_b_i + woe_adjust)/(N_g_i + woe_adjust)
                            ER_i=N_b_i/(N_b_i+N_g_i)
                            n = N_g_i + N_b_i
                            smoothed_woe_i = np.log(DR*(feature.alpha + n)/(n*DR_i + feature.alpha))#*DR))
                        cur_sample_woe.loc[index]=[gr_i, smoothed_woe_i, ER_i]

                    out_woes=out_woes.merge(cur_sample_woe.drop('event_rate', axis=1), on='group').rename({'woe':samples[si].name}, axis=1)
                    out_er=out_er.merge(cur_sample_woe.drop('woe', axis=1), on='group').rename({'event_rate':samples[si].name}, axis=1)
                else:
                    out_woes[sample_names[si]]=np.nan
                    out_er[sample_names[si]]=np.nan

        if datasamples.bootstrap_base is not None:
            if verbose:
                fig = plt.figure(figsize=(15,7))

            bootstrap_correct=[]

            to_keep=[feature.feature, datasamples.bootstrap_base.target]+([datasamples.bootstrap_base.weights] if datasamples.bootstrap_base.weights is not None else [])
            base_with_group=datasamples.bootstrap_base.dataframe[to_keep]
            base_with_group['group']=feature.set_groups(woes=feature.woes, original_values=True, data=base_with_group[feature.feature])

            for bn in range(len(datasamples.bootstrap)):
                cur_sample_woe=pd.DataFrame(columns=['group', 'train_woe', 'woe', 'event_rate'])

                cur_sample=base_with_group.iloc[datasamples.bootstrap[bn]]
                #cur_sample['train_woe']=cur_sample['group'].apply(lambda x: feature.woes[x])
                #cur_sample=cur_sample.sort_values('group')

                if datasamples.bootstrap_base.weights is None:
                    N_b = cur_sample[datasamples.bootstrap_base.target].sum()
                    N_g = (1-cur_sample[datasamples.bootstrap_base.target]).sum()
                else:
                    N_b = cur_sample[cur_sample[datasamples.bootstrap_base.target] == 1][datasamples.bootstrap_base.weights].sum()
                    N_g = cur_sample[cur_sample[datasamples.bootstrap_base.target] == 0][datasamples.bootstrap_base.weights].sum()
                DR = N_b*1.0/N_g
                index=-1
                # for each interval
                for gr_i in sorted(cur_sample['group'].unique()):
                    index=index+1
                    if datasamples.bootstrap_base.weights is None:
                        N_b_i = cur_sample[cur_sample['group']==gr_i][datasamples.bootstrap_base.target].sum()
                        N_g_i = cur_sample[cur_sample['group']==gr_i].shape[0] - N_b_i
                    else:
                        N_b_i = cur_sample[(cur_sample['group']==gr_i)&(cur_sample[datasamples.bootstrap_base.target] == 1)][datasamples.bootstrap_base.weights].sum()
                        N_g_i = cur_sample[(cur_sample['group']==gr_i)&(cur_sample[datasamples.bootstrap_base.target] == 0)][datasamples.bootstrap_base.weights].sum()
                    if not(N_b_i==0 and N_g_i==0):
                        DR_i = (N_b_i + woe_adjust)/(N_g_i + woe_adjust)
                        ER_i=N_b_i/(N_b_i+N_g_i)
                        n = N_g_i + N_b_i
                        smoothed_woe_i = np.log(DR*(feature.alpha + n)/(n*DR_i + feature.alpha))#*DR))
                    cur_sample_woe.loc[index]=[gr_i, feature.woes[gr_i], smoothed_woe_i, ER_i]

                if isinstance(out, str):
                    out_woes=out_woes.merge(cur_sample_woe.drop('event_rate', axis=1), on='group').rename({'woe':'Bootstrap'+str(bn)}, axis=1).drop('train_woe', axis=1)
                    out_er=out_er.merge(cur_sample_woe.drop('woe', axis=1), on='group').rename({'event_rate':'Bootstrap'+str(bn)}, axis=1).drop('train_woe', axis=1)

                cur_sample_woe['trend_train']=np.sign((cur_sample_woe['train_woe']-cur_sample_woe['train_woe'].shift(1)).dropna())
                cur_sample_woe['trend']=np.sign((cur_sample_woe['woe']-cur_sample_woe['woe'].shift(1)).dropna())
                cur_sample_woe['prev_event_rate']=cur_sample_woe['event_rate'].shift(1)

                cur_sample_woe_error=cur_sample_woe[cur_sample_woe['trend_train']!=cur_sample_woe['trend']].dropna(how='all', subset=['trend_train','trend'])
                cur_sample_correct=True
                if cur_sample_woe.shape[0]!=0:
                    for ind, row in cur_sample_woe_error.iterrows():
                        if abs(row['event_rate']-row['prev_event_rate'])>dr_threshold:
                            cur_sample_correct=False
                if miss_is_incorrect:
                    cur_sample_correct=cur_sample_correct and woes_df.merge(cur_sample_woe, on='group', how='left')['woe'].notnull().all()

                if verbose:
                    line_color='green' if cur_sample_correct else 'red'
                    plt.plot(range(woes_df.shape[0]), woes_df.merge(cur_sample_woe, on='group', how='left')['woe'],
                             color=line_color, alpha=0.4)

                bootstrap_correct.append(cur_sample_correct*1)

            bootstrap_correct_part=sum(bootstrap_correct)/len(bootstrap_correct)
            result=(bootstrap_correct_part>=correct_threshold)

            if verbose:
                plt.plot(range(woes_df.shape[0]), woes_df['Train'], color='blue', linewidth=5.0)
                plt.ylabel('WoE')
                plt.xticks(range(woes_df.shape[0]), woes_df['group'])
                plt.suptitle(feature.feature, fontsize = 16)
                fig.autofmt_xdate()
                if out:
                    plt.savefig(out_images+feature.feature+".png", dpi=100, bbox_inches='tight')
                plt.show()
                print('Correct WoE order part = '+str(round(bootstrap_correct_part,4))+' ('+str(sum(bootstrap_correct))+' out of '+str(len(bootstrap_correct))+'), threshold = '+str(correct_threshold))
                if bootstrap_correct_part<correct_threshold:
                    print('Not stable enough WoE order.')

        else:
            if verbose:
                print('No bootstrap samples were found is DataSamples object. Return True.')
            result=True

        if isinstance(out, str):
            return result, out_woes, out_er
        else:
            return result



    def work_all(self, woe, features=None, drop_features=False, dr_threshold=0.01, correct_threshold=0.85, woe_adjust=0.5,
                 miss_is_incorrect=True, verbose=False, out=False, out_images='WOEOrderChecker/',
                 out_woe_low=0, out_woe_high=0, out_er_low=0, out_er_high=0):
        '''
        Checks if WoE order of all features from WOE object remains stable in bootstrap

        Parameters
        -----------
        woe: an object of FeatureWOE type that should be checked
        features: a list of features to check (if None, then all features from woe object will be checked)
        drop_features: should the features be dropped from WOE.feature_woes list in case of failed checks
        dr_threshold: if WoE order is not correct, then default rate difference between swaped bins is checked
        correct_threshold: what part of checks on bootstrap should be correct for feature to pass the check
        woe_adjust: woe adjustment factor (for Default_Rate_i formula)
        miss_is_incorrect: is there is no data for a bin on bootstrap sample, should it be treated as error or not
        verbose: if comments and graphs should be printed
        out: a boolean for image output or a path for xlsx output file to export woe and er values
        out_images: a path for image output (default - WOEOrderChecker/)
        out_woe_low: correcting coefficient for lower edge of WoE gradient scale (if out is str)
        out_woe_high: correcting coefficient for upper edge of WoE gradient scale (if out is str)
        out_er_low: correcting coefficient for lower edge of ER gradient scale (if out is str)
        out_er_high: correcting coefficient for upper edge of ER gradient scale (if out is str)

        Returns
        ----------
        Dictionary with results of check for all features from input WOE object
        '''
        if features is None:
            cycle_features=list(woe.feature_woes)
        else:
            cycle_features=list(features)

        not_in_features_woe=[x for x in cycle_features if x not in woe.feature_woes]
        if len(not_in_features_woe)>0:
            print('No', not_in_features_woe, 'in self.feature_woes. Abort.')
            return None

        if out:
            directory = os.path.dirname(out_images)
            if not os.path.exists(directory):
                os.makedirs(directory)

        woe_order_correct={}
        if isinstance(out, str):
            woes_df=pd.DataFrame(columns=['feature', 'group', 'Train', 'Validate', 'Test']+['Bootstrap'+str(i) for i in range(len(woe.datasamples.bootstrap))])
            er_df=pd.DataFrame(columns=['feature', 'group', 'Train', 'Validate', 'Test']+['Bootstrap'+str(i) for i in range(len(woe.datasamples.bootstrap))])

        for feature in cycle_features:
            if isinstance(out, str):
                woe_order_correct[feature], woes_df_feature, er_df_feature=self.work(woe.feature_woes[feature], datasamples=woe.datasamples, dr_threshold=dr_threshold,
                                                                                     correct_threshold=correct_threshold, woe_adjust=woe_adjust,
                                                                                     miss_is_incorrect=miss_is_incorrect, verbose=verbose, out=out, out_images=out_images)
                woes_df_feature['feature']=feature
                er_df_feature['feature']=feature
                woes_df=woes_df.append(woes_df_feature, ignore_index=True)
                er_df=er_df.append(er_df_feature, ignore_index=True)
            else:
                woe_order_correct[feature]=self.work(woe.feature_woes[feature], datasamples=woe.datasamples, dr_threshold=dr_threshold,
                                                     correct_threshold=correct_threshold, woe_adjust=woe_adjust,
                                                     miss_is_incorrect=miss_is_incorrect, verbose=verbose, out=out, out_images=out_images)

        if isinstance(out, str):
            woes_df=woes_df[['feature', 'group', 'Train', 'Validate', 'Test']+['Bootstrap'+str(i) for i in range(len(woe.datasamples.bootstrap))]].dropna(axis=1)
            er_df=er_df[['feature', 'group', 'Train', 'Validate', 'Test']+['Bootstrap'+str(i) for i in range(len(woe.datasamples.bootstrap))]].dropna(axis=1)
            df_columns=[x for x in woes_df.columns if x not in ['feature', 'group']]
            if out[-4:]=='.csv':
                print('Unappropriate format for exporting two tables. Use .xlsx. Skipping export.')
            elif out[-4:]=='.xls' or out[-5:]=='.xlsx':
                writer = pd.ExcelWriter(out, engine='openpyxl')
                woes_values=woes_df[df_columns].values.reshape(-1,).tolist()
                woes_df.style.apply(color_background,
                                    mn=np.mean(woes_values)-2*np.std(woes_values),
                                    mx=np.mean(woes_values)+2*np.std(woes_values),
                                    cmap='RdYlGn', subset=df_columns,
                                    high=out_woe_high, low=out_woe_low).to_excel(writer, sheet_name='WoE by Samples', index=False)
                er_values=er_df[df_columns].values.reshape(-1,).tolist()
                er_df.style.apply(color_background,
                                  mn=max([0,np.mean(er_values)-2*np.std(er_values)]),
                                  mx=np.mean(er_values)+2*np.std(er_values),
                                  cmap='RdYlGn_r', subset=df_columns,
                                  high=out_er_high, low=out_er_low).to_excel(writer, sheet_name='Event Rate by Samples', index=False)
                # Get the openpyxl objects from the dataframe writer object.
                for worksheet in writer.sheets:
                    for x in writer.sheets[worksheet].columns:
                        writer.sheets[worksheet].column_dimensions[x[0].column].width = 40 if x[0].column=='A' else 12
                writer.save()
            else:
                print('Unknown format for export file. Use .xlsx. Skipping export.')

        if drop_features:
            woe.excluded_feature_woes.update({x:woe.feature_woes[x] for x in woe.feature_woes if woe_order_correct[x]==False})
            woe.feature_woes={x:woe.feature_woes[x] for x in woe.feature_woes if woe_order_correct[x]}
        return woe_order_correct




    def work_tree(self, dtree, input_df=None, er_threshold=0.01, correct_threshold=0.85,
                  miss_is_incorrect=True, to_correct=False, max_corrections=2, verbose=False):
        '''
        Checks if WoE order of the tree remains stable in bootstrap and corrects the tree for it to pass the check

        Parameters
        -----------
        dtree: a cross.DecisionTree object
        input_df: a DataFrame, containing tree description
        er_threshold: if WoE order is not correct, then event rate difference between swaped bins is checked
        correct_threshold: what part of checks on bootstrap should be correct for tree to pass the check
        miss_is_incorrect: if there is no data for a bin on bootstrap sample, should it be treated as error or not
        to_correct: should there be attempts to correct tree by uniting nodes/groups or not
        max_corrections: maximal number of corrections in attempt to change the tree so it will pass the check
        verbose: if comments and graphs should be printed (False - no output, True or 1 - actions log, 2 - actions and interim checks)

        Returns
        ----------
        if to_correct:
            True and a DataFrame with tree description - corrected or initial
        else:
            result of the input tree check and the input tree itself
        '''

        #-----------------------------------------------Subsidiary functions--------------------------------------------------
        def woeo_check(input_df, bootstrap_tree_list, er_threshold=0.01, correct_threshold=0.85, miss_is_incorrect=True, verbose=False):
            '''
            TECH
            WoE order stabitility check

            Parameters
            -----------
            input_df: a DataFrame, containing tree description
            bootstrap_sample_list: a list of DataFrames, cantaining the same tree, as in tree_df, but with stats from bootstrap samples
            er_threshold: if WoE order is not correct, then event rate difference between swaped bins is checked
            correct_threshold: what part of checks on bootstrap should be correct for tree to pass the check
            miss_is_incorrect: if there is no data for a bin on bootstrap sample, should it be treated as error or not
            verbose: if comments and graphs should be printed

            Returns
            ----------
            boolean flag of current tree passing the check and
            the dictionary of {group:number of errors}
            '''
            tree_df=input_df.copy()
            tree_groups_stats=tree_df[['group','group_woe', 'group_target', 'group_amount']].rename({'group_woe':'train_woe'}, axis=1).\
                                dropna().drop_duplicates().reset_index(drop=True).sort_values('group')
            tree_groups_stats['group_er']=tree_groups_stats['group_target']/tree_groups_stats['group_amount']
            features=list(tree_df.columns[:tree_df.columns.get_loc('node')])
            if verbose>True:
                fig = plt.figure(figsize=(15,7))
            bootstrap_correct=[]
            bootstrap_groups_error={}
            for i in range(len(bootstrap_tree_list)):
                #display(bootstrap_tree_list[i][bootstrap_tree_list[i]['leaf']])
                groups_stats=bootstrap_tree_list[i][bootstrap_tree_list[i]['leaf']][['group', 'group_woe']].drop_duplicates().\
                                merge(tree_groups_stats, on='group', how='left').sort_values('group')
                groups_stats['trend_train']=np.sign((groups_stats['train_woe']-groups_stats['train_woe'].shift(1)).dropna())
                groups_stats['trend']=np.sign((groups_stats['group_woe']-groups_stats['group_woe'].shift(1)).dropna())
                groups_stats['prev_er']=groups_stats['group_er'].shift(1)
                #display(groups_stats)

                groups_error=groups_stats[groups_stats['trend_train']!=groups_stats['trend']].dropna(how='all', subset=['trend_train','trend'])
                sample_correct=True
                if groups_stats.shape[0]!=0:
                    for ind, row in groups_error.iterrows():
                        if abs(row['group_er']-row['prev_er'])>er_threshold:
                            if row['group'] in bootstrap_groups_error:
                                bootstrap_groups_error[row['group']]+=1
                            else:
                                bootstrap_groups_error[row['group']]=1
                            sample_correct=False
                if miss_is_incorrect:
                    sample_correct=sample_correct and tree_groups_stats.merge(groups_stats, on='group', how='left')['group_woe'].notnull().all()

                if verbose>True:
                    line_color='green' if sample_correct else 'red'
                    plt.plot(range(tree_groups_stats.shape[0]), tree_groups_stats.merge(groups_stats, on='group', how='left')['group_woe'],
                             color=line_color, alpha=0.4)

                bootstrap_correct.append(sample_correct*1)

            bootstrap_correct_part=sum(bootstrap_correct)/len(bootstrap_correct)
            result=(bootstrap_correct_part>=correct_threshold)

            if verbose>True:
                plt.plot(range(tree_groups_stats.shape[0]), tree_groups_stats['train_woe'], color='blue', linewidth=5.0)
                plt.ylabel('WoE')
                plt.xticks(range(tree_groups_stats.shape[0]), tree_groups_stats['group'])
                plt.suptitle('Tree on '+str(features), fontsize = 16)
                fig.autofmt_xdate()
                plt.show()
            if verbose:
                print('Correct WoE order part = '+str(round(bootstrap_correct_part,4))+' ('+str(sum(bootstrap_correct))+' out of '+str(len(bootstrap_correct))+'), threshold = '+str(correct_threshold))
                if bootstrap_correct_part<correct_threshold:
                    print('Not stable enough WoE order.')

            return result, bootstrap_groups_error

        def woeo_recursive_correct(tree_df, bootstrap_sample_list, worst_group, allowed_corrections=1, corrections=None, verbose=False,
                                  er_threshold=0.01, correct_threshold=0.85, miss_is_incorrect=True,
                                  max_corrections=1):
            '''
            TECH
            Recursive search of corrections needed for tree to pass WoE order stabitility check

            Parameters
            -----------
            tree_df: a DataFrame, containing tree description
            bootstrap_sample_list: a list of DataFrames, cantaining the same tree, as in tree_df, but with stats from bootstrap samples
            worst_group: a number of group, in which the most part of errors was found during WoE order check (so this group switched places with the one before it)
            allowed_corrections: a number of remaining corrections, that are allowed
            corrections: the list of current corrections
            verbose: if comments and graphs should be printed
            er_threshold: if WoE order is not correct, then event rate difference between swaped bins is checked
            correct_threshold: what part of checks on bootstrap should be correct for tree to pass the check
            miss_is_incorrect: if there is no data for a bin on bootstrap sample, should it be treated as error or not
            max_corrections: maximal number of corrections in attempt to change the tree so it will pass the check

            Returns
            ----------
            boolean flag of corrected tree passing the check and
            the list of corrections, that were made
            '''

            if corrections is None:
                corrections=[]

            if allowed_corrections>0:
                possible_nodes_to_correct=[]
                #each group is compared to the previous one and the worst_group has the most number of errors with the previous
                #group (by woe), also groups are numbered by woe, so we need to check the worst, the previous one and also the one
                #before previous and the one after the worst (because maybe we need to unite two groups before the worst, or
                #the worst and the next or both variants)
                for g in range(worst_group-2, worst_group+2):
                    group_filter=(tree_df['group']==g)
                    #if group exists and it is not a united group
                    if group_filter.sum()==1:
                        add_node_series=tree_df[group_filter]
                        possible_nodes_to_correct.append(add_node_series['node'].values[0])
                        #also for groups that were changing their places in woe order we are looking for adjacent nodes that are leaves
                        #and are not in united groups (because there are cases like this one: groups (3,1), (2,0), most errors are
                        #between 0 and 1, but above we are looking only at groups -1, 0, 1 and 2, so we also need to check the factual
                        #tree structure, not only groups woe order)
                        if g in [worst_group-1, worst_group]:
                            adjacent_filter=(tree_df['parent_node']==add_node_series['parent_node'].values[0])&(tree_df['node']!=add_node_series['node'].values[0])
                            if adjacent_filter.sum()==1:
                                adjacent_node_series=tree_df[adjacent_filter]
                                if tree_df[tree_df['group']==adjacent_node_series['group'].values[0]].shape[0]==1 and adjacent_node_series['leaf'].values[0]:
                                    possible_nodes_to_correct.append(adjacent_node_series['node'].values[0])

                possible_nodes_to_correct=sorted(list(set(possible_nodes_to_correct)))
                combinations=[]
                for n1 in range(len(possible_nodes_to_correct)):
                    first_node=possible_nodes_to_correct[n1]
                    for n2 in range(len(possible_nodes_to_correct[n1+1:])):
                        second_node=possible_nodes_to_correct[n1+1:][n2]
                        if dtree.check_unitability(tree_df, [first_node, second_node]):
                            first_node_series=tree_df[tree_df['node']==first_node]
                            parent_node=first_node_series['parent_node'].values[0]
                            split_feature=tree_df[(tree_df['node']==parent_node)]['split_feature'].values[0]
                            first_condition=first_node_series[split_feature].values[0]
                            if not(isinstance(first_condition, list) or isinstance(first_condition, tuple)):
                                nodes_combination=[second_node, first_node]
                            else:
                                nodes_combination=[first_node, second_node]
                            combinations.append([nodes_combination,
                                                abs(first_node_series['woe'].values[0]- \
                                                    tree_df[tree_df['node']==second_node]['woe'].values[0])])
                combinations.sort(key=itemgetter(1))
                if verbose:
                    print('\t'*(max_corrections-allowed_corrections)+'Possible corrections (by nodes):',
                          str([x[0] for x in combinations]))

                for nodes_to_unite, woe in combinations:
                    if verbose:
                        print('\t'*(max_corrections-allowed_corrections+1)+'Checking (level =',max_corrections-allowed_corrections+1,
                              '): uniting nodes', str(nodes_to_unite), 'with woe difference =', woe)
                    corrected_tree_df=dtree.unite_nodes(tree_df, nodes_to_unite)
                    corrected_bootstrap_sample_list=[]
                    for bs_tree in bootstrap_sample_list:
                        corrected_bootstrap_sample=dtree.unite_nodes(bs_tree, nodes_to_unite)
                        corrected_bootstrap_sample=corrected_bootstrap_sample.drop('group', axis=1).merge(corrected_tree_df[['node', 'group']], on='node', how='left')
                        corrected_bootstrap_sample_list.append(corrected_bootstrap_sample)
                    correct, errors = woeo_check(corrected_tree_df, corrected_bootstrap_sample_list, verbose=(verbose>True),
                                                      er_threshold=er_threshold, correct_threshold=correct_threshold,
                                                      miss_is_incorrect=miss_is_incorrect)
                    if correct:
                        if verbose:
                            print('\t'*(max_corrections-allowed_corrections+1)+'Corrections',
                                  str(corrections+[nodes_to_unite]), 'succeeded!')
                        return correct, corrections+[nodes_to_unite]
                    else:
                        if allowed_corrections==1:
                            if verbose:
                                print('\t'*(max_corrections-allowed_corrections+1)+'Maximum correction level reached. Corrections',
                                      str(corrections+[nodes_to_unite]), 'failed.')
                        else:
                            new_worst_group=int(max(errors, key=errors.get))
                            if verbose:
                                print('\t'*(max_corrections-allowed_corrections+1)+'Most errors were produced by',
                                      new_worst_group, 'group. Trying to correct..')
                            correct, final_corrections=woeo_recursive_correct(corrected_tree_df, corrected_bootstrap_sample_list,
                                                                            new_worst_group, allowed_corrections-1,
                                                                            corrections+[nodes_to_unite], verbose,
                                                                            er_threshold=er_threshold, correct_threshold=correct_threshold,
                                                                            miss_is_incorrect=miss_is_incorrect, max_corrections=max_corrections)
                    if correct:
                        return correct, final_corrections
            return False, corrections
        #---------------------------------------------------------------------------------------------------------------

        if input_df is None:
            tree_df=dtree.tree.copy()
        else:
            tree_df=input_df.copy()
        datasamples=dtree.datasamples
        features=[x for x in dtree.features if x in tree_df]
            #list(tree_df.columns[:tree_df.columns.get_loc('node')])

        if max_corrections is None:
            max_corrections=2

        alpha=dtree.alpha
        woe_adjust=dtree.woe_adjust

        bootstrap_sample_list=[]
        if datasamples.bootstrap_base is not None:
            base_for_woe=datasamples.bootstrap_base.keep(features=features).dataframe
            base_for_woe['node']=dtree.transform(base_for_woe, tree_df, ret_values=['node'])
            base_for_woe['---weight---']=base_for_woe[datasamples.bootstrap_base.weights] if datasamples.bootstrap_base.weights is not None else 1
            base_for_woe[datasamples.bootstrap_base.target]=base_for_woe[datasamples.bootstrap_base.target]*base_for_woe['---weight---']
            base_for_woe=base_for_woe[['node', datasamples.bootstrap_base.target, '---weight---']]

            for i in range(len(datasamples.bootstrap)):
                for_woe=base_for_woe.iloc[datasamples.bootstrap[i]]
                groups_stats=for_woe.groupby('node', as_index=False).sum()
                groups_stats.columns=['node', 'target', 'amount']
                sample_tree_df=tree_df[features+['node', 'parent_node', 'depth', 'leaf', 'split_feature', 'group']].merge(groups_stats, on=['node'], how='left')
                for parent_node in sample_tree_df.sort_values('depth', ascending=False)['parent_node'].unique():
                    for v in ['target', 'amount']:
                        sample_tree_df.loc[sample_tree_df['node']==parent_node, v]=sample_tree_df.loc[sample_tree_df['parent_node']==parent_node, v].sum()

                sample_tree_df['nontarget']=sample_tree_df['amount']-sample_tree_df['target']
                good_bad=[sample_tree_df[sample_tree_df['leaf']]['nontarget'].sum(),
                          sample_tree_df[sample_tree_df['leaf']]['target'].sum()]

                sample_tree_df['er']=sample_tree_df['target']/sample_tree_df['amount']
                sample_tree_df['woe']=np.log(((good_bad[1]/good_bad[0])*(alpha + sample_tree_df['amount'])/ \
                                      (sample_tree_df['amount']*((sample_tree_df['target'] + woe_adjust)/(sample_tree_df['nontarget'] + woe_adjust)) + alpha)).astype(float))
                groupped=sample_tree_df[['group', 'target', 'nontarget', 'amount']].groupby('group', as_index=False).sum().rename({'target':'group_target', 'nontarget':'group_nontarget', 'amount':'group_amount'}, axis=1)
                groupped['group_woe']=np.log(((good_bad[1]/good_bad[0])*(alpha + groupped['group_amount'])/ \
                                               (groupped['group_amount']*((groupped['group_target'] + woe_adjust)/(groupped['group_nontarget'] + woe_adjust)) + alpha)).astype(float))
                sample_tree_df=sample_tree_df.merge(groupped, on=['group'], how='left')

                bootstrap_sample_list.append(sample_tree_df)
        else:
            if verbose:
                print('No bootstrap samples were found. Skipping WoE order check..')
            return True, tree_df

        correct, errors = woeo_check(tree_df, bootstrap_sample_list, verbose=verbose,
                                     er_threshold=er_threshold, correct_threshold=correct_threshold,
                                     miss_is_incorrect=miss_is_incorrect)

        if to_correct and correct==False:
            worst_group=int(max(errors, key=errors.get))
            if verbose:
                print('Most errors were produced by', worst_group, 'group. Trying to correct..')
            new_correct=False
            allowed_corrections=min(tree_df[tree_df['leaf']].shape[0]-2, max_corrections)
            #print('allowed_corrections', allowed_corrections)
            for cur_allowed_corrections in range(1,allowed_corrections+1):
                if verbose:
                    print('Current maximum number of corrections =', cur_allowed_corrections)
                new_correct, corrections=woeo_recursive_correct(tree_df, bootstrap_sample_list, worst_group,
                                                                allowed_corrections=cur_allowed_corrections, corrections=[],
                                                                verbose=verbose,
                                                                er_threshold=er_threshold, correct_threshold=correct_threshold,
                                                                miss_is_incorrect=miss_is_incorrect, max_corrections=cur_allowed_corrections)
                #print('new_correct', new_correct)
                if new_correct:
                    break
            if new_correct:
                if verbose:
                    print('Successful corrections:', str(corrections))
                for correction in corrections:
                    tree_df=dtree.unite_nodes(tree_df, correction)
                correct=new_correct
            else:
                if verbose:
                    print('No successful corrections were found. Proceed to uniting groups..')
                while correct==False:
                    if verbose:
                        print('Uniting groups:', worst_group, worst_group-1)
                    tree_df=dtree.unite_groups(tree_df, [worst_group, worst_group-1])
                    corrected_bootstrap_sample_list=[]
                    for bs_tree in bootstrap_sample_list:
                        corrected_bootstrap_sample=dtree.unite_groups(bs_tree, [worst_group, worst_group-1])
                        corrected_bootstrap_sample=corrected_bootstrap_sample.drop('group', axis=1).merge(tree_df[['node', 'group']], on='node', how='left')
                        corrected_bootstrap_sample_list.append(corrected_bootstrap_sample)
                    bootstrap_sample_list=corrected_bootstrap_sample_list.copy()
                    correct, errors = woeo_check(tree_df, bootstrap_sample_list, verbose=verbose,
                                                 er_threshold=er_threshold, correct_threshold=correct_threshold,
                                                 miss_is_incorrect=miss_is_incorrect)
                    worst_group=int(max(errors, key=errors.get)) if len(errors)>0 else None


        if to_correct:
            for g in tree_df['group'].unique():
                group_nodes=tree_df[tree_df['group']==g]['node'].tolist()
                for i in range(len(group_nodes)):
                    for j in range(len(group_nodes[i+1:])):
                        if dtree.check_unitability(tree_df, [group_nodes[i], group_nodes[i+1:][j]]):
                            if verbose:
                                print('Unitable nodes', str([group_nodes[i], group_nodes[i+1:][j]]), 'were found in the same group. Uniting..')
                            tree_df=dtree.unite_nodes(tree_df, [group_nodes[i], group_nodes[i+1:][j]])

            return True, tree_df
        else:
            return correct, tree_df



#---------------------------------------------------------------




# Author - Dmitry Yudochev 24.08.2018
class WaldBSChecker(Processor):
    '''
    Class for coefficient significance checking on bootstrap samples
    '''
    def __init__(self):
        self.stats = pd.DataFrame()


    def work(self, model, woe, refit_data=None, drop_features=False, features_to_leave=None,
             pvalue_threshold=0.05, correctness_threshold=0.85, verbose=False, out=False, out_images='WaldBSChecker/'):
        '''
        Checks model's coefficients significance in bootstrap. If drop_features==True, then features will be
        dropped and the model will be refitted

        Parameters
        -----------
        model: an object of LogisticRegressionModel type, whose coefficients should be checked
        woe: a WOE object, containing feature transformations to WoE
        refit_data: data with woe-transformed feature to refit model in case of drop_features=True
        drop_features: should the features be dropped in case of not stable enough significance of their coefficients
        features_to_leave: features not to be dropped in any case
        pvalue_threshold: maximal p-value for Wald chi-square statistic for coefficient to be considered significant
        correctness_threshold: minimal part of bootstrap samples on which coefficient stays significant for feature
            to be considered having coefficient with stable enough significance
        verbose: if comments and graphs should be printed
        out: a boolean for image output or a path for xlsx output file to export p-value by iteration values
        out_images: a path for image output (default - WaldBSChecker/)

        Returns
        ----------
        a dataframe with standard error, wald statistic and p-value for feature coefficients in case of not dropping them or
        a list of features to stay
        '''

        if features_to_leave is None:
            features_to_leave=[]

        if out:
            directory = os.path.dirname(out_images)
            if not os.path.exists(directory):
                os.makedirs(directory)

        to_check_wald=True

        if woe.datasamples.bootstrap_base is not None:
            if isinstance(out,str) and (out[-4:]=='.xls' or out[-5:]=='.xlsx'):
                writer = pd.ExcelWriter(out, engine='openpyxl')
            with_woe=woe.transform(woe.datasamples.bootstrap_base, original_values=True, calc_gini=False)
            it=0
            while to_check_wald:
                it=it+1
                if out:
                    wald_df=pd.DataFrame(index=['intercept']+model.selected)
                    samples=[woe.datasamples.train, woe.datasamples.validate, woe.datasamples.test]
                    sample_names=['Train', 'Validate', 'Test']
                    for si in range(len(samples)):
                        if samples[si] is not None:
                            sample_wald=model.wald_test(samples[si], woe_transform=woe)
                            wald_df=wald_df.join(sample_wald[['feature', 'p-value']].set_index('feature')).rename({'p-value':sample_names[si]}, axis=1)
                        else:
                            wald_df[sample_names[si]]=np.nan
                    wald_df=wald_df.join(sample_wald[['feature','coefficient']].set_index('feature'))#.rename({0:'Train', 1:'Validate', 2:'Test'}, axis=1)

                to_check_wald=False
                wald_correct={}
                for f in model.selected:
                    wald_correct[f]=0
                for bn in range(len(woe.datasamples.bootstrap)):
                    if woe.datasamples.bootstrap_base.weights is not None:
                        w=model.wald_test(Data(with_woe.dataframe.iloc[woe.datasamples.bootstrap[bn]][model.selected+[woe.datasamples.bootstrap_base.target, woe.datasamples.bootstrap_base.weights]],
                                               woe.datasamples.bootstrap_base.target, features=model.selected, weights=woe.datasamples.bootstrap_base.weights))
                    else:
                        w=model.wald_test(Data(with_woe.dataframe.iloc[woe.datasamples.bootstrap[bn]][model.selected+[woe.datasamples.bootstrap_base.target]],
                                               woe.datasamples.bootstrap_base.target, features=model.selected))
                    if out:
                        wald_df=wald_df.join(w[['feature', 'p-value']].set_index('feature')).rename({'p-value':'Bootstrap'+str(bn)}, axis=1)

                    for f in model.selected:
                        wald_correct[f]=wald_correct[f]+(w[w['feature']==f]['p-value'].values[0]<pvalue_threshold)
                for f in model.selected:
                    wald_correct[f]=wald_correct[f]/len(woe.datasamples.bootstrap)
                if out:
                    #display(wald_df)
                    wald_df=wald_df[['coefficient', 'Train', 'Validate', 'Test']+['Bootstrap'+str(i) for i in range(len(woe.datasamples.bootstrap))]].dropna(axis=1)
                    wald_columns=[x for x in wald_df.columns if x!='coefficient']

                    for ind in wald_df.index:
                        p_value_list=wald_df[[x for x in wald_df.columns if x[:9]=='Bootstrap']].loc[ind].tolist()
                        mean=np.mean(p_value_list)
                        std=np.std(p_value_list)
                        sns.distplot(p_value_list)
                        plt.axvline(x=mean, linestyle='--', alpha=0.5)
                        plt.text(mean, 0, '  Mean = '+str(round(mean,8))+', std = '+str(round(std,8)),
                                 horizontalalignment='right', verticalalignment='bottom', rotation=90)
                        plt.xlabel('Wald p-values in bootstrap')
                        plt.ylabel('Distribution')
                        plt.title(str(ind), fontsize = 16)
                        plt.savefig(out_images+str(ind)+"_"+str(it)+".png", dpi=100, bbox_inches='tight')
                        plt.close()

                    if isinstance(out,str):
                        if out[-4:]=='.xls' or out[-5:]=='.xlsx':
                            wald_df.style.apply(color_background,
                                        mn=0,
                                        mx=pvalue_threshold,
                                        cmap='RdYlGn_r',
                                        subset=pd.IndexSlice[:, wald_columns]).to_excel(writer, sheet_name='Iteration '+str(it))
                             # Get the openpyxl objects from the dataframe writer object.
                            worksheet = writer.sheets['Iteration '+str(it)]
                            for x in worksheet.columns:
                                if x[0].column=='A':
                                    worksheet.column_dimensions[x[0].column].width = 40
                                else:
                                    worksheet.column_dimensions[x[0].column].width = 12
                                    for cell in worksheet[x[0].column]:
                                        cell.number_format = '0.00000000'
                            worksheet.freeze_panes = worksheet['C2']
                        else:
                            print('Unknown or unappropriate format for export several tables. Use .xlsx. Skipping export.')
                if drop_features and refit_data is not None:
                    if verbose:
                        display(wald_correct)
                    insignificant={x:refit_data.ginis[x] for x in wald_correct if wald_correct[x]<correctness_threshold and x not in features_to_leave}
                    if len(insignificant)>0:
                        feature_to_drop=min(insignificant, key=insignificant.get)
                        if verbose:
                            print('Dropping', feature_to_drop, 'because of not stable enough significance of coefficient: ', wald_correct[feature_to_drop], '(gini =', refit_data.ginis[feature_to_drop],')')
                        model.selected.remove(feature_to_drop)
                        model.fit(refit_data, selected_features=True)
                        if verbose:
                            model.draw_coefs()
                        to_check_wald=True
                else:
                    if verbose:
                        print('drop_features==False or no refit_data was specified, so returning wald test results')
                    return wald_correct
            if isinstance(out,str) and (out[-4:]=='.xls' or out[-5:]=='.xlsx'):
                writer.save()
            return model.selected
        else:
            print('No bootstrap samples!')
            return None



#---------------------------------------------------------------



class FullnessAnalyzer():
    '''
    Visualizing data fullness by blocks (if provided), using cluster analysis to split data according
    to its fullness and explaining clusters by training sklearn's Decision Tree on its labels (for
    easier interpretation and determining, to which cluster each observation belongs)

    For full functionality it needs hdbscan and fastcluster packages to be installed (because other methods
    works too slow with highly-dimensional data exceeding 50000 obs). Otherwise only sklearn's k-means
    clustering is available.

    Also for Decision Tree visualization graphviz needs to be installed in OS and graphviz_converter_path
    should be provided to .work and .explain_clusters methods (it is a path to dot.exe, which can transform
    .dot file to .png for it to be displayed in the notebook)

    Because sklearn's Decision Tree is used for explanation only interval values without inf or nan should be used.
    The safe (but maybe not the best) aproach is to base explanation on already transformed features.
    '''

    def __init__(self, data=None, linked=None, clusterer=None, categorical=None):
        '''
        Parameters
        -----------
        stats: a pandas DataFrame, containing technical information for report generation
        data: a Data object with transformed features and cluster labels (if the process has already reached the .get_clusters step)
        linked: a linkage matrix, which is generated by hierarchy clustering (used for getting cluster labels from)
        clusterer: a trained clusterer object (hdbscan.HDBSCAN or sklearn.cluster.KMeans) (used for getting cluster labels from)
        categorical:  a list of categorical features (their alues will be changed to -1 and 1 without 0)
        '''
        self.stats = pd.DataFrame()
        self.data = data
        self.linked = linked
        self.clusterer = clusterer
        self.conditions = None
        self.categorical = None


    def work(self, input_data=None, to_transform=True, to_train=True, to_explain=True,
             features=None, categorical=None, exclude=None, blocks=None, interval_min_unique=None,
             clusterer=None, clusterer_options=None, clusters_number=None,
             explainer_features=None, explainer_max_leaf_nodes=None, explainer_max_depth=None, explainer_min_samples_leaf=None,
             graphviz_converter_path='C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe', verbose=True):
        '''
        Fully process input data for fullness clustering meaning transformation to fullness values, visualizing this data,
        clustering, getting cluster labels, visualizing fullness by clusters and explaining clusters with Decision Tree

        Because sklearn's Decision Tree is used for explanation only interval values without inf or nan should be used. The
        safe (but maybe not the best) aproach is to base explanation on already transformed features.

        Parameters
        -----------
        input_data: a pandas DataFrame or a Data object to analyze
        to_transform: should the input data be transformed to fullness values (-1 for missing, 0 for interval features' zero, 1 for the rest)
        to_train: should the clusterer be trained (it can be omitted if it was already trained and we only need to get/explain clusters)
        to_explain: should the clusters be explained
        features: a list of features to analyze
        categorical: a list of categorical features (their alues will be changed to -1 and 1 without 0)
        exclude: a list of features to exclude from analysis (they can be used later for explanation)
        blocks: a path to Excel file or a pandas DataFrame with blocks information (features will be sorted by blocks during visualization)
        interval_min_unique: a minimal number of unique values for feature to be considered interval (if categorical is None)
        clusterer: a type of clusterer to be used. Default is k-means.
            'k-means' - partitions observations into k clusters in which each observation belongs to the cluster with the nearest mean,
                        serving as a prototype of the cluster. This is the fastest method with usually poor results.
            'hierarchy'- hierarchical agglomerative clustering seeks to build a hierarchy of clusters. This is a "bottom-up" approach:
                        each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. For
                        60000 observations and 500 features it works for about 40 minutes. Number of clusters can be adjusted without
                        need to retrain the clusterer.
            'hdbscan' - Hierarchical Density-Based Spatial Clustering of Applications with Noise. Searches for the areas with the higher
                        density allowing different values of variance for each cluster and existance of points without clusters (label -1).
                        For 60000 observations and 500 features it works for about 55 minutes. Number of clusters cannot be adjusted directly,
                        only by changing clusterer options like min_cluster_size, min_samples etc, clusterer must be retrained.
        clusterer_options: a dictionary of options for clusterer. Contents depend on clusterer type (see main options below):
            'k-means' (for more information look for sklearn.cluster.KMeans):
                init - the method for choosing centroids, 'random' and 'k-means++' are available
                n_clusters - the desired number of clusters
                n_init - the number of iterations to choose best clustering from
            'hierarchy' (for more information look for http://danifold.net/fastcluster.html):
                method - for memory-friendly clustering available methods are centroid (worked best so far), median, ward and single.
                    For memory-hungry clustering other methods can be used: complete, average, weighted. For a dataset with 60000 observations
                    and 500 features 'complete' method used about 10 GB RAM and crushed. Use on your own risk.
                metric - for centroid, median and ward methods only euclidian metric can be used. For other methods these metrics are available:
                    euclidian, sqeuclidean, seuclidean, mahalanobis, cityblock, chebychev, minkowski, cosine, correlation, canberra, braycurtis,
                    hamming, jaccard, yule, dice, rogerstanimoto, russellrao, sokalsneath, kulsinski, matching (sokalmichener), user-defined.
            'hdbscan' (for more information look for https://hdbscan.readthedocs.io/en/latest/):
                min_cluster_size - the minimal number of observations to form a cluster
                min_samples - number of samples in a neighbourhood for a point to be considered a core point. The larger this values is
                    the more conservative clustering will be (more points will be considered as noise)
        clusters_number: the desired number of clusters to get from clusterer (used for k-means during training and for hierarchy after training,
            for hdbscan this option is ignored)
        explainer_features: a list of features to train Decision Tree for cluster labels prediction on (if None all features are used)
        explainer_max_leaf_nodes: a maximal number of explanation tree leaves (if None, then number of clusters is used)
        explainer_max_depth: a maximal depth of explanation tree (if None, then number of clusters - 1 is used)
        explainer_min_samples_leaf: a minimal number of observations in an explanation tree leaf (if None, then minimal cluster size is used)
        graphviz_converter_path: a path to dot.exe, which can transform .dot file to .png for it to be displayed in the notebook
        verbose: a flag for detailed output (including fullness visualization)

        Returns
        ----------
        a dataframe with clusters explanation and statistics
        '''
        to_self=False

        if input_data is None:
            to_self=True
            input_data=self.data

        if to_transform:
            print('Transforming data according to its fullness..')
            processed_data=self.transform(input_data, features=features, categorical=categorical, exclude=exclude,
                                          interval_min_unique=interval_min_unique)
            to_self=True
        else:
            processed_data=copy.deepcopy(input_data)

        if verbose:
            print('Visualizing data fullness..')
            self.visualize(processed_data, features=features, blocks=blocks, to_transform=False, exclude=exclude,
                           text_color = 'k', figsize=(15,5))
        if to_train:
            print('Training', clusterer if clusterer is not None else '', 'clusterer..')
            self.train_clusterer(processed_data, features=features, exclude=exclude, clusterer=clusterer,
                                 options=clusterer_options, clusters_number=clusters_number, verbose=verbose)

        print('Getting clusters..')
        processed_data.dataframe['cluster']=self.get_clusters(clusterer=clusterer, number=clusters_number, to_self=to_self)

        if verbose:
            print('Visualizing data fullness by clusters..')
            self.visualize(processed_data, features=features, blocks=blocks, groups=['cluster'], to_transform=False,
                           text_color = 'k', figsize=(15,5))

        if to_explain:
            print('Training explanation Decision Tree..')
            return self.explain_clusters(processed_data, cluster='cluster', features=explainer_features,
                                         max_leaf_nodes=explainer_max_leaf_nodes,
                                         max_depth=explainer_max_depth, min_samples_leaf=explainer_min_samples_leaf,
                                         graphviz_converter_path=graphviz_converter_path)


    def transform(self, input_data=None, features=None, categorical=None, exclude=None, interval_min_unique=None):
        '''
        Transforms input data features to fullness values (-1 for missing, 0 for interval features' zero, 1 for the rest)

        Parameters
        -----------
        input_data: a pandas DataFrame or a Data object to analyze
        features: a list of features to analyze
        categorical: a list of categorical features (their values will be changed to -1 and 1 without 0)
        exclude: a list of features to exclude from analysis (they can be used later for explanation)
        interval_min_unique: a minimal number of unique values for feature to be considered interval (if categorical is None)

        Returns
        ----------
        a Data object, containing transformed data
        '''

        def lists_contents_equal(a,b):
            return sorted([x for x in a if pd.isnull(x)==False])==sorted([x for x in b if pd.isnull(x)==False]) and \
                ((np.nan in a and np.nan in b) or (np.nan not in a and np.nan not in b))

        if interval_min_unique is None:
            interval_min_unique=30

        if input_data is None:
            input_data=self.data
        if isinstance(input_data, pd.DataFrame):
            data=Data(input_data)
        else:
            data=copy.deepcopy(input_data)

        if features is None:
            features=data.features
        if exclude is not None:
            features=[x for x in features if x not in exclude]

        if categorical is None and self.categorical is not None:
            categorical=self.categorical.copy()

        if categorical is None:
            categorical=[]
            interval=[]
            for f in features:
                unique=data.dataframe[f].unique()
                field_type=data.dataframe[f].dtype
                if field_type==object or unique.shape[0]<interval_min_unique or \
                     (lists_contents_equal(unique, [0,1]) or \
                      lists_contents_equal(unique, [0,1, np.nan])):
                    categorical.append(f)
                else:
                    interval.append(f)
        else:
            interval=[x for x in features if x not in categorical]

        self.categorical=categorical.copy()

        result_data=data.dataframe.copy()
        result_data[categorical]=result_data[categorical].applymap(lambda x: -1 if pd.isnull(x) else 1)
        result_data[interval]=result_data[interval].applymap(lambda x: -1 if pd.isnull(x) else 1 if x!=0 else 0)

        self.data=Data(result_data)

        return self.data


    def visualize(self, input_data=None, features=None, groups=None, blocks=None, to_transform=True, exclude=None,
                  show_features_labels=False, text_color = 'k', figsize=(15,5)):
        '''
        Visualize data fullness by features, sorted in their blocks order (if provided). For each combination of groups' features' values
        separate plot is made.

        Parameters
        -----------
        input_data: a pandas DataFrame or a Data object to analyze
        features: a list of features to analyze
        groups: a list of features to draw separate plots for (a plot for a combination of features' values)
        blocks: a path to Excel file or a pandas DataFrame with blocks information (features will be sorted by blocks during visualization)
        to_transform: should the input data be transformed to fullness values (-1 for missing, 0 for interval features' zero, 1 for the rest)
        exclude: a list of features to exclude from analysis (they can be used later for explanation)
        show_features_labels: should feature labels be shown on the plot or not (if True, then no blocks information will be displayed)
        text_color: a color for all text in plots (including ticks, axis and legend)
        figsize: a size for plots' figure
        '''

        if input_data is None:
            input_data=self.data
        if isinstance(input_data, pd.DataFrame):
            data=Data(input_data)
        else:
            data=copy.deepcopy(input_data)

        if features is None:
            features=data.features

        if groups is not None:
            features=[x for x in features if x not in groups]
            try:
                groups_data=data.dataframe[groups].drop_duplicates().sort_values(groups).reset_index(drop=True).reset_index()
            except Exception:
                print("No specified groups columns in input data. Return None.")
                return None
            groups_number=groups_data.shape[0]
        else:
            groups_number=1

        if exclude is not None:
            features=[x for x in features if x not in exclude]

        if blocks is not None and show_features_labels==False:
            if isinstance(blocks, str):
                blocks=pd.read_excel(blocks)
            blocks.columns=blocks.columns.str.lower()
            for v in ['feature', 'variable', 'var', 'column']:
                if v in blocks:
                    break
            try:
                blocks=blocks.sort_values(['block', v])
            except Exception:
                print("No 'feature' ,'variable', 'var', 'column' or 'block' field in input data. Return None.")
                return None
            blocks=blocks[blocks[v].isin(features)].reset_index(drop=True)
            nd_block=[{v:f, 'block':'Not defined'} for f in features if f not in blocks[v].tolist()]
            blocks=blocks.append(pd.DataFrame(nd_block), ignore_index=True).sort_values(['block', v]).reset_index(drop=True)

            blocks_first=blocks.groupby('block').first().reset_index()
            blocks_edges_x=list(blocks[blocks[v].isin(blocks_first[v])].index)+[blocks.shape[0]]
            blocks_labels_x=[(blocks_edges_x[i]+blocks_edges_x[i+1])/2 for i in range(len(blocks_edges_x)-1)]
            blocks_edges_x=blocks_edges_x[1:-1]

            features=blocks[v].tolist()
        else:
            features=sorted(features)


        if to_transform:
            print('Transforming data according to its fullness..')
            data=self.transform(data, features=features)

        c_text=matplotlib.rcParams['text.color']
        c_axes=matplotlib.rcParams['axes.labelcolor']
        c_xtick=matplotlib.rcParams['xtick.color']
        c_ytick=matplotlib.rcParams['ytick.color']

        matplotlib.rcParams['text.color'] = text_color
        matplotlib.rcParams['axes.labelcolor'] = text_color
        matplotlib.rcParams['xtick.color'] = text_color
        matplotlib.rcParams['ytick.color'] = text_color

        group_stats={}

        f, axes = plt.subplots(groups_number, 1, sharex=True, figsize=figsize)
        if isinstance(axes, np.ndarray)==False:
            axes=[axes]
        for g in range(groups_number):
            if groups is not None:
                current_data=data.dataframe.merge(groups_data[groups_data['index']==g].drop('index', axis=1), on=groups, how='inner')
            else:
                current_data=data.dataframe
            group_stats[g]=current_data[features].apply(pd.value_counts)
            group_stats[g]=group_stats[g]/group_stats[g].sum()
            group_stats[g].T.plot(kind='bar', ax=axes[g], stacked=True, width=1, legend=False, grid=False, ylim=(0,1))
            handles, _ = axes[g].get_legend_handles_labels()
            if groups_number>1:
                axes[g].set_ylabel(str(dict(groups_data[groups].iloc[g])).replace('{', '').replace('}', '').replace("'", '')+'\namount = '+str(current_data.shape[0]),
                                   rotation=0, ha='right', va='center')
            if blocks is not None and show_features_labels==False:
                axes[g].set_xticks(blocks_labels_x)
                axes[g].set_xticklabels(blocks_first.block.tolist())
                for edge in blocks_edges_x:
                    axes[g].axvline(edge-0.5, ymin=-0.5 if g!=groups_number-1 else 0, ymax=1.5 if g!=0 else 1,
                                    linestyle='--', color='red', alpha=1, lw=1, clip_on=False)
            elif show_features_labels:
                axes[g].set_xticks([i for i in range(len(features))])
                axes[g].set_xticklabels(features)
            else:
                axes[g].set_xticks([])

            axes[g].yaxis.set_major_formatter(mtick.PercentFormatter(1))

        labels=['Пропущенные' if x==-1 else '0 (для интервальных)' if x==0 else 'Не пропущенные' if x==1 else x \
                    for x in sorted(np.unique(data.dataframe[features]))]

        f.legend(handles=handles, labels=labels,ncol=len(labels), bbox_to_anchor=(0.5,1), loc='center')
        plt.tight_layout()
        plt.show()

        matplotlib.rcParams['text.color'] = c_text
        matplotlib.rcParams['axes.labelcolor'] = c_axes
        matplotlib.rcParams['xtick.color'] = c_xtick
        matplotlib.rcParams['ytick.color'] = c_ytick


    def train_clusterer(self, input_data=None, features=None, exclude=None, clusterer=None, options=None,
                        clusters_number=None, verbose=True):
        '''
        Trains the chosen clusterer to obtain cluster labels or a linkage matrix for them (for hierarchy clusterer)

        Parameters
        -----------
        input_data: a pandas DataFrame or a Data object to analyze
        features: a list of features to analyze
        exclude: a list of features to exclude from analysis (they can be used later for explanation)
        clusterer: a type of clusterer to be used. Default is k-means.
            'k-means' - partitions observations into k clusters in which each observation belongs to the cluster with the nearest mean,
                        serving as a prototype of the cluster. This is the fastest method with usually poor results.
            'hierarchy'- hierarchical agglomerative clustering seeks to build a hierarchy of clusters. This is a "bottom-up" approach:
                        each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. For
                        60000 observations and 500 features it works for about 40 minutes. Number of clusters can be adjusted without
                        need to retrain the clusterer.
            'hdbscan' - Hierarchical Density-Based Spatial Clustering of Applications with Noise. Searches for the areas with the higher
                        density allowing different values of variance for each cluster and existance of points without clusters (label -1).
                        For 60000 observations and 500 features it works for about 55 minutes. Number of clusters cannot be adjusted directly,
                        only by changing clusterer options like min_cluster_size, min_samples etc, clusterer must be retrained.
        options: a dictionary of options for clusterer. Contents depend on clusterer type (see main options below):
            'k-means' (for more information look for sklearn.cluster.KMeans):
                init - the method for choosing centroids, 'random' and 'k-means++' are available
                n_clusters - the desired number of clusters
                n_init - the number of iterations to choose best clustering from
            'hierarchy' (for more information look for http://danifold.net/fastcluster.html):
                method - for memory-friendly clustering available methods are centroid (worked best so far), median, ward and single.
                    For memory-hungry clustering other methods can be used: complete, average, weighted. For a dataset with 60000 observations
                    and 500 features 'complete' method used about 10 GB RAM and crushed. Use on your own risk.
                metric - for centroid, median and ward methods only euclidian metric can be used. For other methods these metrics are available:
                    euclidian, sqeuclidean, seuclidean, mahalanobis, cityblock, chebychev, minkowski, cosine, correlation, canberra, braycurtis,
                    hamming, jaccard, yule, dice, rogerstanimoto, russellrao, sokalsneath, kulsinski, matching (sokalmichener), user-defined.
            'hdbscan' (for more information look for https://hdbscan.readthedocs.io/en/latest/):
                min_cluster_size - the minimal number of observations to form a cluster
                min_samples - number of samples in a neighbourhood for a point to be considered a core point. The larger this values is
                    the more conservative clustering will be (more points will be considered as noise)
        clusters_number: the desired number of clusters to get from clusterer (used for k-means during training and for hierarchy after training,
            for hdbscan this option is ignored)
        verbose: a flag for detailed output
        '''

        if input_data is None:
            input_data=self.data
        if isinstance(input_data, pd.DataFrame):
            data=Data(input_data)
        else:
            data=copy.deepcopy(input_data)

        if features is None:
            features=data.features.copy()
        if exclude is not None:
            features=[x for x in features if x not in exclude]

        if clusterer is None:
            clusterer='k-means'

        if options is None:
            if clusterer=='hierarchy':
                 options={'method':'centroid'}
            if clusterer=='hdbscan':
                 options={'min_cluster_size':1000, 'min_samples':500}
            if clusterer=='k-means':
                 options={'init':'random', 'n_clusters':3 if clusters_number is None else clusters_number, 'n_init':10}

        if verbose:
            print('-- Starting at: '+str(datetime.datetime.now()))

        if clusterer=='hierarchy':
            if 'method' in options and options['method'] in ['complete', 'average', 'weighted']:
                self.linked = fastcluster.linkage(data.dataframe[features], **options)
            else:
                self.linked = fastcluster.linkage_vector(data.dataframe[features], **options)
        if clusterer=='hdbscan':
            self.clusterer = hdbscan.HDBSCAN(**options)
            self.clusterer.fit(data.dataframe[features])
        if clusterer=='k-means':
            self.clusterer = KMeans(**options)
            self.clusterer.fit(data.dataframe[features])

        if verbose:
            print('-- Finished at: '+str(datetime.datetime.now()))


    def get_clusters(self, clusterer=None, number=None, to_self=True):
        '''
        Gets cluster labels as a pandas Series object for the data, stored in self.data (basicaly for the data, that was used
        to train clusterer on)

        Parameters
        -----------
        clusterer: a type of clusterer that was used (there is a different behavior for retrieving cluster labels for different clusterers).
            Available clusterers: 'k-means' (default), 'hierarchy', 'hdbscan'
        number: the desired number of clusters to get from clusterer (used for hierarchy clusterer, for other clusterers is ignored)
        to_self: should cluster labels be written to self.data.dataframe.cluster

        Returns
        ----------
        a pandas Series object, containing cluster labels
        '''

        if clusterer is None:
            clusterer='k-means'

        if clusterer=='hdbscan' and number is not None:
            print('With hdbscan clusterer there is no way to directly choose the number of clusters. Use k-means of hierarchy instead.')
        if clusterer=='hierarchy':
            result=fcluster(self.linked, 3 if number is None else number , criterion='maxclust')
        if clusterer in ['hdbscan', 'k-means']:
            result=self.clusterer.labels_

        if to_self:
            self.data.dataframe['cluster']=result

        return result


    def change_clusters(self, replace_dict, cluster='cluster', input_data=None):
        '''
        Changes values of clusters based on replace dictionary in format {old_value: new_value}

        Parameters
        -----------
        replace_dict: a dictionary in format {old_value: new_value} to change cluster labels
        cluster: a field with cluster labels
        input_data: a pandas DataFrame or score-kit Data object, containing the filed with cluster labels

        Returns
        ----------
        a changed pandas DataFrame or score-kit Data object
        '''
        to_self=False

        if input_data is None:
            to_self=True
            input_data=self.data
        if isinstance(input_data, pd.DataFrame):
            data=Data(input_data)
        else:
            data=copy.deepcopy(input_data)

        data.dataframe[cluster]=data.dataframe[cluster].replace(replace_dict)

        if to_self:
            self.data=copy.deepcopy(data)

        return data


    def explain_clusters(self, input_data=None, cluster='cluster', features=None,
                         max_leaf_nodes=None, max_depth=None, min_samples_leaf=None,
                         graphviz_converter_path='C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe'):
        '''
        Trains an sklearn's Decision tree to predict cluster labels based on input features, then visualizes it
        and returns a pandas DataFrame with clusters explanation

        For Decision Tree visualization graphviz needs to be installed in OS and graphviz_converter_path
        should be provided to .work and .explain_clusters methods (it is a path to dot.exe, which can transform
        .dot file to .png for it to be displayed in the notebook)


        Parameters
        -----------
        input_data: a pandas DataFrame or a Data object to analyze
        cluster: the name of column, containing cluster labels
        features: a list of features to train Decision Tree for cluster labels prediction on (if None all features are used)
        max_leaf_nodes: a maximal number of explanation tree leaves (if None, then number of clusters is used)
        max_depth: a maximal depth of explanation tree (if None, then number of clusters - 1 is used)
        min_samples_leaf: a minimal number of observations in an explanation tree leaf (if None, then minimal cluster size is used)
        graphviz_converter_path: a path to dot.exe, which can transform .dot file to .png for it to be displayed in the notebook

        Returns
        ----------
        a dataframe with clusters explanation and statistics
        '''

        def recursive_tree_conditions_generator(tree, features, node, leaves=None, input_leaf=None):
            '''
            TECH

            Recursively passes through the tree to get each leaf's splits and statistics
            '''
            if tree.children_left[node]==-1 and tree.children_right[node]==-1:
                current_leaf=copy.deepcopy(input_leaf)
                for cn in range(len(tree.value[node][0])):
                    current_leaf[cn]=tree.value[node][0][cn]
                leaves.append(current_leaf)
                return leaves
            else:
                current_leaf=copy.deepcopy(input_leaf)
                if features[tree.feature[node]] in current_leaf:
                    condition=current_leaf[features[tree.feature[node]]]
                    current_leaf[features[tree.feature[node]]]=[condition[0], tree.threshold[node]]
                else:
                    current_leaf[features[tree.feature[node]]]=[-np.inf, tree.threshold[node]]
                leaves=recursive_tree_conditions_generator(tree, features, tree.children_left[node], leaves, current_leaf)

                current_leaf=copy.deepcopy(input_leaf)
                if features[tree.feature[node]] in current_leaf:
                    condition=current_leaf[features[tree.feature[node]]]
                    current_leaf[features[tree.feature[node]]]=[tree.threshold[node], condition[1]]
                else:
                    current_leaf[features[tree.feature[node]]]=[tree.threshold[node], np.inf]
                leaves=recursive_tree_conditions_generator(tree, features, tree.children_right[node], leaves, current_leaf)
                return leaves

        if input_data is None:
            input_data=self.data
        if isinstance(input_data, pd.DataFrame):
            data=Data(input_data)
        else:
            data=copy.deepcopy(input_data)

        if features is None:
            features=data.features.copy()

        features=[ x for x in features if x!=cluster]

        if max_leaf_nodes is None:
            max_leaf_nodes=data.dataframe[cluster].unique().shape[0]
        if max_depth is None:
            max_depth=data.dataframe[cluster].unique().shape[0]-1
        if min_samples_leaf is None:
            min_samples_leaf=min(data.dataframe[cluster].value_counts())

        clf = DecisionTreeClassifier(criterion='gini',
                                     max_leaf_nodes=max_leaf_nodes,
                                     max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf)
        clf.fit(data.dataframe[features], y=data.dataframe[cluster])

        if graphviz_converter_path is not None:
            export_graphviz(clf, out_file="cluster_tree.dot", feature_names=features, filled = True, rounded = True)
            try:
                system('"'+graphviz_converter_path+'" -Tpng cluster_tree.dot -o cluster_tree.png')
                display(Display_Image(filename='cluster_tree.png'))
            except Exception:
                print('Executable dot.exe was not found at the specified address. \n'+
                      'Please make sure, that graphviz is installed on your system and provide the correct address for dot converter.')

        conditions_df=pd.DataFrame(recursive_tree_conditions_generator(clf.tree_, features, 0, [], {}))
        conditions_df=conditions_df.rename({i:clf.classes_[i] for i in range(len(clf.classes_))}, axis=1)
        conditions_df=conditions_df[[x for x in conditions_df if x not in clf.classes_.tolist()] +clf.classes_.tolist()]

        self.conditions=conditions_df.copy()
        return conditions_df


    def split_data(self, input_data, not_transformed=None, use_index_as_cluster=False):
        '''
        Splits input pandas DataFrame of Data object into clusters according to conditions table and returns the
        dictionary of Data objects. The key of this dictionary is the cluster number and it is determined as
        the cluster with most observations in the current part. Sometimes it may not work correctly, in that case
        cluster number can be set to the conditions table index (with use_index_as_cluster option)


        Parameters
        -----------
        input_data: a pandas DataFrame or a Data object to split
        not_transformed: a list of features that were not transformed to their's fullness values, for them conditions
            will be interpreted as the simple conditions for interval features (for the rest of the features
            conditions will be changed according to the fullness coding)
        use_index_as_cluster: if True the key of the result dictionary will be set to the index value of conditions

        Returns
        ----------
        a dictionary with pandas DataFrame or Data objects
        '''
        if self.conditions is None:
            print('No conditions were found. Please, tun .explain_clusters method. Return None.')
            return None

        if isinstance(input_data, Data):
            data=input_data.dataframe.copy()
        else:
            data=input_data.copy()

        if not_transformed is None:
            not_transformed=[]

        features=[]
        for f in self.conditions:
            if self.conditions[f].apply(lambda x: isinstance(x, list) or pd.isnull(x)).all():
                features.append(f)

        result={}
        for ind, conditions in self.conditions[features].iterrows():
            df_filter=pd.Series([True]*data.shape[0], index=data.index)
            for fn in range(len(features)):
                f=features[fn]
                if isinstance(conditions[fn], list):
                    if f in not_transformed:
                        df_filter=df_filter & ((data[f]>conditions[f][0])&(data[f]<=conditions[f][1]))
                    else:
                        new_filter=pd.Series([False]*data.shape[0], index=data.index)
                        if -1>conditions[f][0] and -1<=conditions[f][1]:
                            new_filter=new_filter | (pd.isnull(data[f]))
                        if 0>conditions[f][0] and 0<=conditions[f][1] and f not in self.categorical:
                            new_filter=new_filter | (data[f]==0)
                        if 1>conditions[f][0] and 1<=conditions[f][1] and f not in self.categorical:
                            new_filter=new_filter | ((pd.isnull(data[f])==False)&(data[f]!=0))
                        if 1>conditions[f][0] and 1<=conditions[f][1] and f in self.categorical:
                            new_filter=new_filter | ((pd.isnull(data[f])==False))
                        df_filter=df_filter & new_filter

            if isinstance(input_data, Data):
                result_data=copy.deepcopy(input_data)
                result_data.dataframe=result_data.dataframe[df_filter].copy()
            else:
                result_data=data[df_filter].copy()
            if use_index_as_cluster==False:
                clusters=dict(self.conditions[[x for x in self.conditions if x not in features]].loc[ind])
                ind=max(clusters, key=clusters.get)
            result[ind]=copy.deepcopy(result_data)

        return result

