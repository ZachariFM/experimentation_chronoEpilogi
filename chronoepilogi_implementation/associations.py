from scipy.stats import pearsonr, spearmanr, beta, rankdata, t, f_oneway, kruskal, alexandergovern
from scipy.special import stdtr
from statsmodels.regression.linear_model import OLS
import pingouin
import bottleneck
import tigramite.independence_tests.regressionCI

from collections import defaultdict
from joblib import Parallel, delayed

#from mass_ts import mass2

import numpy as np
import pandas as pd
import time

##
#
#   Association classes
#
##

class Association:
    def __init__(self, config):
        self.config = config
        self.pvalues = dict()

    def association(self, residuals_df, variable_df):
        pass







def normalize(x):
    #x should have shape (time,) or shape (time,variables)
    a, b = (x-np.mean(x,axis=0)), np.std(x,axis=0)
    return np.divide(a,b,where=(b!=0),out=np.zeros_like(a))  # 'where' to handle nans from std=0
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
def moving_average(a, window=3):
    return np.mean(rolling_window(a, window), -1)
def moving_std(a, window=3):
    return np.std(rolling_window(a, window), -1)
def mass2_modified(ts, query):
    #adapted from the mass-ts module, to allow for 2D ts input.
    #ts is an array of form (time, variables) and query of form (time,).
    ts, query = np.array(ts), np.array(query)
    n = len(ts)
    v = ts.shape[-1]
    m = len(query)
    x = ts.T
    y = query
    
    meany = np.mean(y)
    sigmay = np.std(y)
    meanx = moving_average(x, m)
    padding = np.ones_like(x)
    padding[:,-meanx.shape[-1]:] = meanx
    meanx = padding
    
    # moving_std was inefficient, replaced by bottleneck library move_std.
    #sigmax = moving_std(x, m)
    sigmax = bottleneck.move_std(x,m)[:,m-1:]
    padding = np.zeros_like(x)
    padding[:,-sigmax.shape[-1]:] = sigmax
    sigmax = padding

    y = np.append(np.flip(y), np.zeros([1, n - m]))
    
    # fft computation is currently the major bottleneck of the function
    z = np.fft.fft(x,axis=-1) * np.fft.fft(y,axis=-1)
    z = np.fft.ifft(z,axis=-1)

    # 'where' to handle nans from std=0
    # in this case, we want the correlation to be 0. So instead of nan, we put m.
    a = (z[:,m - 1:n] - m * meanx[:,m - 1:n] * meany)
    b = (sigmax[:,m - 1:n] * sigmay)
    dist = 2 * (m - np.divide(a,b,where=(b!=0),out=np.zeros_like(a)))

    correlation = 1 - np.absolute(dist) / (2 * m)

    return correlation



class PearsonMultivariate(Association):
    """
    Computes for each lag up to <lags> of the given variables, its <return_type> with the residuals.
    The result is then aggregated into a single score using <selection_rule>.

        Prefered use case:
         - many lags have to be computed

        Data assumption:
         - dataframe is sorted by timestamp increasing
         - timestamps are equidistants
         - data has no missing value
         - can currently only process single-sample data.
         - the first <lags> non-na values of the residuals will be excluded.
         - the first values of the tested variable are excluded, depending on the LearningModel lag, to correspond
           to residuals.

        config:
         - return_type (str):
           - correlation: the computed association is the pearson correlation
           - p-value: the computed association is the p-value of the pearson correlation
         - lags (int): the maximal lag of the variable to use.
            if set to 0, only the immediate correlation is computed.
            if > 0, the lag of maximal correlation / minimal p-value amongst the lags is selected.
         - selection_rule: the rule to use to aggregate the lags
           - max: use maximal correlation / minimal p-value
           - average: use average correlation / average p-value
        """

    def _select_correct_rows(self, residuals_df, variables_df):
        # remove nans
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        residuals_indexes = set(residuals_df.index)
        #adjust variable timestamps to residuals since learning process lags will have reduced the length of the series
        variables_ilocs = [i for i in range(variables_df.shape[0]) if (variables_df.index[i] in residuals_indexes)]
        #remove the first <lags> elements of the residuals for mass2_modified computation.
        residuals_ilocs = list(range(residuals_df.shape[0]))
        residuals_ilocs = residuals_ilocs[self.config["lags"]:]
        
        residuals = residuals_df.iloc[residuals_ilocs].values.reshape((-1,))
        variables = variables_df.iloc[variables_ilocs].values
        return residuals, variables

    def association(self, residuals_df, variables_df):
    
        residuals, variables = self._select_correct_rows(residuals_df, variables_df)

        if self.config["lags"] == 1:  # edge case of the fft
            coefficients = [[pearsonr(residuals,variables[:-1,i]).correlation] for i in range(variables.shape[1])]
            coefficients = np.array(coefficients)
        else:
            # mass2 is the computation bottleneck due to fft.
            # Solution: parallelize. Empirically, splitting into sqrt(D) groups is looks best.
            column_split = np.array_split(list(range(variables.shape[1])),int(np.sqrt(variables.shape[1])))
            res = Parallel(n_jobs=-1)(delayed(mass2_modified)(variables[:,list(cols)], residuals) for cols in column_split)
            coefficients = np.concatenate(list(res), axis=0)

        if self.config["return_type"] == "p-value":
            # next 3 lines taken from scipy.stats.pearsonr
            ab = len(residuals)/2 - 1  # len(residuals) is the total sample size over which correlation is computed
            beta_distribution = beta(ab, ab, loc=-1, scale=2)
            pvalues = - 2 * beta_distribution.sf(np.abs(coefficients))
        
            self.pvalues = dict((variable, -pvalues[i])for i,variable in enumerate(variables_df.columns))


        if self.config["selection_rule"] == "max":
            return np.max(pvalues, axis=-1)
        elif self.config["selection_rule"] == "average":
            return np.mean(pvalues, axis=-1)
        else:
            raise(NotImplementedError)
    


class SpearmanMultivariate(PearsonMultivariate):
    """
    Computes for each lag up to <lags> of the given variables, its <return_type> with the residuals.
    The result is then aggregated into a single score using <selection_rule>.

        Prefered use case:
         - many lags have to be computed

        Data assumption:
         - dataframe is sorted by timestamp increasing
         - timestamps are equidistants
         - data has no missing value
         - can currently only process single-sample data.
         - the first <lags> non-na values of the residuals will be excluded.
         - the first values of the tested variable are excluded, depending on the LearningModel lag, to correspond
           to residuals.

        config:
         - return_type (str):
           - correlation: the computed association is the pearson correlation
           - p-value: the computed association is the p-value of the pearson correlation
         - lags (int): the maximal lag of the variable to use.
            if set to 0, only the immediate correlation is computed.
            if > 0, the lag of maximal correlation / minimal p-value amongst the lags is selected.
         - selection_rule: the rule to use to aggregate the lags
           - max: use maximal correlation / minimal p-value
           - average: use average correlation / average p-value
        """
    def _compute_ranks(self,residuals,variables):
        rr = rankdata(residuals)
        rv = rankdata(variables,axis=0)
        return rr,rv

    def association(self, residuals_df, variables_df):
        #align mts
        residuals, variables = self._select_correct_rows(residuals_df, variables_df)

        #spearman computation: ranks are necessary
        residuals, variables = self._compute_ranks(residuals,variables)
        #parallel computation
        column_split = np.array_split(list(range(variables.shape[1])),int(np.sqrt(variables.shape[1])))
        res = Parallel(n_jobs=-1)(delayed(mass2_modified)(variables[:,list(cols)], residuals) for cols in column_split)
        coefficients = np.concatenate(list(res), axis=0)

        #pvalues
        if self.config["return_type"] == "p-value":
            # next lines taken from scipy.stats
            dof = len(residuals) - 2
            # test statistic
            coefficients = coefficients * np.sqrt((dof/((coefficients+1.0)*(1.0-coefficients))).clip(0))
            # comparision with student t
            coefficients = stdtr(dof, -np.abs(coefficients))*2
        
            self.pvalues = {(variable, coefficients[i])for i,variable in enumerate(variables_df.columns)}

        if self.config["selection_rule"] == "max":
            return np.max(coefficients, axis=-1)
        elif self.config["selection_rule"] == "average":
            return np.mean(coefficients, axis=-1)



class ANOVATemporalSlow(PearsonMultivariate):
    def association(self, residuals_df, variables_df):
        residuals, variables = self._select_correct_rows(residuals_df, variables_df)

        self.pvalues = dict()
        pvalues = np.ones(shape=(len(variables_df.columns),))

        #import warnings
        #warnings.filterwarnings("error")

        for variable in range(variables.shape[1]):
            self.pvalues[variables_df.columns[variable]] = []

            ncategories = sorted(np.unique(variables[:,variable]))
            categorical_filters = [variables[:,variable]==category for category in ncategories]
            
            for lag in range(self.config["lags"]):
                samples = [residuals[mask[lag:-self.config["lags"]+lag]] for mask in categorical_filters]
                # due to how f_oneway works, we must avoid length zero samples.
                samples = [s for s in samples if len(s)>0]
                
                if self.config["categorical_method"] == "f_oneway":
                    pval = f_oneway(*samples).pvalue if len(samples)>1 else 1.  # if one sample only, no link.
                elif self.config["categorical_method"] == "kruskal":
                    pval = kruskal(*samples).pvalue if len(samples)>1 else 1.
                elif self.config["categorical_method"] == "alexandergovern":
                    pval = alexandergovern(*samples).pvalue if len(samples)>1 else 1.

                self.pvalues[variables_df.columns[variable]].append(pval)
                pvalues[variable] = min(pvalues[variable], pval)

            self.pvalues[variables_df.columns[variable]] = np.array(self.pvalues[variables_df.columns[variable]])

        return -pvalues


class CrossSectionalAssociation(Association):
    def association(self,residuals_df, variables_df):
        """

        :param residuals_df: pd.Series or pd.DataFrame
        :param variables_df: pd.DataFrame
        :param variable_types:
        :return: np.array
        """
        pvalues=dict()

        variable_types = self.config["variable_types"]
        mass_with_numerical = self.config.get("mass_with_numerical",False)
        if mass_with_numerical:
            name_of_first_lag = self.config["name_of_first_lag"]

        col_names = variables_df.columns.get_level_values(0).unique()
        #numerical
        for variable in col_names:
            if variable_types[variable] == "numerical":
                lagged_df = variables_df[variable]
                if mass_with_numerical and name_of_first_lag:  # use mass in case that the user knows that the data is lagged
                    ts = lagged_df[[name_of_first_lag]].values 
                    query = residuals_df.iloc[self.config["lags"]:].values
                    correlations = mass2_modified(ts, query)
                    ab = len(query)/2 - 1  # len(residuals) is the total sample size over which correlation is computed
                    beta_distribution = beta(ab, ab, loc=-1, scale=2)
                    pval = - 2 * beta_distribution.sf(np.abs(correlations))
                    pvalues[variable] = pval
                else:
                    pval = []
                    for lag in lagged_df.columns:
                        p = pearsonr(lagged_df[lag],residuals_df[residuals_df.columns[0]])
                        pval.append(p.pvalue)
                    
                    pvalues[variable] = np.array(pval)

        #categorical
        for variable in col_names:
            if variable_types[variable] == "categorical":
                pval = []
                lagged_df = variables_df[variable]
                for lag in lagged_df.columns:
                    ncategories = sorted(lagged_df[lag].unique())
                    if isinstance(residuals_df, pd.Series):
                        samples = [residuals_df[lagged_df[lag]==k] for k in ncategories]
                    else:
                        samples = [residuals_df[residuals_df.columns[0]][lagged_df[lag]==k] for k in ncategories]

                    if len(samples)<2:
                        pval.append(1) #constant value so independence
                    else:
                        if self.config["categorical_method"] == "f_oneway":
                            pvalue = f_oneway(*samples).pvalue if len(samples)>1 else 1.  # if one sample only, no link.
                        elif self.config["categorical_method"] == "kruskal":
                            pvalue = kruskal(*samples).pvalue if len(samples)>1 else 1.
                        elif self.config["categorical_method"] == "alexandergovern":
                            pvalue = alexandergovern(*samples).pvalue if len(samples)>1 else 1.
                        pval.append(pvalue)
                pvalues[variable] = np.array(pval)

        self.pvalues = pvalues
        pvalues = [pvalues[v] for v in col_names]
        pvalues = [np.min(lagpval) for lagpval in pvalues] # take minimum of pvalues of each variable
        return -np.array(pvalues)


class TemporalSlowAssociation(Association):
    """Temporal data mixed-type interface.

    For continuous data, we use Pearson Correlation with mass implementation.
    For categorical data, we use an ANOVA test between the residuals and the tested series.

    Data should not have missing values, and the sampling rate must be regular.
    """

    def __init__(self, config):
        """ 
        Parameters
        ----------
        config: dict
            Must contain an entry for:
             - "lags": int, the number of lags to compute the correlation over
             - "categorical_method": str, any of 'f_oneway', 'kruskal', 'alexandergovern'.
                This specifies the kind of test used for categorical data.
             - "variable_types": dict, for each variable name, whether it is "numerical" or "categorical".
                See examples.

        Returns
        -------
        None

        Examples
        --------
        >>> data = pd.DataFrame(np.random.random(size=(1000,5)),columns=["target","1","2","3","4"])
        >>> variable_types = dict([(column, "numerical") for column in data.columns])
        >>> asso = TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso

        Or with mixed types:

        >>> numerical = pd.DataFrame(np.random.random(size=(1000,3)),columns=["target","1","2"])
        >>> categorical = pd.DataFrame(np.random.randint(size=(1000,2)),columns=["3","4"])
        >>> data = pd.concat([numerical,categorical], axis="columns")
        >>> variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
        >>> asso = TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso
        """
        super().__init__(config)


    def association(self,residuals_df, variables_df):
        """ 
        Parameters
        ----------
        residuals_df: pd.DataFrame
            DataFrame of shape (ntimesteps, 1) containing the model residuals of a learning model. 
            The index must be aligned with variables_df.
        variables_df: pd.DataFrame
            DataFrame of shape (ntimesteps, D) containing the D time series to test for association with the residuals.
            The index must be aligned with residuals_df

        Returns
        -------
        pvalues: np.array
            A 1D numpy array containing minus the minimal p-value across lags, for each of the D time series to test.
            The coefficients are in the same order as the columns in variables_df.columns.
            We return minus the p-value by convention, as the maximal -pvalue correspond to the maximal association.

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> data = pd.DataFrame(rng.random(size=(1000,5)),columns=["target","1","2","3","4"])
        >>> variable_types = dict([(column, "numerical") for column in data.columns])
        >>> asso = TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso.association(data[["target"]], data[["1","2","3","4"]])
        array([-0.03384917, -0.02838155, -0.0633841 , -0.15107386])

        Or with mixed types:

        >>> rng = np.random.default_rng(0)
        >>> numerical = pd.DataFrame(rng.random(size=(1000,3)),columns=["target","1","2"])
        >>> categorical = pd.DataFrame(rng.integers(0,3,size=(1000,2)),columns=["3","4"])
        >>> data = pd.concat([numerical,categorical], axis="columns")
        >>> variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
        >>> asso = TemporalSlowAssociation({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso.association(data[["target"]], data[["1","2","3","4"]])
        array([-0.03111284, -0.04568282, -0.03302831, -0.02551908])
        """

        variable_types = self.config["variable_types"]

        # numerical TS
        pearson_obj = PearsonMultivariate({"return_type":"p-value","selection_rule":"max",**self.config})
        numerical_variables = [x for x in variables_df.columns if variable_types[x]=="numerical"]
        if len(numerical_variables)>0:
            numerical_pvalues = pearson_obj.association(residuals_df, variables_df[numerical_variables])
        else:
            numerical_pvalues = []

        # categorical TS
        anova_obj = ANOVATemporalSlow({"categorical_method":self.config["categorical_method"],**self.config})
        categorical_variables = [x for x in variables_df.columns if variable_types[x]=="categorical"]
        if len(categorical_variables)>0:
            categorical_pvalues = anova_obj.association(residuals_df, variables_df[categorical_variables])
        else:
            categorical_pvalues = []

        # mix
        index_num, index_cat = 0,0
        pvalues = []
        for variable in variables_df.columns:
            if variable in numerical_variables:  # unoptimized
                pvalues.append(numerical_pvalues[index_num]) 
                index_num+=1
            elif variable in categorical_variables:  # unoptimized
                pvalues.append(categorical_pvalues[index_cat])
                index_cat+=1

        self.pvalues = {**pearson_obj.pvalues, **anova_obj.pvalues}
        return np.array(pvalues)


# class TemporalSlowAssociation(Association):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # use pearson correlation with fft even with tabular data
#         #self.config["mass_with_numerical"] = True

#     #def reshape_to_cross_sectional(self,df):
#     #    lags = self.config["lags"]
#     #    data = dict()
#     #    for column in df.columns:
#     #        for l in range(lags):
#     #            new_column = df[column].iloc[l:len(df)-lags+l]
#     #            new_column.index = df.index[lags:]
#     #            data[(column,l-lags)]=new_column
#     #    data = pd.DataFrame(data)
#     #    data.index = df.index[lags:]
#     #    data.columns = pd.MultiIndex.from_tuples(data.columns)
#     #    return data
        
#     def association(self,residuals_df, variables_df):

#         # numerical TS
#         variable_types = self.config["variable_types"]
#         pearson_obj = PearsonMultivariate({"return_type":"p-value","selection_rule":"max",**self.config})
#         numerical_variables = [x for x in variables_df.columns if x in variable_types["numerical"]]
#         if len(numerical_variables)>0:
#             numerical_pvalues = pearson_obj.association(residuals_df, variables_df[numerical_variables])
#         else:
#             numerical_pvalues = []

#         # categorical TS
#         categorical_variables = [x for x in variables_df.columns if x in variable_types["categorical"]]
#         #categorical_variables_df = self.reshape_to_cross_sectional(variables_df[categorical_variables])
#         #categorical_pvalues = super(TemporalSlowAssociation,self).association(residuals_df, variables_df)
#         anova_obj = ANOVATemporalSlow({"method":self.config["categorical_method"],**self.config})
#         categorical_variables = [x for x in variables_df.columns if x in variable_types["categorical"]]
#         if len(categorical_variables)>0:
#             categorical_pvalues = anova_obj.association(residuals_df, variables_df[categorical_variables])
#         else:
#             categorical_pvalues = []

#         # mix
#         index_num, index_cat = 0,0
#         pvalues = []
#         for variable in variables_df.columns:
#             if variable in numerical_variables:
#                 pvalues.append(numerical_pvalues[index_num])
#                 index_num+=1
#             elif variable in categorical_variables:
#                 pvalues.append(categorical_pvalues[index_cat])
#                 index_cat+=1

#         return np.array(pvalues) 


    

##########
#
#   Approximate (residual based) partial correlation test for the residuals.
#
##########




class LinearPartialCorrelation():
    def __init__(self, config):
        self.config = config
        self._check_config()
        
    def _check_config(self):
        assert "method" in self.config
        assert "lags" in self.config
        assert "selection_rule" in self.config
    
    def _prepare_data(self, condition_df, residuals_df, candidate_df):
        """
        Pingouin partial_corr asks for the data in form of a dataframe where rows are observation vectors.
        This function formats the lags of condition, candidate and joins them with residuals in a single vector.
        """
        # remove nans eventually occuring in residuals
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        
        # add lags of the condition variable
        col_name = condition_df.columns[0]
        condition_cols = pd.DataFrame()
        for lag in range(1,self.config["lags"]+1):
            condition_cols[col_name+"lag -"+str(lag)] = condition_df[col_name].shift(lag)
        condition_cols = condition_cols.iloc[self.config["lags"]:]
        
        # add lags of the tested variable
        col_name = candidate_df.columns[0]
        candidate_cols = pd.DataFrame()
        for lag in range(1,self.config["lags"]+1):
            candidate_cols[col_name+"lag -"+str(lag)] = candidate_df[col_name].shift(lag)
        candidate_cols = candidate_cols.iloc[self.config["lags"]:]
        
        # create new index
        new_index = residuals_df.index.intersection(condition_cols.index)
        residuals_df = residuals_df.loc[new_index]
        candidate_cols = candidate_cols.loc[new_index]
        condition_cols = condition_cols.loc[new_index]
        
        # concatenate
        df = pd.concat([residuals_df, candidate_cols, condition_cols],axis=1)
        cond_names = condition_cols.columns
        cand_names = candidate_cols.columns
        return df, cond_names, cand_names
    
    def partial_corr(self, residuals_df, candidate_df, condition_df):
        """
        Compute the partial correlation of residuals_df with each lag of candidate_df by taking condition_df variable into account.
        """
        data, cond_names, cand_names = self._prepare_data(condition_df, residuals_df, candidate_df)
        resid_name = residuals_df.columns[0]
        
        pvals = []
        for cand_name in cand_names:
            res = pingouin.partial_corr(data, x=resid_name, y=cand_name, covar=list(cond_names), method=self.config["method"])
            pvals.append(res["p-val"].values[0])
        
        if self.config["selection_rule"] == "min":
            return np.min(pvals, axis=-1)
        elif self.config["selection_rule"] == "average":
            return np.mean(pvals, axis=-1)
            
class ModelBasedPartialCorrelation(LinearPartialCorrelation):
    """
    Compute using two OLS models and a lr-test, whether the candidate is redundant given the condition.
    """
    def _check_config(self):
        assert "lags" in self.config
        assert "large_sample" in self.config

    def partial_corr(self, residuals_df, candidate_df, condition_df):
        residname = residuals_df.columns[0]
        data, cond_names, cand_names = self._prepare_data(condition_df, residuals_df, candidate_df)
        restricted_model = OLS(data[residname], data[cond_names], missing="drop").fit()
        full_model = OLS(data[residname], data[cond_names.tolist()+cand_names.tolist()], missing="drop").fit()
        lr_stat, p_value, df_diff = full_model.compare_lr_test(restricted_model, large_sample=self.config["large_sample"])
        return p_value
        
class CrossSectionalH0(LinearPartialCorrelation):
    """
    Compute using two OLS models and a lr-test, whether the candidate is redundant given the condition.
    """
    def __init__(self, config):
        self.config = config
        self._check_config()

    def _check_config(self):
        assert "large_sample" in self.config

    def _prepare_data(self,condition_df, residuals_df, candidate_df):
        # just rename columns to avoid problems due to multiindex
        condition = condition_df.copy()
        condition.columns = ["cond"+str(i) for i in range(len(condition.columns))]
        candidate = candidate_df.copy()
        candidate.columns = ["cand"+str(i) for i in range(len(candidate.columns))]
        cond_names = condition.columns
        cand_names = candidate.columns
        return pd.concat([condition,candidate],axis=1), cond_names, cand_names
        
    
    def partial_corr(self,residuals_df, candidate_df, condition_df):
        """
        :param residuals_df: pd.Series or pd.DataFrame
        :param candidate_df: pd.DataFrame
        :param condition_df: pd.DataFrame
        :return: np.array
        """
        data, cond_names, cand_names = self._prepare_data(condition_df, residuals_df, candidate_df)
        
        restricted_model = OLS(residuals_df, data[cond_names], missing="drop").fit()
        full_model = OLS(residuals_df, data[cond_names.tolist()+cand_names.tolist()], missing="drop").fit()
        lr_stat, p_value, df_diff = full_model.compare_lr_test(restricted_model, large_sample=self.config["large_sample"])
        return p_value

class HeuristicPartialCorrelation(LinearPartialCorrelation):

    def _check_config(self):
        assert "lags" in self.config

    def _select_correct_rows(self, residuals_df, variables_df):
        """
        Align residuals and variables for the correlation computation
        """
        # remove nans at begining of residuals
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        residuals_indexes = set(residuals_df.index)
        #adjust variable timestamps to residuals since learning process lags will have reduced the length of the series
        variables_ilocs = [i for i in range(variables_df.shape[0]) if (variables_df.index[i] in residuals_indexes)]
        #remove the first <lags> elements of the residuals for mass2_modified computation.
        residuals_ilocs = list(range(residuals_df.shape[0]))
        residuals_ilocs = residuals_ilocs[self.config["lags"]:]
        
        residuals = residuals_df.iloc[residuals_ilocs].values.reshape((-1,))
        variables = variables_df.iloc[variables_ilocs].values
        return residuals, variables
        
    def _correl_pvalue(self,r, n, k):
        """Compute the p-value of a partial correlation coefficient.
        https://pingouin-stats.org/build/html/_modules/pingouin/correlation.html#partial_corr
        """
        # using a student T distribution
        dof = n - k - 2
        tval = r * np.sqrt(dof / (1 - r**2 + 1e-16))  # in case 1-r**2==0
        pval = 2 * t.sf(np.abs(tval), dof)
        return pval

    def partial_corr_mass(self,residuals_df, candidate_df, condition_df):
        """
        Correlation formula for three univariate random variables: corr(x,y|z) = (corr(x,y) - corr(x,z)corr(y,z) ) / sqrt(1-corr²(x,z)) / sqrt(1-corr²(y,z))
        Here, for i,j we want to compute fast corr(Res,Cand_j|Cond_i).
        corr(Res,Cand_j) and corr(Res,Cond_i) can be computed fast for i and j
        corr(Cand_j,Cond_i) is actually the same as corr(Cand_{j-i},Cond_0)
        """
        # correlation of Res and Cand
        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        RCa = mass2_modified(cand, res)
        RCa = RCa[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        # correlation of Res and Cond
        res,cond=self._select_correct_rows(residuals_df, condition_df)
        RCo = mass2_modified(cond, res)
        RCo = RCo[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        
        # effective size over which the above has been estimated:
        sample_length = len(res)
        
        
        # correlation of Cand and Cond
        CoCa = []
        for i in range(self.config["lags"]):
            CoCa.append( mass2_modified(cand,cond[i:-self.config["lags"]+i])[0][:-1] )
        
        # explicit table of CoCa[i,j] = corr of L(i)(cond) and L(j)(cand)
        CoCa_table = np.zeros((self.config["lags"],self.config["lags"]))
        for i in range(self.config["lags"]):
            for j in range(self.config["lags"]):
                CoCa_table[i,j]=CoCa[i][j]
        
        # Compute the final partial correlation
        RCa_Co = np.zeros((self.config["lags"],self.config["lags"]))
        RCo_Ca = np.zeros((self.config["lags"],self.config["lags"]))
        for i in range(self.config["lags"]):
            for j in range(self.config["lags"]): 
                # compute the partial correlation table Corr(Res,Cand_i | Cond_j)
                RCa_Co[i,j] = (RCa[i] - RCo[j]*CoCa_table[j,i]) / np.sqrt( (1 - CoCa_table[j,i]**2)*(1 - RCo[j]**2) ) 
                # compute the partial correlation table Corr(Res,Cond_i | Cand_j)
                RCo_Ca[i,j] = (RCo[i] - RCa[j]*CoCa_table[i,j]) / np.sqrt( (1 - CoCa_table[i,j]**2)*(1 - RCa[j]**2) ) 
        
        # compute pvalue of the two correlation tables
        p_RCa_Co = self._correl_pvalue(RCa_Co,sample_length,1) # conditioning set is size 1
        p_RCo_Ca = self._correl_pvalue(RCo_Ca,sample_length,1) # conditioning set is size 1
        p_RCo = self._correl_pvalue(RCo,sample_length,0) # conditioning set is size 0
        p_RCa = self._correl_pvalue(RCa,sample_length,0) # conditioning set is size 0
        
        # first table is the partial correlation with Cond as conditioning set, second with Cand as conditioning set
        # third is the correlation of Cand with residuals, fourth is correlation of Cond with residuals ==> needed for relevance
        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo

    def partial_corr(self,residuals_df, candidate_df, condition_df):
        """
        Slow version of the partial correlation based on pingouin package
        """
        # correlation of Res and Cand
        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        RCa = mass2_modified(cand, res)
        RCa = RCa[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        # correlation of Res and Cond
        res,cond=self._select_correct_rows(residuals_df, condition_df)
        RCo = mass2_modified(cond, res)
        RCo = RCo[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        
        # effective size over which the above has been estimated:
        sample_length = len(res)
        
        # Compute the final partial correlation
        L = self.config["lags"]
        p_RCa_Co = np.zeros((L,L))
        p_RCo_Ca = np.zeros((L,L))
        for i in range(L):
            for j in range(L): 
                d=pd.DataFrame({"res":res,"cond":cond[j:-L+j,0],"cand":cand[i:-L+i,0]})
                # compute the partial correlation table Corr(Res,Cand_i | Cond_j)
                p_RCa_Co[i,j] = pingouin.partial_corr(data=d,x="res",y="cand",covar="cond")["p-val"].values[0]
                # compute the partial correlation table Corr(Res,Cond_j | Cand_i)
                p_RCo_Ca[j,i] = pingouin.partial_corr(data=d,x="res",y="cond",covar="cand")["p-val"].values[0]
        
        # compute pvalue of the two correlation tables
        p_RCo = self._correl_pvalue(RCo,sample_length,0) # conditioning set is size 0
        p_RCa = self._correl_pvalue(RCa,sample_length,0) # conditioning set is size 0
        
        # first table is the partial correlation with Cond as conditioning set, second with Cand as conditioning set
        # third is the correlation of Cand with residuals, fourth is correlation of Cond with residuals ==> needed for relevance
        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo

class H2PartialCorrelation(HeuristicPartialCorrelation):
    def partial_corr_mass(self,residuals_df, candidate_df, condition_df):
        """
        Correlation formula for three univariate random variables: corr(x,y|z) = (corr(x,y) - corr(x,z)corr(y,z) ) / sqrt(1-corr²(x,z)) / sqrt(1-corr²(y,z))
        Here, for i,j we want to compute fast corr(Res,Cand_j|Cond_i).
        corr(Res,Cand_j) and corr(Res,Cond_i) can be computed fast for i and j
        corr(Cand_j,Cond_i) is actually the same as corr(Cand_{j-i},Cond_0)
        Additionally, we only want the partial correlation on Cond_i with maximal corr(Res,Cond_i), and Cand_j with maximal corr(Res,Cand_j)
        """
        # correlation of Res and Cand
        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        RCa = mass2_modified(cand, res)
        RCa = RCa[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        # correlation of Res and Cond
        res,cond=self._select_correct_rows(residuals_df, condition_df)
        RCo = mass2_modified(cond, res)
        RCo = RCo[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        
        # effective size over which the above has been estimated:
        sample_length = len(res)
        
        
        # correlation of Cand and Cond
        Co_max_index = np.argmax(np.abs(RCo)) # only test the lag of Co with maximal correlation
        
        # We want the jth element of Co, so we move it to the begining.
        # To keep equal length of observations, we have to remove j elements at the end of Ca.
        # We want to compare the first element of Co to "lags" elements of Ca, so we remove elements at the end of Co.
        # This amounts to not removing elements from Ca and removing "lags"-j elements of Co
        CoCa = mass2_modified(cand, cond[Co_max_index:-self.config["lags"]+Co_max_index])[0][:-1]# here, [0] due to cand being a single series, the last element is Lag(0) of cand, to be removed
        #CoCa[i] is the correlation coefficient of L^{L-j}(Cond) with L^{L-i}(Cand)
        
        # explicit table of (Cond_M, Cand_j)
        CoCa = CoCa # already the right table, the ith element correspond to the correlation of lag L-i of Ca vs lag M of Co
        
        
        # Compute the final partial correlation
        RCa_Co = np.zeros((self.config["lags"]))
        RCo_Ca = np.zeros((self.config["lags"]))
        for i in range(self.config["lags"]):
            # compute the partial correlation table Corr(Res,Cand_i | Cond_M)
            RCa_Co[i] = (RCa[i] - RCo[Co_max_index]*CoCa[i]) / np.sqrt( (1 - CoCa[i]**2)*(1 - RCo[Co_max_index]**2) ) 
            # compute the partial correlation table Corr(Res,Cond_M | Cand_i)
            RCo_Ca[i] = (RCo[Co_max_index] - RCa[i]*CoCa[i]) / np.sqrt( (1 - CoCa[i]**2)*(1 - RCa[i]**2) ) 
        
        # compute pvalue of the two correlation tables
        p_RCa_Co = self._correl_pvalue(RCa_Co,sample_length,1) # conditioning set is size 1
        p_RCo_Ca = self._correl_pvalue(RCo_Ca,sample_length,1) # conditioning set is size 1
        p_RCo = self._correl_pvalue(RCo,sample_length,0) # conditioning set is size 0
        p_RCa = self._correl_pvalue(RCa,sample_length,0) # conditioning set is size 0
        
        # first table is the partial correlation with Cond as conditioning set, second with Cand as conditioning set
        # third is the correlation of Cand with residuals, fourth is correlation of Cond with residuals ==> needed for relevance
        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo

    def partial_corr(self,residuals_df, candidate_df, condition_df):
        """
        Slow version of the partial correlation based on pingouin package
        """
        # correlation of Res and Cand
        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        RCa = mass2_modified(cand, res)
        RCa = RCa[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        # correlation of Res and Cond
        res,cond=self._select_correct_rows(residuals_df, condition_df)
        RCo = mass2_modified(cond, res)
        RCo = RCo[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        
        # effective size over which the above has been estimated:
        sample_length = len(res)
        # correlation of Cand and Cond
        Co_max_index = np.argmax(np.abs(RCo)) # only test the lag of Co with maximal correlation
        
        
        # Compute the final partial correlation
        L = self.config["lags"]
        p_RCa_Co = np.zeros((L,))
        p_RCo_Ca = np.zeros((L,))
        for i in range(L):
            d=pd.DataFrame({"res":res,"cond":cond[Co_max_index:-L+Co_max_index,0],"cand":cand[i:-L+i,0]})
            # compute the partial correlation table Corr(Res,Cand_i | Cond_M)
            p_RCa_Co[i] = pingouin.partial_corr(data=d,x="res",y="cand",covar="cond")["p-val"].values[0]
            # compute the partial correlation table Corr(Res,Cond_M | Cand_i)
            p_RCo_Ca[i] = pingouin.partial_corr(data=d,x="res",y="cond",covar="cand")["p-val"].values[0]
        
        # compute pvalue of the two correlation tables
        p_RCo = self._correl_pvalue(RCo,sample_length,0) # conditioning set is size 0
        p_RCa = self._correl_pvalue(RCa,sample_length,0) # conditioning set is size 0
        
        # first table is the partial correlation with Cond as conditioning set, second with Cand as conditioning set
        # third is the correlation of Cand with residuals, fourth is correlation of Cond with residuals ==> needed for relevance
        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo


class H3PartialCorrelation(HeuristicPartialCorrelation):
    def partial_corr_mass(self,residuals_df, candidate_df, condition_df, ):
        """
        Correlation formula for three univariate random variables: corr(x,y|z) = (corr(x,y) - corr(x,z)corr(y,z) ) / sqrt(1-corr²(x,z)) / sqrt(1-corr²(y,z))
        Here, for i,j we want to compute fast corr(Res,Cand_j|Cond_i).
        corr(Res,Cand_j) and corr(Res,Cond_i) can be computed fast for i and j
        Additionally, we only want the partial correlation on Cond_i with maximal corr(Res,Cond_i).
        """
        # correlation of Res and Cand
        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        RCa = mass2_modified(cand, res)
        RCa = RCa[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        # correlation of Res and Cond
        res,cond=self._select_correct_rows(residuals_df, condition_df)
        RCo = mass2_modified(cond, res)
        RCo = RCo[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        
        # effective size over which the above has been estimated:
        sample_length = len(res)
        
        
        # correlation of Cand and Cond
        Co_max_index = np.argmax(np.abs(RCo)) # only test the lag of Co with maximal correlation
        Ca_max_index = np.argmax(np.abs(RCa))
        
        # we align both tables for the correct lags. 
        CoCa = pearsonr(cand[Ca_max_index:-self.config["lags"]+Ca_max_index][:,0],cond[Co_max_index:-self.config["lags"]+Co_max_index][:,0])[0]
        
        # Compute the final partial correlation
        RCa_Co = 0
        RCo_Ca = 0
        # compute the partial correlation table Corr(Res,Cand_M2 | Cond_M1)
        RCa_Co = (RCa[Ca_max_index] - RCo[Co_max_index]*CoCa) / np.sqrt( (1 - CoCa**2)*(1 - RCo[Co_max_index]**2) ) 
        # compute the partial correlation table Corr(Res,Cond_M1 | Cand_M2)
        RCo_Ca = (RCo[Co_max_index] - RCa[Ca_max_index]*CoCa) / np.sqrt( (1 - CoCa**2)*(1 - RCa[Ca_max_index]**2) ) 
        
        # compute pvalue of the two correlation tables
        
        p_RCa_Co = self._correl_pvalue(RCa_Co,sample_length,1) # conditioning set is size 1
        p_RCo_Ca = self._correl_pvalue(RCo_Ca,sample_length,1) # conditioning set is size 1
        p_RCo = self._correl_pvalue(RCo,sample_length,0) # conditioning set is size 0
        p_RCa = self._correl_pvalue(RCa,sample_length,0) # conditioning set is size 0
        
        # first table is the partial correlation with Cond as conditioning set, second with Cand as conditioning set
        # third is the correlation of Cand with residuals, fourth is correlation of Cond with residuals ==> needed for relevance
        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo

    def partial_corr(self,residuals_df, candidate_df, condition_df, ):
        """
        Slow version of the partial correlation based on pingouin package
        """
        # correlation of Res and Cand
        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        RCa = mass2_modified(cand, res)
        RCa = RCa[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        # correlation of Res and Cond
        res,cond=self._select_correct_rows(residuals_df, condition_df)
        RCo = mass2_modified(cond, res)
        RCo = RCo[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        
        # effective size over which the above has been estimated:
        sample_length = len(res)
        # correlation of Cand and Cond
        Co_max_index = np.argmax(np.abs(RCo)) # only test the lag of Co with maximal correlation
        Ca_max_index = np.argmax(np.abs(RCa))
        
        # Compute the final partial correlation
        L = self.config["lags"]
        RCa_Co = 0
        RCo_Ca = 0
        d=pd.DataFrame({"res":res,"cond":cond[Co_max_index:-L+Co_max_index,0],"cand":cand[Ca_max_index:-L+Ca_max_index,0]})
        # compute the partial correlation table Corr(Res,Cand_M2 | Cond_M1)
        p_RCa_Co = pingouin.partial_corr(data=d,x="res",y="cand",covar="cond")["p-val"].values[0]
        # compute the partial correlation table Corr(Res,Cond_M1 | Cand_M2)
        p_RCo_Ca = pingouin.partial_corr(data=d,x="res",y="cond",covar="cand")["p-val"].values[0]
        
        # compute pvalue of the two correlation tables
        p_RCo = self._correl_pvalue(RCo,sample_length,0) # conditioning set is size 0
        p_RCa = self._correl_pvalue(RCa,sample_length,0) # conditioning set is size 0
        
        # first table is the partial correlation with Cond as conditioning set, second with Cand as conditioning set
        # third is the correlation of Cand with residuals, fourth is correlation of Cond with residuals ==> needed for relevance
        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo





class HkPartialCorrelation(HeuristicPartialCorrelation):
    def partial_corr(self,residuals_df, candidate_df, condition_df):
        # correlation of Res and Cand
        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        RCa = mass2_modified(cand, res)
        RCa = RCa[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        # correlation of Res and Cond
        res,cond=self._select_correct_rows(residuals_df, condition_df)
        RCo = mass2_modified(cond, res)
        RCo = RCo[0][:-1] # remove last value as instantaneous correlation should not be taken into account for residuals
        
        # effective size over which the above has been estimated:
        sample_length = len(res)
        
        
        # correlation of Cand and Cond
        k = self.config["k"]
        Co_max_indexes = np.argpartition(np.abs(RCo),-k)[-k:] # only test the lag of Co with maximal correlation
        Ca_max_indexes = np.argpartition(np.abs(RCa),-k)[-k:]
        
        
        RCa_Co = np.zeros((k,k))
        RCo_Ca = np.zeros((k,k))
        # we align both tables for the correct lags. 
        for i,Ca_max_index in enumerate(Ca_max_indexes):
            for j,Co_max_index in enumerate(Co_max_indexes):
            
                CoCa = pearsonr(cand[Ca_max_index:-self.config["lags"]+Ca_max_index][:,0],cond[Co_max_index:-self.config["lags"]+Co_max_index][:,0])[0]
                
                RCa_Co[i,j] = (RCa[Ca_max_index] - RCo[Co_max_index]*CoCa) / np.sqrt( (1 - CoCa**2)*(1 - RCo[Co_max_index]**2) ) 
                RCo_Ca[j,i] = (RCo[Co_max_index] - RCa[Ca_max_index]*CoCa) / np.sqrt( (1 - CoCa**2)*(1 - RCa[Ca_max_index]**2) ) 
        
        
        # compute pvalue of the two correlation tables
        
        p_RCa_Co = self._correl_pvalue(RCa_Co,sample_length,1) # conditioning set is size 1
        p_RCo_Ca = self._correl_pvalue(RCo_Ca,sample_length,1) # conditioning set is size 1
        p_RCo = self._correl_pvalue(RCo[Co_max_indexes],sample_length,0) # conditioning set is size 0
        p_RCa = self._correl_pvalue(RCa[Ca_max_indexes],sample_length,0) # conditioning set is size 0
        
        # first table is the partial correlation with Cond as conditioning set, second with Cand as conditioning set
        # third is the correlation of Cand with residuals, fourth is correlation of Cond with residuals ==> needed for relevance
        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo


class MixedTemporalSlowHk(HeuristicPartialCorrelation):
    """Partial Correlation heuristic for categorical-categorical and continuous-categorical.
    """
    def __init__(self, config):
        self.config = config
        self._check_config()

    def _check_config(self):
        assert "variable_types" in self.config
        assert "k" in self.config
        assert "categorical_method" in self.config
        assert "lags" in self.config
    
    def partial_corr(self, residuals_df, candidate_df, condition_df):
        k = self.config["k"]
        lags = self.config["lags"]
        method = self.config["categorical_method"]
        variable_types = self.config["variable_types"]
        nameCa = candidate_df.columns[0]
        nameCo = condition_df.columns[0]
        typeCa = variable_types[nameCa]
        typeCo = variable_types[nameCo]

        #!TODO: replace by call to predefined class to avoid computing results twice
        asso_instance = TemporalSlowAssociation({"categorical_method":method,"lags":lags,"variable_types":variable_types})
        _ = asso_instance.association(residuals_df, pd.concat([candidate_df,condition_df],axis=1))

        p_RCa = np.array(asso_instance.pvalues[nameCa])
        p_RCo = np.array(asso_instance.pvalues[nameCo])
        Co_max_indexes = np.argpartition(-p_RCo, -k)[-k:]  # only test the lag of Co with maximal association
        Ca_max_indexes = np.argpartition(-p_RCa, -k)[-k:]

        p_RCa = p_RCa[Ca_max_indexes]
        p_RCo = p_RCo[Co_max_indexes]
        
        p_RCa_Co = np.zeros((k, k))
        p_RCo_Ca = np.zeros((k, k))

        res,cand=self._select_correct_rows(residuals_df, candidate_df)
        res,cond=self._select_correct_rows(residuals_df, condition_df)

        #categorical-numerical and categorical-categorical
        if typeCa!="numerical" or typeCo!="numerical":
            n = len(res)
            x_type = np.zeros((n,1))
            y_type = np.zeros((n,1)) if typeCa!="categorical" else np.ones((n,1))
            z_type = np.zeros((n,1)) if typeCo!="categorical" else np.ones((n,1))

            for i, Ca_max_index in enumerate(Ca_max_indexes):
                for j, Co_max_index in enumerate(Co_max_indexes):
                    instance = tigramite.independence_tests.regressionCI.RegressionCI()
                    x = np.expand_dims(res, axis=1)
                    y = cand[Ca_max_index:n+Ca_max_index]
                    z = cond[Co_max_index:n+Co_max_index]
                    p_RCa_Co[i,j] = instance.run_test_raw(x,y,z,x_type=x_type,y_type=y_type,z_type=z_type)[1]
                    p_RCo_Ca[j,i] = instance.run_test_raw(x,z,y,x_type=x_type,y_type=z_type,z_type=y_type)[1]

        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo


class PartialCorrelation():
    def __init__(self, config):
        pass
    def partial_corr(self, residuals_df, candidate_df, condition_df):
        pass

class CrossSectionalHk(PartialCorrelation):
    def __init__(self, config):
        self.config = config
        self._check_config()

    def _check_config(self):
        assert "variable_types" in self.config
        assert "categorical_method" in self.config
        assert "k" in self.config

    def partial_corr(self,residuals_df, candidate_df, condition_df):
        """

        :param residuals_df: pd.Series or pd.DataFrame
        :param candidate_df: pd.DataFrame
        :param condition_df: pd.DataFrame
        :return: np.array
        """
        k = self.config["k"]
        variable_types = self.config["variable_types"]
        categorical_method = self.config["categorical_method"]
        nameCa = candidate_df.columns.get_level_values(0).unique()[0]
        nameCo = condition_df.columns.get_level_values(0).unique()[0]
        typeCa = variable_types[nameCa]
        typeCo = variable_types[nameCo]

        #!TODO: replace by call to predefined class to avoid computing results twice
        asso_config = {"variable_types":variable_types,"categorical_method":categorical_method}
        asso_instance = CrossSectionalAssociation(asso_config)
        _ = asso_instance.association(residuals_df, pd.concat([candidate_df,condition_df],axis=1))

        p_RCa = asso_instance.pvalues[nameCa]
        p_RCo = asso_instance.pvalues[nameCo]
        Co_max_indexes = np.argpartition(-p_RCo, -k)[-k:]  # only test the lag of Co with maximal association
        Ca_max_indexes = np.argpartition(-p_RCa, -k)[-k:]

        p_RCa = p_RCa[Ca_max_indexes]
        p_RCo = p_RCo[Co_max_indexes]
        
        p_RCa_Co = np.zeros((k, k))
        p_RCo_Ca = np.zeros((k, k))

        #numerical-numerical
        if typeCa=="numerical" and typeCo=="numerical":
            # we align both tables for the correct lags.
            for i, Ca_max_index in enumerate(Ca_max_indexes):
                for j, Co_max_index in enumerate(Co_max_indexes):
                    d = pd.concat([residuals_df,candidate_df[nameCa].iloc[:,Ca_max_index],condition_df[nameCo].iloc[:,Co_max_index]],axis=1)
                    d.columns = ["res","cand","cond"]
                    p_RCa_Co[i,j] = pingouin.partial_corr(data=d, x="res", y="cand", covar="cond")["p-val"].values[0]
                    p_RCo_Ca[j,i] = pingouin.partial_corr(data=d, x="res", y="cond", covar="cand")["p-val"].values[0]


        #categorical-numerical and categorical-categorical
        if typeCa!="numerical" or typeCo!="numerical":
            # define outside of loop to avoid allocating space repetitively
            n = len(candidate_df)
            x_type = np.zeros((n,1))
            y_type = np.zeros((n,1)) if typeCa!="categorical" else np.ones((n,1))
            z_type = np.zeros((n,1)) if typeCo!="categorical" else np.ones((n,1))
            # we align both tables for the correct lags.
            for i, Ca_max_index in enumerate(Ca_max_indexes):
                for j, Co_max_index in enumerate(Co_max_indexes):
                    instance = tigramite.independence_tests.regressionCI.RegressionCI()
                    x = residuals_df
                    y = candidate_df[nameCa].iloc[:,[Ca_max_index]]
                    z = condition_df[nameCo].iloc[:,[Co_max_index]]
                    p_RCa_Co[i,j] = instance.run_test_raw(x,y,z,x_type=x_type,y_type=y_type,z_type=z_type)[1]
                    p_RCo_Ca[j,i] = instance.run_test_raw(x,z,y,x_type=x_type,y_type=z_type,z_type=y_type)[1]

        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo



class TemporalSlowHk(PartialCorrelation):
    """ Partial correlation for mixed type data.
    Used in the equivalence phase.

    Given residuals denoted R, a candidate variable Ca, a condition variable Co,
    and noting lag i of variable Ca by Ca_i and lag j of variable Co by Co_j,
    This method computes:
        1) the pvalue of R indep Ca_i for all i
        2) the pvalue of R indep Co_j for all j
        3) the index i1,...,ik corresponding the maximal association between R and Ca_i (minimal pvalues)
        4) the index j1,...,jk corresponding the maximal association between R and Co_j (minimal pvalues)
        5) the pvalue of R indep Ca_iu | Co_jv, for iu in {i1,...,ik} and jv in {j1,...,jk}
        6) the pvalue of R indep Co_jv | Ca_iu, for iu in {i1,...,ik} and jv in {j1,...,jk}
    """
    def __init__(self, config):
        """ Initialize the partial correlation object.

        Parameters
        ----------
        config: dict
            Must contain an entry for:
             - "lags": int, the number of lags to compute the correlation over
             - "categorical_method": str, any of 'f_oneway', 'kruskal', 'alexandergovern'.
                This specifies the kind of test used for categorical data.
             - "variable_types": dict, for each variable name, whether it is "numerical" or "categorical".
                See examples.
             - "k": int, the number of lags to consider for equivalence computation.
               "k" must be lower or equal to "lags".

        Returns
        -------
        None

        Examples
        --------
        >>> data = pd.DataFrame(np.random.random(size=(1000,5)),columns=["target","1","2","3","4"])
        >>> variable_types = dict([(column, "numerical") for column in data.columns])
        >>> asso = TemporalSlowHk({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})

        Or with mixed types:

        >>> variable_types = {"target":"numerical","1":"numerical","2":"numerical","3":"categorical","4":"categorical"}
        >>> asso = TemporalSlowHk({"lags":10,"categorical_method":"f_oneway","variable_types":variable_types})
        """
        self.config = config
        self._check_config()

    def _check_config(self):
        assert "variable_types" in self.config
        assert "k" in self.config
        assert "categorical_method" in self.config
        assert "lags" in self.config
    
    def partial_corr(self, residuals_df, candidate_df, condition_df):
        """ 
        Parameters
        ----------
        residuals_df: pd.DataFrame
            DataFrame of shape (ntimesteps, 1) containing the model residuals of a learning model. 
        candidate_df: pd.DataFrame
            DataFrame of shape (ntimesteps, 1) containing one of the two univariate time series to test for equivalence.
            The index must be aligned with residuals_df
        condition_df: pd.DataFrame
            DataFrame of shape (ntimesteps, 1) containing one of the two univariate time series to test for equivalence.
            The index must be aligned with residuals_df

        Returns
        -------
        p_RCa_Co: np.array
            A 2D numpy array of shape (k,k). It contains the p-values of the tests (R indep Ca_i | Co_j),
            with R the residuals, Ca the candidate TS, Co the condition TS.
            The first dimension correspond to a retained lag of Ca, the second dimension to a lag of Co.
        p_RCa_Co: np.array
            A 2D numpy array of shape (k,k). It contains the p-values of the tests (R indep Co_j | Ca_i).
            The first dimension correspond to a retained lag of Co, the second dimension to a lag of Ca.
        p_RCa: np.array
            A 1D numpy array of shape (k,). It contains the p-value of the correlations (R indep Ca_i).
        p_RCo: np.array
            A 1D numpy array of shape (k,). It contains the p-value of the correlations (R indep Co_i).

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> data = pd.DataFrame(rng.random(size=(1000,5)),columns=["target","1","2","3","4"])
        >>> variable_types = dict([(column, "numerical") for column in data.columns])
        >>> asso = TemporalSlowHk({"lags":10,"k":2,"categorical_method":"f_oneway","variable_types":variable_types})
        >>> asso.partial_corr(data[["target"]],data[["1"]],data[["2"]])
        (array([[0.07208953, 0.05627934],
                [0.03686298, 0.04137501]]),
        array([[0.09649547, 0.10936624],
                [0.02173326, 0.03464236]]),
        array([0.07455153, 0.03384917]),
        array([0.09990165, 0.02838155]))
        """

        variable_types = self.config["variable_types"]
        nameCa = candidate_df.columns[0]
        nameCo = condition_df.columns[0]
        typeCa = variable_types[nameCa]
        typeCo = variable_types[nameCo]

        if typeCa=="numerical" and typeCo=="numerical":
            partial_obj = HkPartialCorrelation(self.config)
            p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = partial_obj.partial_corr(residuals_df, candidate_df, condition_df)
        else:
            partial_obj = MixedTemporalSlowHk(self.config)
            p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = partial_obj.partial_corr(residuals_df, candidate_df, condition_df)

        return p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo






# class TemporalSlowHk(CrossSectionalHk):
#     def reshape_to_cross_sectional(self,df):
#         lags = self.config["lags"]
#         data = dict()
#         for column in df.columns:
#             for l in range(lags):
#                 new_column = df[column].iloc[l:len(df)-lags+l]
#                 new_column.index = df.index[lags:]
#                 data[(column,l-lags)]=new_column
#         data = pd.DataFrame(data)
#         data.index = df.index[lags:]
#         data.columns = pd.MultiIndex.from_tuples(data.columns)
#         return data
    
#     def partial_corr(self,residuals_df, candidate_df, condition_df):
#         residuals_df = residuals_df
#         candidate_df = self.reshape_to_cross_sectional(candidate_df)
#         condition_df = self.reshape_to_cross_sectional(condition_df)
#         return super(TemporalSlowHk,self).partial_corr(residuals_df, candidate_df, condition_df)
