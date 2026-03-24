from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AutoRegResults, AutoRegResultsWrapper
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

import statsmodels.discrete.discrete_model

from scipy.stats import f as fdistrib, wilcoxon, chi2

from sklearn.metrics import mean_absolute_percentage_error

import numpy as np
import pandas as pd
import pydash

# custom error for ARDL model

class NotEnoughDataError(ValueError):
    def __init__(self, datasize, lags, orders):
        self.message = "Model needs more lags of the data than provided to make predictions.\nPlease consider increasing the test data size.\nData length: {}, number of lags: {}, number of orders: {}".format(datasize, lags, orders)
        super().__init__(self.message)
    


##
#
#   Learning wrappers
#
##


class LearningModel:
    def __init__(self, config, target):
        """Initialize a learning model.

        Parameters
        ----------
        config: dict
            Contains the parameter settings of the model. Free for each class to define.
        target: str | tuple[str,str]
            The name of the target. 
            As a string for single level dataframe, a tuple of strings for two-levels dataframes.
        
        Returns
        -------
        None
        """
        self.config = config
        self.target = target
        
        self.data = None
        self.model = None

    #
    # part that need to be implemented for each learning model    
    #
    
    def fit(self, data):
        """Fit the model on the provided data.

        If the model requires a validation split, it may handle it internally.
        The method may save the fitted model in self.model
        The method MUST save the data used to train the model in self.data.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame containing the multivariate time series.
            The index of the DataFrame should correspond to timesteps, and the columns to covariates.
            The column index may have one or two levels depending on user choice.
            It is not required that the model handles both formats.
        
        Returns
        -------
        None
        """
        self.data = data
        
        raise NotImplementedError

    def fittedvalues(self, data=None):
        """Produces the predicted values of the regression task.

        Would correspond to a .predict operation in other libraries.

        Parameters
        ----------
        data: pd.DataFrame, optional
            The data used for prediction. It must have the same columns as the data passed to fit.
            If None is passed, the training data should be used.

        Returns
        -------
        fittedvalues: pd.Series
            1D series with index aligned to the original data.
            In case only an interval of the timesteps is used for prediction (train-test split),
            the predicted values returned may cover only this interval, as long as the index is properly set.
            The fittedvalues must be ordered along time with no missing values in the covered interval.
        """
        if data is None:
            if self.data is None:
                raise(Exception("data was not passed as an argument, and self.data is None.\nHave you set self.data during the call to self.fit?"))
            data = self.data

        raise NotImplementedError
        
    def stopping_metric(self, previous_model, method=""):
        """Metric by which model equivalence is tested.

        The returned metric will be compared to the threshold provided to ChronoEpilogi algorithm.
        If the metric is higher than the threshold, we consider the models equivalent.
        If the metric is lower than the threshold, we consider the current model more powerful than the previous model.

        Parameters
        ----------
        previous_model: LearningModel
            An instance of the same model class, containing a model fitted on data with a subset of the covariates of the current model.
        method: str, optional
            Optional parameter that specifies the method by which to obtain the metric, in case several methods can be considered.
        
        Returns
        -------
        metric: float
            The metric of model equivalence.
        """
        raise NotImplementedError
    
    def has_too_many_parameters(self, ratio):
        """Flag specifying whether the model is too large to be trained.

        Legacy method from linear model applications. 
        Originally, it was used to prevent models from being fitted on datasets with too low sample size compared to number of model parameters.
        May return False constantly if this is no concern for the present model.
        
        Parameters
        ----------
        ratio: float
            Originally, the ratio (sample size)/(number of parameters) that should be minimally respected.
        
        Returns
        -------
        flag: bool
            True if the model is too large for the available data. False otherwise.
        """
        raise NotImplementedError
        
    #
    # part that can be let as-is or modified after inheritance.
    #
    
    def residuals(self, data=None):
        """Compute residuals.

        Uses training data if "data" argument is None, otherwise specified data is used.
        In the case that the fitted values only cover part of the data (for instance when splitting),
        automatically aligns the indexes so that residuals are computed over the provided fitted values only.
        
        Parameters
        ----------
        data: pd.DataFrame, optional
            The data used for prediction. It must have the same columns as the data passed to fit.
            If None is passed, the training data should be used.
        
        Returns
        -------
        residuals_df: pd.DataFrame
            DataFrame with a single column named after the target TS.
            The index corresponds to the index of the fitted values.
            As are the fitted values, residuals should be ordered along time with no missing values in the covered interval.
        
        Examples
        --------
        >>> class NewLearningModel(LearningModel):
        ...     def fittedvalues(self,data=None):
        ...         return data[self.target].iloc[100:]
        >>> data = pd.DataFrame(np.ones((101,12)))
        >>> data.columns = pd.MultiIndex.from_product([[1,2,3],[1,2,3,4]])
        >>> model = NewLearningModel({},(1,1))
        >>> model.residuals(data)
               1
               1
        100  0.0

        With a single level of columns:

        >>> data = pd.DataFrame(np.ones((101,12)))
        >>> data.columns = list(range(12))
        >>> model = NewLearningModel({},1)
        >>> model.residuals(data)
               1
        100  0.0
        """
        if data is None:
            if self.data is None:
                raise(Exception("data was not passed as an argument, and self.data is None.\nHave you set self.data during the call to self.fit?"))
            data = self.data

        fittedvalues = self.fittedvalues(data)
        targetdata = data[self.target]
        targetdata = targetdata.loc[fittedvalues.index]
        residuals = targetdata - fittedvalues
        residuals_df = pd.DataFrame({self.target: residuals})
        return residuals_df

























class ARDLModel(LearningModel):
    """
    Adapted from statsmodels.tsa.ardl.ARDL.
     - due to arguments in both instance declaration and fit routine, config must contain:
       - arguments to pass to ARDL constructor
       - arguments to pass to ARDL.fit
     - does not use lag estimation
     - target variable is included in the observed variables
     - uses minus aic of final model as significance

    config (dict):
     - "constructor" (dict): arguments to pass to the ARDL constructor
       - see https://www.statsmodels.org/dev/generated/statsmodels.tsa.ardl.ARDL.html#statsmodels.tsa.ardl.ARDL
     - "fit" (dict): arguments to pass to the ARDL.fit method
       - see https://www.statsmodels.org/dev/generated/statsmodels.tsa.ardl.ARDL.fit.html#statsmodels.tsa.ardl.ARDL.fit
    """

    def __init__(self, config, target):
        super().__init__(config, target)
        self.results = None  # to store the ARDLResults instance

    def fit(self, data):
        """Make sure that number of parameters are enough compared to the data size
        """
        if isinstance(self.config["constructor"]["order"], int) or isinstance(self.config["constructor"]["order"], float):
            maxlag = self.config["constructor"]["order"]
        else:
            maxlag = max(self.config["constructor"]["order"])
        maxlag = max([maxlag, self.config["constructor"]["lags"]])
        if len(data.index) - maxlag < maxlag*len(data.columns)+4:
            raise NotEnoughDataError(len(data.index), self.config["constructor"]["lags"], self.config["constructor"]["order"])
    
        self.data = data
        self.model = self.createModel(data)
        self.results = self.model.fit(**self.config["fit"])
        
    def createModel(self, data):
        """
        Creates the model from the data provided.
        Created model is not trained.
        """
        if len(data.columns)>1:
            model = ARDL(endog=data[self.target],
                              exog=data.loc[:, data.columns != self.target],
                              **self.config["constructor"])
        else:
            model = ARDL(endog=data[self.target],
                              exog=None,
                              order=None,
                              **pydash.omit(self.config["constructor"], "order"))
        return model

    def stopping_metric(self, previous_model, method):
        """
        Computes the metric associated to the model type.
        The lower the better the new model
        """
        metric = None
        if method == "aic":  # compare models significances
            previous_model_significance = previous_model.aic()
            current_model_significance = self.aic()
            metric = current_model_significance - previous_model_significance
            
        elif method == "f-test":
            constraint_matrix = []
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    new_constraint = np.zeros((len(self.results.params.index),))
                    new_constraint[i]=1
                    constraint_matrix.append(new_constraint)
            r_matrix = np.array(constraint_matrix)
            metric = self.results.f_test(r_matrix).pvalue
            
        elif method == "by_hand_f-test":
            fstat_top = (previous_model.sse() - self.sse()) / (previous_model.dof() - self.dof())
            fstat_bot = self.sse() / self.dof()
            fstat = fstat_top / fstat_bot
            pvalue = 1 - fdistrib.cdf(fstat, previous_model.dof() - self.dof(), self.dof())
            metric = 0 if np.isnan(pvalue) else pvalue
        
        elif method == "wald-test":
            constraint_matrix = []
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    new_constraint = np.zeros((len(self.results.params.index),))
                    new_constraint[i]=1
                    constraint_matrix.append(new_constraint)
            r_matrix = np.array(constraint_matrix)
            metric = self.results.wald_test(r_matrix, use_f=False, scalar=True).pvalue
        
        elif method == "lr-test":
            diff_dof = 0
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    diff_dof+=1
            cstat = -2*(previous_model.llh() - self.llh())
            metric = chi2.sf(cstat,df=diff_dof)
        
        return metric

    def _pad_test_data_to_create_model(self, data):
        """
        The not-so-nice thing about creating a copy model for test data,
        is that model instanciation checks that the data is large enough to learn.
        Hence, when test data is large enough to be evaluated (timesteps > lags)
        but not enough to be learned (timesteps < lags*variables + 1), it blocks.
        The solution I found is to simply pad the data. The fittedvalue method will select
        the right timestamps at the end.
        This method is handling the padding.
        
        Returns:
             the padded or nonpadded data
        Raises NotEnoughDataError if the data is too small for even 1 prediction.
        """
        if len(data)<=self.config["constructor"]["lags"] or len(data)<=self.config["constructor"]["order"]:
            # estimation is impossible, not enough descriptors for 1 prediction
            raise NotEnoughDataError(len(data), self.config["constructor"]["lags"], self.config["constructor"]["order"])
        
        period = self.config["constructor"]["period"] if "period" in self.config["constructor"] else 1
        period = period+1 if period is not None else 2
        needed_regressors = (len(data.columns) - 1)*self.config["constructor"]["order"] 
        needed_regressors += self.config["constructor"]["lags"] + 4
        needed_regressors *= period
        if needed_regressors - len(data)>0:
            zeros = np.zeros((needed_regressors - len(data), len(data.columns)))
            index = range(len(zeros))
            zeros = pd.DataFrame(zeros, columns=data.columns, index=index)
            data = pd.concat([data, zeros])
            
        return data

    def fittedvalues(self,data=None):
        if data is not None:
            index = data.index  # keep track of original index
            pad_data = self._pad_test_data_to_create_model(data)  # pad just in case test size is small
            pad_data = pad_data.reset_index(drop=True)  # predict works better for rangeindex starting at 0
            model = self.createModel(pad_data)
            # use previous parameters 
            fittedvalues = model.predict(self.results._params, start=0,end=len(data)-1, dynamic=False)
            fittedvalues_nona = fittedvalues.dropna()
            fittedvalues_nona.index = index[fittedvalues_nona.index]
            if len(fittedvalues_nona)==0:
                print(fittedvalues_nona)
                print(fittedvalues)
                print(pad_data)
                print("\n")
            return fittedvalues_nona
        else:
            return self.results.fittedvalues

    


    def aic(self):
        return self.results.aic
        
    def llh(self):
        return self.results.llf

    def dof(self):
        return self.results.df_resid
    
    def has_too_many_parameters(self, ratio):
        nbparams = len(self.results.params)
        nobs = self.results.nobs
        return nobs/nbparams<ratio

        


class LogitCrossSectional(LearningModel):
    def __init__(self, config, target):
        self.config = config
        self.target = target

        self.data = None
        self.model = None
        self.results = None

    def _remove_constant_columns(self,data):
        filtered = []
        for column in data.columns:
            if data[column].std()!=0:
                filtered.append(column)
        return data[filtered]
    def _MultiIndex_to_flat(self,data):
        data2 = data.copy()
        data2.columns = data2.columns.to_flat_index()
        return data2
    def _get_endog(self,data):
        data2 = data.loc[:,self.target]
        if not isinstance(data2, pd.Series):
            data2.columns = data2.columns.to_flat_index()
        return data2
    def _get_exog(self,data):
        columns = [x for x in data.columns.get_level_values(0).unique() if x!=self.target[0]]
        exog = data[columns]
        exog = self._MultiIndex_to_flat(exog)
        exog = self._remove_constant_columns(exog)
        exog["intercept"] = 1  # add intercept, as the model is not doing it itself
        return exog

    def fit(self, data):
        self.data = data
        endog = self._get_endog(self.data)
        exog = self._get_exog(self.data)
        self.model = statsmodels.discrete.discrete_model.Logit(endog,exog,**self.config["constructor"])
        if "fit" in self.config:
            self.results = self.model.fit(**self.config["fit"])
        else:
            self.results = self.model.fit_regularized(**self.config["fit_regularized"])

    def fittedvalues(self, data=None):
        # return the fitted values of the model.
        # should be a pd.Series with corresponding index to original data
        # the series should not contain NaN timestamps.
        if data is None:
            data = self.data
        exog = self._get_exog(data)
        values = self.results.predict(exog)
        return values

    def stopping_metric(self, previous_model, method):
        # should return a metric that corresponds more or less to p-values.
        # the lower, the more incentive to keep adding new variables to the selected set
        if method == "lr-test":
            diff_dof = 0
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    diff_dof += 1
            llh_prev = previous_model.loglikelihood()
            llh_new = self.loglikelihood()
            cstat = -2 * (llh_prev - llh_new)
            metric = chi2.sf(cstat, df=diff_dof)
        return metric

    def has_too_many_parameters(self, ratio):
        # part of the stopping criterion: verify if there are ratio times more timestamps in the data
        # than parameters in the model.
        nbparams = len(self.results.params)
        nobs = self.results.nobs
        return nobs / nbparams < ratio

    #
    # part that can be let as-is or modified after inheritance.
    #
    
    def loglikelihood(self):
        return self.model.loglike(self.results.params)
        
    def total_variation(self, data=None):
        if data is None:
            data = self.data
        targetdata = self._get_endog(data)
        targetdata = targetdata[targetdata.columns[0]]
        return np.sum((targetdata-np.mean(targetdata))**2)


    def residuals(self, data=None):
        """

        """
        # output should be a pd.DataFrame, rows index should correspond to the original data
        # the series should not contain NaN timestamps.
        if data is None:
            if self.data is None:
                raise(Exception("data was not passed as an argument, and self.data is None.\nHave you set self.data during the call to self.fit?"))
            data = self.data

        fittedvalues = self.fittedvalues(data)
        targetdata = self._get_endog(data)
        if not isinstance(targetdata, pd.Series):
            targetdata = targetdata[targetdata.columns[0]]

        targetdata = targetdata.loc[fittedvalues.index]

        if self.config['residuals']=='raw':
            residuals = targetdata - fittedvalues
        elif self.config['residuals']=='pearson':
            observed,expected = targetdata,fittedvalues
            residuals = (observed - expected) / np.sqrt(expected * (1 - expected))

        # make the results a dataframe addressable by target.
        df = pd.DataFrame({self.target: residuals})
        return df
        
class PoissonCrossSectional(LogitCrossSectional):
    def fit(self, data):
        self.data = data
        endog = self._get_endog(self.data)
        exog = self._get_exog(self.data)
        self.model = statsmodels.discrete.discrete_model.Poisson(endog,exog,**self.config["constructor"])
        if "fit" in self.config:
            self.results = self.model.fit(**self.config["fit"])
        else:
            self.results = self.model.fit_regularized(**self.config["fit_regularized"])

class NegativeBinomialCrossSectional(LogitCrossSectional):
    def fit(self, data):
        self.data = data
        endog = self._get_endog(self.data)
        exog = self._get_exog(self.data)
        self.model = statsmodels.discrete.discrete_model.NegativeBinomial(endog,exog,**self.config["constructor"])
        if "fit" in self.config:
            self.results = self.model.fit(**self.config["fit"])
        else:
            self.results = self.model.fit_regularized(**self.config["fit_regularized"])

class OLSCrossSectional(LogitCrossSectional):
    def fit(self, data):
        self.data = data
        endog = self._get_endog(self.data)
        exog = self._get_exog(self.data)
        self.model = statsmodels.regression.linear_model.OLS(endog,exog,**self.config["constructor"])
        if "fit" in self.config:
            self.results = self.model.fit(**self.config["fit"])
        else:
            self.results = self.model.fit_regularized(**self.config["fit_regularized"])
    
    def has_too_many_parameters(self, ratio):
        # part of the stopping criterion: verify if there are ratio times more timestamps in the data
        # than parameters in the model.
        nbparams = len(self.results.params)
        nobs = len(self.data)
        return nobs / nbparams < ratio
    

class TemporalAdaptation(LogitCrossSectional):
    def __init__(self, config, target):
        self.config = config
        self.target = target
        self.data = None
        self.model = None
        self.results = None
        if self.config["model_type"]=="OLSCrossSectional":
            self.cross_sectional_instance = OLSCrossSectional(config,(target,None))
        elif self.config["model_type"]=="PoissonCrossSectional":
            self.cross_sectional_instance = PoissonCrossSectional(config,(target,None))
        elif self.config["model_type"]=="LogitCrossSectional":
            self.cross_sectional_instance = LogitCrossSectional(config,(target,None))
        elif self.config["model_type"]=="NegativeBinomialCrossSectional":
            self.cross_sectional_instance = NegativeBinomialCrossSectional(config,(target,None))
        
    def reshape_to_cross_sectional(self, df):
        lags = self.config["lags"]
        autoregressive = self.config["autoregressive"]
        target = self.target
        data = dict()
        for column in df.columns:
            if column==target:
                data[(column, None)]=df[column].iloc[lags:]
                if not autoregressive:
                     continue
            for l in range(lags):
                new_column = df[column].iloc[l:len(df)-lags+l]
                new_column.index = df.index[lags:]
                if column==target:
                    data[(column+"_past",l-lags)]=new_column
                else:
                    data[(column,l-lags)]=new_column
        data = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data.keys()))
        data.index = df.index[lags:]
        return data
        
    def fit(self, data):
        data = self.reshape_to_cross_sectional(data)
        self.data = data
        self.cross_sectional_instance.fit(data)
        self.model = self.cross_sectional_instance.model
        self.results = self.cross_sectional_instance.results
    
    def fittedvalues(self, data=None):
        if data is not None:
            data = self.reshape_to_cross_sectional(data)
            return self.cross_sectional_instance.fittedvalues(data)
        return self.cross_sectional_instance.fittedvalues()
    
    def total_variation(self, data=None):
        if data is not None:
            data = self.reshape_to_cross_sectional(data)
            return self.cross_sectional_instance.total_variation(data)
        return self.cross_sectional_instance.total_variation()
    
    def residuals(self, data=None):
        if data is not None:
            data = self.reshape_to_cross_sectional(data)
            return self.cross_sectional_instance.residuals(data)
        return self.cross_sectional_instance.residuals()
    
    def has_too_many_parameters(self, ratio):
        return self.cross_sectional_instance.has_too_many_parameters(ratio)
         