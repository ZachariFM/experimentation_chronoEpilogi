import numpy as np

import ce_extensions2
import associations
import models

import group_lasso
from statsmodels.tsa.vector_ar.var_model import VAR
import sklearn
import dtw
from joblib import Parallel, delayed



class ChronoEpilogiSingle():
    def __init__(self, config, target, data):
        self.config = self._configuration_completion(config)
        self.target = target
        self.model = ce_extensions2.ChronoEpilogi(self.config, target, data)

    def fit(self, new_config):
        new_config = self._configuration_completion(new_config)
        self.config = new_config
        self.model.fit(config=new_config)

    def get_selected_set(self):
        return self.model.selected_set
    
    def _set_variant_parameters(self, partial_config):
        equivalence_heuristic = partial_config.get("equivalence_heuristic", "full")
        equivalence_greedy_stopping = partial_config.get("equivalence_greedy_stopping", False)
        phases = partial_config.get("phases", "FB")
        return equivalence_heuristic, equivalence_greedy_stopping, phases

    def _configuration_completion(self,partial_config):
        # required parameters
        variable_types = partial_config["variable_types"]
        lags = partial_config["lags"]

        # optional parameters
        association = associations.PearsonMultivariate
        association_config = {'variable_types':variable_types,"lags":lags, "return_type":"p-value", "selection_rule":"max"}
        partial_correlation = associations.HeuristicPartialCorrelation
        partial_correlation_config = {'variable_types':variable_types,"lags":lags,"large_sample":False,
                                      "k":partial_config.get("k",4)}
        model = models.ARDLModel
        model_config = {"constructor":{"order":lags,"lags":lags,"trend":"c","causal":True},
                "fit":{"cov_type":"HC0"}}
        full_test_method = "lr-test"
        max_features = np.inf
        valid_obs_param_ratio = 0.
        significance_threshold = 0.05
        significance_threshold_backward = 0.05
        equivalence_threshold = 0.05
        correlation_threshold = 0.05
        backward_priority = False
        equivalence_heuristic, equivalence_greedy_stopping, phases = self._set_variant_parameters(partial_config)
        data_format = 'temporal'
        default_values = [association,association_config,partial_correlation,
                        partial_correlation_config,model,model_config,
                        full_test_method,max_features,valid_obs_param_ratio,
                        significance_threshold,significance_threshold_backward,
                        equivalence_threshold,correlation_threshold,
                        backward_priority,equivalence_heuristic,equivalence_greedy_stopping,data_format,phases]
        keys = ["association","association.config","partial_correlation",
             "partial_correlation.config","model","model.config",
             "full_test_method","max_features","valid_obs_param_ratio",
             "significance_threshold","significance_threshold_backward",
             "equivalence_threshold","correlation_threshold",
             "backward_priority","equivalence_heuristic","equivalence_greedy_stopping","data_format","phases"]
        d = dict()
        for key, value in zip(keys, default_values):
            d[key] = partial_config.get(key, value)
        return d
    
    def generate_optuna_parameters(trial):
        hp = dict()
        hp["valid_obs_param_ratio"] = trial.suggest_float("feature_selector.valid_obs_param_ratio", 0., 1., step=0.5)
        hp["max_features"] = trial.suggest_int("feature_selector.max_features",50,50)
        hp["significance_threshold"] = trial.suggest_float("feature_selector.significance_threshold", 1e-5, 1., log=True)
        hp["significance_threshold_backward"] = trial.suggest_float("feature_selector.significance_threshold_backward", 1e-5, 1e-1, log=True)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = dict()
        hp_grid["feature_selector.valid_obs_param_ratio"] = [0., 0.5, 1.]
        hp_grid["feature_selector.max_features"] = [50]
        hp_grid["feature_selector.significance_threshold"] = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        hp_grid["feature_selector.significance_threshold_backward"] = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        return hp_grid

class ChronoEpilogiSingle_EndSize(ChronoEpilogiSingle):
    """
    Version of ChronoEpilogiSingle that forces a tunable number of selected TS, without statistical tests.
    Used for comparison purposes with the base version that only uses a test but no maximal number of TS.
    """
    def generate_optuna_parameters(trial):
        hp = dict()
        hp["valid_obs_param_ratio"] = trial.suggest_float("feature_selector.valid_obs_param_ratio", 0., 1., step=0.5)
        hp["max_features"] = trial.suggest_int("feature_selector.max_features",1,30)
        hp["significance_threshold"] = trial.suggest_float("feature_selector.significance_threshold", 2., 2., log=True)
        hp["significance_threshold_backward"] = trial.suggest_float("feature_selector.significance_threshold_backward", 1e-5, 1e-1, log=True)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = dict()
        hp_grid["feature_selector.valid_obs_param_ratio"] = [0., 0.5, 1.]
        hp_grid["feature_selector.max_features"] = list(range(50))
        hp_grid["feature_selector.significance_threshold"] = [2.]
        hp_grid["feature_selector.significance_threshold_backward"] = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        return hp_grid

class ChronoEpilogiFull(ChronoEpilogiSingle):

    def get_total_number_sets(self):
        return self.model.get_total_number_sets()
    def get_equivalence_classes_as_list(self):
        return list(map(lambda x:self.model.equivalent_variables[x],self.model.equivalent_variables))
    
    def _set_variant_parameters(self, partial_config):
        equivalence_heuristic = partial_config.get("equivalence_heuristic", "full")
        equivalence_greedy_stopping = partial_config.get("equivalence_greedy_stopping", False)
        phases = partial_config.get("phases", "FBEV")
        return equivalence_heuristic, equivalence_greedy_stopping, phases
    
    def generate_optuna_parameters(trial):
        hp = dict()
        hp["equivalence_threshold"] = trial.suggest_float("feature_selector.equivalence_threshold", 1e-5, 1e-1, log=True)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = dict()
        hp_grid["feature_selector.equivalence_threshold"] = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        return hp_grid

class ChronoEpilogiFullES(ChronoEpilogiFull):
    def _set_variant_parameters(self, partial_config):
        equivalence_heuristic = partial_config.get("equivalence_heuristic", "full")
        equivalence_greedy_stopping = partial_config.get("equivalence_greedy_stopping", True)
        phases = partial_config.get("phases", "FBEV")
        return equivalence_heuristic, equivalence_greedy_stopping, phases

class ChronoEpilogiH0(ChronoEpilogiFull):
    def _set_variant_parameters(self, partial_config):
        equivalence_heuristic = partial_config.get("equivalence_heuristic", "H0")
        equivalence_greedy_stopping = partial_config.get("equivalence_greedy_stopping", False)
        phases = partial_config.get("phases", "FBEV")
        return equivalence_heuristic, equivalence_greedy_stopping, phases

class ChronoEpilogiH0ES(ChronoEpilogiFull):
    def _set_variant_parameters(self, partial_config):
        equivalence_heuristic = partial_config.get("equivalence_heuristic", "H0")
        equivalence_greedy_stopping = partial_config.get("equivalence_greedy_stopping", True)
        phases = partial_config.get("phases", "FBEV")
        return equivalence_heuristic, equivalence_greedy_stopping, phases

class ChronoEpilogiHk(ChronoEpilogiFull):
    def _set_variant_parameters(self, partial_config):
        equivalence_heuristic = partial_config.get("equivalence_heuristic", "Hk-sym")
        equivalence_greedy_stopping = partial_config.get("equivalence_greedy_stopping", False)
        phases = partial_config.get("phases", "FBEV")
        return equivalence_heuristic, equivalence_greedy_stopping, phases
    
    def generate_optuna_parameters(trial):
        hp = super(ChronoEpilogiHk,ChronoEpilogiHk).generate_optuna_parameters(trial)
        hp["correlation_threshold"] = trial.suggest_float("feature_selector.correlation_threshold", 1e-5, 1e-1, log=True)
        hp["k"] = trial.suggest_int("feature_selector.k", 1, 10, step=1)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = super(ChronoEpilogiHk,ChronoEpilogiHk).generate_optuna_parameters_grid()
        hp_grid["feature_selector.correlation_threshold"] = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        hp_grid["feature_selector.k"] = [1,2,3,4,5,6,7,8,9,10]
        return hp_grid

class ChronoEpilogiHkES(ChronoEpilogiHk):
    def _set_variant_parameters(self, partial_config):
        equivalence_heuristic = partial_config.get("equivalence_heuristic", "Hk-sym")
        equivalence_greedy_stopping = partial_config.get("equivalence_greedy_stopping", True)
        phases = partial_config.get("phases", "FBEV")
        return equivalence_heuristic, equivalence_greedy_stopping, phases


class GroupLasso():
    def __init__(self, config, target, data):
        self.data = data
        self.target = target
        self.config = self._configuration_completion(config)

        X,y,_,groups = self._prepare_data_vectorize(groups=True)
        self.X, self.y, self.groups = X, y, groups
        self.model = group_lasso.GroupLasso(groups, **self.config, supress_warning=True)

    
    def _prepare_data_vectorize(self, groups=False):
        """
        From a pandas dataframe with time in lines and attributes in columns, 
        create a windowed version where each (variable, lag) is a column and 1<=lag<lags.
        
        Params:
            data: pd.DataFrame, the pandas dataframe of the data
            lags: int, the number of lags in the window (window size)
            groups: bool, if true, returns the groups.
        Returns:
            X: np.array, the new predictor matrix
            y: np.array, the new predicted value vector
            indexes: the original indexes of the predicted value (on which we can use pd.DataFrame(y, index=indexes))
            (optional) groups: np.array, the index of each column of data corresponding to each column of X
        """
        # used to vectorize several timesteps in a dimension 1 vector.
        y = self.data[self.target].iloc[self.lags:]
        indexes = y.index
        y = y.values
        window_X = [self.data.values[i:i+self.lags].reshape((-1,)) for i in range(len(self.data)-self.lags)]
        X = np.array(window_X)
        if groups: 
            group_names = list(range(len(self.data.columns)))*self.lags
            return X, y, indexes, group_names
        return X, y, indexes, None

    def _vector_mask_to_columns(self, mask):
        """
        Given a mask (a vector of boolean with True if a feature is selected and False otherwise),
        covering the vectorized feature space (aka, each columns at each lag up to the window size),
        extract each column for which at least one lag was selected by the FS method.
        
        Params:
            mask: 1D np.array of bool, at True if the given vectorized feature is selected, False otherwise.
            data: pd.DataFrame, the data in the original format.
        Returns:
            selected: list of str, the list of column names in the original dataframe that were selected
        """
        indexes = np.any(np.array(mask).reshape((-1,len(self.data.columns))), axis=0)
        selected = self.data.columns[indexes]
        self.selected=list(selected)
        return self.selected
    
    def fit(self, new_config):
        if new_config["lags"]!=self.lags:
            new_config = self._configuration_completion(new_config)
            self.config = new_config
            self.X,self.y,_,self.groups = self._prepare_data_vectorize(groups=True)
            self.model = group_lasso.GroupLasso(self.groups, **self.config, supress_warning=True)
        else: # warm start can be used to speedup hyperparameter search
            new_config = self._configuration_completion(new_config)
            self.config = new_config
            self.model = self.model.set_params(groups=self.groups,**self.config)

        self.model.fit(self.X,self.y)

        mask = self.model.sparsity_mask_
        selected = self._vector_mask_to_columns(mask)
        self.selected = selected
        return
    
    def get_selected_set(self):
        return self.selected
    
    def _configuration_completion(self, partial_config):
        # required parameters
        self.lags = partial_config["lags"]  
        random_state = partial_config["seed"]

        # optional parameters
        group_reg = 0.05
        l1_reg = 0.05
        n_iter = 200
        tol = 1e-5
        warm_start = True

        keys = ["random_state", "group_reg", "l1_reg", "n_iter", "tol", "warm_start"]
        default_values = [random_state, group_reg, l1_reg, n_iter, tol, warm_start]

        d = dict()
        for key, value in zip(keys, default_values):
            d[key] = partial_config.get(key, value)
        return d
    
    def generate_optuna_parameters(trial):
        hp = dict()
        hp["group_reg"] = trial.suggest_float("feature_selector.group_reg",1e-5,1e5,log=True)
        hp["l1_reg"] = trial.suggest_float("feature_selector.l1_reg",1e-5,1e5,log=True)
        hp["n_iter"] = trial.suggest_int("feature_selector.n_iter",100,500,log=True)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = dict()
        hp_grid["feature_selector.group_reg"] = [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e5]
        hp_grid["feature_selector.l1_reg"] = [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e5]
        hp_grid["feature_selector.n_iter"] = [100, 200, 500]
        return hp_grid




class BivariateGranger():
    def __init__(self, config, target, data):
        self.target = target
        self.data = data
        self.memorize_var = dict()
    
    def fit(self, new_config):
        new_config = self._configuration_completion(new_config)
        self.config = new_config
        signif = self.config["alpha_level"]
        maxlags = self.config["lags"]

        selected = []
        for column in self.data.columns:
            if column==self.target:
                continue
            model, results = self._train_var(maxlags, column)
            pvalue = results.test_causality(self.target, causing=column, signif=signif).pvalue
            if pvalue < signif:
                pvalue = results.test_causality(column, causing=self.target, signif=signif).pvalue
                if pvalue > signif:
                    selected.append(column)
        self.selected=selected
        return 
    
    def _train_var(self, maxlags, column):
        key = (maxlags, column)
        if key not in self.memorize_var:
            model = VAR(self.data[[self.target, column]])
            results = model.fit(maxlags=maxlags)
            self.memorize_var[key] = (model, results)
        return self.memorize_var[key]

    
    def get_selected_set(self):
        return self.selected    

    def _configuration_completion(self, partial_config):
        config = {"lags": partial_config["lags"],  # required parameter
                  "alpha_level": partial_config.get("alpha_level", 0.05)}  
        return config
        
    def generate_optuna_parameters(trial):
        hp = dict()
        hp["alpha_level"] = trial.suggest_float("feature_selector.alpha_level",1e-7, 1, log=True)
        return hp
    
    def generate_optuna_parameters_grid():
        hp = dict()
        hp["feature_selector.alpha_level"] = [0.0001,0.001, 0.01, 0.05,0.1]
        return hp
        


class NoSelection():
    def __init__(self, config, target, data):
        self.data = data
    def fit(self, new_config):
        return
    def get_selected_set(self):
        return list(self.data.columns)
    def generate_optuna_parameters(trial):
        hp = dict()
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = dict()
        return hp_grid





class TMRMR():
    def __init__(self, config, target, data):
        self.data = data
        self.target = target
        self._red_dict = dict()
        self._rel_dict = dict()
        self._computed_red_columns = list()  # keep track to not recompute during hyperparameter tuning
    def fit(self, new_config):
        new_config = self._configuration_completion(new_config)
        self.config = new_config
        self._fit_mrmr()
    def get_selected_set(self):
        return self.selected
    def _fit_mrmr(self):
        number_to_select = self.config["number_to_select"]
        alpha_fraction = self.config["alpha"]
        alpha_total = int(alpha_fraction * len(self.data.columns))

        selected = []
        selected_alpha = []
        # relevance computation
        relevance = []
        relevance_dict = dict()
        for column in self.data.columns:
            relevance.append( self._temporalRelevance(column) )
            relevance_dict[column] = relevance[-1]
        selected_alpha_index = np.argpartition(relevance,alpha_total)[::-1][:alpha_total]
        selected_alpha = np.array(self.data.columns)[selected_alpha_index]
        selected_alpha_relevance = np.array(relevance)[selected_alpha_index]
        # redundancy computation
        self._compute_temporalRedundancy(selected_alpha)
        redundancy_dict = dict()
        for i,column1 in enumerate(selected_alpha):
            for column2 in selected_alpha[i:]:
                redundancy_dict[(column1, column2)] = self._get_computed_temporalRedundancy(column1, column2)
                redundancy_dict[(column2, column1)] = redundancy_dict[(column1, column2)]
        # loop mrmr
        if len(selected_alpha) == 0:  # edge case
            self.selected = list()
            return
        first_inclusion = selected_alpha[np.argmax(selected_alpha_relevance)]
        selected.append(first_inclusion)
        while len(selected)<number_to_select and len(selected)<len(selected_alpha):
            set_score_max = -np.inf
            set_column_max = None
            for column in selected_alpha:
                if column in selected: continue
                set_redundancy = (1/(len(selected) + 1)**2) * np.sum([[redundancy_dict[(a,b)] for b in (selected+[column])] for a in (selected+[column])])
                set_relevance = (1/(len(selected) +1)) * np.sum([relevance_dict[a] for a in selected+[column]])
                set_score = set_relevance / set_redundancy
                if set_score > set_score_max:
                    set_score_max = set_score
                    set_column_max = column
            selected.append(set_column_max)
        self.selected = selected


    def _temporalRelevance(self, column):
        if column in self._rel_dict:
            return self._rel_dict[column]
        # average of the F-score or MI over lags
        lags = self.config["lags"]
        r = []
        for i in range(lags):
            X = self.data[[column]].values[i:len(self.data)-lags+i]
            y = self.data[self.target].values[lags:]
            f,p = sklearn.feature_selection.f_regression(X,y)
            r.append(f[0])
        r = np.mean(r)
        self._rel_dict[column] = r
        return r

    def _compute_temporalRedundancy(self, selected_alpha):
        to_compute_columns = [column for column in selected_alpha if column not in self._computed_red_columns]
        for i,column1 in enumerate(to_compute_columns):
            df1 = self.data[[column1]]
            comparison_columns = to_compute_columns[i:]+self._computed_red_columns
            distance_list = []
            step = 20
            for split in range(0,len(comparison_columns),step):  # since it crashes, fast patch to limit memory usage
                partial_distance_list = Parallel(n_jobs=-1)(delayed(dtw.dtw)(df1, self.data[[column2]], distance_only=True, step_pattern="symmetric2")
                                                    for column2 in comparison_columns[split:split+step])
                partial_distance_list = list(map(lambda x:x.distance, partial_distance_list))
                distance_list = distance_list + partial_distance_list
            for j,column2 in enumerate(comparison_columns):
                self._red_dict[(column1, column2)] = distance_list[j]
                self._red_dict[(column2, column1)] = distance_list[j]
        self._computed_red_columns = list(selected_alpha.copy())

    def _get_computed_temporalRedundancy(self, column1, column2):
        return self._red_dict[(column1, column2)]

    def _configuration_completion(self, partial_config):
        config = {"lags": partial_config["lags"],  # required parameter
                  "n_jobs": partial_config["n_jobs"],  # required parameter
                  "alpha": partial_config.get("alpha", 1.),
                  "number_to_select": partial_config.get("number_to_select", 10)}
        return config
        
    def generate_optuna_parameters(trial):
        hp = dict()
        hp["alpha"] = trial.suggest_float("feature_selector.alpha",0, 0.3, log=False)
        hp["number_to_select"] = trial.suggest_int("feature_selector.number_to_select",1, 50, log=False)
        return hp
    
    def generate_optuna_parameters_grid():
        hp = dict()
        hp["feature_selector.alpha"] = [0.1, 0.5, 0.7, 1.]
        hp["feature_selector.number_to_select"] = [5, 10, 15, 20]
        return hp


















# class CorrelationSelection():
#     def __init__(self, config, target, data):
#         self.data = data
#         self.target = target
#         self.config = self._configuration_completion(config)

#         #no need to redo this each time there is a new configuration
#         self.correlation_matrix = data.corr()
#         self.target_corr = self.correlation_matrix[self.target].abs().sort_values(ascending=False)
        
#     def fit(self, new_config):
#         self.config = self._configuration_completion(new_config)
        
#         # Identify features with low correlation to the target
#         low_corr_features = set(self.target_corr[self.target_corr < self.config["target_corr_threshold"]].index.tolist())
#         # Track pairs of highly correlated features
#         corr_matrix = self.correlation_matrix
#         highly_correlated_pairs = []

#         for i in range(len(corr_matrix.columns)):
#             if corr_matrix.columns[i] in low_corr_features:
#                 continue
#             for j in range(i):
#                 if corr_matrix.columns[j] in low_corr_features:
#                     continue
#                 if abs(corr_matrix.iloc[i, j]) > self.config["feature_corr_threshold"]:
#                     highly_correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

#         # Decide which feature to remove from each pair
#         features_to_remove = set()

#         for feature_1, feature_2 in highly_correlated_pairs:
#             # Check if either feature is already marked for removal
#             if feature_1 not in features_to_remove and feature_2 not in features_to_remove:
#                 # Decide which feature to remove (e.g., based on correlation with the target)
#                 corr_1 = abs(corr_matrix.loc[feature_1, self.target])  # Correlation of feature_1 with target
#                 corr_2 = abs(corr_matrix.loc[feature_2, self.target])  # Correlation of feature_2 with target
                
#                 # Remove the feature with the lower correlation to the target
#                 if corr_1 < corr_2:
#                     features_to_remove.add(feature_1)
#                 else:
#                     features_to_remove.add(feature_2)

#         self.selected_set = [c for c in self.data.columns if c not in features_to_remove and c not in low_corr_features]

#     def get_selected_set(self):
#         return self.selected_set
    
#     def _configuration_completion(self,partial_config):
#         d = dict()
#         d["target_corr_threshold"] = partial_config.get("target_corr_threshold", 0.15)
#         d["feature_corr_threshold"] = partial_config.get("feature_corr_threshold", 0.8)
#         return d
#     def generate_optuna_parameters(trial):
#         hp = dict()
#         hp["target_corr_threshold"] = trial.suggest_float("feature_selector.target_corr_threshold", 0., 1.)
#         hp["feature_corr_threshold"] = trial.suggest_float("feature_selector.feature_corr_threshold", 0., 1.)
#         return hp
#     def generate_optuna_parameters_grid():
#         hp_grid = dict()
#         hp_grid["feature_selector.target_corr_threshold"] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
#         hp_grid["feature_selector.feature_corr_threshold"] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
#         return hp_grid
    


def build_base_configuration(fs_name, variable_types, seq_len, seed, n_jobs=-1):
    # CE with single solution
    if fs_name == "CE-single":
        fs_constructor = ChronoEpilogiSingle
        fs_config = {"variable_types": variable_types,
                    "lags": seq_len}
    elif fs_name == "CE-single-endsize":
        fs_constructor = ChronoEpilogiSingle_EndSize
        fs_config = {"variable_types": variable_types,
                    "lags": seq_len}
    
    # CE with multiple solution
    elif fs_name == "CE-exact":
        fs_constructor = ChronoEpilogiFull
        fs_config = {"variable_types": variable_types,
                    "lags": seq_len}
    elif fs_name == "CE-exact-ES":
        fs_constructor = ChronoEpilogiFullES
        fs_config = {"variable_types": variable_types,
                    "lags": seq_len}
    elif fs_name == "CE-resid":
        fs_constructor = ChronoEpilogiH0
        fs_config = {"variable_types": variable_types,
                    "lags": seq_len}
    elif fs_name == "CE-resid-ES":
        fs_constructor = ChronoEpilogiH0ES
        fs_config = {"variable_types": variable_types,
                    "lags": seq_len}
    elif fs_name == "CE-parcorr":
        fs_constructor = ChronoEpilogiHk
        fs_config = {"variable_types": variable_types,
                    "lags": seq_len}
    elif fs_name == "CE-parcorr-ES":
        fs_constructor = ChronoEpilogiHkES
        fs_config = {"variable_types": variable_types,
                    "lags": seq_len}
        
    # Single solution baselines
    elif fs_name == "NoSelection":
        fs_constructor = NoSelection
        fs_config = dict()
    elif fs_name == "GroupLasso":
        fs_constructor = GroupLasso
        fs_config = {"lags": seq_len, "seed":seed}
    elif fs_name == "T-MRMR":
        fs_constructor = TMRMR
        fs_config = {"lags":seq_len, "n_jobs":n_jobs}
    elif fs_name == "CausalPairs":
        fs_constructor = BivariateGranger
        fs_config = {"lags":seq_len}



    return fs_constructor, fs_config