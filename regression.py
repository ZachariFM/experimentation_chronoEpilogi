import numpy as np
import pandas as pd
import importlib
from datetime import datetime
import json

from scipy import stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import xgboost as xgb

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')





class LearningModel:
    """
    A comprehensive class for preparing time-series sensor data for Remaining Useful Life (RUL) prediction.
    Handles data splitting, one-hot encoding, feature selection, scaling, and provides multiple regression models.

    The class is designed to work with a pandas DataFrame containing sensor readings over time for multiple units.
    It ensures no data leakage by splitting data at the unit level and applying transformations (OHE, scaling)
    separately to train and test sets.

    Parameters
    ----------
    target : str
        Name of the target column (RUL). Default 'RUL'.
    db : pd.DataFrame
        The input dataset.
    ep_split : dict or None
        Dictionary with keys 'train' and 'test' containing lists of unit identifiers for each split.
        If None, a new split is generated using `create_split()`.
    OHE : str or None
        Name of the column to apply one-hot encoding to. If None, no explicit OHE column is forced.
    select_features : list or None
        List of feature names to retain after preprocessing. If None, all available features are kept.
    test_size : float
        Fraction of units to use for testing. Default 0.2.
    random_state : int
        Seed for reproducibility. Default 42.
    identifier : tuple
        A tuple (id_column, time_column) where id_column uniquely identifies each unit and time_column is the
        time step or cycle number.
    reject_columns : list
        Columns to exclude from features (e.g., identifier, time, and any other columns passed via `reject`).
    scaler : StandardScaler
        Fitted scaler used for numerical features (fitted on training data only).


    """

    def __init__(self, db, identifier, reject = None, ep_split=None, select_features=None, columnToOHE=None, target='RUL', 
                 test_size=0.2, random_state=42):
        self.target = target
        self.db = db
        self.ep_split = ep_split
        self.OHE = columnToOHE
        self.select_features = select_features
        self.test_size = test_size
        self.random_state = random_state
        self.identifier, self.time = identifier[0], identifier[1]
        if reject != None:
            self.reject_columns = [self.OHE] + reject
        else:
            self.reject_columns = [self.identifier, self.time]
    
        self.scaler = StandardScaler()
    
    
    def create_split(self):
        """
            Split the database along the identifier given
        """
        idColumns = self.db[self.identifier].unique()
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size,  
                                     random_state=self.random_state)                # 
    
        train_idx, test_idx = next(splitter.split(idColumns, groups=idColumns))    # collect the id of the given based on the split
        train_id = idColumns[train_idx]
        test_id = idColumns[test_idx]
        return {'train': train_id, 'test': test_id}
    
    def prepare_data(self):

        if self.ep_split == None:
            self.ep_split = self.create_split()

       
        # we do a OHE if there is a selected columns to perform the OHE on 
        if self.OHE != None:

            # Mask for the train and test split separately to ensure no data leakage
            mask_train = self.db[self.identifier].isin(self.ep_split["train"])
            mask_test = self.db[self.identifier].isin(self.ep_split["test"])

            x_train = self.db[mask_train].drop(columns=[self.time])
            x_test = self.db[mask_test].drop(columns=[self.time])

            print("== Train and Test set size ==")
            
            print(f"Train set:{len(x_train)}")
            print(f"Test set:{len(x_test)}")
            print(f"Initial db:{len(self.db)}\n")

            # we collect the categorical columns in train and test before OHE in order to discard them after OHE to only keep those created after OHE
            categorical_columns_train = [col for col in x_train.columns.tolist() if self.db[col].dtype != np.float64 and col != self.target and col not in self.reject_columns and col != self.OHE]
            categorical_columns_test = [col for col in x_test.columns.tolist() if self.db[col].dtype != np.float64 and col != self.target and col not in self.reject_columns and col != self.OHE]

            x_train = pd.get_dummies(x_train, columns=[self.OHE])
            total_features = len(x_train.columns.tolist())

            x_test =  pd.get_dummies(x_test, columns=[self.OHE])            
            x_train = x_train.drop(columns=categorical_columns_train)
            x_test = x_test.drop(columns=categorical_columns_test)

            
            x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

            # we have to consider the case where some features are not inside the OHE train and test set
            # so we will filter on the available features
            if self.select_features:
                available_features = []
                missing_features = []
                for f in self.select_features:
                    if f in x_train.columns and f in x_test.columns:
                        available_features.append(f)
                    else:
                        missing_features.append(f)

                print(f"there is : {len(available_features)} available features over :{total_features}")
                
                available_features.append(self.target)
                
                # now we can do a selection of all available features for train and test
                x_train = x_train[available_features]
                x_test = x_test[available_features]
            
            # # Simple imputation with median
            for feature in available_features:
                train_median = x_train[feature].median()
                x_train[feature].fillna(train_median, inplace=True)
                x_test[feature].fillna(train_median, inplace=True)
            

        else:

            mask_train = self.db[self.identifier].isin(self.ep_split["train"])
            mask_test = self.db[self.identifier].isin(self.ep_split["test"])


            x_train = self.db[mask_train].drop(columns=[self.time, self.identifier])
            x_test = self.db[mask_test].drop(columns=[self.time, self.identifier])

            categorical_columns_train = [col for col in x_train.columns.tolist() if self.db[col].dtype != np.float64 and col != self.target and col not in self.reject_columns]
            categorical_columns_test = [col for col in x_test.columns.tolist() if self.db[col].dtype != np.float64 and col != self.target and col not in self.reject_columns]

            x_train = pd.get_dummies(x_train, columns=categorical_columns_train)
            x_test =  pd.get_dummies(x_test, columns=categorical_columns_test)            

            x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)


            
        numerical_columns = [col for col in x_train.columns if x_train[col].dtype == np.float64 and col != self.target and col not in self.reject_columns ]
      

        for col in numerical_columns:
            if col in x_train.columns and x_train[col].isna().any():
                train_median = x_train[col].median()
                x_train[col].fillna(train_median, inplace=True)
                if col in x_test.columns:
                    x_test[col].fillna(train_median, inplace=True)
    
        ## We OHE only the train set and test separetely to prevent data_leakage
        x_train[numerical_columns] = self.scaler.fit_transform(x_train[numerical_columns])
        x_test[numerical_columns] = self.scaler.transform(x_test[numerical_columns])

        ## We transform and fit the x_train but not the x_test for the same reason
        X_train, X_test, y_train, y_test = x_train.drop(columns=self.target), x_test.drop(columns=self.target), x_train[self.target], x_test[self.target]
        
        #X_train = X_train[self.select_columns["selected set"]]
        data = [X_train, y_train, X_test, y_test]

        return data
       
    def analyze_data_statistics(self, data, plot=True, top_n_features=10):
        y_train, y_test = data[1], data[3]
        self.analyze_target_variable(y_train, y_test, plot)

    def analyze_target_variable(self, y_train, y_test, plot=True):
        print(f"\nRUL analysis:")
        print(f"  Train mean: {y_train.mean():.2f}, Test mean: {y_test.mean():.2f} cycles")
        print(f"  Train std: {y_train.std():.2f}, Test std: {y_test.std():.2f}")
        print(f"  Mean difference: {(y_train.mean() - y_test.mean()):.2f} cycles")
        print(f"  Relative difference: {((y_train.mean() - y_test.mean()) / y_train.mean() * 100):.1f}%")

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(18, 5))
            axes[0].hist(y_train, bins=30, alpha=0.7, label='Train', color='blue', edgecolor='black')
            axes[0].hist(y_test, bins=30, alpha=0.7, label='Test', color='red', edgecolor='black')
            axes[0].set_title('RUL Distribution - Train vs Test')
            axes[0].set_xlabel('RUL')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            sns.kdeplot(y_train, ax=axes[1], label='Train', color='blue', fill=True, alpha=0.3)
            sns.kdeplot(y_test, ax=axes[1], label='Test', color='red', fill=True, alpha=0.3)
            axes[1].set_title('RUL Density - Train vs Test')
            axes[1].set_xlabel('RUL')
            axes[1].set_ylabel('Density')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def xgboost_train(self, data):
        X_train, y_train, X_test, y_test = data

        xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }
        model = xgb.XGBRegressor(**params)
        reg = model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return reg

    def random_forest_train(self, data):
        X_train, y_train, X_test, y_test = data
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        return rf

    def gradient_boosting_train(self, data):
        X_train, y_train, X_test, y_test = data
        gbm = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        gbm.fit(X_train, y_train)
        return gbm

    def linearRegression(self, data):
        reg = LinearRegression().fit(data[0], data[1])
        return reg

    def metrics(self, reg, data):
        X_train, y_train, X_test, y_test = data
        y_train_pred = reg.predict(X_train)
        R2_train = reg.score(X_train, y_train)

        try:
            y_test_pred = reg.predict(X_test)
            R2_test = reg.score(X_test, y_test)
            mape = mean_absolute_error(y_test_pred, y_test)
        except Exception as e:
            print(f"Test prediction FAILED: {e}")
            return

        n = X_train.shape[0]
        p = X_train.shape[1]
        y_mean = np.mean(y_train)
        sst = np.sum((y_train - y_mean)**2)
        ssr = np.sum((y_train_pred - y_mean)**2)
        sse = np.sum((y_train - y_train_pred)**2)

        msr = ssr / p
        mse = sse / (n - p - 1)
        f_stat = msr / mse
        f_pval = stats.f.sf(f_stat, p, n - p - 1)

        print(f"Mape: {mape}")
        print(f"R² train: {R2_train:.4f}")
        print(f"R² (computed): {(1-sse/sst):.4f}")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"F p-value: {f_pval:.4e}")

        print("\n== TEST (Real performance) ==")
        print(f"Test R²: {R2_test:.4f}\n")
    



class ARIMAModel:
    """
    A wrapper class for SARIMAX time-series modeling designed for Remaining Useful Life (RUL) prediction. 

    Parameters:
    db : The input database containing time-series data for multiple units.
        
    identifier : A list containing the column names for (id, time_step). 

    target : The name of th variable to be predicted.
        
    exog_features : A list of column names to be used as exogenous variables in the SARIMAX model.
        
    ohe_columns : Subset of exog_features that are categorical and require One-Hot Encoding.
        
    scale_exog :If True, applies StandardScaler to the numerical exogenous features.
        
    test_size : The fraction of total units to reserve for the test set if ep_split is not provided.
        
    ep_split : An explicit split provided as (train_unit_ids, test_unit_ids) if there is a need to custom splitting
        
    random_state :The seed used for the random split to ensure reproducibility.
        
    order : The (p, d, q) order of the ARIMA model, representing the Autoregressive, 
        Integrated, and Moving Average components.
        
    seasonal_order : The (P, D, Q, s) seasonal component of the model.
        
    gap_size : The number of NaN values to insert between different unit sequences. 
        
    naming : A descriptive prefix used for plot titles and identification.
    """
    def __init__(self, db, identifier, target='RUL', exog_features=None,
                 ohe_columns=None, scale_exog=False,
                 test_size=0.2, ep_split = None,  random_state=42,
                 order=(1,0,0), seasonal_order=(0,0,0,0),
                 gap_size=None, naming = None):
        self.db = db
        self.id_col, self.time_col = identifier
        self.target = target
        self.exog_features = exog_features if exog_features else []
        self.ohe_columns = ohe_columns if ohe_columns else []
        self.scale_exog = scale_exog
        self.test_size = test_size
        self.random_state = random_state
        self.order = order
        self.seasonal_order = seasonal_order

        self.ohe = None               
        self.scaler = None             
        self.ohe_feature_names_ = None

        if ep_split:
            self.train_units = ep_split[0]
            self.test_units = ep_split[1]
        else:
            self.train_units = None
            self.test_units = None            

        self.ep_split = None
        self._create_unit_split()

        #Model storage
        self.model_fit = None

        p = self.order[0]
        P = self.seasonal_order[0] if self.seasonal_order else 0
        max_lag = max(p, P)
        self.gap_size = gap_size if gap_size is not None else max_lag + 1

        self.naming = naming

    
    def _create_unit_split(self):

        if self.train_units is not None and self.test_units is not None:
               
            self.ep_split ={   
                'train': np.array(self.train_units),
                'test': np.array(self.test_units)
        }
            print(f"Train units: {len(self.ep_split['train'])} | Test units: {len(self.ep_split['test'])}")
            return
            
        else:
            units = self.db[self.id_col].unique()
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size,
                                        random_state=self.random_state)
            train_idx, test_idx = next(splitter.split(units, groups=units))
            self.ep_split = {'train': units[train_idx],'test': units[test_idx]}

        print(f"Train units: {len(self.ep_split['train'])} | Test units: {len(self.ep_split['test'])}")

    def _prepare_series(self, unit_ids, fit_preprocessor=False):

        if len(unit_ids) == 0:
            return np.array([]), None

        df_sub = self.db[self.db[self.id_col].isin(unit_ids)].copy()
        df_sub.sort_values([self.id_col, self.time_col], inplace=True)

        if self.exog_features:
        # Forward fill numerical columns per unit
            for unit in unit_ids:
                mask = df_sub[self.id_col] == unit
                # For numerical columns, forward fill
                df_sub.loc[mask, self.exog_features] = df_sub.loc[mask, self.exog_features].fillna(method='ffill')
            # After forward fill, any remaining NaNs (at the very beginning of a unit) fill with 0
                df_sub[self.exog_features] = df_sub[self.exog_features].fillna(0)

        if self.exog_features:
            # Separate numerical and categorical features
            all_exog_cols = self.exog_features
            cat_cols = [c for c in all_exog_cols if c in self.ohe_columns]
            num_cols = [c for c in all_exog_cols if c not in self.ohe_columns]

            # OHE
            if cat_cols:
                if fit_preprocessor:
                    self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    # Fit on the subset
                    cat_data = df_sub[cat_cols]
                    self.ohe.fit(cat_data)
                    ohe_array = self.ohe.transform(cat_data)
                    self.ohe_feature_names_ = self.ohe.get_feature_names_out(cat_cols)
                else:
                    if self.ohe is None:
                        raise ValueError("OneHotEncoder not fitted. Call fit() first.")
                    cat_data = df_sub[cat_cols]
                    ohe_array = self.ohe.transform(cat_data)
                # Convert to DataFrame for later processing
                ohe_df = pd.DataFrame(ohe_array, columns=self.ohe_feature_names_, index=df_sub.index)
            else:
                ohe_df = pd.DataFrame(index=df_sub.index)

            if num_cols:
                num_data = df_sub[num_cols].values
                if fit_preprocessor and self.scale_exog:
                    self.scaler = StandardScaler()
                    num_scaled = self.scaler.fit_transform(num_data)
                elif self.scale_exog:
                    if self.scaler is None:
                        raise ValueError("Scaler not fitted. Call fit() first.")
                    num_scaled = self.scaler.transform(num_data)
                else:
                    num_scaled = num_data
                num_df = pd.DataFrame(num_scaled, columns=num_cols, index=df_sub.index)
            else:
                num_df = pd.DataFrame(index=df_sub.index)

            # Combine all exogenous feaatures
            exog_df = pd.concat([num_df, ohe_df], axis=1)
            # Ensure we have a row for every original row 
            exog_df = exog_df.loc[df_sub.index] 
            exog_values = exog_df.values
        else:
            exog_values = None

        endog_list = []
        exog_list = [] if exog_values is not None else None

        for unit in unit_ids:
            unit_mask = df_sub[self.id_col] == unit
            unit_endog = df_sub.loc[unit_mask, self.target].values
            endog_list.extend(unit_endog)

            if exog_values is not None:
                unit_exog = exog_values[unit_mask.values, :]
                exog_list.extend(unit_exog)

            #Add gap
            if unit != unit_ids[-1]:
                endog_list.extend([np.nan] * self.gap_size)
                if exog_values is not None:
                    # we add rows of NaNs with the same number of columns
                    exog_list.extend([[0] * exog_values.shape[1]] * self.gap_size)

        endog = np.array(endog_list)
        exog = np.array(exog_list) if exog_values is not None else None


        return endog, exog

    def fit(self):
        train_ids = self.ep_split['train']
        endog, exog = self._prepare_series(train_ids, fit_preprocessor=True)

        print(f"Fitting model on {len(endog)} time steps (including gaps) from {len(train_ids)} units...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(endog,
                            exog=exog,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            self.model_fit = model.fit(disp=False)
        print("Model fitting completed.")
        return self

    def predict_test(self):

        if self.model_fit is None:
            raise ValueError("Model must be fitted first. Call fit().")

        test_ids = self.ep_split['test']
        all_preds = []

        for unit in test_ids:
            # Prepare the series for this single unit (no gaps, no fitting)
            endog_unit, exog_unit = self._prepare_series([unit], fit_preprocessor=False)

            if len(endog_unit) == 0:
                continue

            unit_model = SARIMAX(endog_unit, exog=exog_unit,
                                 order=self.order,
                                 seasonal_order=self.seasonal_order)
            unit_res = unit_model.filter(self.model_fit.params)

            preds = np.asarray(unit_res.fittedvalues)

            unit_data = self.db[self.db[self.id_col] == unit].sort_values(self.time_col)
            time_vals = unit_data[self.time_col].values

            for t, y_true, y_pred in zip(time_vals, endog_unit, preds):
                all_preds.append({
                    self.id_col: unit,
                    self.time_col: t,
                    'true': y_true,
                    'predicted': y_pred
                })

        return pd.DataFrame(all_preds)

    def evaluate(self, predictions_df):
        y_true = predictions_df['true']
        y_pred = predictions_df['predicted']

        mape = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"Test MAPE: {mape:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R² : {r2:.4f}")

        return {'MAPE': mape, 'RMSE': rmse, 'R2': r2}

    def plot_predictions(self, predictions_df, num_units=3):

        units = predictions_df[self.id_col].unique()[:num_units]

        for i in units:
            fig, ax = plt.subplots(figsize=(9, 4))

            data = predictions_df[predictions_df[self.id_col] == i].sort_values(self.time_col)

            ax.plot(data[self.time_col], data['true'], label='True', marker='o')
            ax.plot(data[self.time_col], data['predicted'], label='Predicted', marker='x')

            ax.set_title(f'{self.naming} {i}') 
            ax.set_xlabel(self.time_col)
            ax.set_ylabel(self.target)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()