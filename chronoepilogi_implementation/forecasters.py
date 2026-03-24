from torch.utils.data import DataLoader
import numpy as np

#from TSLmain.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
#from TSLmain.utils.dataset import DatasetFromDataFrames
#from TSLmain.utils.tools import dotdict



import darts
import darts.models.forecasting.tft_model
import darts.models.forecasting.rnn_model
import darts.models.forecasting.nlinear
import darts.models.forecasting.dlinear
import darts.models.forecasting.xgboost
import darts.models.forecasting.linear_regression_model
import torch.nn
import pytorch_lightning.callbacks.early_stopping




#######################################
#                                     #
#           Darts wrappers            #
#                                     #
#######################################


class BaseDartsWrapper():
    """
    Base class containing unified training/predict methods.
    No hyperparameter.
    """
    constructor = None
    def __init__(self, config, target, **kwargs):
        config = self._configuration_completion(config)
        self.pred_len = config["forecasting_horizon"]
        self.config = dict([(k,config[k]) for k in config if k!="forecasting_horizon"])
        self.target = target

        self.model = self.constructor(**self.config)

    def _prepare_data(self, data, data_time):
        d = darts.TimeSeries.from_dataframe(data,value_cols=[x for x in data.columns])
        if len(data_time.columns) == 0:
            dt = darts.TimeSeries.from_times_and_values(times=data_time.index, values=np.empty((len(data_time), 0)))
        else:
            dt = darts.TimeSeries.from_dataframe(data_time,value_cols=[x for x in data_time.columns])
        return d[self.target], d[[x for x in data.columns if x!=self.target]], dt

    def _full_fit_params(self, data_train_target, data_train_past, data_train_future, data_val_target, data_val_past, data_val_future):
        "Routine in charge of choosing which covariate to include"
        config = dict()
        config["series"]=data_train_target
        if self.model.supports_past_covariates:
            config["past_covariates"]=data_train_past
        if self.model.supports_future_covariates:
            config["future_covariates"]=data_train_future
        if isinstance(self.model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel) or \
            isinstance(self.model, darts.models.forecasting.xgboost.XGBModel):
            config["val_series"] = data_val_target
            if self.model.supports_past_covariates:
                config["val_past_covariates"]=data_val_past
            if self.model.supports_future_covariates:
                config["val_future_covariates"]=data_val_future
        return config

    def fit(self, data_train, data_train_time, data_val, data_val_time):
        data_train_target, data_train_past, data_train_future = self._prepare_data(data_train, data_train_time)
        data_val_target, data_val_past, data_val_future = self._prepare_data(data_val, data_val_time)
        fit_config = self._full_fit_params(data_train_target, data_train_past, data_train_future, data_val_target, data_val_past, data_val_future)
        self.model.fit(**fit_config)
        
    def _full_predict_params(self, data_test_target, data_test_past, data_test_future):
        "Routine in charge of finding which covariate to include and add other fixed parameters"
        config = dict()
        config["series"]=data_test_target
        if self.model.supports_past_covariates:
            config["past_covariates"]=data_test_past
        if self.model.supports_future_covariates:
            config["future_covariates"]=data_test_future
        config["forecast_horizon"] = self.pred_len
        config["stride"] = 1
        config["retrain"] = False
        config["overlap_end"] = False
        config["last_points_only"] = False
        return config

    def predict(self, data_test, data_test_time):
        data_test_target, data_test_past, data_test_future = self._prepare_data(data_test, data_test_time)

        test_config = self._full_predict_params(data_test_target, data_test_past, data_test_future)
        # pred is a list of time-series, each containing a forecast.
        pred = self.model.historical_forecasts(**test_config)
        # demerit of doing the following: we lose the time index.
        y_true = np.array([data_test_target[pred[i].time_index].values() for i in range(len(pred))])
        y_pred = np.array([pred[i].values() for i in range(len(pred))])
        return y_pred, y_true
        
    def _configuration_completion(self, partial_config):
        d = dict()
        pass
    def infer_data_shape_parameters(partial_config, data, data_time):
        return partial_config
    def generate_optuna_parameters(trial):
        pass
    def generate_optuna_parameters_grid():
        pass

class TorchDartsWrapper(BaseDartsWrapper):
    """
    Class specific to models using TorchLightning.
    Hyperparameter specification concerns lightning trainer utilities
    """
    def _configuration_completion(self, partial_config):
        d = dict()
        d["batch_size"] =  partial_config.get("", 32)
        d["n_epochs"] =  partial_config.get("n_epochs", 100)
        d["random_state"] =  partial_config.get("random_state", 0)  # needs to be provided by the user
        d["nr_epochs_val_period"] = None  # must be set that way to have early stopping
        d["pl_trainer_kwargs"] = dict()
        es_instance = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
            monitor=partial_config.get("pl_trainer_kwargs.callbacks.monitor","val_loss"),
            patience=partial_config.get("pl_trainer_kwargs.callbacks.patience",3),  # needs to be tuned I think
            min_delta=partial_config.get("pl_trainer_kwargs.callbacks.min_delta",0.),
            mode=partial_config.get("pl_trainer_kwargs.callbacks.mode","min"),
        )
        d["pl_trainer_kwargs"]["callbacks"] = partial_config.get("pl_trainer_kwargs.callbacks",[])+[es_instance]
        return d
    def infer_data_shape_parameters(partial_config, data, data_time):
        return partial_config
    def generate_optuna_parameters(trial):
        hp = dict()
        pass
        hp["n_epochs"] = trial.suggest_int("forecaster.n_epochs", 10, 500, step=10, log=False)
        hp["pl_trainer_kwargs.callbacks.patience"] = trial.suggest_int("forecaster.pl_trainer_kwargs.callbacks.patience", 1, 10, step=1, log=True)
        hp["pl_trainer_kwargs.callbacks.min_delta"] = trial.suggest_float("forecaster.pl_trainer_kwargs.callbacks.min_delta", 0.0001, 0.1, log=True)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = dict()
        hp_grid["forecaster.n_epochs"] = [10, 100, 500]
        hp_grid["forecaster.pl_trainer_kwargs.callbacks.patience"] = [1, 3, 10]
        hp_grid["forecaster.pl_trainer_kwargs.callbacks.min_delta"] = [0.0001, 0.05]
        return hp_grid


class TFTDartsWrapper(TorchDartsWrapper):
    """
    Temporal Fusion Transformer. Hyperparameters define the network used.
    """
    constructor = darts.models.forecasting.tft_model.TFTModel
    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon",1)
        d["input_chunk_length"] = partial_config.get("input_chunk_length", 32)  # should be given by the user
        d["output_chunk_length"] = partial_config.get("output_chunk_length",d["forecasting_horizon"])
        d["output_chunk_shift"] =  partial_config.get("output_chunk_shift",0)
        d["hidden_size"] =  partial_config.get("hidden_size",16)
        d["lstm_layers"] =  partial_config.get("lstm_layers",1)
        d["num_attention_heads"] =  partial_config.get("num_attention_heads",4)
        d["full_attention"] =  partial_config.get("full_attention",False)
        d["feed_forward"] =  partial_config.get("feed_forward","GatedResidualNetwork")
        d["dropout"] =  partial_config.get("dropout",0.1)
        d["hidden_continuous_size"] =  partial_config.get("hidden_continuous_size",8)
        d["categorical_embedding_sizes"] =  partial_config.get("categorical_embedding_sizes",None)  # if necessary, should be given by the user
        d["add_relative_index"] =  partial_config.get("add_relative_index", False)
        d["loss_fn"] =  partial_config.get("loss_fn", torch.nn.MSELoss())  # will depend on the predicted TS type I think. Only used if likelihood is None
        d["likelihood"] =  partial_config.get("likelihood", None)
        d["norm_type"] =  partial_config.get("norm_type", 'LayerNorm')
        d["use_static_covariates"] =  partial_config.get("use_static_covariates", True)
        return d
    def generate_optuna_parameters(trial):
        hp = super(TFTDartsWrapper,TFTDartsWrapper).generate_optuna_parameters(trial)
        hp["hidden_size"] = trial.suggest_int("forecaster.hidden_size", 1, 64, step=1, log=True)
        hp["lstm_layers"] = trial.suggest_int("forecaster.lstm_layers", 1, 3, step=1, log=False)
        hp["num_attention_heads"] = trial.suggest_int("forecaster.num_attention_heads", 1, 8, step=1, log=False)
        hp["dropout"] = trial.suggest_float("forecaster.dropout", 0., 0.5, log=False)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = super(TFTDartsWrapper,TFTDartsWrapper).generate_optuna_parameters_grid()
        hp_grid["forecaster.hidden_size"] = [4, 16, 32]
        hp_grid["forecaster.lstm_layers"] = [1, 2]
        hp_grid["forecaster.num_attention_heads"] = [1, 4]
        hp_grid["forecaster.dropout"] = [0., 0.1, 0.2]
        return hp_grid

class RNNDartsWrapper(TorchDartsWrapper):
    """
    RNN network (RNN, LSTM, GRU). Hyperparameters define the network used.
    """
    constructor = darts.models.forecasting.rnn_model.RNNModel
    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon",1)
        d["input_chunk_length"] =  partial_config.get("input_chunk_length", 32)  # should be given by the user
        d["model"] =  partial_config.get("model", "LSTM")  # one of LSTM, RNN, GRU
        d["hidden_dim"] =  partial_config.get("hidden_dim", 25)
        d["n_rnn_layers"] =  partial_config.get("n_rnn_layers", 1)
        d["dropout"] =  partial_config.get("dropout", 0.1)
        d["training_length"] =  partial_config.get("training_length", d["input_chunk_length"]+d["forecasting_horizon"])
        d["loss_fn"] =  partial_config.get("loss_fn", torch.nn.MSELoss())  # will depend on the predicted TS type I think
        d["likelihood"] =  partial_config.get("likelihood", None) 
        return d
    def generate_optuna_parameters(trial):
        hp = super(RNNDartsWrapper,RNNDartsWrapper).generate_optuna_parameters(trial)
        hp["hidden_dim"] = trial.suggest_int("forecaster.hidden_dim", 1, 64, step=1, log=True)
        hp["n_rnn_layers"] = trial.suggest_int("forecaster.n_rnn_layers", 1, 3, step=1, log=False)
        hp["dropout"] = trial.suggest_float("forecaster.dropout", 0., 0.5, log=False)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = super(RNNDartsWrapper,RNNDartsWrapper).generate_optuna_parameters_grid()
        hp_grid["forecaster.hidden_dim"] = [4, 16, 32]
        hp_grid["forecaster.n_rnn_layers"] = [1, 2]
        hp_grid["forecaster.dropout"] = [0., 0.1]
        return hp_grid

class BlockRNNDartsWrapper(RNNDartsWrapper):
    constructor = darts.models.forecasting.block_rnn_model.BlockRNNModel
    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["output_chunk_length"] = partial_config.get("output_chunk_length",d["forecasting_horizon"])
        return d

class TCNDartsWrapper(TorchDartsWrapper):
    """
    TCN network. Hyperparameters define the network used.
    """
    constructor = darts.models.forecasting.tcn_model.TCNModel
    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon",1)
        d["input_chunk_length"] = partial_config.get("input_chunk_length", 32)  # should be given by the user
        d["output_chunk_length"] = partial_config.get("output_chunk_length",d["forecasting_horizon"])
        d["output_chunk_shift"] =  partial_config.get("output_chunk_shift",0)
        d["kernel_size"] =  partial_config.get("kernel_size", 3)
        d["num_filters"] =  partial_config.get("num_filters", 3)
        d["weight_norm"] = partial_config.get("weight_norm", False)
        d["dilatation_base"] = partial_config.get("dilatation_base", 2)
        d["num_layers"] = partial_config.get("num_layers", 2)
        d["dropout"] =  partial_config.get("dropout",0.2)
        d["loss_fn"] =  partial_config.get("loss_fn", torch.nn.MSELoss())  # will depend on the predicted TS type I think
        d["likelihood"] =  partial_config.get("likelihood", None) 
        return d
    def generate_optuna_parameters(trial):
        hp = super(TCNDartsWrapper,TCNDartsWrapper).generate_optuna_parameters(trial)
        hp["kernel_size"] = trial.suggest_int("forecaster.kernel_size", 3, 20, step=1, log=True)
        hp["dropout"] = trial.suggest_float("forecaster.dropout", 0., 0.5, log=False)
        hp["num_filters"] = trial.suggest_int("forecaster.num_filters",3,20, log=True)
        hp["dilatation_base"] = trial.suggest_int("forecaster.dilatation_base",2,3)
        hp["num_layers"] = trial.suggest_int("forecaster.num_layers",1,3,step=1)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = super(TCNDartsWrapper,TCNDartsWrapper).generate_optuna_parameters_grid()
        hp_grid["forecaster.kernel_size"] = [3, 5, 8, 12, 20]
        hp_grid["forecaster.dropout"] = [0., 0.1, 0.2, 0.5]
        hp_grid["forecaster.num_filters"] = [3, 8, 20]
        hp_grid["forecaster.dilatation_base"] = [2, 3]
        hp_grid["forecaster.num_layers"] = [1,2,3]
        return hp_grid
    
class NLinearDartsWrapper(TorchDartsWrapper):
    """
    NLinear network. Hyperparameters define the network used.
    """
    constructor = darts.models.forecasting.nlinear.NLinearModel
    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon",1)
        d["input_chunk_length"] = partial_config.get("input_chunk_length", 32)  # should be given by the user
        d["output_chunk_length"] = partial_config.get("output_chunk_length",d["forecasting_horizon"])
        d["output_chunk_shift"] =  partial_config.get("output_chunk_shift",0)
        d["use_static_covariates"] =  partial_config.get("use_static_covariates", True)
        d["loss_fn"] =  partial_config.get("loss_fn", torch.nn.MSELoss())  # will depend on the predicted TS type I think
        d["likelihood"] =  partial_config.get("likelihood", None) 
        return d
    def generate_optuna_parameters(trial):
        hp = super(NLinearDartsWrapper,NLinearDartsWrapper).generate_optuna_parameters(trial)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = super(NLinearDartsWrapper,NLinearDartsWrapper).generate_optuna_parameters_grid()
        return hp_grid

class DLinearDartsWrapper(NLinearDartsWrapper):
    """
    DLinear network, inheriting NLinear except for "kernel size".
    """
    constructor = darts.models.forecasting.dlinear.DLinearModel
    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["kernel_size"] =  partial_config.get("kernel_size", 25)  # should be odd number
        return d
    def generate_optuna_parameters(trial):
        hp = super(DLinearDartsWrapper,DLinearDartsWrapper).generate_optuna_parameters(trial)
        hp["kernel_size"] = trial.suggest_int("forecaster.kernel_size", 10, 200, step=1, log=True)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = super(DLinearDartsWrapper,DLinearDartsWrapper).generate_optuna_parameters_grid()
        hp_grid["forecaster.kernel_size"] = [10, 25, 50]
        return hp_grid

class XGBDartsWrapper(BaseDartsWrapper):
    constructor = darts.models.forecasting.xgboost.XGBModel
    def _configuration_completion(self, partial_config):
        d = dict()
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon",1)
        d["output_chunk_length"] = partial_config.get("output_chunk_length",d["forecasting_horizon"])
        d["output_chunk_shift"] =  partial_config.get("output_chunk_shift",0)
        d["lags"] = partial_config.get("lags", 32)  # should be given by the user
        d["lags_past_covariates"] = partial_config.get("lags_past_covariates", d["lags"])
        d["lags_future_covariates"] = partial_config.get("lags_future_covariates", (d["lags"], d["output_chunk_length"]))
        d["use_static_covariates"] =  partial_config.get("use_static_covariates", True)
        d["likelihood"] =  partial_config.get("likelihood", None) 

        d["objective"] = partial_config.get("objective", "reg:squarederror") 
        d["n_estimators"] = partial_config.get("n_estimators", 10000)
        d["max_depth"] = partial_config.get("max_depth", 6)
        d["max_leaves"] = partial_config.get("max_leaves", 0)
        d["learning_rate"] = partial_config.get("learning_rate", 0.3)
        d["verbosity"] = partial_config.get("verbosity", 0)
        d["random_state"] = partial_config.get("random_state", 0)  # should be given by the user
        d["reg_alpha"] = partial_config.get("reg_alpha", 0)
        d["reg_lambda"] = partial_config.get("reg_lambda", 1.)
        d["eval_metric"] = partial_config.get("eval_metric","rmse")
        d["early_stopping_rounds"] = partial_config.get("early_stopping_rounds", 3)
        d["device"] = partial_config.get("device", "cuda")

        return d
    def generate_optuna_parameters(trial):
        hp = dict()
        hp["max_depth"] = trial.suggest_int("forecaster.max_depth", 5, 20, step=5, log=False)
        hp["reg_alpha"] = trial.suggest_float("forecaster.reg_alpha", 0.001, 10., log=True)
        hp["reg_lambda"] = trial.suggest_float("forecaster.reg_lambda", 0.001, 10., log=True)
        hp["early_stopping_rounds"] = trial.suggest_int("forecaster.early_stopping_rounds", 1, 10, step=1, log=True)
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = dict()
        hp_grid["forecaster.max_depth"] = [5, 10]
        hp_grid["forecaster.reg_alpha"] = [0.001, 1.]
        hp_grid["forecaster.reg_lambda"] = [0.001, 1.]
        hp_grid["forecaster.early_stopping_rounds"] = [3, 10]
        return hp_grid

class LinearDartsWrapper(BaseDartsWrapper):
    constructor = darts.models.forecasting.linear_regression_model.LinearRegressionModel
    def _configuration_completion(self, partial_config):
        d = dict()
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon",1)
        d["output_chunk_length"] = partial_config.get("output_chunk_length",d["forecasting_horizon"])
        d["output_chunk_shift"] =  partial_config.get("output_chunk_shift",0)
        d["lags"] = partial_config.get("lags", 32)  # should be given by the user
        d["lags_past_covariates"] = partial_config.get("lags_past_covariates", d["lags"])
        d["lags_future_covariates"] = partial_config.get("lags_future_covariates", (d["lags"], d["output_chunk_length"]))
        d["use_static_covariates"] =  partial_config.get("use_static_covariates", True)
        d["likelihood"] =  partial_config.get("likelihood", None) 
        return d
    def generate_optuna_parameters(trial):
        hp = dict()
        return hp
    def generate_optuna_parameters_grid():
        hp_grid = dict()
        return hp_grid



def build_base_configuration(forecaster_name, pred_len, seq_len):

    if forecaster_name == "TFTDartsWrapper":
        es_constructor = TFTDartsWrapper
        es_config = {"forecasting_horizon":pred_len,
                "input_chunk_length":seq_len}

    elif forecaster_name == "RNNDartsWrapper":
        es_constructor = RNNDartsWrapper
        es_config = {"forecasting_horizon":pred_len,
                "input_chunk_length":seq_len}
        
    elif forecaster_name == "BlockRNNDartsWrapper":
        es_constructor = RNNDartsWrapper
        es_config = {"forecasting_horizon":pred_len,
                "input_chunk_length":seq_len}

    elif forecaster_name == "TCNDartsWrapper":
        es_constructor = NLinearDartsWrapper
        es_config = {"forecasting_horizon":pred_len,
                "input_chunk_length":seq_len}
        
    elif forecaster_name == "NLinearDartsWrapper":
        es_constructor = NLinearDartsWrapper
        es_config = {"forecasting_horizon":pred_len,
                "input_chunk_length":seq_len}

    elif forecaster_name == "DLinearDartsWrapper":
        es_constructor = DLinearDartsWrapper
        es_config = {"forecasting_horizon":pred_len,
                "input_chunk_length":seq_len}

    elif forecaster_name == "LinearDartsWrapper":
        es_constructor = LinearDartsWrapper
        es_config = {"forecasting_horizon":pred_len,
                    "lags":seq_len}

    elif forecaster_name == "XGBDartsWrapper":
        es_constructor = XGBDartsWrapper
        es_config = {"forecasting_horizon":pred_len,
                    "lags":seq_len}
    return es_constructor, es_config



#######################################
#                                     #
#           TLSMain wrappers          #
#                                     #
#######################################




# class BaseLongTermWrapper():
#     def __init__(self, config, target, unique_model_id: str):
#         self.config = self._configuration_completion(config)
#         self.target = target
#         self.config["target"] = self.target # dataset class needs the target for multivariate to univariate forecasting
#         self.unique_model_id = unique_model_id  # will be used to save checkpoints

#         self.training_instance = Exp_Long_Term_Forecast(self.config)
    
#     def _prepare_data_loader(self, data_set, shuffle_flag):
#         return DataLoader(
#             dataset=data_set,
#             batch_size=self.config.batch_size,
#             shuffle=shuffle_flag)

#     def fit(self, data_train, data_train_time, data_val, data_val_time):
#         self.dataset_train = DatasetFromDataFrames(data_train, data_train_time, self.config)
#         self.dataset_val = DatasetFromDataFrames(data_val, data_val_time, self.config)
#         self.data_loader_train = self._prepare_data_loader(self.dataset_train, True)
#         self.data_loader_val = self._prepare_data_loader(self.dataset_val, False)

#         self.model = self.training_instance.train(self.data_loader_train, self.data_loader_val, self.unique_model_id)

#     def predict(self, data_test, data_test_time):
#         self.dataset_test = DatasetFromDataFrames(data_test, data_test_time, self.config)
#         self.data_loader_test = self._prepare_data_loader(self.dataset_test, False)

#         # [Batch size, Predicted variates size, Predicted horizon length]
#         y_pred_batch, y_true_batch = self.training_instance.test(self.data_loader_test, self.unique_model_id)

#         return y_pred_batch, y_true_batch
    
#     def _configuration_completion(self, partial_config):
#         # for hyperparameter specification, see DatasetFromDataFrames and Exp_Long_Term_Forecast.
#         d = dotdict()
#         d["features"] = partial_config.get("features", "MS")
#         d["seq_len"] = partial_config.get("seq_len", 32)  # should be given by the user
#         d["label_len"] = partial_config.get("label_len", 0)  
#         d["pred_len"] = partial_config.get("pred_len", 1)  # should be given by the user
#         d["batch_size"] = partial_config.get("batch_size", 32)
#         d["use_gpu"] = partial_config.get("use_gpu", True)
#         d["gpu_type"] = partial_config.get("gpu_type", "cuda")
#         d["gpu"] = partial_config.get("gpu", 0)
#         d["use_multi_gpu"] = partial_config.get("use_multi_gpu", False)
#         d["devices"] = partial_config.get("devices", "0")
#         d["devices_ids"] = partial_config.get("devices_ids", [0])
#         d["learning_rate"] = partial_config.get("learning_rate", 0.0001)
#         d["lradj"] = partial_config.get("lradj", "cosine")
#         d["patience"] = partial_config.get("patience", 3)
#         d["checkpoints"] = partial_config.get("checkpoints", "./checkpoints/")
#         d["train_epochs"] = partial_config.get("train_epochs", 10)
#         d["task_name"] = partial_config.get("task_name", 'long_term_forecast')
#         return d
#     def infer_data_shape_parameters(partial_config, data, data_time):
#         d = dotdict()
#         for key in partial_config:
#             d[key] = partial_config[key]
#         return d
#     def generate_optuna_parameters(trial):
#         pass
#     def generate_optuna_parameters_grid():
#         pass


# class DLinearLongTermWrapper(BaseLongTermWrapper):
#     def _configuration_completion(self, partial_config):
#         d = super()._configuration_completion(partial_config)
#         d["model"] = partial_config.get("model", "DLinear")
#         d["individual"] = partial_config.get("individual", False)   # should be provided with domain knowledge
#         d["moving_avg"] = partial_config.get("moving_avg", 25)  # should be provided with domain knowledge
#         d["enc_in"] = partial_config.get("enc_in", None)  # has to be infered from the data
#         d["num_class"] = partial_config.get("num_class", 0)  # unused
#         return d
#     def infer_data_shape_parameters(partial_config, data, data_time):
#         d = super(DLinearLongTermWrapper,DLinearLongTermWrapper).infer_data_shape_parameters(partial_config, data, data_time)
#         d["enc_in"] = len(data.columns)
#         return d
#     def generate_optuna_parameters(trial):
#         d = super(DLinearLongTermWrapper,DLinearLongTermWrapper).generate_optuna_parameters(trial)
#         pass
#     def generate_optuna_parameters_grid():
#         d = super(DLinearLongTermWrapper,DLinearLongTermWrapper).generate_optuna_parameters_grid()
#         pass

# class TFTLongTermWrapper(BaseLongTermWrapper):
#     def _configuration_completion(self, partial_config):
#         d = super()._configuration_completion(partial_config)
#         d["model"] = partial_config.get("model", "TemporalFusionTransformer")
#         d["data"] = partial_config.get("data", "custom")
#         d["static"] = partial_config.get("static", None)  # should be provided with domain knowledge
#         d["observed"] = partial_config.get("observed", None)  # should be provided with domain knowledge
#         d["d_inp"] = partial_config.get("d_inp", None)  # has to be infered from the data
#         d["embed"] = partial_config.get("embed", "custom")
#         d["freq"] = partial_config.get("freq", None)  # Not used when custom embed
#         d["d_model"] = partial_config.get("d_model", 16)
#         d["dropout"] = partial_config.get("dropout", 0.2)
#         d["n_heads"] = partial_config.get("n_heads", 1)
#         d["c_out"] = partial_config.get("c_out", 1)  # is 1 in the MS forecasting task
#         return d
#     def infer_data_shape_parameters(partial_config, data, data_time):
#         d = super(TFTLongTermWrapper,TFTLongTermWrapper).infer_data_shape_parameters(partial_config, data, data_time)
#         d["d_inp"] = len(data_time.columns)
#         return d
#     def generate_optuna_parameters(trial):
#         d = super(TFTLongTermWrapper,TFTLongTermWrapper).generate_optuna_parameters(trial)
#         pass
#     def generate_optuna_parameters_grid():
#         d = super(TFTLongTermWrapper,TFTLongTermWrapper).generate_optuna_parameters_grid()
#         pass

