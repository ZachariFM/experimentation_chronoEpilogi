import pandas as pd
import pyarrow.parquet as pq
import sys
import os
import numpy as np
import random as r
import json 
sys.path.insert(0, os.path.join(os.getcwd(), 'chronoepilogi_implementation'))


from ce_extensions2 import ChronoEpilogi


def OHE_chrono(db, col_to_ohe,target, lag, identifier, keep_dummies = False):

    """Implement chronoepilogi on a dataBase with a one-hot-encoding (ohe)

    Parameters: 
        db : database on wich the the algorithm is run 

        col_to_ohe : set of columns upon which the ohe is implemented (the numericals columns are left untouched)

        target : column's name that are used as target for prediction
        
        lag: Value of the lag chosen

        identifier : A list containing the column names for (id, time_step)

    Notes : 

    """

    ##Cleaning of all the row having a nan for a columns 
    ##One-Hot encoding 
    

    original_columns = db.columns.tolist()
    print(original_columns)

    db_encoded = pd.get_dummies(db, prefix=col_to_ohe[0], columns=col_to_ohe)
    print(f"Size of the sampled database: {db_encoded.shape}")

    ##Dropping of EPISODE_ID and DateTime to not let them appear in the Markov Boundaries
    db_encoded = db_encoded.drop(columns=identifier)

    column_names = db_encoded.columns.tolist()

    ## we take all the columns that are of type float and the categorical one we want for the model
    numerical_columns = [col for col in column_names if db_encoded[col].dtype == np.float64 and col != target]
    categorical_columns = [col for col in column_names if col.startswith(col_to_ohe[0]+"_")]

    ## and add them to our features database
    features_db = numerical_columns + categorical_columns + [target]


    #X = pd.concat([features_db, db_encoded[target]], axis=1)
    X = db_encoded[features_db].copy()

    ## to prevent dtype issue, we set the type to a suitable one, just in case it hasnt done it by itself

    X[numerical_columns] = X[numerical_columns].astype(np.float64)
    X[categorical_columns] = X[categorical_columns].astype(np.int64)
    
    ## Cleaning the NaN and the +- inf
    for col in numerical_columns:
        X[col] = X[col].fillna(X[col].mean())


    variable_types = {
        **{col: "numerical" for col in numerical_columns},
        **{col: "categorical" for col in categorical_columns}
    }

    # Calling of CE
    fs_instance = ChronoEpilogi(
        X,
        target,
        phases="FBEV",
        target_type= "continuous",
        start_with_univariate_autoregressive_model=False,
        default_max_lag=lag,  
        variable_types=variable_types
    )

    fs_instance.fit()

    data = {
        "selected set": fs_instance.selected_set,
        "length of selected set" :len(fs_instance.selected_set),
        "and number of covariates in the dataset" :len(X.columns) -1,
    }

    json_str = json.dumps(data, indent=3)
    filename = f"results/{col_to_ohe[0]+"_lag_"+str(lag)}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        f.write(json_str)
    print(f'results saved here {filename}')