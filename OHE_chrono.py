import pandas as pd
import pyarrow.parquet as pq
import sys
import os
import numpy as np
import random as r
import json 
sys.path.insert(0, os.path.join(os.getcwd(), 'chronoepilogi_implementation'))


sys.path.insert(0, os.path.join(os.getcwd(), 'chronoepilogi_implementation'))

# Now import the class
from ce_extensions2 import ChronoEpilogi


def OHE_chrono(db, col_to_ohe,target, target_type, lag, reduce_data):

    """Implement chronoepilogi on a dataBase with a one-hot-encoding (ohe)

    Parameters: 
        db : database on wich the the algorithm is run 
        columns : set of columns the ohe is implemented (the numericals columns are left untouched)
                  format : list
        target : column's name that are used as target for prediction
                 format : list
        target_type : continuous or ???

    Notes : 

    """

    ##Cleaning of all the row having a nan for a columns 
    ##One-Hot encoding 
    column_to_OHE = col_to_ohe

    #########################################################
    # THIS CAN BE USED TO REDUCE THE DATASET SIZE IF NEEDED
    if reduce_data == True:
        non_na_rows = db[~db[column_to_OHE[0]].isna()]
        na_rows = db[db[column_to_OHE[0]].isna()].sample(n=3000, random_state=42)  # Adjust 'n' as needed
        db_sampled = pd.concat([non_na_rows, na_rows])
        db_sampled.sort_values(by=["Date_Time", "EPISODE_ID"], inplace=True)
        db=db_sampled.copy()
    ##########################################################

    db_encoded = pd.get_dummies(db, prefix=col_to_ohe[0], columns=col_to_ohe)
    print(f"Size of the sampled database: {db_encoded.shape}")

    ##Dropping of EPISODE_ID and DateTime to not let them appear in the Markov Boundaries
    db_encoded = db_encoded.drop(columns="EPISODE_ID")

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

    ## Data summary
    
    print(f"Keeping {len(numerical_columns)+len(categorical_columns)+len(target)} columns:")
    print(f"Categorical columns: {len(categorical_columns)}")
    print(f"Numerical columns: {len(numerical_columns)}")
    print(f"Target:{target}")

    print(target)

    variable_types = {
        **{col: "numerical" for col in numerical_columns},
        **{col: "categorical" for col in categorical_columns}
    }

    fs_instance = ChronoEpilogi(
        X,
        target,
        phases="FB",
        target_type= "continuous",
        start_with_univariate_autoregressive_model=False,
        default_max_lag=lag,  
        variable_types=variable_types
    )

    fs_instance.fit()

    print("selected set:", fs_instance.selected_set)
    print("length of selected set:",len(fs_instance.selected_set),"and number of covariates in the dataset:",len(X.columns) -1)
    
    data = {
    "selected set": fs_instance.selected_set,
    "length of selected set" :len(fs_instance.selected_set),
    "and number of covariates in the dataset" :len(X.columns) -1,
}

    json_str = json.dumps(data, indent=3)
    with open(col_to_ohe[0]+"_lag_"+str(lag)+".json", "w") as f:
        f.write(json_str)
    