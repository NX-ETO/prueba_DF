from teradataml import *
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import numpy as np

def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)

    print("Starting training...")

    model = DecisionForest(data=train_df,
                            input_columns = feature_names, 
                            response_column = target_name, 
                            max_depth = 12, 
                            num_trees = 4, 
                            min_node_size = 1, 
                            mtry = 3, 
                            mtry_seed = 1, 
                            seed = 1, 
                            tree_type = 'REGRESSION')
    
    model.result.to_sql(f"model_${context.model_version}", if_exists="replace")    
    print("Saved trained model")

    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        context=context
    )
