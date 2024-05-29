import warnings
from logging import INFO

import flwr as fl
from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
import pandas as pd
from sklearn.model_selection import train_test_split
from dataPreparation import data_splitting


from utils import server_args_parser
from server_utils import (
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    get_evaluate_fn,
    CyclicClientManager,
)
from dataset import resplit, transform_dataset_to_dmatrix


warnings.filterwarnings("ignore", category=UserWarning)


# Parse arguments for experimental settings
args = server_args_parser()
train_method = args.train_method
pool_size = args.pool_size
num_rounds = args.num_rounds
num_clients_per_round = args.num_clients_per_round
num_evaluate_clients = args.num_evaluate_clients
centralised_eval = args.centralised_eval

# Load centralised test set
if centralised_eval:
    # fds = FederatedDataset(
    #     dataset="jxie/higgs", partitioners={"train": 20}, resplitter=resplit
    # )
    
    housing_dataset = pd.read_csv('/home/iman/projects/kara/Projects/T-Rise/xgboost-comprehensive/global_test_data.csv')

    unique_values = housing_dataset['ocean_proximity'].unique()
    value_to_number = {value: idx for idx, value in enumerate(unique_values)}

    # Replace unique values with numbers
    housing_dataset['ocean_proximity'] = housing_dataset['ocean_proximity'].map(value_to_number)

    # Ensure all columns are numeric and drop rows with missing values
    housing_dataset = housing_dataset.apply(pd.to_numeric, errors='coerce').dropna()
    # X = housing_dataset.drop(columns='median_house_value')
    # y = housing_dataset['median_house_value']
    # X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(X, y, test_size = 0.15,
    #                                                                                 random_state = 42)
    
    
    log(INFO, "Loading centralised test set...")
    # test_set = fds.load_split("test")
    # test_set.set_format("numpy")
    test_set = dict()
    test_set['inputs'] = housing_dataset.drop(columns=['median_house_value'])
    test_set['label'] = housing_dataset['median_house_value']
    test_dmatrix = transform_dataset_to_dmatrix(test_set)
    
    # log(INFO, 'test set: ')
    # log(INFO, test_dmatrix.head())


# Define strategy
if train_method == "bagging":
    # Bagging training
    strategy = FedXgbBagging(
        evaluate_function=get_evaluate_fn(test_dmatrix) if centralised_eval else None,
        fraction_fit=(float(num_clients_per_round) / pool_size),
        min_fit_clients=num_clients_per_round,
        min_available_clients=pool_size,
        min_evaluate_clients=num_evaluate_clients if not centralised_eval else 0,
        fraction_evaluate=1.0 if not centralised_eval else 0.0,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=(
            evaluate_metrics_aggregation if not centralised_eval else None
        ),
    )
else:
    # Cyclic training
    strategy = FedXgbCyclic(
        fraction_fit=1.0,
        min_available_clients=pool_size,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
    )

dataset = '/home/iman/projects/kara/Projects/T-Rise/xgboost-comprehensive/housing.csv'
data_root_path = '/home/iman/projects/kara/Projects/T-Rise/xgboost-comprehensive/'
data_splitting(pool_size, dataset, data_root_path, 0.3)
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_manager=CyclicClientManager() if train_method == "cyclic" else None,
)
