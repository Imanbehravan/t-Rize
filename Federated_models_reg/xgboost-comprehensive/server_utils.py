from typing import Dict, List, Optional
from logging import INFO
import xgboost as xgb
from flwr.common.logger import log
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from utils import BST_PARAMS
from sklearn.metrics import r2_score
import pickle



def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    print('--------------eval metrics server------------')
    print(eval_metrics)
    mse_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    
    r2_aggregated = (
        sum([metrics["r2-client"] * num for num, metrics in eval_metrics]) / total_num
    )
    # mse_aggregated = (
    #     sum([metrics["MSE"] * num for num, metrics in eval_metrics]) / total_num
    # )
    metrics_aggregated = {"mse": mse_aggregated, "r2": r2_aggregated}
    return metrics_aggregated


def get_evaluate_fn(test_data):
    """Return a function for centralised evaluation."""

    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        # If at the first round, skip the evaluation
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=BST_PARAMS)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            # Run evaluation
            eval_results = bst.eval_set(
                evals=[(test_data, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            print('-------eval_results-------')
            print(eval_results)
            auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            log(INFO, f"AUC = {auc} at round {server_round}")
            
            y_true = test_data.get_label()
            y_pred = bst.predict(test_data)

            # Calculate R-squared score
            r2 = r2_score(y_true, y_pred)
            
            print('-------- r2 score of global model----')
            print(r2)
            
            model_filename = f"/home/iman/projects/kara/Projects/T-Rize/Federated_models_reg/xgboost-comprehensive/global_model/global_model_round_{server_round}.pkl"
            with open(model_filename, "wb") as file:
                pickle.dump(bst, file)
            log(INFO, f"Global model saved as {model_filename}")

            return 0, {"mse": auc, 'r2': r2}

    return evaluate_fn


class CyclicClientManager(SimpleClientManager):
    """Provides a cyclic client selection rule."""

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # Return all available clients
        return [self.clients[cid] for cid in available_cids]
