import pandas as pd
from sklearn.model_selection import train_test_split

def data_splitting(num_splits: int, datasetPath: str, sub_data_root, testsize):
    

    # Load the dataset from the CSV file
    dataset = pd.read_csv(datasetPath)
    
    global_trainData, global_testData = train_test_split(dataset, test_size=testsize)
    global_testData.to_csv(f'{sub_data_root}/global_test_data.csv', index=False)

    # Specify the number of sub-datasets
    num_subsets = 5

    # Calculate the number of samples in each subset
    subset_size = len(global_trainData) // num_subsets

    # Split the dataset into sub-datasets based on sample indices
    for i in range(num_subsets):
        start_idx = i * subset_size
        end_idx = start_idx + subset_size
        subset = global_trainData.iloc[start_idx:end_idx]
        subset.to_csv(f'{sub_data_root}/subset_{i+1}.csv', index=False)

    # Handle the remaining samples (if any)
    remaining_samples = global_trainData.iloc[end_idx:]
    if not remaining_samples.empty:
        remaining_samples.to_csv(f'subset_{num_subsets+1}.csv', index=False)