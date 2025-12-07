from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List
import pandas as pd
import datasets
import subprocess
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import random

# ======================================================================
# GLOBAL CONFIGURATION
# ======================================================================

# Define version and metadata for the corpus
_VERSION = "1.0.0"
_DESCRIPTION = "Time Series Corpus containing multiple univariate and multivariate datasets"
_CITATION = "Citations for respective datasets from original source"

# Define the base directory for dataset downloads
BASE_DIR = Path("./KaggleData")
BASE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the base directory exists

# ======================================================================
# CONFIGURATION LOADER
# ======================================================================

def load_datasets_config():

    """
    Loads dataset configuration from a CSV file hosted on Hugging Face Hub.
    
    Returns:
        dict: Nested dictionary containing all dataset configurations,
              structured as {dataset_name: [dataset_config_entries]}
    """

    # Download and read config file from Hugging Face
    csv_file_path = hf_hub_download(repo_id="ddrg/kaggle-time-series-datasets", filename="time-series-datasets.csv", repo_type="dataset")
    config_df = pd.read_csv(csv_file_path, delimiter=';')
    config_dict = {}

    # Process each row in the configuration DataFrame
    for _, row in config_df.iterrows():
        # Create the file path within the BASE_DIR
        file_path = BASE_DIR / Path(row['file_name']).name  # Ensure all files are saved to BASE_DIR

        data_columns = [col.strip() for col in row['data_column'].split(',')] if ',' in row['data_column'] else row['data_column'].strip()
        multivariate = str(row['multivariate']).strip().upper() == 'TRUE'
        DataPoints = str(row['DataPoints']).strip()

        # Create dataset entry dictionary
        dataset_entry = {
            "datasetID": row['datasetID'].strip(),
            "file_name": str(file_path),
            "date_column": row['date_column'].strip(),
            "data_column": data_columns,
            "multivariate": multivariate,
            "multivariate": multivariate,
            "variance": row['variance'].strip(),
            "domain": row['Tags'].strip(),
            "DataPoints": DataPoints
        }

        dataset_name = row['name'].strip()

        # Instead of overwriting, store multiple datasets in a list
        if dataset_name in config_dict:
            config_dict[dataset_name].append(dataset_entry)
        else:
            config_dict[dataset_name] = [dataset_entry]
            
    return config_dict

# ======================================================================
# DATASET CONFIGURATION CLASS
# ======================================================================

@dataclass
class TimeSeriesDatasetConfig(datasets.BuilderConfig):
    """
    Custom configuration class extending Hugging Face's BuilderConfig.
    Stores the loaded dataset configurations.
    """
    datasets_config: Optional[Dict[str, Dict[str, Any]]] = None

# ======================================================================
# MAIN DATASET BUILDER CLASS
# ======================================================================

class TimeSeriesDataset(datasets.GeneratorBasedBuilder):
    """
    Main dataset builder class that handles:
    - Downloading datasets from Kaggle
    - Processing the data
    - Generating train/test splits
    - Formatting the final dataset structure
    """
    VERSION = datasets.Version(_VERSION)

    # Define dataset configuration for Hugging Face
    BUILDER_CONFIGS = [
        TimeSeriesDatasetConfig(
            name="TIME_SERIES",
            version=datasets.Version(_VERSION),
            description="Multiple univariate and multivariate datasets",
            datasets_config=load_datasets_config()
        )
    ]

    def _info(self):
        """
        Defines the dataset schema and metadata.
        
        Returns:
            datasets.DatasetInfo: Contains feature definitions and metadata
        """
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features({
                "name": datasets.Value("string"),
                "date": datasets.Value("string"),
                "value": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),  # List of lists of floats for multivariate
                "variance": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "DataPoints": datasets.Value("string"),
            }),
            version=self.VERSION
        )

    def _split_generators(self, dl_manager):
        """
        Downloads datasets and creates train/test splits.
        
        Returns:
            List[datasets.SplitGenerator]: Split generators for train and test sets
        """
        downloaded_files = {}
        dataset_list = list(self.config.datasets_config.items())  # Convert dict to list for tqdm
        
        blacklist = [] # DS where Download failed (to not try over and over again)
        with tqdm(total=len(dataset_list), desc="Downloading datasets", unit="dataset") as pbar:
            for dataset_name, dataset_entries in dataset_list: 
                for dataset_info in dataset_entries:  # Iterate over the list
                    dest = Path(dataset_info["file_name"])  # Access the file_name
                    dest.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
                    key = f"{dataset_name}|{dataset_info['file_name']}"  # Use | as delimiter

                    if dest.with_suffix('.csv').exists():
                            downloaded_files[key] = str(dest.with_suffix('.csv'))  
                            print(f"Already downloaded {dataset_name} to {dest.with_suffix('.csv')}.")
                    elif dataset_info['datasetID'] in blacklist:
                        print(f"Skipping {dataset_info['datasetID']} as it is in the blacklist.")
                    else:
                        # Download the dataset using Kaggle API 
                        kaggle_command = f"kaggle datasets download -d {dataset_info['datasetID']} -p {dest.parent} --force --unzip"
                        try:
                            result = subprocess.run(kaggle_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            print(result.stdout.decode())  # Print the output of the download command
        
                            # Check if CSV exists directly; if not, skip
                            if dest.with_suffix('.csv').exists():
                                #key = f"{dataset_name}|{dataset_info['file_name']}"  # Use | as delimiter
                                downloaded_files[key] = str(dest.with_suffix('.csv'))  
                                print(f"Successfully downloaded {dataset_name} to {dest.with_suffix('.csv')}.")
                            else:
                                print(f"No CSV found for {dataset_name} at expected location: {dest.with_suffix('.csv')}")
                                continue  # Skip if no CSV file found
        
                        except subprocess.CalledProcessError as e:
                            blacklist.append(dataset_info['datasetID'])  # Add to blacklist if download failed
                            print(f"Failed to download {dataset_name}: {e}\n{e.stderr.decode()}")
                            continue  # Skip to the next dataset if download fails
    
                    pbar.update(1)  # Update progress bar after each dataset
    
        print("Blacklisted datasets due to download issues:\n", '\n'.join(blacklist))
        # Split the downloaded files into train and test sets
        filepaths = list(downloaded_files.items())
        random.Random(42).shuffle(filepaths)
        train_size = int(0.8 * len(filepaths))  # 80% for training, 20% for testing
        train_files = dict(filepaths[:train_size])
        test_files = dict(filepaths[train_size:])
    
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": train_files}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepaths": test_files}
            )
        ]

    def _generate_examples(self, filepaths):
        """
        Processes downloaded files into the final dataset format.
            
        Yields:
            Tuple[int, dict]: Index and processed dataset example
        """
        all_datasets = []
    
        for key, filepath in filepaths.items():  # Use the full key with file_name
            print(f"Processing key: {key}")
            try:
                dataset_name, file_name = key.split("|", 1)  # Use | as delimiter
                print(f"Extracted dataset_name: {dataset_name}, file_name: {file_name}")
            except ValueError:
                print(f"Invalid key format: {key}. Skipping.")
                continue
    
            dataset_info_list = self.config.datasets_config.get(dataset_name, [])  # Get the list of dataset entries
            
            # Find the correct dataset entry by file_name
            dataset_info = next((d for d in dataset_info_list if d["file_name"] == file_name), None)
            if dataset_info is None:
                print(f"Skipping {file_name}: Not found in config.")
                continue
    
            try:
                if Path(filepath).exists():
                    df = pd.read_csv(filepath, on_bad_lines='skip')
                else:
                    print(f"File {filepath} does not exist.")
                    continue
    
                date_col = dataset_info["date_column"]
    
                # Handle date parsing correctly
                if date_col not in df.columns:
                    print(f"Specified date column '{date_col}' not found in the dataset {dataset_name}. Skipping.")
                    continue
    
                # Convert date column to string format
                if df[date_col].dtype in ['int64', 'float64']:  # If column contains years (e.g., 2020)
                    df[date_col] = df[date_col].astype(int).astype(str) + "-01-01 00:00:00"
                else:
                    # Convert date column to string format
                    if df[date_col].dtype in ['int64', 'float64']:  # If column contains only years (e.g., 2020)
                        df[date_col] = df[date_col].astype(int).astype(str) + "-01-01 00:00:00"  # Convert to "01-01-YYYY 00:00:00"
                    else:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  # Convert to datetime
                    
                        # If the date is missing or only contains a year, ensure it's in the correct format
                        df[date_col] = df[date_col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else "0000-01-01 00:00:00")

    
                dates = df[date_col].tolist()
                
                data_columns = dataset_info["data_column"]
                domain = dataset_info["domain"]
                DataPoints = dataset_info["DataPoints"]
                variance = dataset_info["variance"]
                values = []
    
                if isinstance(data_columns, list):
                    for col in data_columns:
                        if col in df.columns:
                            values.append(df[col].astype(float).tolist())  # Ensure values are floats
                        else:
                            print(f"Specified data column '{col}' not found in the dataset {dataset_name}. Skipping.")
                            continue
                else:
                    if data_columns in df.columns:
                        values = [df[data_columns].astype(float).tolist()]  # Wrap single column in a list
                    else:
                        print(f"Specified data column '{data_columns}' not found in the dataset {dataset_name}. Skipping.")
                        continue
    
                # Store the dataset information in the desired format
                all_datasets.append({
                    "name": dataset_name,
                    "date": dates,  
                    "value": values,  
                    "variance": variance,
                    "domain": domain,
                    "DataPoints": DataPoints,
                })
    
            except Exception as e:
                print(f"Error processing {dataset_name} ({filepath}): {e}")
                continue
    
        # Yield all datasets
        for idx, data in enumerate(all_datasets):
            yield idx, data
