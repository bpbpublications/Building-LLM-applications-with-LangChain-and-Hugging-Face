"""
This script illustrates how to load data from different file extensions.

Over here we have illustrated simple scenarios but based on the requirement the
format of data in txt/csv/json files can be different.

Here we have not provided sample txt/csv/json files hence pls make sure to replace
the code with your respective files location.
"""

from datasets import load_dataset
from datasets import (
    load_dataset_builder,
)  # To inspect the data before downloading it from HuggingFaceHub
from datasets import (
    get_dataset_split_names,
)  # To check how many splits available in the data from HuggingFaceHub

# ======================================================================================================================
# Load data from HuggingFaceHub

# https://huggingface.co/datasets
# ======================================================================================================================

# For Wikipedia or similar data we need to mention which data files we want to download from the list on below URL
# https://huggingface.co/datasets/wikimedia/wikipedia/tree/main
# ds_builder = load_dataset_builder(
#     "wikimedia/wikipedia", cache_dir="E:\\Repository\\Book\\data", "20231101.chy"
# )

ds_builder = load_dataset_builder(
    "rotten_tomatoes", cache_dir="E:\\Repository\\Book\\data"
)  # dataset name is rotten_tomatoes
print(ds_builder.info.description)
print(ds_builder.info.features)
print(ds_builder.info.dataset_name)
print(ds_builder.info.dataset_size)
print(ds_builder.info.download_size)

# Get split names
get_dataset_split_names("rotten_tomatoes")

# Now download the data to specific directory. .........................................................................
# cach_dir = dir where data needs to be stored
# split = Which split of the data to load.
dataset_with_split = load_dataset(
    "rotten_tomatoes", split="validation", cache_dir="E:\\Repository\\Book\\data"
)
print(dataset_with_split)
"""
Here the data has 2 columns/features.
text: contains the raw text
label: contains the label/prediction of the text

Output:
-------
Dataset({
    features: ['text', 'label'],
    num_rows: 1066
})
"""

print(dataset_with_split[4])
"""
Output:
-------
{'text': 'bielinsky is a filmmaker of impressive talent .', 'label': 1}
"""

# No split has been defined ............................................................................................
dataset_without_split = load_dataset(
    "rotten_tomatoes", cache_dir="E:\\Repository\\Book\\data"
)
print(dataset_without_split)
"""
Output:
-------
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
})
"""

print(dataset_without_split["train"][0])
"""
Output:
-------
{'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even
greater than arnold schwarzenegger , jean-claud van damme or steven segal .', 'label': 1}
"""

print(dataset_without_split["validation"][0])
"""
Output:
-------
{'text': 'compassionately explores the seemingly irreconcilable situation between conservative christian parents and
their estranged gay and lesbian children .', 'label': 1}
"""

# ======================================================================================================================
"""
Load data from TXT file from Local

In the function load_dataset
"text" means we want to load text data
data_files:: single file location or list of different files from different or same locations
data_dit:: dir which contains all the txt files
"""
# ======================================================================================================================
txt_file_path = "E:\\Repository\\Book\\data\\txt_files\\rotten_tomatoes.txt"

# Single File ..........................................................................................................
# Default split will be train
dataset_txt = load_dataset(
    "text", data_files=txt_file_path, cache_dir="E:\\Repository\\Book\\data_cache"
)
print(dataset_txt)
"""
Output:
-------
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 1066
    })
})
"""

print(dataset_txt["train"]["text"][0])
"""
Output:
-------
lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages sweetness largely without
stickiness .
"""


# Multiple Files - Provide as list .....................................................................................
# Default split will be train
# For simplicity we have taken same file path twice but here you can mention files from same folder or different folders
dataset_txt_list = load_dataset(
    "text",
    data_files=[txt_file_path, txt_file_path],
    cache_dir="E:\\Repository\\Book\\data_cache",
)

## OR ##

# In case you have all the txt files in the same folder you can mention data_dir as well.
txt_file_dir = "E:\\Repository\\Book\\data\\txt_files"
dataset_txt_list = load_dataset(
    "text", data_dir=txt_file_dir, cache_dir="E:\\Repository\\Book\\data_cache"
)

print(dataset_txt_list)
"""
Output:
-------
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 2132
    })
})
"""

print(dataset_txt_list["train"]["text"][2131])
"""
Output:
-------
enigma is well-made , but it's just too dry and too placid .
"""

# Multiple Files with Train, Test and Validation Split
# ..........................................................................................................
# For simplicity we have taken same file path thrice but here you can mention files from same folder or different
# folders

# Here in case if you have single file for each category you can mention without list as well for example,
# data_files = {"train": txt_file_path, "test": txt_file_path, "validation": txt_file_path}

dataset_txt_splits = load_dataset(
    "text",
    data_files={
        "train": [txt_file_path],
        "test": [txt_file_path],
        "validation": [txt_file_path],
    },
    cache_dir="E:\\Repository\\Book\\data_cache",
)

print(dataset_txt_splits)
"""
Output:
-------
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text'],
        num_rows: 1066
    })
    validation: Dataset({
        features: ['text'],
        num_rows: 1066
    })
})
"""

print(dataset_txt_splits["train"]["text"][1065])
print(dataset_txt_splits["test"]["text"][1065])
print(dataset_txt_splits["validation"]["text"][1065])
"""
Here output will be same for all the 3 splits i.e., train, test and validation
Because we have used the same file for train, test and validation

Output:
-------
enigma is well-made , but it's just too dry and too placid .
"""


# ======================================================================================================================
"""
Load data from CSV file from Local

Please note that
    1. the implementation of multiple files from same or different folders
    2. the implementation of train/test/validation splits    
will remain same as described above in the text file section.
Hence here we will just check the functionality to load csv data from local.

In the function load_dataset
"csv" means we want to load csv data
data_files:: single file location or list of different files from different or same locations
data_dit:: dir which contains all the csv files
"""
# ======================================================================================================================
csv_file_path = "E:\\Repository\\Book\\data\\csv_files\\rotten_tomatoes.csv"
dataset_csv = load_dataset(
    "csv", data_files=csv_file_path, cache_dir="E:\\Repository\\Book\\data_cache"
)

print(dataset_csv)
"""
Output:
-------
features: ['reviews'] ===> it is the column name of the csv file. CSV file contain single column having name 'reviews'
DatasetDict({
    train: Dataset({
        features: ['reviews'],
        num_rows: 1066
    })
})
"""

print(dataset_csv["train"][0])
"""
Output:
-------
{'reviews': 'lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages sweetness
largely without stickiness .'}
"""

# ======================================================================================================================
"""
Load data from JSON file from Local

Please note that
    1. the implementation of multiple files from same or different folders
    2. the implementation of train/test/validation splits    
will remain same as described above in the text file section.
Hence here we will just check the functionality to load json data from local.

In the function load_dataset
"json" means we want to load csv data
data_files:: single file location or list of different files from different or same locations
data_dit:: dir which contains all the json files
"""
# ======================================================================================================================
json_file_path = "E:\\Repository\\Book\\data\\json_files\\rotten_tomatoes.json"
dataset_json = load_dataset(
    "json", data_files=json_file_path, cache_dir="E:\\Repository\\Book\\data_cache"
)

print(dataset_json)
"""
Output:
-------
features: ['reviews'] ===> it is the key name of the json file. JSON file contain single key having name 'reviews'.
As we have everything under single key hence here "num_rows" parameter shows "1" only.

DatasetDict({
    train: Dataset({
        features: ['reviews'],
        num_rows: 1
    })
})
"""

print(dataset_json["train"][0])
"""
The output has been truncated.

Output:
-------
Output Truncated:

{'reviews': {'0': 'lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages
 sweetness largely without stickiness .', '1': 'consistently clever and suspenseful .', '2': 'it\'s like a " big chill "
 reunion of the baader-meinhof gang , only these guys are more harmless pranksters than political activists .',
 '3': 'the story .................................................}}
"""
