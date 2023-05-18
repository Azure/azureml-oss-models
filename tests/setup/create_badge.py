

from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
)
import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)
import json
import os
import argparse

# argument, markdown file name
parser = argparse.ArgumentParser()
parser.add_argument("--markdown_file", type=str, default="../../dashboard/HuggingFace/README.md")
# argument, models list json, logged by create_queue.py
parser.add_argument("--models_list_json", type=str, default="../logs/get_model_containers/17May2023-231031.json")
args = parser.parse_args()

templates=['transformers-cpu-small', 'transformers-cpu-medium', 'transformers-cpu-large','transformers-cpu-extra-large', 'transformers-gpu-medium']

# read models_list_json file into a list
with open(args.models_list_json) as f:
    model_containers = json.load(f)

lines=[]
i=0  
# Getting latest model version from registry is not working, so get all versions and find latest
for model in model_containers:
    lines.append(f"{model}|[![{model}](https://github.com/Azure/azureml-oss-models/actions/workflows/{model}.yml/badge.svg)](https://github.com/Azure/azureml-oss-models/actions/workflows/{model}.yml)")
    i= i + 1

# write to markdown file
with open(args.markdown_file, 'w') as f:
    f.write(f"### Total models: {i}\n")
    f.write("|Model|Status|\n")
    f.write("|-----|------|\n")   
    for line in lines:
        f.write(line + "\n")



