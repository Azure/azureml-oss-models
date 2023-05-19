
import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import time, sys, os
from datetime import datetime


# function to load model_list_file
def load_model_list_file(model_list_file):
    # if model_list_file is extension is json, load json file
    if model_list_file.endswith(".json"):
        with open(model_list_file) as f:
            return json.load(f)
    # read all other files as text files, assuming one model per line
    with open(model_list_file) as f:
        return f.read().splitlines()

# function to query models from registry
def get_model_containers(registry_name, templates):
    counter=0
    print (f"Getting models from registry {registry_name}")
    models=[]
    model_details={}

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)
    registry_ml_client = MLClient(credential, registry_name=registry_name)
    model_containers=registry_ml_client.models.list()
    
    for model_container in model_containers:
        # if model_container.name is in templates, skip
        if model_container.name in templates:
            continue

        # bug - registry_ml_client.models.list() is not supposed to return archived models
        # workaround to check if model is archived - get all versions and check if count is 0 
        model_versions=registry_ml_client.models.list(name=model_container.name)
        model_version_count=0
        # can't just check len(model_versions) because it is a iterator
        latest_model=None
        for model in model_versions:
            model_version_count = model_version_count + 1
            latest_model=model
        if model_version_count == 0:
            continue
        models.append(model_container.name)
        model_details[model_container.name] = latest_model
        # print progress
        counter=counter+1
        sys.stdout.write(f'{counter}\r')
        sys.stdout.flush()
    print (f"\nFound {counter} models in registry")

    # dump models to ../logs/get_model_containers/{registry_name} with filename as DDMMMYYYY-HHMMSS.json
    if not os.path.exists(f"../logs/get_model_containers/{registry_name}"):
        os.makedirs(f"../logs/get_model_containers/{registry_name}")
    with open(f"../logs/get_model_containers/{registry_name}/{datetime.now().strftime('%d%b%Y-%H%M%S')}.json", "w") as f:
        json.dump(models, f, indent=4)

    return models

