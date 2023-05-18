

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
import sys

# constants
LOG = True

# parse command line argument to specify the directory to write the workflow files to
parser = argparse.ArgumentParser()
# mode - options are file or registry
parser.add_argument("--mode", type=str, default="file")
# registry name if model is in registry
parser.add_argument("--registry_name", type=str, default="HuggingFace")
# argument to specify Github workflow directory. can write to local dir for testing
# !!! main workflow files will be overwritten if set to "../../.github/workflows" !!!
parser.add_argument("--workflow_dir", type=str, default="../../.github/workflows")
# argument to specify queue directory
parser.add_argument("--queue_dir", type=str, default="../config/queue")
# queue set name (will create a folder under queue_dir with this name)
# !!! backup files in this folder will be overwritten !!!
parser.add_argument("--test_set", type=str, default="huggingface-rerun")
# file containing list of models to test, one per line
parser.add_argument("--model_list_file", type=str, default="../config/rerun.txt")
# test_keep_looping, to keep looping through the queue after all models have been tested
parser.add_argument("--test_keep_looping", type=str, default="false")
# test_trigger_next_model, to trigger next model in queue after each model is tested
parser.add_argument("--test_trigger_next_model", type=str, default="true")
# test_sku_type, to specify sku type to use for testing
parser.add_argument("--test_sku_type", type=str, default="cpu")
# parallel_tests, to specify number of parallel tests to run per workspace. 
# this will be used to create multiple queues
parser.add_argument("--parallel_tests", type=int, default=3)
# workflow-template.yml file to use as template for generating workflow files
parser.add_argument("--workflow_template", type=str, default="../config/workflow-template-huggingface.yml")
# workspace_list file get workspace metadata
parser.add_argument("--workspace_list", type=str, default="../config/workspaces.json")
# directory to write logs
parser.add_argument("--log_dir", type=str, default="../logs")

args = parser.parse_args()
parallel_tests = int(args.parallel_tests)

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print ("::error Auth failed, DefaultAzureCredential not working: \n{e}")
    exit (1)
# Connect to the HuggingFaceHub registry
registry_ml_client = MLClient(credential, registry_name="HuggingFace")

queue = []

# move this to config file later
templates=['transformers-cpu-small', 'transformers-cpu-medium', 'transformers-cpu-large','transformers-cpu-extra-large', 'transformers-gpu-medium']


def load_workspace_config():
    with open(args.workspace_list) as f:
        return json.load(f)
    
# function to load model_list_file
def load_model_list_file():
    # if model_list_file is extention is json, load json file
    if args.model_list_file.endswith(".json"):
        with open(args.model_list_file) as f:
            return json.load(f)
    # read all other files as text files, assuming one model per line
    with open(args.model_list_file) as f:
        return f.read().splitlines()

# function to query models from registry
def get_model_containers():
    counter=0
    print (f"Getting models from registry {args.registry_name}")
    models=[]
    model_details={}
    registry_ml_client = MLClient(credential, registry_name=args.registry_name)
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

    if LOG:
        # create log_dir if it does not exist
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        # create get_model_containers directory under log_dir if it does not exist
        if not os.path.exists(f"{args.log_dir}/get_model_containers"):
            os.makedirs(f"{args.log_dir}/get_model_containers")
        # generate filename as DDMMMYYYY-HHMMSS.json
        timestamp = time.strftime("%d%b%Y-%H%M%S.json")
        # write models list to file
        with open(f"{args.log_dir}/get_model_containers/list_{timestamp}", 'w') as f:
            json.dump(models, f, indent=4)    
        with open(f"{args.log_dir}/get_model_containers/details_{timestamp}", 'w') as f:
            json.dump(model_details, f, indent=4)   
    return models

# function to assign models to queues
# assign each model from models to a thread per workspace in a round robin fashion by appending to a list called 'models' in the queue dictionary
def assign_models_to_queues(models, workspace_list):
    print (f"\nAssigning models to queues")
# get count of workspaces in workspace_list
    queue = {}
    i=0
    while i < len(models):
        for workspace in workspace_list:
            #print (f"workspace instance: {workspace}")
            for thread in range(parallel_tests):
                #print (f"thread instance: {thread}")
                if i < len(models):
                    if workspace not in queue:
                        queue[workspace] = {}
                    if thread not in queue[workspace]:
                        queue[workspace][thread] = []
                    queue[workspace][thread].append(models[i])
                    i=i+1
                    #print (f"Adding model {models[i]} at index {i} to queue {workspace}-{thread}")
                else:
                    #print (f"Reached end of models list, breaking out of loop")
                    if LOG:
                        # if assign_models_to_queues under log_dir does not exist, create it
                        if not os.path.exists(f"{args.log_dir}/assign_models_to_queues"):
                            os.makedirs(f"{args.log_dir}/assign_models_to_queues")
                        # generate filename as DDMMMYYYY-HHMMSS.json
                        timestamp = time.strftime("%d%b%Y-%H%M%S.json")
                        # write queue to file
                        with open(f"{args.log_dir}/assign_models_to_queues/{timestamp}", 'w') as f:
                            json.dump(queue, f, indent=4)
                    # validate that count of models across all queues is equal to count of models in models list
                    model_count=0
                    for workspace in queue:
                        for thread in queue[workspace]:
                            model_count=model_count+len(queue[workspace][thread])
                    if model_count != len(models):
                        print (f"Error: Model count mismatch. Expected {len(models)} but found {model_count}")
                        exit (1)
                    else:
                        print (f"Found {model_count} models across {len(queue)} queues, which is equal to count of models in models list")
                    return queue




# function to create queue files
def create_queue_files(queue, workspace_list):
    print (f"\nCreating queue files")
    # create folder queue if it does not exist
    if not os.path.exists(args.queue_dir):
        os.makedirs(args.queue_dir)
    # check if test_set folder exists
    if not os.path.exists(f"{args.queue_dir}/{args.test_set}"):
        os.makedirs(f"{args.queue_dir}/{args.test_set}")
    # delete any files in test_set folder
    os.system(f"rm -rf {args.queue_dir}/{args.test_set}/*")
    # generate queue files
    for workspace in queue:
        for thread in queue[workspace]:
            #print (f"Generating queue file {args.queue_dir}/{args.test_set}/{workspace}-{thread}.json")
            q_dict = {"queue_name": f"{workspace}-{thread}", "models": queue[workspace][thread]}
            # get the workspace from workspace_list
            q_dict["workspace"] = workspace
            q_dict["subscription"] = workspace_list[workspace]["subscription"]
            q_dict["resource_group"] = workspace_list[workspace]["resource_group"]
            q_dict["registry"] = args.registry_name
            with open(f"{args.queue_dir}/{args.test_set}/{workspace}-{thread}.json", 'w') as f:
                json.dump(q_dict, f, indent=4)

# function to create workflow files
# !!! any existing workflow files in workflow_dir will be overwritten. backup... !!!
def create_workflow_files(queue, workspace_list):
    counter=0
    print (f"Creating workflow files")
    # check if workflow_dir exists
    if not os.path.exists(args.workflow_dir):
        os.makedirs(args.workflow_dir)
    # generate workflow files
    for workspace in queue:
        for thread in queue[workspace]:
            for model in queue[workspace][thread]:
                write_single_workflow_file(model, f"{workspace}-{thread}", workspace_list[workspace]['secret_name'])
                # print progress
                counter=counter+1
                sys.stdout.write(f'{counter}\r')
                sys.stdout.flush()
    print (f"\nCreated {counter} workflow files")

# function to write a single workflow file
def write_single_workflow_file(model, q, secret_name):
    # print a single dot without a newline to show progress
    print (".", end="", flush=True)
    workflow_file=f"{args.workflow_dir}/{model}.yml"
    #print (f"Generating workflow file: {workflow_file}")
    os.system(f"cp {args.workflow_template} {workflow_file}")
    # replace <test_queue> with q
    os.system(f"sed -i 's/<test_queue>/{q}/g' {workflow_file}")
    # replace <test_sku_type> with test_sku_type in workflow_file
    os.system(f"sed -i 's/<test_sku_type>/{args.test_sku_type}/g' {workflow_file}")
    # replace <test_registry> with test_registry in workflow_file
    os.system(f"sed -i 's/<test_trigger_next_model>/{args.test_trigger_next_model}/g' {workflow_file}")
    # replace <test_keep_looping> with test_keep_looping in workflow_file
    os.system(f"sed -i 's/<test_keep_looping>/{args.test_keep_looping}/g' {workflow_file}")
    # replace <test_model_name> with model_container.name in workflow_file
    os.system(f"sed -i 's/<test_model_name>/{model}/g' {workflow_file}")
    # replace <test_set> with test_set in workflow_file
    os.system(f"sed -i 's/<test_set>/{args.test_set}/g' {workflow_file}")
    # replace <test_secret_name> 
    os.system(f"sed -i 's/<test_secret_name>/{secret_name}/g' {workflow_file}")

# main function
def main():
    # get list of models from registry
    if args.mode == "registry":
        models = get_model_containers()
    elif args.mode == "file":
        models = load_model_list_file()
    else:
        print (f"::error Invalid mode {args.mode}")
        exit (1)
    print (f"Found {len(models)} models")
    # load workspace_list_json
    workspace_list = load_workspace_config()
    print (f"Found {len(workspace_list)} workspaces")
    # assign models to queues
    queue = assign_models_to_queues(models, workspace_list)
    print (f"Created queues")
    # create queue files
    create_queue_files(queue, workspace_list)
    print (f"Created queue files")
    # create workflow files
    create_workflow_files(queue, workspace_list)
    print (f"Created workflow files")
    print (f"Summary:")
    print (f"  Models: {len(models)}")
    print (f"  Workspaces: {len(workspace_list)}")
    print (f"  Parallel tests: {parallel_tests}")
    print (f"  Total queues: {len(workspace_list)*parallel_tests}")
    print (f"  Average models per queue: {int(len(models)/(len(workspace_list)*parallel_tests))}")

        
if __name__ == "__main__":
    main()










