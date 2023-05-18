from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential
import json
import argparse

credential=DefaultAzureCredential()

# parse --workspace_list_json argument, with default value of ../config/workspaces.json
parser = argparse.ArgumentParser()
parser.add_argument("--workspace_list_json", type=str, default="../config/workspaces.json")
args = parser.parse_args()  

# constants
STOP_ON_ERROR = True
DEBUG = False


# function to load the ./config/workspace.json file as dictionary
def load_workspace_config():
    with open(args.workspace_list_json) as f:
        return json.load(f)

# define a function to check if workspace exists with get
def check_workspace_exists(ws_name, ml_client):
    try:
        ws = ml_client.workspaces.get(name=ws_name)
        print(f"Workspace: {ws_name} exists")
        return True
    except Exception as e:
        print(f"Workspace {ws_name} does not exist.")
        if (DEBUG):
            print(f"Error: \n{e}")
        return False

# function to create a workspace
def create_workspace(ml_client, ws):
    return ml_client.workspaces.begin_create(ws).result()

def main():

    workspace_list = load_workspace_config()
    for workspace in workspace_list:
        #print(f"Creating workspace {workspace_list[workspace]}")
        ws = Workspace(
            name=workspace,
            location=workspace_list[workspace]["region"]
        )
        # create ml client using subscription id and resource group from workspace object
        ml_client = MLClient(
            credential=credential,
            subscription_id=workspace_list[workspace]["subscription"],
            resource_group_name=workspace_list[workspace]["resource_group"]
        )
        # check if workspace exists, else create it using the create_workspace function
        if not check_workspace_exists(ws.name, ml_client):
            try:
                ws = create_workspace(ml_client, ws)
                print(f"Created workspace {ws.name} in region {ws.location}")
            except Exception as ex:
                print(f"Failed to create workspace {ws.name} in region {ws.location}: {ex}")
                if (STOP_ON_ERROR):
                    exit(1)
       
if __name__ == "__main__":
    main()


