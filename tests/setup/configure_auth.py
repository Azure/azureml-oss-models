# pre reqs - users lot of cli tools, should switch to python sdk in the future
# az cli installed
# gh cli installed (conda install gh --channel conda-forge)
# Make you have logged in with az cli: `az login`


from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential
import json
import argparse
import sys
import subprocess

# constants
PRINT_TOKEN = True

credential=DefaultAzureCredential()

# parse --workspace_list_json argument, with default value of ../config/workspaces.json
parser = argparse.ArgumentParser()
parser.add_argument("--workspace_list_json", type=str, default="../config/workspaces.json")
# parse --service_principal_name argument, with default value of huggingface_registry_test
parser.add_argument("--service_principal_name", type=str, default="huggingface_registry_test_sp")
# parse --github_workflow_cred argument, with default value of AZ_CRED
parser.add_argument("--github_workflow_cred", type=str, default="AZ_CRED")
# prase --role_to_assign argument, with default value of Contributor
parser.add_argument("--role_to_assign", type=str, default="Contributor")


args = parser.parse_args()  



# function to load the ./config/workspace.json file as dictionary
def load_workspace_config():
    with open(args.workspace_list_json) as f:
        return json.load(f)

# define a function to check to set azure subscription
def set_azure_subscription(subscription_id):
    cmd = f"az account set --subscription {subscription_id}"
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE, shell=True).decode(sys.getfilesystemencoding())
        print (f"'{cmd}' output:\n  {output}")
    except subprocess.CalledProcessError as e:
        print('exit code: {}'.format(e.returncode))
        print('stdout: {}'.format(e.output.decode(sys.getfilesystemencoding())))
        print('stderr: {}'.format(e.stderr.decode(sys.getfilesystemencoding())))

def create_service_principal(sp_name):
    cmd = f"az ad sp create-for-rbac --name {sp_name} --sdk-auth"
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE, shell=True).decode(sys.getfilesystemencoding())
        return output
    except subprocess.CalledProcessError as e:
        print('exit code: {}'.format(e.returncode))
        print('stdout: {}'.format(e.output.decode(sys.getfilesystemencoding())))
        print('stderr: {}'.format(e.stderr.decode(sys.getfilesystemencoding())))

# saves token to the current github repo's secrets. 
# can enhance this to take in a github repo name and org: https://cli.github.com/manual/gh_secret_set 
def save_token_to_github_secret(token, secret_name):
    cmd = f"gh secret set {secret_name} -b {token}"
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE, shell=True).decode(sys.getfilesystemencoding())
        #print (f"'{cmd}' output:\n  {output}")
    except subprocess.CalledProcessError as e:
        print('exit code: {}'.format(e.returncode))
        print('stdout: {}'.format(e.output.decode(sys.getfilesystemencoding())))
        print('stderr: {}'.format(e.stderr.decode(sys.getfilesystemencoding())))

# function to get the service principal id
def get_service_principal_app_id(sp_name):
    cmd = f"az ad sp list --display-name {sp_name} --query [].appId -o tsv"
    try:
        sp_id = subprocess.check_output(cmd, stderr=subprocess.PIPE, shell=True).decode(sys.getfilesystemencoding())
        return sp_id.rstrip()
    except subprocess.CalledProcessError as e:
        print('exit code: {}'.format(e.returncode))
        print('stdout: {}'.format(e.output.decode(sys.getfilesystemencoding())))
        print('stderr: {}'.format(e.stderr.decode(sys.getfilesystemencoding())))

# function to assign role to service principal
def assign_role_to_service_principal(sp_id, subscription_id, resource_group, role):
    cmd = f"az role assignment create --assignee {sp_id} --role {role} --scope /subscriptions/{subscription_id}/resourceGroups/{resource_group}"
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE, shell=True).decode(sys.getfilesystemencoding())
        print (f"'{cmd}' output:\n  {output}")
    except subprocess.CalledProcessError as e:
        print('exit code: {}'.format(e.returncode))
        print('stdout: {}'.format(e.output.decode(sys.getfilesystemencoding())))
        print('stderr: {}'.format(e.stderr.decode(sys.getfilesystemencoding())))


def main():
# load the workspace config file
    workspace_list = load_workspace_config()

# find unique auth_scopes from the workspace_list
    auth_scopes = {}
    for workspace in workspace_list:
        auth_scopes[workspace_list[workspace]["subscription"]] = workspace_list[workspace]["resource_group"]
    
    output = create_service_principal(args.service_principal_name)
    print(f"Created service principal {args.service_principal_name}")

    if PRINT_TOKEN: 
        print(f"Token: {output}")

    # save token to github secret
    save_token_to_github_secret(output, args.github_workflow_cred)

 
    print(f"Saved token to github secret {args.github_workflow_cred}")

    #get service principal id
    appId = get_service_principal_app_id(args.service_principal_name)
    print(f"Service principal app id: {appId}")

    # for each subscription in auth_scope, assign role to service principal
    for subscription in auth_scopes:
        set_azure_subscription(subscription)
        assign_role_to_service_principal(appId, subscription, auth_scopes[subscription], args.role_to_assign)
        print(f"Assigned role {args.role_to_assign} to service principal {args.service_principal_name} in subscription {subscription} and resource group {auth_scopes[subscription]}")

if __name__ == "__main__":
    main()

