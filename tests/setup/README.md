### Run the below scripts in order to setup generate test files and check in the repo for running tests

> ⚠️ Scripts in this folder need to be run manually


#### [configure_auth.py](./configure_auth.py)
See: https://github.com/Azure/actions-workflow-samples/blob/master/assets/create-secrets-for-GitHub-workflows.md
Pre-req: Install Az cli, Gh cli, and login to both
Setup auth between the github repo workflow and resource groups that have workspaces. Steps
* Create service principle (SP)
* Get SP token
* Save SP token to Github secrets
* Assign SP to each subscription + resource group

#### [create_workspaces.py](./create_workspaces.py)
Pre-req: [configure_auth.py](./configure_auth.py) 
Create workspaces listed in [workspaces.json](../config/workspaces.json). Skips if workspace already exists.
> Has some bugs, use [configure-auth.sh](./configure-auth.sh)

#### [create_queue.py](./create_queue.py)
Pre-req:
* [workspaces.json](../config/workspaces.json)
This is the most important script that generates the test setup as follows:
1. Assigns models to queues. A queue is defined as list of models that are tested in sequence. You can have multiple queues if you want to test in parallel. 
2. Assigns queues to workspaces for testing. One workspace can have one or more queues. If you assign multiple skus to a workspace, endpoints will be created in parallel in that workspace, so make sure you have quota. 
3. Generates one github workflow yaml file for each model. the yaml file is hard coded with the queue it has to run on. the queue json has details such as workspace, sub, rg, etc. 

Parameter|Description
--|--
mode|options are `file` or `registry`. `file` creates test queues from local file `model_list_file`. `registry` pulls models from `registry_name`
model_list_file|file name for `file` option in `mode`
registry_name|AzureML registry that has models to test
workflow_dir|location where github workflow yaml files must be generated. Default is `../../.github/workflows`, so be careful about overwriting the original workflows. 
queue_dir|root dir where queue files must be written. [deploy_huggingface_models.py](../src/deploy_huggingface_models.py) which is the test driver expects these to be in [../config/queue/](../config/queue/), so cannot change.
test_set|dir under `queue_dir`. Supports creating various test sets such as `all`, `smoke`, etc. But a model workflow can use only one `test_set` at a time.
test_keep_looping|to keep looping through the queue after all models have been tested
test_trigger_next_model|to trigger next model in queue after each model is tested
test_sku_type|cpu or gpu
parallel_tests| to specify number of parallel tests to run per workspace. will create multiple queues per workspace if greater than 1. set value depending on quota in workspace.
workflow_template|the most important file that ties together everything. each model has a github workflow file that is generated using this template.
workspace_list|list of workspaces to use for testing, default: [workspaces.json](../config/workspaces.json)
log_dir|dir to cache models fetched from registry as it takes several minutes to get 1000s of models. the logged file can then be passed as input to `model_list_file` when using `mode` as `file`.

#### [create_badge.py](./create_badge.py)
light weight script to generate markdown file with model workflow status badges. Currently only supports models as a local file, need to add support for pulling from registry.



#### Running tests
* Run individual queue: To kick of a queue, you need to find the first model in a queue and start the workflow for that model. You can do this with gh cli: `gh workflow run <workflow-name>`. Or you can check in a workflow file that automates this. 
* Run all queues: `gh workflow run TRIGGER_TESTS`. See [TRIGGER_TESTS.yml](../../.github/workflows/TRIGGER_TESTS.yml)

#### Note on scaling
* Quota is defined per region per subscription. You can browse quota in AzureML studio global UI. The current infra has about 100 cores per region per subscription. As such, we are creating 1 workspace per region. Since a subscription can have at max 10 regions, we are using 3 subscriptions * 10 workspaces per subscription in different regions = 30 test workspaces. Each workspace runs 3 queues in parallel. As such the through put is about 90 models in parallel. So if it takes 30min to test a model, you can test 90 * 2 = 180 models per hour or 180 * 24 = ~4000 models a day. 