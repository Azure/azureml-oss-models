from azure.ai.ml import MLClient, UserIdentityConfiguration
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)
from azure.ai.ml.dsl import pipeline
try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    print("workspace_name : ", queue.workspace)
    try:
        workspace_ml_client = MLClient.from_config(credential=credential)
    except:
        workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
import_model = ml_client_registry.components.get(name="import_model_oss_test", label="latest")
test_model_name = os.environ.get('test_model_name')
def get_task(self) -> str:
        """ This method will read the huggin face api url data in a dataframe. Then it will findout 
        the model which is of transformer type . Then it will find that particular model and its task

        Returns:
            str: task name
        """
        # response = urlopen(URL)
        # # Load all the data with the help of json
        # data_json = json.loads(response.read())
        # # Convert it into dataframe and mention the specific column
        # df = pd.DataFrame(data_json, columns=COLUMNS_TO_READ)
        # # Find the data with the model which will be having trasnfomer tag
        # df = df[df.tags.apply(lambda x: STRING_TO_CHECK in x)]
        # # Find the data with that particular name
        # required_data = df[df.modelId.apply(lambda x: x == self.model_name)]
        # # Get the task
        # required_data = required_data["pipeline_tag"].to_string()
        # pattern = r'[0-9\s+]'
        # final_data = re.sub(pattern, '', required_data)
        # return final_data

        hf_api = HfApi()
        # Get all the1 models in the list
        models = hf_api.list_models(
            full=True, sort='lastModified', direction=-1)
        # Unpack all values from the generator object
        required_data = [i for i in models]

        daata_dict = {}
        # Loop through the list
        for data in required_data:
            # Loop through all the column present in the list
            for key in data.__dict__.keys():
                if key in LIST_OF_COLUMNS:
                    # Check the dictionary already contains a value for that particular column
                    if daata_dict.get(key) is None:
                        # If the column and its value is not present then insert column and an empty list pair to the dictionary
                        daata_dict[key] = []
                    # Get the value for that particular column
                    values = daata_dict.get(key)
                    if key == 'tags':
                        # If its tag column extract value if it is nonne then bydefault return a list with string Empty
                        values.append(data.__dict__.get(key, ["Empty"]))
                    else:
                        values.append(data.__dict__.get(key, "Empty"))
                    daata_dict[key] = values
        # Convert dictionary to the dataframe
        df = pd.DataFrame(daata_dict)
        # Find the data with the model which will be having trasnfomer tag
        df = df[df.tags.apply(lambda x: STRING_TO_CHECK in x)]
        # Retrive the data whose task is in the list
        df = df[df['pipeline_tag'].isin(TASK_NAME)]

        # Find the data with that particular name
        required_data = df[df.modelId.apply(lambda x: x == self.model_name)]
        # Get the task
        required_data = required_data["pipeline_tag"].to_string()
        # Create pattern fiel number and space
        pattern = r'[0-9\s+]'
        # Replace number and space
        final_data = re.sub(pattern, '', required_data)
        return final_data
		
TASK_NAME = get_task()
update_existing_model=True
Reg_Model=test_model_name.replace('/','-')
# version_list = list(ml_client_ws.models.list(Reg_Model))
# foundation_model = ''
# if len(version_list) == 0:
#     print("Model not found in registry")
huggingface_model_exists_in_registry = False
# else
#     model_version = version_list[0].version
#     foundation_model = ml_client_ws.models.get(Reg_Model, model_version)
#     print(
#         "\n\nUsing model name: {0}, version: {1}, id: {2} for F.T".format(
#             foundation_model.name, foundation_model.version, foundation_model.id
#         )
#     )
#     huggingface_model_exists_in_registry = True
# print (f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
@pipeline
def model_import_pipeline(model_id,update_existing_model, task_name):
    """
    Create model import pipeline using pipeline component.

    Parameters
    ----------
    model_id : str
    compute : str

    Returns
    -------
    model_registration_details : dict
    """
    import_model_job = import_model(
        model_id=test_model_name, task_name=task_name,update_existing_model=update_existing_model
    )

    # Set job to not continue on failure
    import_model_job.settings.continue_on_step_failure = False

    return {
        "model_registration_details": import_model_job.outputs.model_registration_details
    }
	
pipeline_object = model_import_pipeline(
    model_id=test_model_name,
    # compute=COMPUTE,
    task_name=TASK_NAME,
    # registry_name=REGISTRY_NAME,
    update_existing_model=update_existing_model,
    
)
pipeline_object.identity = UserIdentityConfiguration()

pipeline_object.settings.force_rerun = True


# pipeline_object.settings.default_compute = COMPUTE
schedule_huggingface_model_import = (
    not huggingface_model_exists_in_registry
    and test_model_name not in [None, "None"]
    and len(test_model_name) > 1
)
print(
    f"Need to schedule run for importing {test_model_name}: {schedule_huggingface_model_import}")

huggingface_pipeline_job = None
# if schedule_huggingface_model_import:
    # submit the pipeline job
huggingface_pipeline_job = ml_client_ws.jobs.create_or_update(
    pipeline_object, experiment_name=experiment_name
)
# wait for the pipeline job to complete
ml_client_ws.jobs.stream(huggingface_pipeline_job.name)

import os
import shutil

download_path = "./pipeline_outputs/"

# delete the folder if already exists
if os.path.exists(download_path):
    shutil.rmtree(download_path)

# if pipeline job was not scheduled, skip
if huggingface_pipeline_job  is not None:

    print("Pipeline job: " + huggingface_pipeline_job.name)
    print("Downloading pipeline job output: model_registration_details")

    pipeline_download_path = os.path.join(download_path, huggingface_pipeline_job.name)
    os.makedirs(pipeline_download_path, exist_ok=True)

    ml_client_ws.jobs.download(
        name=huggingface_pipeline_job.name,
        download_path=pipeline_download_path,
        output_name="model_registration_details",
    )
import json

# if pipeline job was not scheduled, skip
if huggingface_pipeline_job is not None:

    with open(
        f"./pipeline_outputs/{huggingface_pipeline_job.name}/named-outputs/model_registration_details/model_registration_details.json",
        "r",
    ) as f:
        registration_details = json.load(f)

    model_name = registration_details["name"]
    model_version = registration_details["version"]

    # Get the model object from workspace
    model = ml_client_ws.models.get(name=model_name, version=model_version)
    print(f"\n{model_name}")
    print(model.__dict__)
