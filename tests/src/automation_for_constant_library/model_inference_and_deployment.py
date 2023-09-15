import time
import json
import os
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings,
    Model,
    ModelConfiguration,
    ModelPackage,
    AzureMLOnlineInferencingServer
)
import mlflow
from box import ConfigBox
import re
import sys


class ModelInferenceAndDeployemnt:
    def __init__(self, test_model_name, workspace_ml_client, registry) -> None:
        self.test_model_name = test_model_name
        self.workspace_ml_client = workspace_ml_client
        self.registry = registry

    def get_error_messages(self):
        # load ../../config/errors.json into a dictionary
        with open('../../config/errors.json') as f:
            return json.load(f)

    def prase_logs(self, logs):
        error_messages = self.get_error_messages()
        # split logs by \n
        logs_list = logs.split("\n")
        # loop through each line in logs_list
        for line in logs_list:
            # loop through each error in errors
            for error in error_messages:
                # if error is found in line, print error message
                if error['parse_string'] in line:
                    print(
                        f"::error:: {error_messages['error_category']}: {line}")

    def get_online_endpoint_logs(self, deployment_name, online_endpoint_name):
        print("Deployment logs: \n\n")
        logs = self.workspace_ml_client.online_deployments.get_logs(
            name=deployment_name, endpoint_name=online_endpoint_name, lines=100000)
        print(logs)
        self.prase_logs(logs)

    def get_latest_model_version(self, workspace_ml_client, model_name):
        print("In get_latest_model_version...")
        version_list = list(workspace_ml_client.models.list(model_name))
        if len(version_list) == 0:
            print("Model not found in registry")
        else:
            model_version = version_list[0].version
            foundation_model = workspace_ml_client.models.get(
                model_name, model_version)
            print(
                "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                    foundation_model.name, foundation_model.version, foundation_model.id
                )
            )
        print(
            f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
        #print(f"Model Config : {latest_model.config}")
        return foundation_model

    def cloud_inference(self, scoring_file, scoring_input, online_endpoint_name, deployment_name):
        try:
            print("endpoint_name : ", online_endpoint_name)
            print("deployment_name : ", deployment_name)
            print("Input data is this one :", scoring_input)
            response = self.workspace_ml_client.online_endpoints.invoke(
                endpoint_name=online_endpoint_name,
                deployment_name=deployment_name,
                request_file=scoring_file,
            )
            response_json = json.loads(response)
            output = json.dumps(response_json, indent=2)
            print(f"response: \n\n{output}")
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as fh:
                print(f'####Sample input', file=fh)
                print(f'```json', file=fh)
                print(f'{scoring_input}', file=fh)
                print(f'```', file=fh)
                print(f'####Sample output', file=fh)
                print(f'```json', file=fh)
                print(f'{output}', file=fh)
                print(f'```', file=fh)
        except Exception as e:
            print(f"::error:: Could not invoke endpoint: \n")
            print(f"{e}\n\n check logs:\n\n")

    def create_model_package(self, latest_model, endpoint):
        print("In create_model_package...")
        model_configuration = ModelConfiguration(mode="download")
        package_name = f"package-v2-{latest_model.name}"
        package_config = ModelPackage(
            target_environment_name=package_name,
            inferencing_server=AzureMLOnlineInferencingServer(),
            model_configuration=model_configuration
        )
        model_package = self.workspace_ml_client.models.package(
            latest_model.name,
            latest_model.version,
            package_config
        )
        try:
            self.workspace_ml_client.begin_create_or_update(endpoint).result()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            print(f"::error:: Could not create endpoint: \n")
            print(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                  " the exception is this one :", e)
            print(f"{e}\n\n check logs:\n\n")
            self.prase_logs(str(e))
            exit(1)
        return model_package

    def create_online_deployment(self, latest_model, online_endpoint_name, model_package, instance_type):
        print("In create_online_deployment...")
        print("latest_model.name is this : ", latest_model.name)
        # Expression need to be replaced with hyphen
        expression_to_ignore = ["/", "\\", "|", "@", "#", ".",
                                "$", "%", "^", "&", "*", "<", ">", "?", "!", "~", "_"]
        # Create the regular expression to ignore
        regx = re.compile('|'.join(map(re.escape, expression_to_ignore)))
        # Check the model_name contains any of there character
        expression_check = re.findall(regx, latest_model.name)
        if expression_check:
            # Replace the expression with hyphen
            latest_model_name = regx.sub("-", latest_model.name)
        else:
            latest_model_name = latest_model.name

        # Check if the model name starts with a digit
        if latest_model_name[0].isdigit():
            num_pattern = "[0-9]"
            latest_model_name = re.sub(num_pattern, '', latest_model_name)
            latest_model_name = latest_model_name.strip("-")
        # Check the model name is more then 32 character
        if len(latest_model.name) > 32:
            model_name = latest_model_name[:31]
            deployment_name = model_name.rstrip("-")
        else:
            deployment_name = latest_model_name
        print("deployment name is this one : ", deployment_name)
        deployment_config = ManagedOnlineDeployment(
            name=deployment_name,
            model=latest_model,
            endpoint_name=online_endpoint_name,
            environment=model_package,
            instance_type=instance_type,
            instance_count=1
        )
        try:
            deployment = self.workspace_ml_client.online_deployments.begin_create_or_update(
                deployment_config).result()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            print(f"::error:: Could not create deployment\n")
            print(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                  " the exception is this one :", e)
            print(f"{e}\n\n check logs:\n\n")
            self.prase_logs(str(e))
            self.get_online_endpoint_logs(
                deployment_name, online_endpoint_name)
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=online_endpoint_name).wait()
            exit(1)

        return deployment_name

    def delete_online_endpoint(self, online_endpoint_name):
        try:
            print("\n In delete_online_endpoint.....")
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=online_endpoint_name).wait()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            print(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                  " the exception is this one :", e)
            print(f"::warning:: Could not delete endpoint: : \n{e}")
            exit(0)

    def get_task_specified_input(self, task):
        scoring_file = f"../../config/sample_inputs/{self.registry}/{task}.json"
        #scoring_file = f"sample_inputs/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = ConfigBox(json.load(f))
                print(f"scoring_input file:\n\n {scoring_input}\n\n")
        except Exception as e:
            print(
                f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")
        return scoring_file, scoring_input

    def local_inference(self, task, latest_model, scoring_input):
        model_sourceuri = latest_model.properties["mlflow.modelSourceUri"]
        loaded_model_pipeline = mlflow.transformers.load_model(
            model_uri=model_sourceuri)
        print(
            f"Latest model name : {latest_model.name} and latest model version : {latest_model.version}", )
        if task == "fill-mask":
            pipeline_tokenizer = loaded_model_pipeline.tokenizer
            for index in range(len(scoring_input.input_data)):
                scoring_input.input_data[index] = scoring_input.input_data[index].replace(
                    "<mask>", pipeline_tokenizer.mask_token).replace("[MASK]", pipeline_tokenizer.mask_token)

        output = loaded_model_pipeline(scoring_input.input_data)
        print("My outupt is this : ", output)

    def model_infernce_and_deployment(self, instance_type):
        model_name = self.test_model_name.replace("/", "-")
        latest_model = self.get_latest_model_version(
            self.workspace_ml_client, model_name)
        task = latest_model.flavors["transformers"]["task"]
        print("latest_model:", latest_model)
        print("Task is : ", task)
        scoring_file, scoring_input = self.get_task_specified_input(task=task)
        # self.local_inference(task=task, latest_model=latest_model, scoring_input=scoring_input)
        # endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
        timestamp = int(time.time())
        online_endpoint_name = task + str(timestamp)
        #online_endpoint_name = "Testing" + str(timestamp)
        print(f"online_endpoint_name: {online_endpoint_name}")
        endpoint = ManagedOnlineEndpoint(
            name=online_endpoint_name,
            auth_mode="key",
        )
        model_package = self.create_model_package(
            latest_model=latest_model, endpoint=endpoint)
        deployment_name = self.create_online_deployment(
            latest_model=latest_model,
            online_endpoint_name=online_endpoint_name,
            model_package=model_package,
            instance_type=instance_type
        )
        # self.cloud_inference(
        #     scoring_file=scoring_file,
        #     scoring_input=scoring_input,
        #     online_endpoint_name=online_endpoint_name,
        #     deployment_name=deployment_name
        # )
        self.delete_online_endpoint(online_endpoint_name=online_endpoint_name)
