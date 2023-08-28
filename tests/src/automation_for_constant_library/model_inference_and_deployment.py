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

class ModelInferenceAndDeployemnt:
    def __init__(self, test_model_name, workspace_ml_client, registry_ml_client, registry) -> None:
        self.test_model_name = test_model_name
        self.workspace_ml_client = workspace_ml_client
        self.registry_ml_client = registry_ml_client
        self.registry = registry

    def get_latest_model_version(self, registry_ml_client, model_name):
        print("In get_latest_model_version...")
        version_list = list(registry_ml_client.models.list(model_name))
        if len(version_list) == 0:
            print("Model not found in registry")
        else:
            model_version = version_list[0].version
            foundation_model = registry_ml_client.models.get(
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

    def sample_inference(self, latest_model, registry, workspace_ml_client, online_endpoint_name):
        # get the task tag from the latest_model.tags
        tags = str(latest_model.tags)
        # replace single quotes with double quotes in tags
        tags = tags.replace("'", '"')
        # convert tags to dictionary
        tags_dict = json.loads(tags)
        task = tags_dict['task']
        print(f"task: {task}")
        scoring_file = f"../../config/sample_inputs/{registry}/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = json.load(f)
                print(f"scoring_input file:\n\n {scoring_input}\n\n")
        except Exception as e:
            print(
                f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")

        # invoke the endpoint
        try:
            response = workspace_ml_client.online_endpoints.invoke(
                endpoint_name=online_endpoint_name,
                deployment_name="demo",
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

    def create_online_endpoint(self, latest_model, endpoint):
        print("In create_online_endpoint...")
        # try:
        #     workspace_ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
        # except Exception as e:
        #     print (f"::error:: Could not create endpoint: \n")
        #     print (f"{e}\n\n check logs:\n\n")
        #     prase_logs(str(e))
        #     exit (1)

        # print(workspace_ml_client.online_endpoints.get(name=endpoint.name))

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
        # timestamp = int(time.time())
        # online_endpoint_name = "Testing" + str(timestamp)
        # print (f"online_endpoint_name: {online_endpoint_name}")
        # endpoint = ManagedOnlineEndpoint(
        #      name=online_endpoint_name,
        #      auth_mode="key",
        #  )
        self.workspace_ml_client.begin_create_or_update(endpoint).result()
        return model_package

    def create_online_deployment(self, latest_model, online_endpoint_name, model_package, instance_type):
        print("In create_online_deployment...")
        # demo_deployment = ManagedOnlineDeployment(
        #     name="demo",
        #     endpoint_name=endpoint.name,
        #     model=latest_model.id,
        #     instance_type="Standard_D13",
        #     instance_count=1,
        # )
        # demo_deployment = ManagedOnlineDeployment(
        #     name="default",
        #     endpoint_name=endpoint.name,
        #     model=latest_model.id,
        #     instance_type="Standard_DS2_v2",
        #     instance_count="1",
        #     request_settings=OnlineRequestSettings(
        #         max_concurrent_requests_per_instance=1,
        #         request_timeout_ms=50000,
        #         max_queue_wait_ms=500,
        #     ),
        #     liveness_probe=ProbeSettings(
        #         failure_threshold=10,
        #         timeout=10,
        #         period=10,
        #         initial_delay=480,
        #     ),
        #     readiness_probe=ProbeSettings(
        #         failure_threshold=10,
        #         success_threshold=1,
        #         timeout=10,
        #         period=10,
        #         initial_delay=10,
        #     ),
        # )
        # workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
        # endpoint.traffic = {"demo": 100}
        # workspace_ml_client.begin_create_or_update(endpoint).result()
        print("latest_model.name is this : ", latest_model.name)
        latest_model_name = latest_model.name.replace("_", "-")
        if latest_model_name[0].isdigit():
            num_pattern = "[0-9]"
            latest_model_name = re.sub(num_pattern, '', latest_model_name)
            latest_model_name = latest_model_name.strip("-")
            
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
        deployment = self.workspace_ml_client.online_deployments.begin_create_or_update(
            deployment_config).result()

    def delete_online_endpoint(self, online_endpoint_name):
        try:
            print("\n In delete_online_endpoint.....")
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=online_endpoint_name).wait()
        except Exception as e:
            print(f"::warning:: Could not delete endpoint: : \n{e}")
            exit(0)

    def model_inference(self, task, latest_model):
        scoring_file = f"../../config/sample_inputs/{self.registry}/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = ConfigBox(json.load(f))
                print(f"scoring_input file:\n\n {scoring_input}\n\n")
        except Exception as e:
            print(
                f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")
        print(
            f"Latest model name : {latest_model.name} and latest model version : {latest_model.version}", )
        downloaded_model = self.workspace_ml_client.models.download(
            name=latest_model.name, version=latest_model.version, download_path=f"./model_download")
        loaded_model = mlflow.transformers.load_model(
            model_uri=f"./model_download/{latest_model.name}/{latest_model.name}-artifact", return_type="pipeline")
        print(type(loaded_model))

        if task == "fill-mask":
            pipeline_tokenizer = loaded_model.tokenizer
            for index in range(len(scoring_input.inputs)):
                scoring_input.inputs[index] = scoring_input.inputs[index].replace(
                    "<mask>", pipeline_tokenizer.mask_token).replace("[MASK]", pipeline_tokenizer.mask_token)

        output = loaded_model(scoring_input.inputs)
        print("My outupt is this : ", output)

    def model_infernce_and_deployment(self, instance_type):
        model_name = self.test_model_name.replace("/", "-")
        # if len(self.test_model_name) > 22:
        #     model_name = self.test_model_name.replace("/", "-")[:22]
        #     model_name = model_name.rstrip("-")
        # else:
        #     model_name = self.test_model_name
        latest_model = self.get_latest_model_version(
            self.registry_ml_client, model_name)
        task = latest_model.flavors["transformers"]["task"]
        print("latest_model:", latest_model)
        print("Task is : ", task)
        self.model_inference(task=task, latest_model=latest_model)
        # endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
        timestamp = int(time.time())
        online_endpoint_name = task + str(timestamp)
        #online_endpoint_name = "Testing" + str(timestamp)
        print(f"online_endpoint_name: {online_endpoint_name}")

        endpoint = ManagedOnlineEndpoint(
            name=online_endpoint_name,
            auth_mode="key",
        )
        model_package = self.create_online_endpoint(
            latest_model=latest_model, endpoint=endpoint)
        self.create_online_deployment(
            latest_model=latest_model,
            online_endpoint_name=online_endpoint_name,
            model_package=model_package,
            instance_type=instance_type
        )
        self.delete_online_endpoint(online_endpoint_name=online_endpoint_name)
        # endpoint = ManagedOnlineEndpoint(
        #     name=online_endpoint_name,
        #     auth_mode="key",
        # )

        #print("endpoint name:",endpoint)
        #self.create_online_endpoint(self.workspace_ml_client, endpoint)
        #self.create_online_deployment(self.workspace_ml_client, endpoint, latest_model)
        # task = latest_model.flavors["transformers"]["task"]
        #model_for_package = Model(name=latest_model.name, version=latest_model.version, type=AssetTypes.MLFLOW_MODEL)

        # timestamp = int(time.time())
        # online_endpoint_name = "fill-mask" + str(timestamp)
        # endpoint = ManagedOnlineEndpoint(
        #     name=online_endpoint_name,
        #     description="Online endpoint for "
        #     + latest_model.name
        #     + ", for fill-mask task",
        #     auth_mode="key",
        # )
        # self.workspace_ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        # demo_deployment = ManagedOnlineDeployment(
        #     name="demo",
        #     endpoint_name=online_endpoint_name,
        #     model=latest_model.id,
        #     #instance_type="Standard-D13",
        #     instance_count=1,
        #     request_settings=OnlineRequestSettings(
        #         request_timeout_ms=60000,
        #     ),
        # )
        # self.workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).result()
        # endpoint.traffic = {"demo": 100}
        # self.workspace_ml_client.begin_create_or_update(endpoint).result()
