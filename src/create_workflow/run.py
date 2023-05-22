from huggingface_hub import HfApi, ModelFilter
import re
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

inference_task_type_dict = {
    "fill-mask": {"real-time": "fill-mask-online-endpoint.ipynb", "batch": "fill-mask-batch-endpoint.ipynb"},
    "automatic-speech-recognition" : {"real-time": "asr-online-endpoint", "batch": "asr-batch-endpoint"},
    "question-answering": {"real-time": "question-answering-online-endpoint", "batch": "question-answering-batch-endpoint"},
    "summarization":{"real-time": "summarization-online-endpoint", "batch": "summarization-batch-endpoint"},
    "text-classification": {"real-time": "text-classification-online-endpoint", "batch": "entailment-contradiction-batch"},
    "text-generation": {"real-time": "text-generation-online-endpoint", "batch": "text-generation-batch-endpoint"},
    "token-classification": {"real-time": "token-classification-online-endpoint", "batch": "token-classification-batch-endpoint"},
    "translation": {"real-time": "translation-online-endpoint", "batch": "translation-batch-endpoint"}
}

evaluation_task_dict = {
    "fill-mask": "fill-mask.ipynb",
    "question-answering": "question-answering.ipynb",
    "summarization":"abstractive-and-extractive-summarization.ipynb",
    "text-classification": "entailment-contradiction.ipynb",
    "text-generation": "text-generation.ipynb",
    "token-classification": "news-articles-entity-recognition.ipynb",
    "translation": "translation-romanian-to-english.ipynb"
}

ft_task_dict = {
    "question-answering": "extractive-qa.ipynb",
    "summarization":"news-summary.ipynb",
    "text-classification": "emotion-detection.ipynb",
    "token-classification": "token-classification.ipynb",
    "translation": "translation.ipynb"  
}

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

registry_ml_client = MLClient(
    credential, registry_name="azureml"
)


def get_latest_model_version(model_id):
    models = registry_ml_client.models.list(name = model_id)
    max_version = (max(models, key=lambda x: int(x.version))).version
    return max_version

def get_directory_model_id(directory):
    return os.listdir(directory)

def get_top_model_ids(task, sort_key="downloads", direction=-1, limit=10):
    api = HfApi()
    models = api.list_models(
        filter=ModelFilter(
            task=task,
        ),
        sort=sort_key,
        direction=direction,
        limit=limit,
    )
    models = list(models)
    return [x.modelId for x in models]


def remove_existing_workflow(output_path):
    file_names = os.listdir(output_path)
    for file in file_names:
        if file.startswith("import-"):
            os.remove(f"{output_path}/{file}")

def generate_eval_workflow(task, template = 'workflow_templates/evaluation_workflow.yaml', output_path = "../../.github/workflows"):
    try:
        with open(template, "r") as f:
            template_content = f.read()
        notebook = evaluation_task_dict[task]
        template_content = template_content.replace("<notebook>", notebook)
        template_content = template_content.replace("<task>", task)   

        output_file = f"{output_path}/evaluation-{task}_nb.yaml"
        with open(output_file, "w") as f:
            f.write(template_content)

        print(f"Evaluation workflow written for {task}")
    except exception as e:
        print(e)
        print("Not fetched notebook for evaluation task ", task)


def generate_finetune_workflow(model_name, model_version, fttask, template = 'workflow_templates/finetuning_workflow.yaml', output_path = "../../.github/workflows"):

    try:
        notebook = ft_task_dict[fttask]
        with open(template, "r") as f:
            template_content = f.read()

        template_content = template_content.replace("<model_name>", model_name)
        template_content = template_content.replace("<model_version>", model_version)
        template_content = template_content.replace("<notebook>", notebook)
        template_content = template_content.replace("<fttask>", fttask)

        output_file = f"{output_path}/finetuning-{fttask}-{model_name}_nb"
        with open(output_file, "w") as f:
            f.write(template_content)
        print(f"Finetune workflow written for {model_name} and {fttask}")
    except exception as e:
        print(e)
        print("Not fetched notebook for finetune model_name, fttask", model_name, fttask)

def generate_inference_workflow(model_name, model_version, task, template = 'workflow_templates/inference_workflow.yaml', output_path = "../../.github/workflows"):
    inference_types = ['real-time', 'batch']
    for infer_type in inference_types:
        with open(template, "r") as f:
            template_content = f.read()
        try:
            notebook = inference_task_type_dict[task][infer_type]
            template_content = template_content.replace("<inference-type>", infer_type)
            template_content = template_content.replace("<model_name>", model_name)
            template_content = template_content.replace("<model_version>", model_version)
            template_content = template_content.replace("<notebook>", notebook)
            template_content = template_content.replace("<task>", task)

            output_file = f"{output_path}/{infer_type}-inference-{model_name}_nb.yaml"
            with open(output_file, "w") as f:
                f.write(template_content)
            print(f"{infer_type} Inference workflow written for {model_name}")
        except:
            print("Not fetched notebook for inference model_name, task", model_name, task)


def generate_workflow_files(dictionary):

    model_name = dictionary["model_name"]
    model_version = dictionary["model_version"]
    task = dictionary["task"]

    
    generate_inference_workflow(model_name, model_version, task)
    generate_eval_workflow(task)

    for fttask in dictionary["finetuning-tasks"]:
        generate_finetune_workflow(model_name, model_version, fttask)

def create_md_table(data):
    table_header = "| Task | Model ID | Status |\n"
    table_divider = "| --- | --- | --- |\n"

    table_rows = ""
    for row in data:
        task_supported = row["task_supported"]
        model_id = row["model_id"]
        table_rows += f"| {task_supported} | {model_id} | [![{model_id.replace('/','-')} workflow](https://github.com/Azure/azureml-examples/actions/workflows/import-{model_id.replace('/','-')}.yaml/badge.svg?branch=hrishikesh/model-import-workflows)](https://github.com/Azure/azureml-examples/actions/workflows/import-{model_id.replace('/','-')}.yaml?branch=hrishikesh/model-import-workflows) |\n"

    table = table_header + table_divider + table_rows

    with open("../README.md", "w") as file:
        file.write(table)
    print("README file created which will show the status of import workflow.....")


def replace_special_characters(string):
    pattern = r"[^a-zA-Z0-9-_]"
    return re.sub(pattern, "", string)

directory = "C:/Users/hgeed/Documents/model_status_experiment/azureml-assets/models/system"
model_ids = get_directory_model_id(directory)

data = []
for model_id in model_ids:
    try:
        version = get_latest_model_version(model_id)
        model = registry_ml_client.models.get(name = model_id, version = version)
        task = model.tags["task"]
        finetuning_task = [x.strip() for x in  model.properties["finetuning-tasks"].split(',')]
        data.append({
            "model_name":model_id,
            "model_version":version,
            "task":task,
            "finetuning-tasks":finetuning_task
            })
    except:
        print("failed getting model object for ", model_id)


# Usage example
# template_file = "workflow_template.yaml"  # Path to your template workflow file
# output_directory = "../../../../../../.github/workflows"  # Directory where the generated workflow files will be saved

for dictionary in data:
    generate_workflow_files(dictionary)

######################################################################################################################################################




















# task_supported = [
#     "fill-mask",
#     "token-classification",
#     "question-answering",
#     "summarization",
#     "text-generation",
#     "text-classification",
#     "translation",
#     "image-classification",
#     "text-to-image",
# ]
# remove_existing_workflow(output_directory)

# data = []
# for task in task_supported:
#     model_ids = get_top_model_ids(task=task)
#     generate_workflow_file(template_file, model_ids, output_directory)
#     for model_id in model_ids:
#         data.append({"model_id": model_id, "task_supported": task})
#     print(f"Workflow file generated for task {task}")

# create_md_table(data)

#### To do
## Remove workflow files if not in top n models
## Add better function to create md file