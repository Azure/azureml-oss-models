import os
import json
import requests
from datetime import datetime

import argparse
# parameter to get models list from file
parser = argparse.ArgumentParser()
parser.add_argument("--model_list_file", type=str, default="../logs/get_model_containers/17May2023-231031.json")
# parameter to get github workflows from file
parser.add_argument("--github_workflows_file", type=str, default="../logs/get_github_workflows/18May2023-211807.json")
# mode parameter to get workflow status from api or file
parser.add_argument("--mode", type=str, default="api")
# parameter to get markfown file name
parser.add_argument("--markdown_file", type=str, default="../../dashboard/HuggingFace/README.md")
args = parser.parse_args()

# constants
RUN_API="https://api.github.com/repos/Azure/azureml-oss-models/actions/runs"

def get_github_token():
    # fetch the github token from `gh auth token` command
    return os.popen("gh auth token").read().rstrip()


# function to load model_list_file
def load_model_list_file():
    # if model_list_file is extention is json, load json file
    if args.model_list_file.endswith(".json"):
        with open(args.model_list_file) as f:
            return json.load(f)
    # read all other files as text files, assuming one model per line
    with open(args.model_list_file) as f:
        return f.read().splitlines()

# function to get the github workflows using github rest api

def get_github_workflows(token):
    print (f"Getting github workflows from {RUN_API}")
    total_pages = None
    current_page = 1
    per_page = 100
    runs = []
    while total_pages is None or current_page <= total_pages:
        # create a requests session object with 
        headers = { "Authorization": f"Bearer {token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "Accept": "application/vnd.github+json"
        }
        params = { "per_page": per_page, "page": current_page }
        response = requests.get(RUN_API, headers=headers, params=params)
        if response.status_code == 200:
            json_response = response.json()
            # append workflow_runs to runs list
            runs.extend(json_response['workflow_runs'])
            if current_page == 1:
            # divide total_count by per_page and round up to get total_pages
                total_pages = int(json_response['total_count'] / per_page) + 1
            current_page += 1
            # print a single dot to show progress
            print (f"\rRuns fetched: {len(runs)}", end="", flush=True)
        else:
            print (f"Error: {response.status_code} {response.text}")
            exit(1)
    print (f"\n")
    # create ../logs/get_github_workflows/ if it does not exist
    if not os.path.exists("../logs/get_github_workflows"):
        os.makedirs("../logs/get_github_workflows")
    # dump runs as json file in ../logs/get_github_workflows folder with filename as DDMMMYYYY-HHMMSS.json
    with open(f"../logs/get_github_workflows/{datetime.now().strftime('%d%b%Y-%H%M%S')}.json", "w") as f:
        json.dump(runs, f, indent=4)
    return runs
    
# function to calculate test status based on models - total tests, success, failure, not_tested, total test duration
def calculate_test_status(runs, models):
# get the latest run for each model
# for each run, get the status, conclusion and duration as updated_at - created_at
# calculate total, success, failure, not_tested, total test duration
    results_per_model = {}
    min_time = max_time = runs[0]['updated_at']
    for run in runs:
        model = run['name']
        status = run['status']
        conclusion = run['conclusion']
        if model not in results_per_model:
            results_per_model[model] = {"success": 0, "failure": 0, "unknown": 0,"not_tested": 0, "duration": 0, "last_tested": ""}
            if status == "completed":
                # if min_time is greater than run['created_at'], set min_time to run['created_at']
                if min_time > run['created_at']:
                    min_time = run['created_at']
                if conclusion == "success":
                    results_per_model[model]["success"] = 1
                elif conclusion == "failure":
                    results_per_model[model]["failure"] = 1
                else:
                    results_per_model[model]["unknown"] = 1
                results_per_model[model]['last_tested'] = run['updated_at']
                # load updated_at and created_at as datetime objects and assign the difference to duration in minutes
                updated_at = datetime.strptime(run['updated_at'], "%Y-%m-%dT%H:%M:%SZ")
                created_at = datetime.strptime(run['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                results_per_model[model]['duration'] = (updated_at - created_at).total_seconds() / 60
            else:
                results_per_model[model]["not_tested"] = 1
                results_per_model[model]['last_tested'] = None
    # for models not in results_per_model, set not_tested to 1
    for model in models:
        if model not in results_per_model:
            results_per_model[model] = {"success": 0, "failure": 0, "unknown": 0,"not_tested": 1, "duration": 0, "last_tested": None}
    # dump results_per_model as json to stdout
    print (json.dumps(results_per_model['bert-base-cased'], indent=4))
    # load max_time and min_time as datetime objects and assign the difference to clock_time in minutes
    max_time = datetime.strptime(max_time, "%Y-%m-%dT%H:%M:%SZ")
    min_time = datetime.strptime(min_time, "%Y-%m-%dT%H:%M:%SZ")
    clock_time = (max_time - min_time).total_seconds() / 60

    return results_per_model, clock_time

def summarize_test_status(results_per_model):
    status = {"total": 0, "success": 0, "failure": 0, "unknown": 0, "not_tested": 0, "total_duration": 0}
    print (json.dumps(status, indent=4))
    for model in results_per_model:
        status["total"] += 1
        status["success"] += results_per_model[model]["success"]
        status["failure"] += results_per_model[model]["failure"]
        status["unknown"] += results_per_model[model]["unknown"]
        status["not_tested"] += results_per_model[model]["not_tested"]
        if results_per_model[model]["last_tested"]:
            status["total_duration"] += results_per_model[model]["duration"]
    return status

def create_badge(results_per_model, status, clock_time):
    lines=[]
    # generate test_duration_srt as hours and minutes from total_duration
    test_duration_str = f"{int(status['total_duration'] / 60)}h {int(status['total_duration'] % 60)}m"
    # generate clock_time_str as hours and minutes from clock_time
    clock_time_str = f"{int(clock_time / 60)}h {int(clock_time % 60)}m"
    # print status as markdown table
    lines.append(f"### Summary\n")
    lines.append(f"🚀Total|✅Success|❌Failure|❔Unknown|🧪Not Tested|🕰️Total Duration|⏱️Clock duration\n")
    lines.append(f"-----|-------|-------|-------|----------|--------------\n")
    lines.append(f"{status['total']}|{status['success']}|{status['failure']}|{status['unknown']}|{status['not_tested']}|{test_duration_str}|{clock_time_str}\n")
   
    labels = ["Success", "Failure", "Unknown", "Not Tested"]
    for label in labels:
        lines.append(f"### {label}\n")
        # covert label to lowercase and replace space with underscore
        lower_label = label.lower().replace(" ", "_")
        lines.append("|Model|Last tested at|\n")
        lines.append("|-----|--------------|\n")
        for model in results_per_model:
            if results_per_model[model][lower_label] == 1:
                if results_per_model[model]['last_tested'] is not None:
                # load last_tested as datetime object and format as DD MMM YYYY HH:MM:SS
                    last_tested = datetime.strptime(results_per_model[model]['last_tested'], "%Y-%m-%dT%H:%M:%SZ").strftime("%d %b %Y %H:%M:%S")
                else:
                    last_tested = "No status"
                lines.append(f"{model}|[![{last_tested}](https://github.com/Azure/azureml-oss-models/actions/workflows/{model}.yml/badge.svg)](https://github.com/Azure/azureml-oss-models/actions/workflows/{model}.yml)")


    # write to markdown file
    print (f"Writing to {args.markdown_file}")
    with open(args.markdown_file, 'w') as f:
        for line in lines:
            f.write(line + "\n")

def main():
    # print mode
    print (f"Mode: {args.mode}")
    # if mode is api, get github workflows using github rest api
    if args.mode == "api":
        runs = get_github_workflows(get_github_token())
    elif args.mode == "file":
    # else, load github workflows from file
        with open(args.github_workflows_file) as f:
            runs = json.load(f)
    else:
        print (f"Error: Invalid mode {args.mode}")
        exit(1)
    models = load_model_list_file()
    results_per_model, clock_time = calculate_test_status(runs, models)
    # print
    status = summarize_test_status(results_per_model)
    # dump status to STDOUT
    create_badge(results_per_model, status, clock_time)


if __name__ == "__main__":
    main()
