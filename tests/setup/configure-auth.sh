# pre reqs
# az cli installed
# gh cli installed (conda install gh --channel conda-forge)
# Make your you have `az login` and `az account set`

#GROUP="huggingface-registry-test"
#SUBSCRIPTION="c11df4e4-eb38-4e11-b3dd-e8874de77d0a"

#GROUP1="huggingface-registry-test1"
#SUBSCRIPTION1="80c77c76-74ba-4c8c-8229-4c3b2957990c"

#GROUP2="huggingface-registry-test22"
#SUBSCRIPTION2="ec1644c7-587f-4e34-bc8d-6c162a042cc2"

SUBSCRIPTION=$1
GROUP=$2
CRED=$3

echo "Setting subscription to $SUBSCRIPTION..."
az account set -s $SUBSCRIPTION

SP_DISPLAY_NAME=$GROUP-sp
#CRED="AZ_CRED"

echo "Creating service principal $SP_DISPLAY_NAME and setting repository secret $CRED..."

#az ad sp create-for-rbac --name $SP_DISPLAY_NAME --role Contributor --sdk-auth | gh secret set $CRED

az ad sp create-for-rbac --name $SP_DISPLAY_NAME --role Contributor --scopes /subscriptions/$SUBSCRIPTION/resourceGroups/$GROUP --sdk-auth | gh secret set $CRED

echo  "granting Contributor access to $GROUP and $SUBSCRIPTION access to the service principal..."

# Get the app id:
appid=$( az ad sp list --display-name $SP_DISPLAY_NAME | grep appId | awk -F: '{print $2}' | sed s/\"//g | sed s/,//g )

az role assignment create --assignee $appid --role Contributor --scope /subscriptions/$SUBSCRIPTION/resourceGroups/$GROUP 
