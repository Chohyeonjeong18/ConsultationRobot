import requests

# This function will pass your text to the machine learning model
# and return the top result with the highest confidence
def YesOrNO_classify(text):
    key = "3ddb25e0-51ee-11ea-a23e-3f15abe9a3ade5b3619c-8090-4842-bbab-a4777084cb4b"
    url = "https://machinelearningforkids.co.uk/api/scratch/"+ key + "/classify"

    response = requests.get(url, params={ "data" : text })

    if response.ok:
        responseData = response.json()
        topMatch = responseData[0]
        return topMatch["class_name"]
    else:
        response.raise_for_status()
