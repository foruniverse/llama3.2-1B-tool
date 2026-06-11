from modelscope.hub.api import HubApi
api = HubApi()
api.login("ms-5b9ac5f7-788f-4fa4-96e8-8068b24f1ebb")
api.upload_folder(
    repo_id="whichcy/llama3.2-1B-tool",
    folder_path="./models/sft-360",
    repo_type="model"
)
