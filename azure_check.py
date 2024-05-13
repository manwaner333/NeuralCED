import os
from openai import AzureOpenAI

# print(os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT"))
# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2024-02-01",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )

# AZURE_OPENAI_API_KEY = "836efdfda2fb4413a2b57ff26f303f0c"
# AZURE_OPENAI_ENDPOINT = "https://azure-openai-ukp-004.openai.azure.com/"
# deployment_name = 'gpt-4'

AZURE_OPENAI_API_KEY = "5de57ad92cb94857a335d4d92bebd4cf"
AZURE_OPENAI_ENDPOINT = "https://azure-openai-ukp-003.openai.azure.com/"

deployment_name = "azure-openai-ukp-003"


client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
 # This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment.

# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a tagline for an ice cream shop. '
response = client.completions.create(model=deployment_name, prompt=start_phrase, max_tokens=10)
print(start_phrase + response.choices[0].text)
#

# client = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     api_version=api_version,
#     base_url=f"{api_base}openai/deployments/{deployment_name}/extensions",
# )
# response = client.chat.completions.create(
#     model=deployment_name,
#     messages=[
#         { "role": "system", "content": "You are a helpful assistant." },
#         { "role": "user", "content": [
#             {
#                 "type": "text",
#                 "text": "Describe this picture:"
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": "<image URL>"
#                 }
#             }
#         ] }
#     ],
#     max_tokens=2000
# )
# print(response)