import boto3
import json

# Define your prompt data
prompt_data = """
Tell me something useful about generative AI!
"""

# Initialize the Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Create the body for the API call
payload = {
    "inputText": prompt_data,
    "textGenerationConfig": {
        "maxTokenCount": 8192,  # Max tokens as per your API example
        "stopSequences": [],    # No stop sequences
        "temperature": 0,       # Temperature for creativity control
        "topP": 1               # Top-P sampling value
    }
}

# Convert payload to JSON string for the body field
body = json.dumps(payload)

# Specify the model ID for Amazon's Titan model
model_id = "amazon.titan-text-express-v1"

# Invoke the model and print the response
try:
    response = bedrock.invoke_model(
        modelId=model_id,
        body=body,
        accept="application/json",
        contentType="application/json"
    )

    # Process the response
    response_body = json.loads(response.get("body").read())
    response_text = response_body['results']  # Adjust based on response structure
    print(response_text)

except Exception as e:
    print("Error occurred:", e)
