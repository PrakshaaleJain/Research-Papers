import autogen
import google.generativeai as genai
import os
from dotenv import load_dotenv
from autogen.code_utils import  infer_lang


# Load environment variables
load_dotenv()

# Retrieve the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API key is missing! Check your .env file.")

# Setting up configuration for model and API key
config_model = [{
        "model" : "gemini-1.5-pro", # can use any other model by running models.py 
        "api_key" : GOOGLE_API_KEY,
        "api_type" : "google"
    }]

# Creating an assistant agent that codes the solutions and revises the code from the output given by userproxy.
assistant = autogen.AssistantAgent(
    name="assistant",
    # llm_config=gemini_config,  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
    llm_config={
        "seed" : 42,     # for reproducibility of results
        "config_list" : config_model,
        "cache_seed" : 41,
    },
)


# Creating a user-proxy that runs the code and return the output of the code to assistant
userproxy = autogen.UserProxyAgent(
    name="User_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,      # MAX_AUTO_REPLY can be changed based on the number of rounds of conv. between both the agents
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "Mechanics",
        "use_docker": False,  
    }
)


prompt = """A 1m-by-1m elastic plate has Young's modulus of 1GPa and Poisson ration of 0.3.
It has zero displacement on the left edge and 0.1m displacement along x direction on the right edge.
Solve for the displacement using FENICS and store the displacement result in a PNG file."""

chat_response = userproxy.initiate_chat(
    assistant,
    message = prompt,
)

chat_id = chat_response.chat_id
convs = chat_response.chat_history
total_cost = chat_response.cost[1]

print(f"CHAT REF: {chat_id}")
print(f"COST OF TRANSACTION: {total_cost}")

for conv in convs:
  content = conv['content']
  active_role = conv['role']

  print(f"{active_role}: {content}")


print("COMPLETE")



