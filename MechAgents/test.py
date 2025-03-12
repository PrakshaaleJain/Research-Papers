import autogen
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API key is missing! Check your .env file.")

# Setting up configuration for model and API key
config_model = [{
        "model": "gemini-1.5-pro",
        "api_key": GOOGLE_API_KEY,
        "api_type": "google"
    }]

# Creating an assistant agent that codes the solutions and revises the code from the output given by userproxy.
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "seed": 42,
        "config_list": config_model,
        "cache_seed": 41,
    },
)

# Creating a user-proxy that runs the code and provides feedback
userproxy = autogen.UserProxyAgent(
    name="User_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "Mechanics",
        "use_docker": False,
    }
)

# Initial Task
initial_task = (
    "A 1m-by-1m elastic plate has Young's modulus of 1GPa and Poisson ratio of 0.3. "
    "It has zero displacement on the left edge and 0.1m displacement along the x direction on the right edge. "
    "Solve for the displacement using FENICS and store the displacement result in a PNG file."
)

# User proxy asks the assistant to generate exactly 4 more progressive modifications (total 5 rounds)
question_prompt = (
    "Given the following initial FEniCS problem:\n\n"
    f"{initial_task}\n\n"
    "Generate exactly four additional progressive modifications that build upon this task. "
    "Each modification should introduce a meaningful change, including boundary condition adjustments, mesh refinement, geometry changes, stress calculations, and error corrections."
)

# Generate exactly 4 follow-up tasks dynamically
chat_response = userproxy.initiate_chat(assistant, message=question_prompt)

# Extract the generated follow-up tasks
generated_tasks = []
for conv in chat_response.chat_history:
    if conv["role"] == "assistant":
        generated_tasks.append(conv["content"])

# Ensure we only take 4 tasks (if more are generated, we truncate)
generated_tasks = generated_tasks[:4]

# Combine the initial task with generated tasks (ensuring 5 rounds)
all_tasks = [initial_task] + generated_tasks

# Run the iterative conversation for exactly 5 rounds
for round_num, task in enumerate(all_tasks[:5], start=1):  # Ensuring exactly 5 rounds
    print(f"\n--- ROUND {round_num} ------------------------------------------------------------------------------------------")
    chat_response = userproxy.initiate_chat(assistant, message=task)
    
    # Get chat history
    chat_id = chat_response.chat_id
    convs = chat_response.chat_history
    total_cost = chat_response.cost["usage_including_cached_inference"]["gemini-1.5-pro"]["cost"]

    print(f"CHAT REF: {chat_id}")
    print(f"COST OF TRANSACTION: {total_cost}")

    # Print conversation history
    for conv in convs:
        content = conv['content']
        active_role = conv['role']
        print(f"{active_role}: {content}")

print("\nCOMPLETE")
