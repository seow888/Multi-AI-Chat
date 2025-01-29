import requests
import json

def call_ollama_api(user_prompt, system_prompt="You are a helpful assistant", model="deepseek-r1:1.5b"):
    """
    Makes an API call to a local Ollama LLM instance.

    Args:
        user_prompt (str): The prompt from the user.
        system_prompt (str, optional): The system prompt to guide the LLM. Defaults to None.
        model (str, optional): The Ollama model to use. Defaults to "llama2".

    Returns:
        str: The generated text response from the Ollama model, or None if there was an error.
    """
    api_url = "http://localhost:11434/api/chat" # ≡ƒæê Using the /api/chat endpoint!

    headers = {'Content-Type': 'application/json'}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt}) # System prompt added here!
    messages.append({"role": "user", "content": user_prompt})       # User prompt is always added!

    data = {
        "model": model,
        "messages": messages,
        "stream": False # Set to True if you want streaming responses
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        json_response = response.json()
        # For /api/chat endpoint, the response is in 'message' -> 'content'
        return json_response['message']['content']

    except requests.exceptions.RequestException as e:
        print(f"*Error calling Ollama API:* {e}")
        return None
    except json.JSONDecodeError:
        print("*Error decoding JSON response from Ollama.*")
        return None
    except KeyError:
        print("*Error accessing 'message' -> 'content' in JSON response.*")
        return None

if __name__ == "__main__":
    user_query = "What is the capital of France?"
    system_instruction = "You are a helpful geography expert. Provide concise answers." # Example system prompt!

    ollama_response = call_ollama_api(user_query, system_prompt=system_instruction)

    if ollama_response:
        print("*User Query:*", user_query)
        print("*System Instruction:*", system_instruction)
        print("*Ollama Response:*")
        print(ollama_response)
    else:
        print("*Failed to get response from Ollama.*")


    print("\n--- *Example without System Prompt* ---")
    user_query_no_system = "Tell me a joke."
    ollama_response_no_system = call_ollama_api(user_query_no_system)

    if ollama_response_no_system:
        print("*User Query:*", user_query_no_system)
        print("*Ollama Response:*")
        print(ollama_response_no_system)
    else:
        print("*Failed to get response from Ollama.*")
