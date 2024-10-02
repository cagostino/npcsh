import subprocess
import requests
import os
import json
import ollama
import sqlite3

npcsh_model = os.environ.get("NPCSH_MODEL", "phi3")
npcsh_provider = os.environ.get("NPCSH_PROVIDER", "ollama")


def get_ollama_conversation(messages, model):
    response = ollama.chat(model=model, messages=messages)
    messages.append(response)
    # print(response["message"])
    return messages


"""
test_messages = [
{"role": "user", "content": "hows it going"}]
model = "llama3.1"
ea =   get_ollama_conversion(ea, model)

ea.append({"role": "user", "content": "then can you help me design something really spectacular?"})


"""


def get_ollama_response(prompt, model, format=None, **kwargs):
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if format is not None:
            data["format"] = format

        # print(f"Requesting LLM response for prompt: {prompt}")
        response = requests.post(url, json=data)
        response.raise_for_status()
        llm_response = json.loads(response.text)["response"]

        # If format is JSON, try to parse the response as JSON
        if format == "json":
            try:
                return json.loads(llm_response)
            except json.JSONDecodeError:
                print(f"Warning: Expected JSON response, but received: {llm_response}")
                return {"error": "Invalid JSON response"}
        else:
            return llm_response
    except Exception as e:
        return f"Error interacting with LLM: {e}"


def get_openai_response(prompt, model, functional_requirements=None):
    pass


def get_claude_response(prompt, model, format=None):
    pass


def get_llm_response(prompt, provider=npcsh_provider, model=npcsh_model, **kwargs):
    if provider == "ollama":
        return get_ollama_response(prompt, model, **kwargs)
    elif provider == "openai":
        return get_openai_response(prompt, model, **kwargs)
    elif provider == "claude":
        return get_claude_response(prompt, model, **kwargs)
    else:
        return "Error: Invalid provider specified."


def execute_data_operations(query, command_history):
    location = os.getcwd()
    prompt = f"""
    A user submitted this query: {query}
    You need to generate a script using python, R, or SQL that will accomplish the user's intent.
    
    Respond ONLY with the procedure that should be executed.

    Here are some examples:
    {{"data_operation": "<sql query>", 'engine': 'SQL'}}
    {{'data_operation': '<python script>', 'engine': 'PYTHON'}}
    {{'data_operation': '<r script>', 'engine': 'R'}} 

    You must reply with only ONE output.
    """

    response = get_llm_response(prompt, format="json")
    output = response
    command_history.add(query, [], json.dumps(output), location)
    print(response)

    if response["engine"] == "SQL":
        db_path = os.path.expanduser("~/.npcsh_history.db")
        query = response["data_operation"]
        try:
            print(f"Executing query in SQLite database: {query}")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                result = cursor.fetchall()
                for row in result:
                    print(row)
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        except Exception as e:
            print(f"Error executing query: {e}")
    elif response["engine"] == "PYTHON":
        engine = "python"
        script = response["data_operation"]
        try:
            result = subprocess.run(
                f"echo '{script}' | {engine}",
                shell=True,
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error executing script: {result.stderr}")
        except Exception as e:
            print(f"Error executing script: {e}")
    elif response["engine"] == "R":
        engine = "Rscript"
        script = response["data_operation"]
        try:
            result = subprocess.run(
                f"echo '{script}' | {engine}",
                shell=True,
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error executing script: {result.stderr}")
        except Exception as e:
            print(f"Error executing script: {e}")
    else:
        print("Error: Invalid engine specified.")

    return response


def execute_llm_command(command, command_history):
    max_attempts = 5
    attempt = 0
    subcommands = []

    location = os.getcwd()
    while attempt < max_attempts:
        prompt = f"""
        A user submitted this query: {command}.
        You need to generate a bash command that will accomplish the user's intent.
        Respond ONLY with the command that should be executed.
        in the json key "bash_command".
        You must reply with valid json and nothing else.
        """

        response = get_llm_response(prompt, format="json")
        print(f"LLM suggests: {response}")

        if isinstance(response, dict) and "bash_command" in response:
            bash_command = response["bash_command"]
        else:
            print("Error: Invalid response format from LLM")
            attempt += 1
            continue

        try:
            subcommands.append(bash_command)
            result = subprocess.run(
                bash_command, shell=True, text=True, capture_output=True
            )
            if result.returncode == 0:
                # simplify the output
                prompt = f"""
                    Here was the output of the result for the {command} inquiry  
                    which ran this bash command {bash_command}:

                    {result.stdout}

                    Provide a simple short description that provides the most
                    useful information about the command's result.

                        """
                response = get_llm_response(prompt)
                print(response)
                output = response
                command_history.add(command, subcommands, output, location)

                return response
            else:
                print(f"Command failed with error:")
                print(result.stderr)

            error_prompt = f"""
            The command '{bash_command}' failed with the following error:
            {result.stderr}
            Please suggest a fix or an alternative command.
            Respond with a JSON object containing the key "bash_command" with the suggested command.
            """
            fix_suggestion = get_llm_response(error_prompt, format="json")
            if isinstance(fix_suggestion, dict) and "bash_command" in fix_suggestion:
                print(f"LLM suggests fix: {fix_suggestion['bash_command']}")
                command = fix_suggestion["bash_command"]
            else:
                print("Error: Invalid response format from LLM for fix suggestion")
        except Exception as e:
            print(f"Error executing command: {e}")

        attempt += 1
    command_history.add(command, subcommands, "Execution failed", location)

    print("Max attempts reached. Unable to execute the command successfully.")
    return "Max attempts reached. Unable to execute the command successfully."


def execute_llm_question(command, command_history):
    location = os.getcwd()

    prompt = f"""
    A user submitted this query: {command}
    You need to generate a response to the user's inquiry.
    Respond ONLY with the response that should be given.
    in the json key "response".
    You must reply with valid json and nothing else.
    """
    response = get_llm_response(prompt, format="json")
    print(f"LLM suggests: {response}")
    output = response["response"]
    command_history.add(command, [], output, location)
    return response["response"]


def execute_llm_thought(command, command_history):
    location = os.getcwd()
    prompt = f"""
    A user submitted this query: {command} . 
    Please generate a response to the user's inquiry.
    Respond ONLY with the response that should be given.
    in the json key "response".
    You must reply with valid json and nothing else.
    """
    response = get_llm_response(prompt, format="json")
    print(f"LLM suggests: {response}")
    output = response["response"]
    command_history.add(command, [], output, location)
    return response["response"]


def check_llm_command(command, command_history):
    # check what kind of request the command is
    prompt = f"""
    A user submitted this query: {command}
    What kind of request is this? 
    [Command, Thought]
    Thoughts are simply musings or ideas.
    Commands are requests for a specific action to be performed.
    
    Most requests are likely to be commands so only use the other options if
    the request is clearly not a command.


    Return your response in valid json key "request_type". 
    
    Provide an explanation in key "explanation".
    Your answer must be like
    
    {{"request_type": "Command", 
        "explanation": "This is a command because..."}}
    or
    {{"request_type": "Thought", 
        "explanation": "This is a thought because..."}}
    """

    response = get_llm_response(prompt, format="json")
    correct_request_type = response["request_type"]
    print(f"LLM initially suggests: {response}")
    # does the user agree with the request type? y/n

    if correct_request_type == "Command":
        return execute_llm_command(command, command_history)
    elif correct_request_type == "Question":
        return execute_llm_question(command, command_history)
    elif correct_request_type == "Thought":
        return execute_llm_thought(command, command_history)
