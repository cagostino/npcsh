import subprocess
import requests
import os
import json
import ollama
import sqlite3
import pandas as pd
import openai
from dotenv import load_dotenv
from openai import OpenAI
import anthropic


# Load environment variables from .env file
def load_env_from_execution_dir():
    # Get the directory where the script is being executed
    execution_dir = os.path.abspath(os.getcwd())
    # print(f"Execution directory: {execution_dir}")
    # Construct the path to the .env file
    env_path = os.path.join(execution_dir, ".env")

    # Load the .env file if it exists
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded .env file from {execution_dir}")
    else:
        print(f"Warning: No .env file found in {execution_dir}")


load_env_from_execution_dir()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
openai_api_key = os.getenv("OPENAI_API_KEY", None)

npcsh_model = os.environ.get("NPCSH_MODEL", "phi3")
npcsh_provider = os.environ.get("NPCSH_PROVIDER", "ollama")
npcsh_db_path = os.path.expanduser(
    os.environ.get("NPCSH_DB_PATH", "~/npcsh_history.db")
)


def get_system_message(npc):
    system_message = f"""
    .
    ..
    ...
    ....
    .....
    ......
    .......
    ........
    .........
    ..........
    Hello!
    Welcome to the team.
    You are an NPC working as part of our team. 
    You are the {npc.name} NPC with the following primary directive: {npc.primary_directive}.
    In some cases, users may request insights into data contained in a local database.
    For these purposes, you may use any data contained within these sql tables 
    {npc.tables}        
    which are contained in the database at {npcsh_db_path}.
    
    So if you need to obtain data you may use sqlite3 and write queries
    to obtain the data you need. 
    
    When formulating SQLite queries:

        1. Remember that SQLite doesn't support TOP. Use LIMIT instead for selecting a limited number of rows.
        2. To get the last row, use: "SELECT * FROM table_name ORDER BY rowid DESC LIMIT 1;"
        3. To check if a table exists, use: "SELECT name FROM sqlite_master WHERE type='table' AND name='table_name';"
        4. Always use single quotes for string literals in SQLite queries.
        5. When executing SQLite commands from bash, ensure proper escaping of quotes.

        For bash commands interacting with SQLite:
        1. Use this format: sqlite3 /path/to/database.db 'SQL query here'
        2. Example: sqlite3 /home/caug/npcsh_history.db 'SELECT * FROM command_history ORDER BY rowid DESC LIMIT 1;'

        When encountering errors:
        1. "no such function: TOP" - Remember to use LIMIT instead.
        2. "syntax error" - Double-check quote usage and escaping in bash commands.
        3. "no such table" - Verify the table name and check if it exists first.

        Always consider data integrity, security, and privacy in your operations. Offer clear explanations and examples for complex data concepts and SQLite queries.

    
    --------------
    --------------
    
    """

    return system_message


def get_ollama_conversation(messages, model, npc=None):
    messages_copy = messages.copy()
    if messages_copy[0]["role"] != "system":
        if npc is not None:
            system_message = get_system_message(npc)
            messages_copy.insert(0, {"role": "system", "content": system_message})

    response = ollama.chat(model=model, messages=messages_copy)
    messages_copy.append(response["message"])
    return messages_copy


def get_openai_conversation(messages, model, npc=None, api_key=None, **kwargs):
    try:
        if api_key is None:
            api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)

        system_message = (
            get_system_message(npc) if npc else "You are a helpful assistant."
        )

        # If no messages are provided, start a new conversation
        if messages is None or len(messages) == 0:
            messages = [{"role": "system", "content": system_message}]

        # Extract the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        if last_user_message is None:
            raise ValueError("No user message found in the conversation history.")

        # messages.append({"role": "user", "content": last_user_message})

        completion = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

        response_message = completion.choices[0].message

        messages.append({"role": "assistant", "content": response_message.content})

        return messages

    except Exception as e:
        return f"Error interacting with OpenAI: {e}"


def get_anthropic_conversation(messages, model, npc=None, api_key=None, **kwargs):
    try:
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY", None)
        system_message = get_system_message(npc) if npc else ""
        client = anthropic.Anthropic(api_key=api_key)

        # If no messages are provided, start a new conversation
        if messages is None or len(messages) == 0:
            messages = [{"role": "system", "content": system_message}]

        # Extract the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        if last_user_message is None:
            raise ValueError("No user message found in the conversation history.")

        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_message,  # Include system message in each turn for Anthropic
            messages=[
                {"role": "user", "content": last_user_message}
            ],  # Send only the last user message
            **kwargs,
        )

        messages.append({"role": "assistant", "content": message.content[0].text})

        return messages

    except Exception as e:
        return f"Error interacting with Anthropic: {e}"


def get_conversation(
    messages, provider=npcsh_provider, model=npcsh_model, npc=None, **kwargs
):
    print(provider, model)
    if provider == "ollama":
        return get_ollama_conversation(messages, model, npc=npc, **kwargs)
    elif provider == "openai":
        return get_openai_conversation(messages, model, npc=npc, **kwargs)
    elif provider == "anthropic":
        return get_anthropic_conversation(messages, model, npc=npc, **kwargs)
    else:
        return "Error: Invalid provider specified."


def debug_loop(prompt, error, model):
    response = get_ollama_response(prompt, model)
    print(response)
    if error in response:
        print(response[error])
        return True
    return False


def get_data_response(
    request,
    db_conn,
    tables=None,
    n_try_freq=5,
    extra_context=None,
    history=None,
    npc=None,
):
    prompt = f"""
            Here is a request from a user: 

            ```
            {request}
            ```
            
            You need to satisfy their request. 
            
            """

    if tables is not None:
        prompt += f"""
        Here are the tables you have access to: 
        {tables}
        """
    if extra_context is not None:
        prompt += f"""
        {extra_context}
        """
    if history:
        prompt += f"""
        The history of the queries that have been run so far is:
        {history}
        """

    prompt += f"""

            Please either write :
            1) an SQL query that will return the requested data.
            or 
            2) a query that will provide you more information so that you can answer the request.

            Return the query and the choice you made, i.e. whether you chose 1 or 2.
            Please return a json with the following keys: "query" and "choice".

            This is a description of the types and requirements for the outputs.
            {{
                "query": {"type": "string", 
                          "description": "a valid SQL query that will accomplish the task"},
                "choice": {"type":"int", 
                           "enum":[1,2]}
                "choice_explanation": {"type": "string", 
                                        "description": "a brief explanation of why you chose 1 or 2"}
            }}
            Do not return any extra information. Respond only with the json.
            """
    llm_response = get_llm_response(prompt, format="json", npc=npc)

    list_failures = []
    success = False
    n_tries = 0
    while not success:
        response_json = process_data_output(
            llm_response,
            db_conn,
            request,
            tables=tables,
            history=list_failures,
            npc=npc,
        )
        if response_json["code"] == 200:
            return response_json["response"]

        else:
            list_failures.append(response_json["response"])
        n_tries += 1
        if n_tries % n_try_freq == 0:
            print(
                f"The attempt to obtain the data has failed {n_tries} times. Do you want to continue?"
            )
            user_input = input('Press enter to continue or enter "n" to stop: ')
            if user_input.lower() == "n":
                return list_failures

    return llm_response


def check_output_sufficient(
    request, response, query, model=None, provider=None, npc=None
):
    prompt = f"""

            A user made this request:
                            ```
                {request}
                    ```
            You have extracted a result from the database using the following query:
                            ```
                {query}
                    ``` 
            Here is the result:
                            ```
                {response.head(),
                    response.describe()}
                    ``` 
            Is this result sufficient to answer the user's request?

            Return a json with the following
            key: 'IS_SUFFICIENT' 
            Here is a description of the types and requirements for the output.
                            ```
                {{
                    "IS_SUFFICIENT": {"type": "bool", 
                                        "description": "whether the result is sufficient to answer the user's request"}
                }}
                            ```   
            
            
                                                        
            """
    llm_response = get_llm_response(
        prompt, format="json", model=model, provider=provider, npc=npc
    )
    if llm_response["IS_SUFFICIENT"] == True:
        return response
    else:
        return False


def process_data_output(
    llm_response, db_conn, request, tables=None, n_try_freq=5, history=None, npc=None
):
    if llm_response["choice"] == 1:
        query = llm_response["query"]
        try:
            response = db_conn.execute(query).fetchall()
            # make into pandas
            response = pd.DataFrame(response)
            result = check_output_sufficient(request, response, query)
            if result:
                return {"response": response, "code": 200}
            else:
                output = {
                    "response": f"""The result in the response : ```{response.head()} ``` 
                        was not sufficient to answer the user's request. """,
                    "code": 400,
                }
                return output

            return response
        except Exception as e:
            return {"response": f"Error executing query: {str(e)}", "code": 400}
    elif llm_response["choice"] == 2:
        if "query" in llm_response:
            query = llm_response["query"]
            try:
                response = db_conn.execute(query).fetchall()
                # make into pandas
                response = pd.DataFrame(response)
                extra_context = f"""


                You indicated that you would need to run 
                the following query to provide a more complete response:
                        ```
                    {query}
                        ``` 
                Here is the result:
                                ```
                    {response.head(),
                        response.describe()}
                        ``` 
                                                            
                """
                if history is not None:
                    extra_context += f"""
                    The history of the queries that have been run so far is:
                    {history}
                    """

                llm_response = get_data_response(
                    request,
                    db_conn,
                    tables=tables,
                    extra_context=extra_context,
                    n_try_freq=n_try_freq,
                    history=history,
                )
                return {"response": llm_response, "code": 200}
            except Exception as e:
                return {"response": f"Error executing query: {str(e)}", "code": 400}
        else:
            return {"response": "Error: Missing query in response", "code": 400}
    else:
        return {"response": "Error: Invalid choice in response", "code": 400}


"""

messages = [
{"role": "user", "content": "hows it going"}]
model='phi3'
response = ollama.chat(model=model, messages=messages)

model = "llama3.1"
messages =   get_ollama_conversation(messages, model)

ea.append({"role": "user", "content": "then can you help me design something really spectacular?"})


"""
import base64  # For image encoding


def get_ollama_response(
    prompt, model, image=None, npc=None, format=None, messages=None, **kwargs
):
    try:
        url = "http://localhost:11434/api/generate"

        system_message = get_system_message(npc) if npc else ""
        full_prompt = f"{system_message}\n\n{prompt}"
        data = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
        }
        if image:  # Add image to the request if provided
            with open(image["file_path"], "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                data["image"] = base64_image

        if format is not None:
            data["format"] = format

        response = requests.post(url, json=data)
        response.raise_for_status()
        llm_response = json.loads(response.text)["response"]
        items_to_return = {"response": llm_response}
        if messages is not None:
            messages.append({"role": "assistant", "content": llm_response})
            items_to_return["messages"] = messages

        if format == "json":
            try:
                items_to_return["response"] = json.loads(llm_response)
                return items_to_return
            except json.JSONDecodeError:
                print(f"Warning: Expected JSON response, but received: {llm_response}")
                return {"error": "Invalid JSON response"}
        else:
            return items_to_return
    except Exception as e:
        return f"Error interacting with LLM: {e}"


def get_openai_response(
    prompt,
    model,
    image=None,
    npc=None,
    format=None,
    api_key=None,
    messages=None,
    **kwargs,
):
    try:
        if api_key is None:
            api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)

        system_message = (
            get_system_message(npc) if npc else "You are a helpful assistant."
        )
        if messages is None:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        if image:  # Add image if provided
            with open(image["file_path"], "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            }
                        ],
                    }
                )

        completion = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

        llm_response = completion.choices[0].message.content

        items_to_return = {"response": llm_response}

        if messages is not None:
            messages.append({"role": "assistant", "content": llm_response})
            items_to_return["messages"] = messages
        if format == "json":
            try:
                items_to_return["response"] = json.loads(llm_response)
                return items_to_return
            except json.JSONDecodeError:
                print(f"Warning: Expected JSON response, but received: {llm_response}")
                return {"error": "Invalid JSON response"}
        else:
            return items_to_return
    except Exception as e:
        return f"Error interacting with OpenAI: {e}"


def get_anthropic_response(
    prompt,
    model,
    image=None,
    npc=None,
    format=None,
    api_key=None,
    messages=None,
    **kwargs,
):
    try:
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY", None)
        system_message = (
            get_system_message(npc) if npc else "You are a helpful assistant."
        )
        client = anthropic.Anthropic(api_key=api_key)
        if messages is None:
            messages = []
            messages.append({"role": "system", "content": system_message})

        if image:
            with open(image["file_path"], "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",  # Or the appropriate media type
                                    "data": base64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,  # Include the prompt here with the image
                            },
                        ],
                    }
                )
        else:
            messages.append({"role": "user", "content": prompt})

        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_message,
            messages=messages[1:],
        )

        llm_response = message.content[0].text  # This is the AI's text response
        items_to_return = {"response": llm_response}

        if messages is not None:
            messages.append({"role": "assistant", "content": llm_response})
            items_to_return["messages"] = messages

        if format == "json":
            try:
                items_to_return["response"] = json.loads(llm_response)
                return items_to_return

            except json.JSONDecodeError:
                print(f"Warning: Expected JSON response, but received: {llm_response}")
                # If parsing fails, return the raw response wrapped in a dictionary
                return {"response": llm_response, "error": "Invalid JSON response"}
        else:
            return items_to_return

    except Exception as e:
        return f"Error interacting with Anthropic: {e}"


def lookupprovider(model):
    if model in [
        "phi3",
        "llama3.2",
        "llama3.1",
        "gemma2:9b",
    ]:  # replace with ollama attribute
        return "ollama"
    elif model in ["gpt-4o-mini", "gpt-4o", "gpt-4o"]:  # replace with openai attribute
        return "openai"
    elif model in ["claude-3.5-haiku", "claude-3.5"]:  # replace with claude attribute
        return "claude"


def get_llm_response(
    prompt,
    provider=npcsh_provider,
    model=npcsh_model,
    npc=None,
    messages=None,
    **kwargs,
):
    # print(provider)
    # print(model)
    if provider is None and model is None:
        provider = "ollama"
        model = "phi3"
    elif provider is None and model is not None:
        provider = lookupprovider(model)
    if provider == "ollama":
        if model is None:
            model = "phi3"
        return get_ollama_response(prompt, model, npc=npc, messages=messages, **kwargs)
    elif provider == "openai":
        if model is None:
            model = "gpt-4o-mini"
        return get_openai_response(prompt, model, npc=npc, messages=messages, **kwargs)
    elif provider == "anthropic":
        if model is None:
            model = "claude-3-haiku-20240307"
        return get_anthropic_response(
            prompt, model, npc=npc, messages=messages, **kwargs
        )
    else:
        return "Error: Invalid provider specified."


def execute_data_operations(query, command_history, model=None, provider=None):
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

    response = get_llm_response(prompt, model=model, provider=provider, format="json")
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


def execute_llm_command(
    command, command_history, model=None, provider=None, npc=None, messages=None
):
    max_attempts = 5
    attempt = 0
    subcommands = []
    if npc is None:
        npc_name = "sibiji"

    location = os.getcwd()
    print(f"{npc_name} generating command")
    while attempt < max_attempts:
        prompt = f"""
        A user submitted this query: {command}.
        You need to generate a bash command that will accomplish the user's intent.
        Respond ONLY with the command that should be executed.
        in the json key "bash_command".
        You must reply with valid json and nothing else.
        """

        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            messages=messages,
            npc=npc,
            format="json",
        )
        messages = response["messages"]
        response = response["response"]

        print(f"LLM suggests the following bash command: {response['bash_command']}")
        print(f"running command")
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

                    Provide a simple response to the user that explains to them
                    what you did and how it accomplishes what they asked for. 
                    

                        """
                response = get_llm_response(
                    prompt,
                    model=model,
                    provider=provider,
                    npc=npc,
                    messages=messages,
                )
                messages = response["messages"]
                response = response["response"]

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
            fix_suggestion = get_llm_response(
                error_prompt,
                model=model,
                provider=provider,
                npc=npc,
                format="json",
                messages=messages,
            )
            messages = fix_suggestion["messages"]
            fix_suggestion = fix_suggestion["response"]

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


def check_llm_command(
    command,
    command_history,
    model=npcsh_model,
    messages=None,
    provider=npcsh_provider,
    npc=None,
):
    location = os.getcwd()

    if messages is None:
        messages = []
    prompt = f"""
    A user submitted this query: {command}
    Is this query a specific request for a task to be accomplished?
      
    In considering how to answer this, consider whether it is 
        something that can be answered via a bash command on the users computer? 
        Assume that the user has access to internet 
        and basic command line tools like curl, wget, 
        etc so you are not limited to just the local machine.
        Additionally, the user may have access to a local database so
        you can use sqlite3 to query the database.

    If so, respond with a JSON object containing the key "is_command" with the value "yes".

    Provide an explanation in the json key "explanation" .

    You must reply with valid json and nothing else.
    The format and requirements of the output are as follows:
    {{
        "is_command": {{"type": "string", 
                       "enum": ["yes", "no"],
                        "description": "whether the query is a command"}},
        "explanation": {{"type": "string",
                        "description": "a brief explanation of why the query is or is not a command"}}
    }}
    The types of the outputs must strictly adhere to these requirements.
    Use these to hone your output and only respond with the actual filled-in json.
    Do not return any extra information. Respond only with the json.
    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        npc=npc,
        messages=messages,
        format="json",
    )
    # import pdb

    # pdb.set_trace()
    messages = response["messages"]
    response = response["response"]

    # Handle potential errors and non-JSON responses
    if not isinstance(response, dict):
        print(f"Error: Expected a dictionary, but got {type(response)}")
        return "Error: Invalid response from LLM"
    if "error" in response:
        print(f"LLM Error: {response['error']}")
        return f"LLM Error: {response['error']}"

    # Check for the 'is_command' key, handle cases where it's missing
    if "is_command" not in response:
        print("Error: 'is_command' key missing in LLM response")
        return "Error: 'is_command' key missing in LLM response"
    if response["is_command"] == "yes":
        cmd_stt = "a command"
    else:
        cmd_stt = "a question"

    print(f"The request is {cmd_stt} .")

    output = response
    command_history.add(command, [], json.dumps(output), location)

    if response["is_command"] == "yes":
        return execute_llm_command(
            command,
            command_history,
            model=model,
            provider=provider,
            messages=messages,
            npc=npc,
        )
    else:
        return execute_llm_question(
            command,
            command_history,
            model=model,
            provider=provider,
            # messages=messages,
            npc=npc,
        )


def execute_llm_question(
    command,
    command_history,
    model=npcsh_model,
    provider=npcsh_provider,
    npc=None,
    messages=None,
):
    location = os.getcwd()

    # Use get_conversation if messages are provided
    if messages:
        response = get_conversation(messages, model=model, provider=provider, npc=npc)

        if isinstance(response, str) and "Error" in response:  # Check for errors
            output = response
        elif (
            isinstance(response, list) and len(response) > 0
        ):  # Check for valid response
            output = response[-1]["content"]  # Extract assistant's reply
        else:
            output = "Error: Invalid response from conversation function"
        # print(response)

    else:  # Use get_llm_response for single turn queries
        response = get_llm_response(
            command,
            model=model,
            provider=provider,
            npc=npc,
            messages=messages,
        )
        # print(response["response"])
        output = response

    command_history.add(command, [], json.dumps(output), location)
    return response  # return the full conversation
