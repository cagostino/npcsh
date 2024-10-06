import subprocess
import requests
import os
import json
import ollama
import sqlite3
import pandas as pd

npcsh_model = os.environ.get("NPCSH_MODEL", "phi3")
npcsh_provider = os.environ.get("NPCSH_PROVIDER", "ollama")
npcsh_db_path = os.path.expanduser(os.environ.get("NPCSH_DB_PATH", "~/npcsh_history.db"))

def get_ollama_conversation(messages, model):
    messages_copy = messages.copy()
    response = ollama.chat(model=model, messages=messages)
    messages_copy.append(response["message"])
    return messages_copy


def debug_loop(prompt, error, model):
    response = get_ollama_response(prompt, model)
    print(response)
    if error in response:
        print(response[error])
        return True
    return False


def get_data_response(
    request, db_conn, tables=None, n_try_freq=5, extra_context=None, history=None, npc = None
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
    llm_response = get_llm_response(prompt, format="json", npc= npc)

    list_failures = []
    success = False
    n_tries = 0
    while not success:
        response_json = process_data_output(
            llm_response, db_conn, request, tables=tables, history=list_failures, 
            npc = npc
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


def check_output_sufficient(request, response, query, model=None, provider=None, npc = None):
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
        prompt, format="json", model=model, provider=provider, npc = npc
    )
    if llm_response["IS_SUFFICIENT"] == True:
        return response
    else:
        return False


def process_data_output(
    llm_response, db_conn, request, tables=None, n_try_freq=5, history=None, npc = None
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


def get_ollama_response(prompt, model, npc=None, format=None, **kwargs):
    try:
        url = "http://localhost:11434/api/generate"
        

        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if format is not None:
            data["format"] = format

        response = requests.post(url, json=data)
        response.raise_for_status()
        llm_response = json.loads(response.text)["response"]

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

def test_get_ollama_response():
    prompt = "This is a test prompt."
    model = "phi3"
    response = get_ollama_response(prompt, model)
    print(response)
    assert response is not None
    assert isinstance(response, str)

    prompt = """A user submitted this query: "SELECT * FROM table_name". 
    You need to generate a script that will accomplish the user\'s intent. 
    Respond ONLY with the procedure that should be executed. Place it in a JSON object with the key 
    "script_to_test".
    The format and requiremrents of the output are as follows:
    {
    "script_to_test": {"type": "string", 
    "description": "a valid SQL query that will accomplish the task"}
    }

    """
    model = "phi3"

    response = get_ollama_response(prompt, model, format="json")
    print(response)
    assert response is not None
    assert isinstance(response, dict)
    assert "script_to_test" in response


def get_openai_response(prompt, model, functional_requirements=None):
    pass


def get_claude_response(prompt, model, format=None):
    pass


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


def get_llm_response(prompt, provider=npcsh_provider, model=npcsh_model, npc=None, **kwargs):
    # Prepare the system message
    system_message = ""
    #print('NPC=', npc)
    if npc:
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
        # Prepare the full prompt
        full_prompt = f"{system_message}\n\n{prompt}" 
    else:
        full_prompt = prompt
        
    if provider is None and model is None:
        provider = "ollama"
        model = "phi3"
    elif provider is None and model is not None:
        provider = lookupprovider(model)
    if provider == "ollama":
        if model is None:
            model = "phi3"
        return get_ollama_response(full_prompt, model, npc=npc, **kwargs)
    elif provider == "openai":
        if model is None:
            model = "gpt-4o-mini"
        return get_openai_response(full_prompt, model, npc=npc, **kwargs)
    elif provider == "claude":
        if model is None:
            model = "claude-3.5-haiku"
        return get_claude_response(full_prompt, model, npc=npc, **kwargs)
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



def execute_llm_command(command, command_history, model=None, provider=None, npc=None):
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

        response = get_llm_response(
            prompt, model=model, provider=provider, npc=npc, format="json"
        )
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
                response = get_llm_response(
                    prompt,
                    model=model,
                    provider=provider,
                    npc = npc,

                )
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
                error_prompt, model=model, provider=provider, npc = npc,  format="json"
            )
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


def check_llm_command(command, command_history, model=None, provider=None, npc=None):
    location = os.getcwd()
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
    response = get_llm_response(prompt, model=model, provider=provider, npc=npc, format="json")
    print(f"LLM suggests: {response}")
    
    output = response
    command_history.add(command, [], json.dumps(output), location)
    if response["is_command"] == "yes":
        return execute_llm_command(
            command, command_history, model=model, provider=provider, npc=npc
        )
    else:
        return execute_llm_question(
            command, command_history, model=model, provider=provider, npc=npc
        )

def execute_llm_question(
    command, command_history, model=None, provider=None, npc=None, messages=None
):
    location = os.getcwd()

    prompt = f"""
    A user submitted this query: {command}
    You need to generate a response to the user's inquiry.
    Respond ONLY with the response that should be given.
    in the json key "response".
    You must reply with valid json and nothing else.
    """
    response = get_llm_response(prompt, model=model, provider=provider, npc=npc, format="json")
    print(f"LLM suggests: {response}")
    output = response["response"]
    command_history.add(command, [], output, location)
    return response["response"]
