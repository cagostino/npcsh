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

import markdown
import re
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter

import textwrap
from rich.console import Console
from rich.markdown import Markdown, CodeBlock

class LeftAlignedCodeBlock(CodeBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = (0, 0)  # Set left margin to 0

class LeftAlignedMarkdown(Markdown):
    elements = Markdown.elements.copy()
    elements['code_block'] = LeftAlignedCodeBlock

def render_markdown(text):
    console = Console()
    md = LeftAlignedMarkdown(text)
    console.print(md)

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

npcsh_model = os.environ.get("NPCSH_MODEL", "llama3.2")
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
    # print(provider, model)
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
        if messages is None or len(messages) == 0:
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
        # print("openai_messages", messages)
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
        if messages is None or len(messages) == 0:
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
        # print(messages)

        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_message,
            messages=messages[1:],
        )

        # print(message)
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
        model = "llama3.2"
    elif provider is None and model is not None:
        provider = lookupprovider(model)
    if provider == "ollama":
        if model is None:
            model = "llama3.2"
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


import matplotlib.pyplot as plt  # Import for showing plots
from IPython.display import (
    display,
)  # For displaying DataFrames in a more interactive way
import numpy as np
import pandas as pd


def load_data(file_path, name):
    dataframes[name] = pd.read_csv(file_path)
    print(f"Data loaded as '{name}'")


def execute_data_operations(
    query, command_history, dataframes, npc=None, db_path="~/npcsh_history.db"
):
    location = os.getcwd()
    db_path = os.path.expanduser(db_path)

    try:
        try:
            # Create a safe namespace for pandas execution
            namespace = {
                "pd": pd,
                "np": np,
                "plt": plt,
                **dataframes,  # This includes all our loaded dataframes
            }
            # Execute the query
            result = eval(query, namespace)

            # Handle the result
            if isinstance(result, (pd.DataFrame, pd.Series)):
                print(render_markdown(result))
                return result, "pd"
            elif isinstance(result, plt.Figure):
                plt.show()
                return result, "pd"
            elif result is not None:
                print(render_markdown(result))

                return result, "pd"

        except Exception as exec_error:
            print(f"Pandas Error: {exec_error}")

        # 2. Try SQL
        # print(db_path)
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                print(query)
                print(get_available_tables(db_path))

                cursor.execute(query)
                # get available tables

                result = cursor.fetchall()
                if result:
                    for row in result:
                        print(row)
                    return result, "sql"
        except Exception as e:
            print(f"SQL Error: {e}")

        # 3. Try R
        try:
            result = subprocess.run(
                ["Rscript", "-e", query], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(result.stdout)
                return result.stdout, "r"
            else:
                print(f"R Error: {result.stderr}")
        except Exception as e:
            pass

        # If all engines fail, ask the LLM
        print("Direct execution failed. Asking LLM for SQL query...")
        llm_prompt = f"""
        The user entered the following query which could not be executed directly using pandas, SQL, R, Scala, or PySpark:
        ```
        {query}
        ```

        The available tables in the SQLite database at {db_path} are:
        ```sql
        {get_available_tables(db_path)}
        ```

        Please provide a valid SQL query that accomplishes the user's intent.  If the query requires data from a file, provide instructions on how to load the data into a table first.
        Return only the SQL query, or instructions for loading data followed by the SQL query.
        """

        llm_response = get_llm_response(llm_prompt, npc=npc)

        print(f"LLM suggested SQL: {llm_response}")
        command = llm_response.get("response", "")
        if command == "":
            return "LLM did not provide a valid SQL query.", None
        # Execute the LLM-generated SQL
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(command)
                result = cursor.fetchall()
                if result:
                    for row in result:
                        print(row)
                    return result, "llm"
        except Exception as e:
            print(f"Error executing LLM-generated SQL: {e}")
            return f"Error executing LLM-generated SQL: {e}", None

    except Exception as e:
        print(f"Error executing query: {e}")
        return f"Error executing query: {e}", None


def get_available_tables(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'command_history'"
            )
            tables = cursor.fetchall()

            return tables
    except Exception as e:
        print(f"Error getting available tables: {e}")
        return ""

from typing import Optional, List, Dict, Any

def execute_llm_command(
    command: str,
    command_history: Any,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    npc: Optional[Any] = None,
    messages: Optional[List[Dict[str, str]]] = None,
retrieved_docs=None, n_docs = 5) -> str:
    max_attempts = 5
    attempt = 0
    subcommands = []
    npc_name = npc.name if npc else "sibiji"
    location = os.getcwd()
    print(f"{npc_name} generating command")
    # Create context from retrieved documents
    context = ""
    if retrieved_docs:
        for filename, content in retrieved_docs[:n_docs]:
            #print(f"Document: {filename}")
            #print(content)
            context+=f"Document: {filename}\n{content}\n\n"
        context = f"Refer to the following documents for context:\n{context}\n\n"
    while attempt < max_attempts:
        prompt = f"""
        A user submitted this query: {command}.
        You need to generate a bash command that will accomplish the user's intent.
        Respond ONLY with the command that should be executed.
        in the json key "bash_command".
        You must reply with valid json and nothing else. Do not include markdown formatting
        """
        if len(context) > 0:
            prompt += f"""
            What follows is the context of the text files in the user's directory that are potentially relevant to their request
            Use these to help inform your decision.
            {context}
            """
        
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            messages=[],
            npc=npc,
            format="json",
        )
        
        if messages is not None:
            messages = response.get("messages", [])
        
        llm_response = response.get("response", {})
        #print(f"LLM response type: {type(llm_response)}")
        #print(f"LLM response: {llm_response}")

        try:
            if isinstance(llm_response, str):
                llm_response = json.loads(llm_response)
            
            if isinstance(llm_response, dict) and "bash_command" in llm_response:
                bash_command = llm_response["bash_command"]
            else:
                raise ValueError("Invalid response format from LLM")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            attempt += 1
            continue

        print(f"LLM suggests the following bash command: {bash_command}")
        subcommands.append(bash_command)

        try:
            print(f"Running command: {bash_command}")
            result = subprocess.run(
                bash_command, shell=True, text=True, capture_output=True, check=True
            )
            print(f"Command executed with output: {result.stdout}")
            
            prompt = f"""
                Here was the output of the result for the {command} inquiry  
                which ran this bash command {bash_command}:

                {result.stdout}

                Provide a simple response to the user that explains to them
                what you did and how it accomplishes what they asked for. 
                """
            if len(context) > 0:
                prompt += f"""
                What follows is the context of the text files in the user's directory that are potentially relevant to their request
                Use these to help inform how you respond. 
                You must read the context and use it to provide the user with a more helpful answer related to their specific text data.

                CONTEXT:

                {context}
                """
            
            response = get_llm_response(
                prompt,
                model=model,
                provider=provider,
                npc=npc,
                messages=[],
            )
            
            if messages is not None:
                messages = response.get("messages", [])
            output = response.get("response", "")

            print(render_markdown(output))
            command_history.add(command, subcommands, output, location)

            return output

        except subprocess.CalledProcessError as e:
            print(f"Command failed with error:")
            print(e.stderr)

            error_prompt = f"""
            The command '{bash_command}' failed with the following error:
            {e.stderr}
            Please suggest a fix or an alternative command.
            Respond with a JSON object containing the key "bash_command" with the suggested command.
            Do not include any additional markdown formatting.
            
            """

            if len(context) > 0: 
                error_prompt+=f"""           
                    What follows is the context of the text files in the user's directory that are potentially relevant to their request
                    Use these to help inform your decision.
                    {context}
                    """


            
            fix_suggestion = get_llm_response(
                error_prompt,
                model=model,
                provider=provider,
                npc=npc,
                format="json",
                messages=messages,
            )
            
            if messages is not None:
                messages = fix_suggestion.get("messages", [])
            
            fix_suggestion_response = fix_suggestion.get("response", {})
            
            try:
                if isinstance(fix_suggestion_response, str):
                    fix_suggestion_response = json.loads(fix_suggestion_response)
                
                if isinstance(fix_suggestion_response, dict) and "bash_command" in fix_suggestion_response:
                    print(f"LLM suggests fix: {fix_suggestion_response['bash_command']}")
                    command = fix_suggestion_response["bash_command"]
                else:
                    raise ValueError("Invalid response format from LLM for fix suggestion")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing LLM fix suggestion: {e}")

        attempt += 1

    command_history.add(command, subcommands, "Execution failed", location)
    return "Max attempts reached. Unable to execute the command successfully."

def check_llm_command(
    command,
    command_history,
    model=npcsh_model,
    messages=None,
    provider=npcsh_provider,
    npc=None,
    retrieved_docs=None,
    n_docs=5,
):
    location = os.getcwd()

    if messages is None:
        messages = []
    # Create context from retrieved documents
    context = ""
    if retrieved_docs:
        for filename, content in retrieved_docs[:n_docs]:
            #print(f"Document: {filename}")
            #print(content)
            context+=f"Document: {filename}\n{content}\n\n"
        context = f"Refer to the following documents for context:\n{context}\n\n"
        


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

    You must reply with valid json and nothing else. Do not include any markdown formatting.
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
    if len(context) > 0:
        prompt += f"""
        What follows is the context of the text files in the user's directory that are potentially relevant to their request
        Use these to help inform your decision.
        {context}
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
    # print(response)
    if messages is not None:
        messages = response.get("messages", None)
    response = response.get("response", None)

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
    command_history.add(command, [], output, location)

    if response["is_command"] == "yes":
        return execute_llm_command(
            command,
            command_history,
            model=model,
            provider=provider,
            messages=messages,
            npc=npc,
            retrieved_docs=retrieved_docs,
            
        )
    else:
        return execute_llm_question(
            command,
            command_history,
            model=model,
            provider=provider,
            # messages=messages,
            npc=npc,
            retrieved_docs=retrieved_docs,
        )


def execute_llm_question(
    command,
    command_history,
    model=npcsh_model,
    provider=npcsh_provider,
    npc=None,
    messages=None,
    retrieved_docs=None,
    n_docs=5
):
    location = os.getcwd()
    context=""
    if retrieved_docs:
        for filename, content in retrieved_docs[:n_docs]:
            #print(f"Document: {filename}")
            #print(content)
            context+=f"Document: {filename}\n{content}\n\n"
        context = f"Refer to the following documents for context:\n{context}\n\n"
        command = f"""{command}\n       
        What follows is the context of the text files in the user's directory that are potentially relevant to their request
        Use these to help inform your decision.

        CONTEXT:
    {context}"""

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
    print(render_markdown(output["response"]))
    command_history.add(command, [], output, location)
    return response  # return the full conversation
