import pandas as pd
import random
import os
from faker import Faker

fake = Faker()

import pandas as pd
import sqlite3
from datetime import datetime
from npcsh.npc_compiler import NPCCompiler, NPCSQLOperations, NPCDBTAdapter
from typing import List

import json
import os

import os
import sqlite3
import pandas as pd
from pathlib import Path


import sqlite3
import pandas as pd
import re
import os
import re
import random
import sqlite3
import pandas as pd
from faker import Faker
from collections import defaultdict, deque
from npcsh.npc_compiler import NPCSQLOperations


def extract_variables(input_string):
    """
    Extract variable names from a string that are enclosed in curly braces.

    :param input_string: The string to extract variables from.
    :return: A list of extracted variable names.
    """
    # Regular expression to match content within curly braces
    pattern = r"\{([^}]+)\}"

    # Find all matches using re.findall
    matches = re.findall(pattern, input_string)

    # Return the matches as a list
    return matches


def load_csv_to_db(csv_path, table_name, db_path):
    conn = sqlite3.connect(db_path)
    try:
        print(f"Loading {csv_path} into table {table_name}")
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Table {table_name} successfully created in the database.")
    except Exception as e:
        print(f"Failed to load {csv_path} into table {table_name}: {e}")
    finally:
        conn.close()


# Initialize Faker for synthetic data generation
fake = Faker()


def generate_csv(schema, num_rows, output_path):
    """
    Generate a CSV file with synthetic data based on a schema.

    :param schema: List of dictionaries defining column names and types.
    :param num_rows: Number of rows to generate.
    :param output_path: Path to save the generated CSV.
    """
    data = {}
    for column in schema:
        name, col_type = column["name"], column["type"]
        if col_type == "TEXT":
            data[name] = [fake.text(max_nb_chars=20) for _ in range(num_rows)]
        elif col_type == "INTEGER":
            data[name] = [random.randint(1, 1000) for _ in range(num_rows)]
        elif col_type == "TIMESTAMP":
            data[name] = [fake.date_time_this_decade() for _ in range(num_rows)]
        elif col_type == "FLOAT":
            data[name] = [random.uniform(0.0, 1000.0) for _ in range(num_rows)]
        else:
            raise ValueError(f"Unsupported type: {col_type}")

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated CSV at: {output_path}")


def generate_all_csvs(schemas, output_directory, num_rows=100):
    """
    Generate all CSVs based on predefined schemas.

    :param schemas: Dictionary of schemas.
    :param output_directory: Directory to save the generated CSV files.
    :param num_rows: Number of rows to generate for each CSV.
    """
    for table_name, schema in schemas.items():
        output_path = os.path.join(output_directory, f"{table_name}.csv")
        generate_csv(schema, num_rows, output_path)


def load_csv_to_db(csv_path, table_name, db_path):
    """
    Load CSV data into a SQLite database.

    :param csv_path: Path to the CSV file.
    :param table_name: The table name in the database.
    :param db_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    try:
        print(f"Loading {csv_path} into table {table_name}")
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Table {table_name} successfully created in the database.")
    except Exception as e:
        print(f"Failed to load {csv_path} into table {table_name}: {e}")
    finally:
        conn.close()


# Define schemas for tables
schemas = {
    "raw_customer_feedback": [
        {"name": "feedback", "type": "TEXT"},
        {"name": "customer_id", "type": "INTEGER"},
        {"name": "timestamp", "type": "TIMESTAMP"},
    ],
    "raw_orders": [
        {"name": "order_id", "type": "INTEGER"},
        {"name": "product_name", "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
        {"name": "order_date", "type": "TIMESTAMP"},
    ],
}
import pandas as pd
import sqlite3
import os


class SQLCompiler:
    def __init__(self, models_directory):
        self.models_directory = models_directory

    def compile(self, sql_file):
        if not sql_file.endswith(".sql"):
            raise ValueError("File must have .sql extension")
        sql_file = os.path.abspath(sql_file)

        try:
            with open(sql_file, "r") as file:
                sql_content = file.read()
            return self.resolve_model_references(sql_content)
        except Exception as e:
            raise ValueError(f"Error processing SQL file: {str(e)}")

    def resolve_model_references(self, sql_content):
        ref_pattern = r"ref\(['\"]([\w_]+)['\"]\)"

        def replace_ref(match):
            model_name = match.group(1)
            return model_name  # Use the model name directly as the table name

        return re.sub(ref_pattern, replace_ref, sql_content)


class ModelRunner:
    def __init__(self, models_directory, sql_compiler, db_path, npcsql):
        self.models_directory = models_directory
        self.db_path = db_path
        self.sql_compiler = sql_compiler
        self.npcsql = npcsql

    def build_dependency_graph(self):
        dependency_graph = {}
        model_files = [
            f for f in os.listdir(self.models_directory) if f.endswith(".sql")
        ]

        for model_file in model_files:
            model_name = os.path.splitext(model_file)[0]
            dependency_graph[model_name] = []
            model_path = os.path.join(self.models_directory, model_file)
            with open(model_path, "r") as f:
                sql = f.read()

            refs = self.extract_refs(sql)
            dependency_graph[model_name].extend(refs)

        return dependency_graph

    def topological_sort(self, dependency_graph):
        in_degree = defaultdict(int)
        for model, dependencies in dependency_graph.items():
            in_degree[model]
            for dep in dependencies:
                in_degree[dep] += 1

        queue = deque([model for model in in_degree if in_degree[model] == 0])
        execution_order = []

        while queue:
            model = queue.popleft()
            execution_order.append(model)

            for dep in dependency_graph.get(model, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(execution_order) != len(in_degree):
            raise ValueError("Cyclic dependency detected in the models.")

        return execution_order

    def execute_model(self, model_name):
        model_file = os.path.join(self.models_directory, f"{model_name}.sql")
        compiled_sql = self.sql_compiler.compile(model_file)

        if "synthesize(" in compiled_sql or "spread_and_sync(" in compiled_sql:
            print(f"Running NPC operations for model: {model_name}")
            self.execute_npc_operations(compiled_sql)
        else:
            print(f"Executing SQL execution result for model: {model_name}:")
            self.execute_sql(compiled_sql)

    def extract_refs(self, sql):
        ref_pattern = r"ref\(['\"]([\w_]+)['\"]\)"
        return re.findall(ref_pattern, sql)

    def execute_sql(self, sql_query):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        conn.commit()
        print("SQL executed successfully.")

    def execute_npc_operations(self, sql_query):
        table_info = sql_query.split("FROM")[1].split("GROUP BY")[0].strip()
        args = sql_query.split("synthesize(")[1].split(")")[0].split(",")

        task = args[0].strip()
        columns = extract_variables(args[0].strip().strip('"'))
        npc = args[1].strip().replace("'", "")
        context = args[2].strip().replace("'", "")
        framework = args[3].strip().replace("'", "") if len(args) > 3 else "default"

        cols_string = ", ".join(columns)
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f"SELECT {cols_string} FROM {table_info}", conn)

        result_df = self.npcsql.synthesize(
            task,
            df=df,
            columns=columns,
            npc=npc,
            context=context,
            framework=framework,
        )
        print(result_df)

    def run_all_models(self):
        dependency_graph = self.build_dependency_graph()
        execution_order = self.topological_sort(dependency_graph)

        for model in execution_order:
            self.execute_model(model)


def main():
    # Define your CSV schemas here
    schemas = {
        "raw_customer_feedback": [
            {"name": "feedback", "type": "TEXT"},
            {"name": "customer_id", "type": "INTEGER"},
            {"name": "timestamp", "type": "TIMESTAMP"},
        ],
        "raw_orders": [
            {"name": "order_id", "type": "INTEGER"},
            {"name": "product_name", "type": "TEXT"},
            {"name": "price", "type": "FLOAT"},
            {"name": "order_date", "type": "TIMESTAMP"},
        ],
    }

    output_directory = "data"
    models_directory = "models"
    db_path = os.path.expanduser("~/npcsh_history.db")

    # Generate CSVs
    generate_all_csvs(schemas, output_directory)

    # Load CSVs into the database
    for table_name in schemas.keys():
        load_csv_to_db(f"{output_directory}/{table_name}.csv", table_name, db_path)

    # Initialize the NPC operations
    npcsql = NPCSQLOperations(npc_directory="./npc_team", db_path=db_path)

    # Initialize the SQL compiler and model runner
    sql_compiler = SQLCompiler(models_directory)
    runner = ModelRunner(models_directory, sql_compiler, db_path, npcsql)

    # Run all models
    runner.run_all_models()


if __name__ == "__main__":
    main()
