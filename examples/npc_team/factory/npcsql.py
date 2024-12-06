import os
import re
import sqlite3
import pandas as pd
from typing import Dict, List, Set, Union
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime


def extract_variables(query_template: str) -> List[str]:
    """Extract variable names from a string that are enclosed in curly braces"""
    pattern = r"\{([^}]+)\}"
    return re.findall(pattern, query_template)


class AIFunctionParser:
    """Handles parsing and extraction of AI function calls from SQL"""

    @staticmethod
    def extract_function_params(sql: str) -> Dict[str, Dict]:
        """Extract AI function parameters from SQL"""
        ai_functions = {}

        # Pattern to match function calls like synthesize("text", "npc", "context")
        pattern = r"(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)"
        matches = re.finditer(pattern, sql)

        for match in matches:
            func_name = match.group(1)
            if func_name in ["synthesize", "spread_and_sync"]:
                params = match.group(2).split(",")
                ai_functions[func_name] = {
                    "query": params[0].strip().strip("\"'"),
                    "npc": params[1].strip().strip("\"'"),
                    "context": params[2].strip().strip("\"'"),
                }

        return ai_functions


import os
import re
import sqlite3
import pandas as pd
from typing import Dict, List, Set, Union
from pathlib import Path
from collections import defaultdict, deque


class SQLModel:
    def __init__(self, name: str, content: str, path: str):
        self.name = name
        self.content = content
        self.path = path
        self.dependencies = self._extract_dependencies()
        self.has_ai_function = self._check_ai_functions()
        self.ai_functions = self._extract_ai_functions()  # Store AI function data

    def _extract_dependencies(self) -> Set[str]:
        """Extract model dependencies using ref() calls"""
        pattern = r"\{\{\s*ref\(['\"]([^'\"]+)['\"]\)\s*\}\}"
        return set(re.findall(pattern, self.content))

    def _check_ai_functions(self) -> bool:
        """Check if the model contains AI function calls"""
        ai_functions = [
            "synthesize",
            "spread_and_sync",
            "delegate",
            "dilate",
            "erode",
            "strategize",
            "validate",
            "decompose",
            "criticize",
            "summarize",
            "advocate",
            "bootstrap",
            "resample",
            "mediate",
            "contrast",
            "reconcile",
            "integrate",
            "harmonize",
            "orchestrate",
        ]
        return any(func in self.content for func in ai_functions)

    def _extract_ai_functions(self) -> Dict[str, Dict]:
        """Extract all AI functions and their parameters from the SQL content"""
        ai_functions = {}
        pattern = r"(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)"  # To match function calls like synthesize("text", "npc", "context")
        matches = re.finditer(pattern, self.content)

        for match in matches:
            func_name = match.group(1)
            if func_name in [
                "synthesize",
                "spread_and_sync",
                "delegate",
                "dilate",
                "erode",
                "strategize",
                "validate",
                "decompose",
                "criticize",
                "summarize",
                "advocate",
                "bootstrap",
                "resample",
                "mediate",
                "contrast",
                "reconcile",
                "integrate",
                "harmonize",
                "orchestrate",
            ]:
                params = match.group(2).split(",")
                params = [
                    param.strip().strip("\"'") for param in params
                ]  # Clean up each parameter

                # Ensure we have at least 4 parameters (or fill with None)
                while len(params) < 4:
                    params.append(None)

                # Handle extra parameter for functions that require more than 4 arguments
                extra = params[4] if len(params) > 4 else None

                ai_functions[func_name] = {
                    "column": params[0],
                    "npc": params[1],
                    "query": params[2],
                    "context": params[3],
                    "extra": extra,
                }

        return ai_functions

    def _extract_dependencies(self) -> Set[str]:
        """Extract model dependencies using ref() calls"""
        pattern = r"\{\{\s*ref\(['\"]([^'\"]+)['\"]\)\s*\}\}"
        return set(re.findall(pattern, self.content))

    def _check_ai_functions(self) -> bool:
        """Check if the model contains AI function calls"""
        ai_functions = ["synthesize", "spread_and_sync"]
        return any(func in self.content for func in ai_functions)


class ModelCompiler:
    def __init__(self, models_dir: str, db_path: str):
        self.models_dir = Path(models_dir)
        self.db_path = db_path
        self.models: Dict[str, SQLModel] = {}

    def discover_models(self):
        """Discover all SQL models in the models directory"""
        self.models = {}
        for sql_file in self.models_dir.glob("**/*.sql"):
            model_name = sql_file.stem
            with open(sql_file, "r") as f:
                content = f.read()
            self.models[model_name] = SQLModel(model_name, content, str(sql_file))
            print(f"Discovered model: {model_name}")  # Debug print
        return self.models

    def build_dag(self) -> Dict[str, Set[str]]:
        """Build dependency graph"""
        dag = {}
        for model_name, model in self.models.items():
            dag[model_name] = model.dependencies
        print(f"Built DAG: {dag}")  # Debug print
        return dag

    def topological_sort(self) -> List[str]:
        """Generate execution order using topological sort"""
        dag = self.build_dag()
        in_degree = defaultdict(int)

        # Calculate in-degrees
        for node, deps in dag.items():
            for dep in deps:
                in_degree[dep] += 1
                if dep not in dag:
                    dag[dep] = set()  # Add leaf nodes to DAG

        # Find all nodes with no dependencies
        queue = deque(
            [node for node in dag.keys() if len(dag[node]) == 0]
        )  # Changed this line
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Find nodes that depend on the current node
            for dependent, deps in dag.items():
                if node in deps:
                    deps.remove(node)
                    if len(deps) == 0:
                        queue.append(dependent)

        if len(result) != len(dag):
            raise ValueError("Circular dependency detected")

        print(f"Execution order: {result}")  # Debug print
        return result

    def _replace_model_references(self, sql: str) -> str:
        """
        Replace `{{ ref(...) }}` placeholders with corresponding model table names.
        """
        ref_pattern = r"\{\{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}"

        def replace_ref(match):
            model_name = match.group(1)
            if model_name not in self.models:
                raise ValueError(
                    f"Model '{model_name}' not found during ref replacement."
                )
            return model_name  # Replace with the actual model table name

        # Replace all model references with the actual model names
        replaced_sql = re.sub(ref_pattern, replace_ref, sql)

        return replaced_sql  # Return the correctly replaced SQL without unnecessary modifications

    def compile_model(self, model_name: str) -> str:
        """
        Compile a single model, resolving refs.
        """
        model = self.models[model_name]
        compiled_sql = model.content

        # Replace refs using the new replacement method
        compiled_sql = self._replace_model_references(compiled_sql)

        print(f"Compiled SQL for {model_name}:\n{compiled_sql}")  # Debug print
        return compiled_sql

    def _extract_base_query(self, sql: str) -> str:
        """Extract the base query without AI functions"""
        # First, replace all ref() calls with actual table names
        for dep in self.models[self.current_model].dependencies:
            sql = sql.replace(f"{{{{ ref('{dep}') }}}}", dep)

        # Split into SELECT and FROM parts
        parts = sql.split("FROM", 1)
        if len(parts) != 2:
            raise ValueError("Invalid SQL syntax")

        select_part = parts[0].replace("SELECT", "").strip()
        from_part = "FROM" + parts[1]

        # Split columns, handling nested parentheses
        columns = []
        current = []
        paren_level = 0

        for char in select_part:
            if char == "(":
                paren_level += 1
            elif char == ")":
                paren_level -= 1
            elif char == "," and paren_level == 0:
                columns.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        if current:
            columns.append("".join(current).strip())

        # Process columns
        final_columns = []
        for col in columns:
            if "synthesize(" not in col:
                final_columns.append(col)
            else:
                # Extract alias if present
                alias_match = re.search(r"as\s+(\w+)\s*$", col, re.IGNORECASE)
                if alias_match:
                    final_columns.append(f"NULL as {alias_match.group(1)}")

        # Reconstruct query
        final_sql = f"SELECT {', '.join(final_columns)} {from_part}"
        print(f"Extracted base query:\n{final_sql}")  # Debug print
        return final_sql

    def execute_model(self, model_name: str) -> pd.DataFrame:
        """Execute a model and materialize it to the database"""
        self.current_model = model_name  # Add this line to track current model
        model = self.models[model_name]
        compiled_sql = self.compile_model(model_name)

        try:
            if model.has_ai_function:
                df = self._execute_ai_model(compiled_sql, model)
            else:
                df = self._execute_standard_sql(compiled_sql)

            # Materialize results to database
            self._materialize_to_db(model_name, df)
            return df

        except Exception as e:
            print(f"Error executing model {model_name}: {str(e)}")
            raise

    def _execute_standard_sql(self, sql: str) -> pd.DataFrame:
        """Execute standard SQL query"""
        with sqlite3.connect(self.db_path) as conn:
            try:
                sql = re.sub(r"--.*?\n", "\n", sql)  # Remove comments
                sql = re.sub(r"\s+", " ", sql).strip()  # Clean whitespace
                return pd.read_sql(sql, conn)
            except Exception as e:
                print(f"Failed to execute SQL: {sql}")
                print(f"Error: {str(e)}")
                raise

    def _execute_ai_model(self, sql: str, model: SQLModel) -> pd.DataFrame:
        """Execute SQL with AI functions"""
        try:
            # Get base data
            base_sql = self._extract_base_query(sql)
            print(f"Executing base SQL:\n{base_sql}")
            df = self._execute_standard_sql(base_sql)

            # Process each AI function
            for func_name, params in model.ai_functions.items():
                column = params["column"]
                npc = params["npc"]
                query = params["query"]
                context = params["context"]
                extra = params["extra"]

                # Process row by row
                results = []
                for _, row in df.iterrows():
                    # Format the query with row values
                    formatted_query = query.format(**row.to_dict())

                    # Here you would call your actual AI processing
                    # For now, just adding a placeholder
                    result = f"AI processed: {formatted_query}"
                    results.append(result)

                # Add results back to dataframe
                df[f"{func_name}_result"] = results

            return df

        except Exception as e:
            print(f"Error in AI model execution: {str(e)}")
            raise

    def _materialize_to_db(self, model_name: str, df: pd.DataFrame):
        """Save the DataFrame as a table in the database"""
        with sqlite3.connect(self.db_path) as conn:
            # Drop the table if it exists
            conn.execute(f"DROP TABLE IF EXISTS {model_name}")

            # Write the new data
            df.to_sql(model_name, conn, index=False)
            print(f"Materialized model {model_name} to database")

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?;
            """,
                (table_name,),
            )
            return cursor.fetchone() is not None

    def run_all_models(self):
        """Execute all models in dependency order"""
        self.discover_models()
        execution_order = self.topological_sort()
        print(f"Running models in order: {execution_order}")

        results = {}
        for model_name in execution_order:
            print(f"\nExecuting model: {model_name}")

            # Check if all dependencies are materialized
            model = self.models[model_name]
            for dep in model.dependencies:
                if not self._table_exists(dep):
                    raise ValueError(
                        f"Dependency {dep} not found in database for model {model_name}"
                    )

            results[model_name] = self.execute_model(model_name)

        return results


def create_example_models(models_dir: str):
    """Create example SQL model files"""
    os.makedirs(models_dir, exist_ok=True)

    # Basic feedback model
    customer_feedback = """
    SELECT
        feedback,
        customer_id,
        timestamp
    FROM raw_customer_feedback
    WHERE LENGTH(feedback) > 10;
    """

    # Model with AI function
    customer_insights = """
    SELECT
        customer_id,
        feedback,
        timestamp,
        synthesize(
            "feedback text: {feedback}",
            "analyst",
            "feedback_analysis"
        ) as ai_analysis
    FROM {{ ref('customer_feedback') }};
    """

    # Write models in correct order
    models = {
        "customer_feedback.sql": customer_feedback,
        "customer_insights.sql": customer_insights,
    }

    for name, content in models.items():
        path = os.path.join(models_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created model: {name}")


def main():
    # Setup
    models_dir = "example_models"
    db_path = os.path.expanduser("~/npcsh_history.db")

    # Create raw data
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame(
        {
            "feedback": ["Great product!", "Could be better", "Amazing service"],
            "customer_id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )

    # Create the raw table
    df.to_sql("raw_customer_feedback", conn, index=False, if_exists="replace")
    print("Created raw_customer_feedback table")

    # Create example models
    create_example_models(models_dir)

    # Initialize compiler and run models
    compiler = ModelCompiler(models_dir, db_path)
    results = compiler.run_all_models()

    # Print results
    for model_name, df in results.items():
        print(f"\nResults for {model_name}:")
        print(df.head())


if __name__ == "__main__":
    main()
