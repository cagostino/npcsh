import pandas as pd
import numpy as np
import os
from npcsh.npc_compiler import NPC, NPCTeam, Tool


# Create test data and save to CSV
def create_test_data(filepath="sales_data.csv"):
    sales_data = pd.DataFrame(
        {
            "date": pd.date_range(start="2024-01-01", periods=90),
            "revenue": np.random.normal(10000, 2000, 90),
            "customer_count": np.random.poisson(100, 90),
            "avg_ticket": np.random.normal(100, 20, 90),
            "region": np.random.choice(["North", "South", "East", "West"], 90),
            "channel": np.random.choice(["Online", "Store", "Mobile"], 90),
        }
    )

    # Add patterns to make data more realistic
    sales_data["revenue"] *= 1 + 0.3 * np.sin(
        np.pi * np.arange(90) / 30
    )  # Seasonal pattern
    sales_data.loc[sales_data["channel"] == "Mobile", "revenue"] *= 1.1  # Mobile growth
    sales_data.loc[
        sales_data["channel"] == "Online", "customer_count"
    ] *= 1.2  # Online customer growth

    sales_data.to_csv(filepath, index=False)
    return filepath, sales_data


code_execution_tool = Tool(
    {
        "tool_name": "execute_code",
        "description": """Executes a Python code block with access to pandas,
                          numpy, and matplotlib.
                          Results should be stored in the 'results' dict to be returned.
                          The only input should be a single code block with \n characters included.
                          The code block must use only the  libraries or methods contained withen the
                            pandas, numpy, and matplotlib libraries or using builtin methods.
                          do not include any json formatting or markdown formatting.

                          When generating your script, the final output must be encoded in a variable
                          named "output". e.g.

                          output  = some_analysis_function(inputs, derived_data_from_inputs)
                            Adapt accordingly based on the scope of the analysis

                          """,
        "inputs": ["script"],
        "steps": [
            {
                "engine": "python",
                "code": """{{script}}""",
            }
        ],
    }
)

# Analytics team definition
analytics_team = [
    {
        "name": "analyst",
        "primary_directive": "You analyze sales performance data, focusing on revenue trends, customer behavior metrics, and market indicators. Your expertise is in extracting actionable insights from complex datasets.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
    {
        "name": "researcher",
        "primary_directive": "You specialize in causal analysis and experimental design. Given data insights, you determine what factors drive observed patterns and design tests to validate hypotheses.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
    {
        "name": "engineer",
        "primary_directive": "You implement data pipelines and optimize data processing. When given analysis requirements, you create efficient workflows to automate insights generation.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
]


def create_analytics_team():
    # Initialize NPCs with just the code execution tool
    npcs = []
    for npc_data in analytics_team:
        npc = NPC(
            name=npc_data["name"],
            primary_directive=npc_data["primary_directive"],
            model=npc_data["model"],
            provider=npc_data["provider"],
            tools=[code_execution_tool],  # Only code execution tool
        )
        npcs.append(npc)

    # Create coordinator with just code execution tool
    coordinator = NPC(
        name="coordinator",
        primary_directive="You coordinate the analytics team, ensuring each specialist contributes their expertise effectively. You synthesize insights and manage the workflow.",
        model="gpt-4o-mini",
        provider="openai",
        tools=[code_execution_tool],  # Only code execution tool
    )

    # Create team
    team = NPCTeam(npcs=npcs, foreman=coordinator)
    return team


def main():
    # Create and save test data
    data_path, sales_data = create_test_data()

    # Initialize team
    team = create_analytics_team()

    # Run analysis - updated prompt to reflect code execution approach
    results = team.orchestrate(
        f"""
    Analyze the sales data at {data_path} to:
    1. Identify key performance drivers
    2. Determine if mobile channel growth is significant
    3. Recommend tests to validate growth hypotheses

    Here is a header for the data file at {data_path}:
    {sales_data.head()}

    When working with dates, ensure that date columns are converted from raw strings. e.g. use the pd.to_datetime function.


    When working with potentially messy data, handle null values by using nan versions of numpy functions or
    by filtering them with a mask .

    Use Python code execution to perform the analysis - load the data and perform statistical analysis directly.
    """
    )

    print(results)

    # Cleanup
    os.remove(data_path)


if __name__ == "__main__":
    main()
