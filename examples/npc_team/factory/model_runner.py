# Usage example:
def main():
    # Initialize
    npc_compiler = NPCCompiler("~/.npcsh/npc_team", "~/npcsh_history.db")
    runner = NPCModelRunner(npc_compiler)

    # Run first model
    with open("models/customer_feedback.sql", "r") as f:
        feedback_model = runner.run_model("customer_feedback", f.read())
        print("First model results:")
        print(feedback_model.head())

    # Run second model that depends on the first
    with open("models/customer_insights.sql", "r") as f:
        insights_model = runner.run_model("customer_insights", f.read())
        print("\nSecond model results:")
        print(insights_model.head())

    # Check history
    with sqlite3.connect(runner.history_db) as conn:
        history = pd.read_sql(
            "SELECT * FROM model_runs ORDER BY run_timestamp DESC", conn
        )
        print("\nModel run history:")
        print(history)


if __name__ == "__main__":
    main()
