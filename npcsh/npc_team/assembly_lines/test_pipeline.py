import pandas as pd
from sqlalchemy import create_engine
import os

# Sample market events data
market_events_data = {
    "datetime": [
        "2023-10-15 09:00:00",
        "2023-10-16 10:30:00",
        "2023-10-17 11:45:00",
        "2023-10-18 13:15:00",
        "2023-10-19 14:30:00",
    ],
    "headline": [
        "Stock Market Rallies Amid Positive Economic Data",
        "Tech Giant Announces New Product Line",
        "Federal Reserve Hints at Interest Rate Pause",
        "Oil Prices Surge Following Supply Concerns",
        "Retail Sector Reports Record Q3 Earnings",
    ],
}

# Create a DataFrame
market_events_df = pd.DataFrame(market_events_data)

# Define database path relative to user's home directory
db_path = os.path.expanduser("~/npcsh_history.db")

# Create a connection to the SQLite database
engine = create_engine(f"sqlite:///{db_path}")
with engine.connect() as connection:
    # Write the data to a new table 'market_events', replacing existing data
    market_events_df.to_sql(
        "market_events", con=connection, if_exists="replace", index=False
    )

print("Market events have been added to the database.")

email_data = {
    "datetime": [
        "2023-10-10 10:00:00",
        "2023-10-11 11:00:00",
        "2023-10-12 12:00:00",
        "2023-10-13 13:00:00",
        "2023-10-14 14:00:00",
    ],
    "subject": [
        "Meeting Reminder",
        "Project Update",
        "Invoice Attached",
        "Weekly Report",
        "Holiday Notice",
    ],
    "sender": [
        "alice@example.com",
        "bob@example.com",
        "carol@example.com",
        "dave@example.com",
        "eve@example.com",
    ],
    "recipient": [
        "bob@example.com",
        "carol@example.com",
        "dave@example.com",
        "eve@example.com",
        "alice@example.com",
    ],
    "body": [
        "Don't forget the meeting tomorrow at 10 AM.",
        "The project is progressing well, see attached update.",
        "Please find your invoice attached.",
        "Here is the weekly report.",
        "The office will be closed on holidays, have a great time!",
    ],
}

# Create a DataFrame
emails_df = pd.DataFrame(email_data)

# Define database path relative to user's home directory
db_path = os.path.expanduser("~/npcsh_history.db")

# Create a connection to the SQLite database
engine = create_engine(f"sqlite:///{db_path}")
with engine.connect() as connection:
    # Write the data to a new table 'emails', replacing existing data
    emails_df.to_sql("emails", con=connection, if_exists="replace", index=False)

print("Sample emails have been added to the database.")


from npcsh.npc_compiler import PipelineRunner
import os

pipeline_runner = PipelineRunner(
    pipeline_file="morning_routine.pipe",
    npc_root_dir=os.path.abspath("."),  # Use absolute path to parent directory
    db_path="~/npcsh_history.db",
)
pipeline_runner.execute_pipeline()


import pandas as pd
from sqlalchemy import create_engine
import os

# Sample data generation for news articles
news_articles_data = {
    "news_article_id": list(range(1, 21)),
    "headline": [
        "Economy sees unexpected growth in Q4",
        "New tech gadget takes the world by storm",
        "Political debate heats up over new policy",
        "Health concerns rise amid new disease outbreak",
        "Sports team secures victory in last minute",
        "New economic policy introduced by government",
        "Breakthrough in AI technology announced",
        "Political leader delivers speech on reforms",
        "Healthcare systems pushed to limits",
        "Celebrated athlete breaks world record",
        "Controversial economic measures spark debate",
        "Innovative tech startup gains traction",
        "Political scandal shakes administration",
        "Healthcare workers protest for better pay",
        "Major sports event postponed due to weather",
        "Trade tensions impact global economy",
        "Tech company accused of data breach",
        "Election results lead to political upheaval",
        "Vaccine developments offer hope amid pandemic",
        "Sports league announces return to action",
    ],
    "content": ["Article content here..." for _ in range(20)],
    "publication_date": pd.date_range(start="1/1/2023", periods=20, freq="D"),
}

# Create a DataFrame
news_df = pd.DataFrame(news_articles_data)

# Define the database path
db_path = os.path.expanduser("~/npcsh_history.db")

# Create a connection to the SQLite database
engine = create_engine(f"sqlite:///{db_path}")
with engine.connect() as connection:
    # Write the data to a new table 'news_articles', replacing existing data
    news_df.to_sql("news_articles", con=connection, if_exists="replace", index=False)

print("News articles have been added to the database.")

from npcsh.npc_compiler import PipelineRunner
import os

runner = PipelineRunner(
    "./news_analysis.pipe",
    db_path=os.path.expanduser("~/npcsh_history.db"),
    npc_root_dir=os.path.abspath("."),
)
results = runner.execute_pipeline()

print("\nResults:")
print("\nClassifications (processed row by row):")
print(results["classify_news"])
print("\nAnalysis (processed in batch):")
print(results["analyze_news"])


from npcsh.npc_compiler import PipelineRunner
import os

runner = PipelineRunner(
    "./news_analysis_mixa.pipe",
    db_path=os.path.expanduser("~/npcsh_history.db"),
    npc_root_dir=os.path.abspath("."),
)
results = runner.execute_pipeline()

print("\nResults:")
print("\nClassifications (processed row by row):")
print(results["classify_news"])
print("\nAnalysis (processed in batch):")
print(results["analyze_news"])
