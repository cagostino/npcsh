import pysqlite3 as sqlite3
import sqlite_vec


db = sqlite3.connect("/home/caug/npcsh_history.db.sqlite")
db.enable_load_extension(True)
sqlite_vec.load(db)

# (version,) = db.execute("select vss_version()").fetchone()
# print(version)


# Create virtual table
db.execute(
    """
CREATE VIRTUAL TABLE  if not exists
vec_examples
USING vec0(
    sample_embedding float[8]
);
"""
)
db.commit()

# Insert test vectors with proper formatting
db.execute(
    """
insert into vec_examples( sample_embedding)
  values
    ( '[-0.200, 0.250, 0.341, -0.211, 0.645, 0.935, -0.316, -0.924]'),
    ( '[0.443, -0.501, 0.355, -0.771, 0.707, -0.708, -0.185, 0.362]'),
    ( '[0.716, -0.927, 0.134, 0.052, -0.669, 0.793, -0.634, -0.162]'),
    ( '[-0.710, 0.330, 0.656, 0.041, -0.990, 0.726, 0.385, -0.958]');
               """,
)
db.commit()


results = db.execute(
    """
    SELECT
        *
    FROM vec_examples a
"""
).fetchall()
print("\nTest 1 - All rows:")
for row in results:
    print(row)
k = 2  # Set k to the number of nearest neighbors you want
query = """
SELECT
  rowid,
  distance
from vec_examples
where sample_embedding match '[0.890, 0.544, 0.825, 0.961, 0.358, 0.0196, 0.521, 0.175]'
order by distance limit 3;
"""
results = db.execute(query).fetchall()


print("\nTest 2 - Pairwise distances:")
for id1, id2, distance in results:
    print(f"Between row {id1} and {id2}: distance = {distance:.4f}")

# Test 3: Find vectors within a certain distance
threshold = 2.5
results = db.execute(
    """
    SELECT rowid, distance
    FROM vec_examples
    WHERE sample_embedding MATCH ?
    AND distance < ?
    ORDER BY distance
""",
    [formatted_query, threshold],
).fetchall()

print(f"\nTest 3 - Vectors within distance {threshold}:")
for rowid, distance in results:
    print(f"Row {rowid}: distance = {distance:.4f}")
