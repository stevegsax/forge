import os
import sqlite3

db_path = os.path.join(os.environ["XDG_STATE_HOME"], "forge", "forge.db")

con = sqlite3.connect(db_path)
con.create_function("basename", 1, os.path.basename)

rows = con.execute("SELECT basename(file_path) FROM ocr_results ORDER BY file_path").fetchall()

for (name,) in rows:
    print(name)

con.close()

