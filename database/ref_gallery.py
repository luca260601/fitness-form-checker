import sqlite3, shutil
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS exercises(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS images(
  id INTEGER PRIMARY KEY,
  exercise_id INTEGER NOT NULL,
  path TEXT NOT NULL UNIQUE,
  split TEXT NOT NULL CHECK(split IN ('good','bad')),
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(exercise_id) REFERENCES exercises(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS labels(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS image_labels(
  image_id INTEGER NOT NULL,
  label_id INTEGER NOT NULL,
  PRIMARY KEY(image_id,label_id),
  FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
  FOREIGN KEY(label_id) REFERENCES labels(id) ON DELETE CASCADE
);
"""

def connect(db_path: str):
    conn = sqlite3.connect(db_path); conn.execute("PRAGMA foreign_keys=ON;"); return conn

def init_db(db_path: str):
    conn = connect(db_path)
    with conn: conn.executescript(SCHEMA)
    conn.close()

def get_or_create_exercise(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM exercises WHERE name=?", (name,)).fetchone()
    if row: return row[0]
    with conn: conn.execute("INSERT INTO exercises(name) VALUES(?)", (name,))
    return conn.execute("SELECT id FROM exercises WHERE name=?", (name,)).fetchone()[0]

def add_image(conn, exercise: str, abs_src: str, dest_dir: str, split: str, labels: list[str]) -> str:
    ex_id = get_or_create_exercise(conn, exercise)
    dest = Path(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    src_p = Path(abs_src); dst = dest / src_p.name
    i = 1
    while dst.exists():
        dst = dest / f"{src_p.stem}_{i}{src_p.suffix}"; i += 1
    shutil.copy2(src_p, dst)
    rel = str(dst)
    with conn:
        conn.execute("INSERT OR IGNORE INTO images(exercise_id,path,split) VALUES(?,?,?)",(ex_id, rel, split))
        img_id = conn.execute("SELECT id FROM images WHERE path=?", (rel,)).fetchone()[0]
        for tag in (labels or []):
            tag = tag.strip()
            if not tag: continue
            lid = conn.execute("SELECT id FROM labels WHERE name=?", (tag,)).fetchone()
            if not lid:
                conn.execute("INSERT INTO labels(name) VALUES(?)", (tag,))
                lid = conn.execute("SELECT id FROM labels WHERE name=?", (tag,)).fetchone()
            conn.execute("INSERT OR IGNORE INTO image_labels(image_id,label_id) VALUES(?,?)", (img_id, lid[0]))
    return rel
