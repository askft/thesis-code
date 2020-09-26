DROP TABLE IF EXISTS Relations;

CREATE TABLE Relations (
    id        INTEGER PRIMARY KEY,
    left      TEXT NOT NULL,
    relation  TEXT NOT NULL,
    right     TEXT NOT NULL,
    outputId  INTEGER NOT NULL
);

-- Possibly add
-- * source url