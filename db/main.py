import random
import sqlite3
from neo4j import GraphDatabase
import neo

PROTEINS = ["S", "T", "U", "V", "W", "X", "Y", "Z"]

RELATIONS = [
    "INHIBITOR",
    "SUBSTRATE",
    "INDIRECT-DOWNREGULATOR",
    "INDIRECT-UPREGULATOR",
    "ACTIVATOR",
    "ANTAGONIST",
    "PRODUCT-OF",
    "AGONIST",
    "DOWNREGULATOR",
    "UPREGULATOR",
    "AGONIST-ACTIVATOR",
    "SUBSTRATE_PRODUCT-OF",
    "AGONIST-INHIBITOR",
    "NO-RELATION",  # ??????
]


def generate_example_outputs(n):
    outputs = []
    for outputId in range(n):
        entities = random.sample(PROTEINS, random.randint(2, 3))
        relations = []
        for e1 in entities:
            for e2 in entities:
                if e1 == e2:
                    continue
                relations.append({
                    "type": random.choice(RELATIONS),
                    "left": e1,
                    "right": e2,
                })
        outputs.append({
            "relations": relations,
            "outputId": outputId,
        })
    return outputs


def main2():
    driver = GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "ppi"))

    outputs = generate_example_outputs(5)

    with driver.session() as session:
        for o in outputs:
            for r in o["relations"]:
                session.write_transaction(
                    neo.add_relation, r["left"], r["right"], r["type"])
        # session.read_transaction(neo.show_stuff)

    driver.close()


if __name__ == "__main__":
    main2()


# def main():
#     with open('model.sql', 'r') as sql_file:
#         sql_script = sql_file.read()

#     conn = sqlite3.connect('ppi.db')
#     cursor = conn.cursor()
#     cursor.executescript(sql_script)

#     outputs = generate_example_outputs(5)

#     vs = [(None, r["left"], r["type"], r["right"], o["outputId"])
#           for o in outputs
#           for r in o["relations"]]
#     cursor.executemany("INSERT INTO Relations VALUES (?,?,?,?,?)", vs)

#     conn.commit()
#     conn.close()
