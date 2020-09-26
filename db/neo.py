def add_relation(tx, p1, p2, r):
    tx.run("""MERGE (a:Protein {name: $p1})""", p1=p1)
    tx.run("""MERGE (b:Protein {name: $p2})""", p2=p2)
    tx.run("""
        MATCH (a:Protein {name: $p1}), (b:Protein {name: $p2})
        MERGE (a)-[r:`""" + r + """`]->(b)
            ON MATCH SET
                r.counter=coalesce(r.counter, 0) + 1
        """, p1=p1, p2=p2)


def show_stuff(tx):
    for record in tx.run("MATCH (p:Protein) RETURN p"):
        print(record["p.name"])


# def print_friends(tx, name):
#     for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
#                          "RETURN friend.name ORDER BY friend.name", name=name):
#         print(record["friend.name"])
