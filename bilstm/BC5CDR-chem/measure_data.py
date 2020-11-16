import json

with open("parsed_data.txt", "r") as f:
    data = json.loads(f.read())

    count = 0
    for item in data:
        count +=1

    print("Count: " + str(count))