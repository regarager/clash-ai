import json
from cards import format_name, ELIXIR_COST, CARDS

obj = json.loads(open("cards.json", "r").read())

idk = {}
for item in obj["items"]:
    try:
        idk[item["name"]] = item["elixirCost"]
    except KeyError:
        idk[item["name"]] = 0

a = list(map(format_name, ELIXIR_COST.keys()))
c = a + list(map(lambda s: "enemy-" + s, a))

for x in c:
    if x not in CARDS:
        print(x)

for x in CARDS:
    if x not in c:
        print(x)
