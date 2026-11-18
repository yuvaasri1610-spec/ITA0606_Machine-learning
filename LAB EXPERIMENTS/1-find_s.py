
# FIND-S Algorithm

data = [
    ["sunny", "hot", "high", "weak", "no"],
    ["sunny", "hot", "high", "strong", "no"],
    ["overcast", "hot", "high", "weak", "yes"],
    ["rain", "mild", "high", "weak", "yes"],
    ["rain", "cool", "normal", "weak", "yes"],
    ["rain", "cool", "normal", "strong", "no"],
    ["overcast", "cool", "normal", "strong", "yes"],
    ["sunny", "mild", "high", "weak", "no"],
    ["sunny", "cool", "normal", "weak", "yes"],
    ["rain", "mild", "normal", "weak", "yes"],
    ["sunny", "mild", "normal", "strong", "yes"],
    ["overcast", "mild", "high", "strong", "yes"],
    ["overcast", "hot", "normal", "weak", "yes"],
    ["rain", "mild", "high", "strong", "no"]
]

attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
hypothesis = None

print("FIND-S Algorithm\n")

for row in data:
    example = row[:-1]
    target = row[-1]

    if target == "yes":
        if hypothesis is None:
            hypothesis = example.copy()
        else:
            for i in range(len(hypothesis)):
                if hypothesis[i] != example[i]:
                    hypothesis[i] = "?"

        print("Example:", example)
        print("Hypothesis:", hypothesis)

print("\nFinal Most Specific Hypothesis:")

for attribute, value in zip(attributes, hypothesis):
    print(attribute, "=", value)
