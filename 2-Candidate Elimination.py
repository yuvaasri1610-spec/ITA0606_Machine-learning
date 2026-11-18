import csv

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

with open("training.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

with open("training.csv", "r") as f:
    examples = list(csv.reader(f))

n = len(examples[0]) - 1

S = ['0'] * n
G = [['?'] * n]

for row in examples:
    x = row[:-1]

    if row[-1] == 'Yes':
        G = [g for g in G if all(
            g[i] == '?' or g[i] == x[i] for i in range(n)
        )]

        for i in range(n):
            if S[i] == '0':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'

    else:
        new_G = []

        for g in G:
            for i in range(n):
                if g[i] == '?' and S[i] != '?' and S[i] != x[i]:
                    h = g.copy()
                    h[i] = S[i]
                    new_G.append(h)

        G = new_G

print("Specific Boundary S:", S)
print("General Boundary G:", G)
