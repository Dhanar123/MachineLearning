# EASY Candidate Elimination Algorithm

# Training data
data = [
    ["Technical", "Senior", "Excellent", "Good", "Urban", "Yes"],
    ["Technical", "Junior", "Excellent", "Good", "Urban", "Yes"],
    ["Non-Technical", "Junior", "Average", "Poor", "Rural", "No"],
    ["Technical", "Senior", "Average", "Good", "Rural", "No"],
    ["Technical", "Senior", "Excellent", "Good", "Rural", "Yes"]
]

# Number of attributes (excluding output)
n = 5

# Step 1: Initialize S and G
S = ["Ø"] * n
G = ["?"] * n

print("Initial S =", S)
print("Initial G =", G)
print("--------------------------------")

# Step 2: Process each example
for example in data:
    attributes = example[:-1]
    label = example[-1]

    print("Example:", attributes, "->", label)

    # POSITIVE example
    if label == "Yes":
        for i in range(n):
            if S[i] == "Ø":
                S[i] = attributes[i]
            elif S[i] != attributes[i]:
                S[i] = "?"

    # NEGATIVE example
    else:
        for i in range(n):
            if S[i] != "?" and S[i] != attributes[i]:
                G[i] = S[i]

    print("S =", S)
    print("G =", G)
    print("--------------------------------")

# Final result
print("Final Specific Boundary (S):", S)
print("Final General Boundary (G):", G)
