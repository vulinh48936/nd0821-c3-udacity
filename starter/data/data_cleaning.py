new_lines = []

with open("census.csv", "r") as f:
    for line in f:
        new_line = line.replace(", ", ",")
        new_lines.append(new_line)

with open("clean_census.csv", "w") as f:
    for line in new_lines:
        f.write(line)