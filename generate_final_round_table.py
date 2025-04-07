import re
import pandas as pd
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python generate_results_table.py <log1.txt> <log2.txt> ...")
    sys.exit(1)

log_files = sys.argv[1:]

rows = []

for file in log_files:
    if not os.path.isfile(file):
        print(f"Skipping missing file: {file}")
        continue

    with open(file, "r") as f:
        lines = f.readlines()

    round_info = {}

    # Gather all round data
    for i, line in enumerate(lines):
        round_match = re.search(r"Round (\d+) accuracy: ([0-9.]+)", line)
        poison_match = re.search(r"Round (\d+) poison accuracy: ([0-9.]+)", line)
        threshold_match = re.search(r"Using percentile thresholds: p1=(\d+), p2=(\d+), p3=(\d+), p4=(\d+)", line)

        if round_match:
            r = int(round_match.group(1))
            acc = float(round_match.group(2))
            round_info.setdefault(r, {})["accuracy"] = acc

        if poison_match:
            r = int(poison_match.group(1))
            poison = float(poison_match.group(2))
            round_info.setdefault(r, {})["poison"] = poison

        if threshold_match:
            for j in range(i - 1, -1, -1):
                if "Round " in lines[j]:
                    round_search = re.search(r"Round (\d+)", lines[j])
                    if round_search:
                        r = int(round_search.group(1))
                        round_info.setdefault(r, {})["p1"] = int(threshold_match.group(1))
                        round_info[r]["p2"] = int(threshold_match.group(2))
                        round_info[r]["p3"] = int(threshold_match.group(3))
                        round_info[r]["p4"] = int(threshold_match.group(4))
                        break

    if not round_info:
        print(f"No round data found in: {file}")
        continue

    last_round = max(round_info.keys())
    info = round_info[last_round]

    last_acc = info.get("accuracy")
    last_poison = info.get("poison")
    p1 = info.get("p1")
    p2 = info.get("p2")
    p3 = info.get("p3")
    p4 = info.get("p4")

    dataset = "fmnist" if "fmnist" in file else "cifar10"
    trojan = "plus_distributed" if "plus_distributed" in file else ("plus" if "plus" in file else "square")

    if p1 is not None:
        defense = f"ours ({p1},{p2},{p3},{p4})"
    else:
        defense = "unknown"

    if last_acc is not None and last_poison is not None:
        rows.append([dataset, trojan, defense, last_poison, last_acc])
    else:
        print(f"Couldn't extract final round data from: {file}")

# Save only the clean columns
output_file = "final_results_table.csv"
columns = ["dataset", "trojan", "defense", "poison_accuracy", "validation_accuracy"]

if os.path.isfile(output_file):
    existing_df = pd.read_csv(output_file, usecols=columns)
    new_df = pd.DataFrame(rows, columns=columns)
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    final_df = pd.DataFrame(rows, columns=columns)

final_df.to_csv(output_file, index=False)
print(f"Appended cleaned rows to {output_file}")

