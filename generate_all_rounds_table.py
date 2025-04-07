import re
import pandas as pd
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python generate_all_rounds_table.py <log1.txt>")
    sys.exit(1)

log_file = sys.argv[1]
if not os.path.isfile(log_file):
    print(f"Missing file: {log_file}")
    sys.exit(1)

with open(log_file, "r") as f:
    lines = f.readlines()

round_info = {}

# Parse the log file
for i, line in enumerate(lines):
    acc_match = re.search(r"Round (\d+) accuracy: ([0-9.]+)", line)
    poison_match = re.search(r"Round (\d+) poison accuracy: ([0-9.]+)", line)
    thresh_match = re.search(r"Using percentile thresholds: p1=(\d+), p2=(\d+), p3=(\d+), p4=(\d+)", line)

    if acc_match:
        r = int(acc_match.group(1))
        round_info.setdefault(r, {})["accuracy"] = float(acc_match.group(2))

    if poison_match:
        r = int(poison_match.group(1))
        round_info.setdefault(r, {})["poison"] = float(poison_match.group(2))

    if thresh_match:
        for j in range(i - 1, -1, -1):
            if "Round " in lines[j]:
                round_search = re.search(r"Round (\d+)", lines[j])
                if round_search:
                    r = int(round_search.group(1))
                    round_info.setdefault(r, {})["p1"] = int(thresh_match.group(1))
                    round_info[r]["p2"] = int(thresh_match.group(2))
                    round_info[r]["p3"] = int(thresh_match.group(3))
                    round_info[r]["p4"] = int(thresh_match.group(4))
                    break

# Detect dataset and trojan type
dataset = "fmnist" if "fmnist" in log_file else "cifar10"
trojan = "plus_distributed" if "plus_distributed" in log_file else ("plus" if "plus" in log_file else "square")

# Build rows
rows = []
for r, info in sorted(round_info.items()):
    acc = info.get("accuracy")
    poison = info.get("poison")
    p1 = info.get("p1")
    p2 = info.get("p2")
    p3 = info.get("p3")
    p4 = info.get("p4")
    defense = f"ours ({p1},{p2},{p3},{p4})" if p1 is not None else "unknown"

    if acc is not None and poison is not None:
        rows.append([f"Round {r}", dataset, trojan, defense, poison, acc])

# Add separator row
rows.append(["---", "---", "---", "---", "", ""])

# Save to dynamic filename
num_rounds = len([r for r in round_info if isinstance(r, int)])

if p1 is not None:
    filename = f"round_results_p{p1}_{p2}_{p3}_{p4}_r{num_rounds}.csv"
else:
    filename = f"round_results_unknown_r{num_rounds}.csv"

columns = ["round", "dataset", "trojan", "defense", "poison_accuracy", "validation_accuracy"]
final_df = pd.DataFrame(rows, columns=columns)

if os.path.isfile(filename):
    final_df.to_csv(filename, mode="a", index=False, header=False)
else:
    final_df.to_csv(filename, index=False)

print(f"Saved results to {filename}")
