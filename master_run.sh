bash runner_poison.sh square cifar10 ours
sleep 2
pkill -u "$(whoami)" pt_main_thread
#below line generates/appends to a file named final_table_results.csv where it has the final round of the most recent txt file and records the data
python generate_final_round_table.py "$(ls -t ./results/cifar10/*.txt | head -n 1)"
#below line generates/appends to a file named "round_results_p1value_p2value_p3value_p4value_r#rounds.csv" where it has all rounds generated from the above bash line code "bash runner_poison.sh square cifar10 ours" so it lists out the results from the most recent txt file and makes it into an easy to graph data format
python generate_all_rounds_table.py "$(ls -t ./results/cifar10/*.txt | head -n 1)"


bash runner_poison.sh plus cifar10 ours
sleep 2
pkill -u "$(whoami)" pt_main_thread
python generate_final_round_table.py "$(ls -t ./results/cifar10/*.txt | head -n 1)"
python generate_all_rounds_table.py "$(ls -t ./results/cifar10/*.txt | head -n 1)"

#commented out test below because code for plus_distributed poisoning pattern is still being fixed/updated
#bash runner_poison.sh plus_distributed cifar10 ours
#sleep 2
#pkill -u "$(whoami)" pt_main_thread


bash runner_poison.sh square fmnist ours
sleep 2
pkill -u "$(whoami)" pt_main_thread
python generate_final_round_table.py "$(ls -t ./results/fmnist/*.txt | head -n 1)"
python generate_all_rounds_table.py "$(ls -t ./results/fmnist/*.txt | head -n 1)"


bash runner_poison.sh plus fmnist ours
sleep 2
pkill -u "$(whoami)" pt_main_thread
python generate_final_round_table.py "$(ls -t ./results/fmnist/*.txt | head -n 1)"
python generate_all_rounds_table.py "$(ls -t ./results/fmnist/*.txt | head -n 1)"

#commented out test below because code for plus_distributed poisoning pattern is still being fixed/updated
#bash runner_poison.sh plus_distributed fmnist ours
#sleep 2
#pkill -u "$(whoami)" pt_main_thread
