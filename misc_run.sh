
bash runner_poison.sh plus cifar10 ours_hybrid
sleep 2
pkill -u "$(whoami)" pt_main_thread

bash runner_poison.sh plus fmnist ours_hybrid
sleep 2
pkill -u "$(whoami)" pt_main_thread

bash runner_poison_distributed.sh cifar10 ours_hybrid
sleep 2
pkill -u "$(whoami)" pt_main_thread

