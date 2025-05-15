bash runner_poison_20clients.sh plus cifar10 ours
sleep 2
pkill -u "$(whoami)" pt_main_thread

bash runner_poison_20clients.sh plus cifar10 ours_hybrid
sleep 2
pkill -u "$(whoami)" pt_main_thread

bash runner_poison_20clients.sh plus cifar10 rlr
sleep 2
pkill -u "$(whoami)" pt_main_thread



