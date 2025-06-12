bash runner_poison.sh square cifar10 ours resnet
sleep 2
pkill -u "$(whoami)" pt_main_thread

bash runner_poison.sh square cifar10 ours_hybrid resnet
sleep 2
pkill -u "$(whoami)" pt_main_thread

bash runner_poison.sh square cifar10 rlr resnet
sleep 2
pkill -u "$(whoami)" pt_main_thread

bash runner_poison.sh square cifar10 nodefense resnet
sleep 2
pkill -u "$(whoami)" pt_main_thread

