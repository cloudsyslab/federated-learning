#bash runner_poison.sh square cifar10 ours
#sleep 2
#pkill pt_main_thread
bash runner_poison.sh plus cifar10 ours
sleep 2
pkill pt_main_thread
bash runner_poison.sh square fmnist ours
sleep 2
pkill pt_main_thread
bash runner_poison.sh plus fmnist ours
sleep 2
pkill pt_main_thread
