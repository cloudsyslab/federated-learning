#!/bin/bash

bash runner_poison_noniid.sh plus ours
sleep 2
pkill -u plama pt_main_thread
#bash runner_poison_noniid.sh plus ours_hybrid
#sleep 2
#pkill -u plama pt_main_thread
#bash runner_poison_noniid.sh plus rlr
#sleep 2
#pkill -u plama pt_main_thread
#bash runner_poison_noniid.sh square ours
#sleep 2
#pkill -u plama pt_main_thread
#bash runner_poison_noniid.sh square ours_hybrid
#sleep 2
#pkill -u plama pt_main_thread
#bash runner_poison_noniid.sh square rlr
#sleep 2
#pkill -u plama pt_main_thread
#bash runner_poison_noniid.sh plus oracle
#sleep 2
#pkill -u plama pt_main_thread
bash runner_poison_noniid.sh plus nodefense
sleep 2
pkill -u plama pt_main_thread
#bash runner_poison_noniid.sh square oracle
#sleep 2
#pkill -u plama pt_main_thread
#bash runner_poison_noniid.sh square nodefense
#sleep 2
#pkill -u plama pt_main_thread

