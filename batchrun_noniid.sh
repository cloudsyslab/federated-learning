#!/bin/bash
set -e

# Always go to the directory where this script lives
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

bash runner_poison_noniid.sh plus ours_hybrid
pkill -u plama pt_main_thread
bash runner_poison_noniid.sh plus rlr
pkill -u plama pt_main_thread
bash runner_poison_noniid.sh square ours_hybrid
pkill -u plama pt_main_thread
bash runner_poison_noniid.sh square rlr
pkill -u plama pt_main_thread
