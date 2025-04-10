#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./dataset', download=True)"

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data> <defense technique>"
    exit 1
fi

pattern="plus_distributed"
data=$1
defense=$2

# Check if the pattern and data variables are not empty
if [ -z "$pattern" ] || [ -z "$data" ] || [ -z "$defense" ]; then
    echo "Error: Pattern, data and defense must not be empty."
    echo "defense options: ours, rlr, none etc."
    exit 1
fi

# start server
if [[ "$defense" == "ours" ]]; then
    python server.py --data "$data" --pattern "$pattern" --percentile True --num_agents 40 &
elif [[ "$defense" == "nodefense" ]]; then
    python server.py --data "$data" --pattern "$pattern" --num_agents 40 &
elif [[ "$defense" == "rlr" ]]; then
    python server.py --data "$data" --pattern "$pattern" --UTDDetect True --num_agents 40 &
elif [[ "$defense" == "oracle" ]]; then
    python server.py --data "$data" --pattern "$pattern" --perfect True --num_agents 40 &
    echo "incorrect option"
    exit 1
fi

sleep 10

#Start the poisoned clients
for i in `seq 0 3`; do
    echo "Starting poison client $i"
    python client.py --poison POISON --pattern "$pattern" --clientID $i --data "$data" --use_cuda True --device "cuda:4" --num_agents 40 &
done

#Distrubute the remaining clients accross the 4 GPUs
for i in `seq 4 9`; do
    echo "Starting client $i"
    python client.py --clientID $i --use_cuda True --device "cuda:0" --data "$data" --num_agents 40 &
done


for i in `seq 10 19`; do
    echo "Starting client $i"
    python client.py --clientID $i --use_cuda True --device "cuda:1" --data "$data" --num_agents 40 &
done

for i in `seq 20 29`; do
    echo "Starting client $i"
    python client.py --clientID $i --use_cuda True --device "cuda:2" --data "$data" --num_agents 40 &
done

for i in `seq 30 39`; do
    echo "Starting client $i"
    python client.py --clientID $i --use_cuda True --device "cuda:3" --data "$data" --num_agents 40 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
