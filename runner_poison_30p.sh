#!/bin/bash
# attack ratio of 30% : 30% of 10 clients = 3 malicious clients
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./dataset', download=True)"

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <pattern> <data> <defense technique>"
    exit 1
fi

pattern=$1
data=$2
defense=$3

# Check if the pattern and data variables are not empty
if [ -z "$pattern" ] || [ -z "$data" ] || [ -z "$defense" ]; then
    echo "Error: Pattern, data and defense must not be empty."
    echo "defense options: ours, rlr, none etc."
    exit 1
fi

# start server
if [[ "$defense" == "ours" ]]; then
    python server.py --data "$data" --pattern "$pattern" --percentile True --num_agents 10 &
elif [[ "$defense" == "ours_hybrid" ]]; then
    python server.py --data "$data" --pattern "$pattern" --percentileHybrid True --num_agents 10 &
elif [[ "$defense" == "nodefense" ]]; then
    python server.py --data "$data" --pattern "$pattern" --num_agents 10 &
elif [[ "$defense" == "rlr" ]]; then
    python server.py --data "$data" --pattern "$pattern" --UTDDetect True --num_agents 10 &
elif [[ "$defense" == "oracle" ]]; then
    python server.py --data "$data" --pattern "$pattern" --perfect True --num_agents 10 &
else
    echo "incorrect option"
    exit 1
fi

sleep 10

for i in `seq 0 2`; do
    echo "Starting client $i with poisoning"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:4" --poison POISON --pattern "$pattern" --num_agents 10 &
done

for i in `seq 3 4`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:0" --num_agents 10 &
done


for i in `seq 5 6`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:1" --num_agents 10 &
done


for i in `seq 7 9`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:2" --num_agents 10 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
