#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./dataset', download=True)"

# Check if the required arguments are provided
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <pattern> <data> <defense technique> [model]"
    exit 1
fi

pattern=$1
data=$2
defense=$3
model="convnet"

# Optional 4th argument
if [ "$#" -eq 4 ]; then
    model=$4
fi

# Check if the pattern and data variables are not empty
if [ -z "$pattern" ] || [ -z "$data" ] || [ -z "$defense" ]; then
    echo "Error: Pattern, data and defense must not be empty."
    echo "defense options: ours, rlr, none etc."
    exit 1
fi

# start server
if [[ "$defense" == "ours" ]]; then
    python server.py --data "$data" --pattern "$pattern" --percentile True --num_agents 10 --model "$model" &
elif [[ "$defense" == "ours_hybrid" ]]; then
    python server.py --data "$data" --pattern "$pattern" --percentileHybrid True --num_agents 10 --model "$model" &
elif [[ "$defense" == "nodefense" ]]; then
    python server.py --data "$data" --pattern "$pattern" --num_agents 10 --model "$model" --model "$model" &
elif [[ "$defense" == "rlr" ]]; then
    python server.py --data "$data" --pattern "$pattern" --UTDDetect True --num_agents 10 --model "$model" &
elif [[ "$defense" == "oracle" ]]; then
    python server.py --data "$data" --pattern "$pattern" --perfect True --num_agents 10 --model "$model" &
else
    echo "incorrect option"
    exit 1
fi

#sleep 10
echo "Waiting for server to be ready..."

SERVER_HOST="localhost"
SERVER_PORT=8080

for i in {1..30}; do
    if nc -z "$SERVER_HOST" "$SERVER_PORT"; then
        echo "Server is up!"
        break
    fi
    echo "Server not ready yet, retrying in 1 second..."
    sleep 1
done


for i in `seq 0 2`; do
    echo "Starting client $i with poisoning" 
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:4" --poison POISON --pattern "$pattern" --num_agents 10 --model "$model" &
done

for i in `seq 3 4`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:0" --num_agents 10 --model "$model" &
done


for i in `seq 5 6`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:1" --num_agents 10 --model "$model" &
done


for i in `seq 7 9`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:2" --num_agents 10 --model "$model" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
