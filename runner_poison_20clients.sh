#!/bin/bash
# with 20 clients and 20% poisoning

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
num_clients=20

# Check if the pattern and data variables are not empty
if [ -z "$pattern" ] || [ -z "$data" ] || [ -z "$defense" ]; then
    echo "Error: Pattern, data and defense must not be empty."
    echo "defense options: ours, rlr, none etc."
    exit 1
fi

# start server
if [[ "$defense" == "ours" ]]; then
    python server.py --data "$data" --pattern "$pattern" --percentile True --num_agents $num_clients --model "resnet" &
elif [[ "$defense" == "ours_hybrid" ]]; then
    python server.py --data "$data" --pattern "$pattern" --percentileHybrid True --num_agents $num_clients --model "resnet" &
elif [[ "$defense" == "nodefense" ]]; then
    python server.py --data "$data" --pattern "$pattern" --num_agents $num_clients --model "resnet" &
elif [[ "$defense" == "rlr" ]]; then
    python server.py --data "$data" --pattern "$pattern" --UTDDetect True --num_agents $num_clients --model "resnet" &
elif [[ "$defense" == "oracle" ]]; then
    python server.py --data "$data" --pattern "$pattern" --perfect True --num_agents $num_clients --model "resnet" &
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


for i in `seq 0 3`; do
    echo "Starting client $i with poisoning"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:3" --poison POISON --pattern "$pattern" --num_agents $num_clients --model "resnet" &
done


for i in `seq 4 9`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:0" --num_agents $num_clients --model "resnet" &
done

for i in `seq 10 14`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:1" --num_agents $num_clients --model "resnet" &
done


for i in `seq 15 19`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "$data" --use_cuda True --device "cuda:2" --num_agents $num_clients --model "resnet" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
