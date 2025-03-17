#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./dataset', download=True)"

pattern=$1

echo "Starting client 0 with poisoning"
python client.py --clientID 0 --data "cifar10" --use_cuda True --device "cuda:4" --poison POISON --pattern "$pattern" &


for i in `seq 1 3`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "cifar10" --use_cuda True --device "cuda:0" &
done


for i in `seq 4 6`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "cifar10" --use_cuda True --device "cuda:1"  &
done


for i in `seq 7 9`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "cifar10" --use_cuda True --device "cuda:2" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
