#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <pattern> <defense technique>"
    exit 1
fi

pattern=$1
data="fedmnist"
defense=$2

# Check if the pattern and data variables are not empty
if [ -z "$pattern" ] || [ -z "$data" ] || [ -z "$defense" ]; then
    echo "Error: Pattern, data and defense must not be empty."
    echo "defense options: ours, rlr, none etc."
    exit 1
fi


# start server
if [[ "$defense" == "ours" ]]; then
	python server.py --data "$data" --pattern "$pattern" --percentile True --num_agents 33 &
elif [[ "$defense" == "ours_hybrid" ]]; then
	python server.py --data "$data" --pattern "$pattern" --percentileHybrid True --num_agents 33 &
elif [[ "$defense" == "nodefense" ]]; then
    python server.py --data "$data" --pattern "$pattern" --num_agents 33 &
elif [[ "$defense" == "rlr" ]]; then
    python server.py --data "$data" --pattern "$pattern" --UTDDetect True --num_agents 33 &
elif [[ "$defense" == "oracle" ]]; then
    python server.py --data "$data" --pattern "$pattern" --perfect True --num_agents 33 &
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

# --- Configuration ---
TOTAL_CLIENTS=33
NUM_POISON=4
NUM_NORMAL=$((TOTAL_CLIENTS - NUM_POISON)) # Should be 29
# GPUs
POISON_GPU="cuda:4"
NORMAL_GPU_1="cuda:0"
NORMAL_GPU_2="cuda:1"
NORMAL_GPU_3="cuda:2"
# --- End Configuration ---

# --- Launch Clients ---
# Start the poisoned clients
echo "--- Starting ${NUM_POISON} Poisoned Clients on ${POISON_GPU} ---"
for (( i=0; i<4; i++ )); do
    clientID=$i	
    echo "Starting poison client ${clientID}"
    python client.py \
        --poison POISON \
        --pattern "$pattern" \
        --clientID "$clientID" \
        --data "$data" \
        --use_cuda True \
        --device "$POISON_GPU" \
        --num_agents "$TOTAL_CLIENTS" &
done

# Start the normal clients, distributing manually across GPUs
# We need to launch 29 normal clients. Let's split them: 10, 10, 9

echo "--- Starting 10 Normal Clients on ${NORMAL_GPU_1} ---"
for (( i=4; i<14; i++ )); do
    clientID=$i
    echo "Starting normal client ${clientID}"
    python client.py \
        --clientID "$clientID" \
        --data "$data" \
        --use_cuda True \
        --device "$NORMAL_GPU_1" \
        --num_agents "$TOTAL_CLIENTS" &
done

echo "--- Starting 10 Normal Clients on ${NORMAL_GPU_2} ---"
for (( i=14; i<24; i++ )); do
    clientID=$i
    echo "Starting normal client ${clientID}"
    python client.py \
        --clientID "$clientID" \
        --data "$data" \
        --use_cuda True \
        --device "$NORMAL_GPU_2" \
        --num_agents "$TOTAL_CLIENTS" &
done

echo "--- Starting 9 Normal Clients on ${NORMAL_GPU_3} ---"
for (( i=24; i<33; i++ )); do
    clientID=$i
    echo "Starting normal client ${clientID}"
    python client.py \
        --clientID "$clientID" \
        --data "$data" \
        --use_cuda True \
        --device "$NORMAL_GPU_3" \
        --num_agents "$TOTAL_CLIENTS" &
done

echo "--- All ${TOTAL_CLIENTS} clients launched ---"
