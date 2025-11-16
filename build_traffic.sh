#! /bin/bash  
set -euo pipefail
# Main information
echo "Legal traffic: build"
echo ""
echo "Generating traffic..."

cd skyfall || exit 1
# total_timeslots=$1
# cpu_cores=$2
# seq 0 $total_timeslots | xargs -n 1 -P $cpu_cores -I{} bash -c "sudo python3 generate_flow_grid.py {} & sudo python3 generate_flow_circle.py {}"

total_timeslots="${1:?missing total_timeslots}"
cpu_cores="${2:?missing cpu_cores}"


# 각 타임슬롯마다 +grid만 실행
seq 0 "$total_timeslots" | xargs -n 1 -P "$cpu_cores" -I{} \
  bash -c '
    sudo python3 generate_flow_grid.py "$1"
  ' _ {}
cd .. || exit 1

# Calculation finished
echo "Traffic calculation finished!"
