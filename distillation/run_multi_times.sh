#!/usr/bin/bash
for i in {1..100}; do
    echo "Number $i: $(date +%Y-%m-%d-%H:%M:%S)"
    ./run_torchdistill_main.sh
done | tee timing.log