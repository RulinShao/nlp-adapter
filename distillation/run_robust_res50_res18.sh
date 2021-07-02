#!/usr/bin/bash
for i in {1..100}; do
    echo "Number $i: $(date +%Y-%m-%d-%H:%M:%S)"
    python3 torchdistill_main.py --config configs/ilsvrc2012/kd/resnet18_from_robust_resnet50.yaml --log log/ilsvrc2012/kd/resnet18_from_robust_resnet50.txt
done | tee timing.log
