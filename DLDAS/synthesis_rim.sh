#!/bin/bash
mkdir rim_paper_synthesis


trainum=3000

mkdir rim_paper_synthesis/$trainum
python test_batch.py --config configs/rim_paper.yaml --input_folder datasets/rim/testA --output_folder rim_paper_synthesis/$trainum --num_style 9 --checkpoint outputs/rim_paper/checkpoints/gen_0000$trainum.pt --a2b 1

python transfer_data.py $trainum rim 9
python elasctic_trans.py rim

