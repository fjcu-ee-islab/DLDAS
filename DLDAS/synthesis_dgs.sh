#!/bin/bash



trainum=28000

mkdir -p dgs_paper_synthesis/$trainum
python test_batch.py --config configs/dgs_paper.yaml --input_folder datasets/dgs/testA --output_folder dgs_paper_synthesis/$trainum --num_style 9 --checkpoint outputs/dgs_paper/checkpoints/gen_000$trainum.pt --a2b 1

python transfer_data.py $trainum dgs 9
python elasctic_trans.py dgs

