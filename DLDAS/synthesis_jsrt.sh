#!/bin/bash



trainum=7000

mkdir -p jsrt_paper_synthesis/$trainum
python test_batch.py --config configs/jsrt_paper.yaml --input_folder datasets/jsrt/testA --output_folder jsrt_paper_synthesis/$trainum --num_style 9 --checkpoint outputs/jsrt_paper/checkpoints/gen_0000$trainum.pt --a2b 1

python transfer_data.py $trainum jsrt 9
python elasctic_trans.py jsrt
