mkdir npydata
mkdir results

python data.py \
--data dgs \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma2 \
--result paper \
--test_dir ./data/dgs/paper/test/specific_paper_test/image \
--test_type specific_paper

CUDA_VISIBLE_DEVICES=1 python main.py \
--data dgs \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma2 \
--result paper \
--train_dir ./data/dgs/paper/train \
--val_dir ./data/dgs/paper/val \
--test_type specific_paper \
--gt_dir ./dgs_answer \
--model_n specific_paper
