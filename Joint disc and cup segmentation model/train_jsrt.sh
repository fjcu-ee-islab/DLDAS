mkdir npydata
mkdir results

python data.py \
--data jsrt \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 1.0 \
--g_aug Contrast+Gamma2 \
--result paper \
--test_dir ./data/jsrt/paper/test/specific_paper_test \
--test_type specific_paper

CUDA_VISIBLE_DEVICES=1 python main.py \
--data jsrt \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma2 \
--result paper \
--train_dir ./data/jsrt/paper/train \
--val_dir ./data/jsrt/paper/val \
--test_type specific_paper \
--gt_dir ./jsrt_answer \
--model_n specific_paper
