mkdir npydata
mkdir results

python data.py \
--data rim \
--SupTL_alpha 1.0 \
--SupTL_beta 1.0 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma6 \
--result few_paper \
--test_dir ./data/rim/few/test/specific_paper_test \
--test_type specific_paper

CUDA_VISIBLE_DEVICES=1 python main.py \
--data rim \
--SupTL_alpha 1.0 \
--SupTL_beta 1.0 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma6 \
--result few_paper \
--train_dir ./data/rim/few/train \
--val_dir ./data/rim/few/val \
--test_type specific_paper \
--gt_dir ./rim_answer \
--model_n specific_paper
