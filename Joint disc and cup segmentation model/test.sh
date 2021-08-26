mkdir npydata
mkdir results

python test.py \
--data dgs \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.9 \
--g_aug Contrast+Gamma2 \
--result few_paper \
--test_type specific_paper \
--gt_dir ./dgs_answer \
--model_n specific_paper
