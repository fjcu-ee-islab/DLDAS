mkdir npydata
mkdir results


python data.py \
--data rim \
--SupTL_alpha 1.0 \
--SupTL_beta 1.0 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma6 \
--result best_few_rim \
--test_dir ./data/rim/few/test/specific_paper_test/image \
--test_type specific_paper


CUDA_VISIBLE_DEVICES=0 python test.py \
--data rim \
--SupTL_alpha 1.0 \
--SupTL_beta 1.0 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma6 \
--result best_few_rim \
--test_type specific_paper \
--gt_dir ./rim_answer \
--model_n rim_paper_few_best
