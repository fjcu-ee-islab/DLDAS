mkdir npydata
mkdir results


python data.py \
--data dgs \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.9 \
--g_aug Contrast+Gamma2 \
--result best_few_dgs \
--test_dir ./2roi_test/dgs/specific_paper_test \
--test_type specific_paper


CUDA_VISIBLE_DEVICES=0 python test.py \
--data dgs \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.9 \
--g_aug Contrast+Gamma2 \
--result best_few_dgs \
--test_type specific_paper \
--gt_dir ./dgs_answer \
--model_n dgs_paper_few_best
