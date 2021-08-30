mkdir npydata
mkdir results


python data.py \
--data dgs \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma2 \
--result best_dgs \
--test_dir ./data/dgs/paper/test/specific_paper_test/image \
--test_type specific_paper


CUDA_VISIBLE_DEVICES=0 python test.py \
--data dgs \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.8 \
--g_aug Contrast+Gamma2 \
--result best_dgs \
--test_type specific_paper \
--gt_dir ./dgs_answer \
--model_n dgs_paper_best
