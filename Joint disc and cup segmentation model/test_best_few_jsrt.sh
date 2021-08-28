mkdir npydata
mkdir results


python data.py \
--data jsrt \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.9 \
--g_aug Contrast+Gamma2 \
--result best_few_jsrt \
--test_dir ./data/jsrt/few/test/specific_paper_test \
--test_type specific_paper


CUDA_VISIBLE_DEVICES=0 python test.py \
--data jsrt \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 0.9 \
--g_aug Contrast+Gamma2 \
--result best_few_jsrt \
--test_type specific_paper \
--gt_dir ./jsrt_answer \
--model_n jsrt_paper_few_best
