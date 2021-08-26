mkdir npydata
mkdir results


python data.py \
--data jsrt \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 1.0 \
--g_aug Contrast+Gamma2 \
--result best_jsrt \
--test_dir ./2roi_test/jsrt/specific_paper_test \
--test_type specific_paper


CUDA_VISIBLE_DEVICES=0 python test.py \
--data jsrt \
--SupTL_alpha 0.5 \
--SupTL_beta 0.5 \
--SupTL_gamma 0.0 \
--SupTL_k 1.0 \
--g_aug Contrast+Gamma2 \
--result best_jsrt \
--test_type specific_paper \
--gt_dir ./jsrt_answer \
--model_n jsrt_paper_best
