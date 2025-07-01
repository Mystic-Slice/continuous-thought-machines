python -m tasks.sort.train --seed 42 --N_to_sort 30 --training_iterations 50001 --device 0 --memory_length 1 --log_dir logs/sort/ctm_1_mem --reload

python -m tasks.sort.train --seed 42 --N_to_sort 30 --training_iterations 50001 --device 0 --memory_length 25 --log_dir logs/sort/ctm_25_mem --reload

python -m tasks.sort.train --seed 42 --N_to_sort 30 --training_iterations 50001 --device 0 --memory_length 15 --log_dir logs/sort/ctm_15_mem --reload