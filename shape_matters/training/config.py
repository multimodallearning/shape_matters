

cfg = {
    "num_runs": 1,
    "its_per_run": 15000,
    "batch_size": 64,
    "cosine_annealing_T_0": 6000,
    "cosine_annealing_T_mult": 2,
    "learning_rate": .0005,
    "patch_size": [96,64,80],
    "rand_cutout_size": [16,8,12],
    "rand_cutout_num": 8,
    "rand_shift_max_offset": [16,8,12]
}

cfg_test = {
    "num_runs": 1,
    "its_per_run": 1500,
    "batch_size": 8,
    "cosine_annealing_T_0": 3,
    "cosine_annealing_T_mult": 2,
    "learning_rate": .0005,
    "patch_size": [96,64,80],
    "rand_cutout_size": [16,8,12],
    "rand_cutout_num": 8,
    "rand_shift_max_offset": [16,8,12]
}

cfg_mlp = {
    "num_runs": 2,
    "its_per_run": 6000,
    "batch_size": 8,
    "learning_rate": .001,
    "patch_size": [96,64,80],
}



