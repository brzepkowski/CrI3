EXAMPLE LAUNCH

(1) First we have to generate all possible combinations of parameters (J, L and Sz):

$ python3 generate_all_parameters.py Lx Ly D J_min J_max L_min L_max

This generates files "parameters_Lx_....csv" and "lattice.png". We can change one filename:

$ mv parameters_Lx_....csv all_params.csv

to make things simpler in the future. WARNING: it's important to include the "all_" prefix at
the beginning, because further we will use script compress_data.sh, which iterates over all files,
whose names begin with "parameters"!

(2) Copy generated parameters in other file (e.g. all_params_copy.csv), so that it is easy
to compare finished calculations with all, that should be conducted.

(3) Create file, in which we will be storing the values of parameters, for which calculations finished succesfully:

$ touch succ_fin_params.csv

(4) LAUNCH "SCREEN"!!!

$ screen

(5) Launch "run_multiple_parameters.sh" in the following way:

$ ./run_multiple_parameters.sh all_params.csv N succ_fin_params.csv,

where N is the number of states, that you want obtain (for each subspace corresponding to given quantum number). For example
N = 3 gives the ground state and first two excited ones (again, IN EACH SUBSPACE).

(!) WARNING: to change the amount of resources needed on BEM, you have to change the values in "run_one_parameter_set.sh" script.

(6) Check, if everything was computed succesfully:

$  diff <(sort all_params_copy.csv -u) <(sort succ_fin_params.csv -u) | tail -n +2 | sed 's/^..//' | cat > all_params.csv

and launch step (5)

(7) Run step (6) as long as there are some parameters left to launch the calculations for.
_______________________________________________________________________________
(8) Make sure, that there are no empty lines in the succ_fin_params.csv.

(9) When all calculations are finished, concatenate separate results into according files by launching:

$ ./compress_data.sh

(10) Make sure, that there are no dummy folders like "parameters___" or "parameters_succ_fin_params". If there are, delete them.

(11) Generate final images by launching:

$ python3 generate_maps.py succ_fin_params.csv
