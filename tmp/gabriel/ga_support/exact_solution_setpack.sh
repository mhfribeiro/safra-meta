#!/bin/bash
# silhouette with cosine distance experiments
./exact_solution_setpack.py --lang pt --maxs 9 --metric cosine --save_log files/output_ga/exact_setpack_cosine.csv
./exact_solution_setpack.py --lang pt --maxs 9 --metric cosine --save_log files/output_ga/exact_setpack_cosine_outliers.csv --wo

# silhouette with euclidean distance experiments
./exact_solution_setpack.py --lang pt --maxs 9 --metric euclidean --save_log files/output_ga/exact_setpack_euclidean.csv
./exact_solution_setpack.py --lang pt --maxs 9 --metric euclidean --save_log files/output_ga/exact_setpack_euclidean_outliers.csv --wo

# variance experiments
./exact_solution_setpack.py --lang pt --maxs 9 --metric variance --save_log files/output_ga/exact_setpack_variance.csv
./exact_solution_setpack.py --lang pt --maxs 9 --metric variance --save_log files/output_ga/exact_setpack_variance_outliers.csv --wo

# inertia experiments
./exact_solution_setpack.py --lang pt --maxs 9 --metric inertia --save_log files/output_ga/exact_setpack_inertia.csv
./exact_solution_setpack.py --lang pt --maxs 9 --metric inertia --save_log files/output_ga/exact_setpack_inertia_outliers.csv --wo

