#!/bin/bash
# Silhouette with cosine
./exact_solution_setpack.py --lang pt --metric cosine --save_log files/output_ga/exact_setpack_cosine.csv
./ga_fs_setpack.py --lang pt --metric sc --selec bin --save_log --ngen 300 --pop 50 --mincxpb 0.8 --elsize 0.5 --mutpb 0.02 --divfac 0.5 --verbose files/output_ga/ga_setpack_cosine.csv
./exact_solution_setpack.py --lang pt --metric cosine --save_log files/output_ga/exact_setpack_cosine_outliers.csv --wo
./ga_fs_setpack.py --lang pt --metric sc --selec bin --save_log --ngen 300 --pop 50 --mincxpb 0.8 --elsize 0.5 --mutpb 0.02 --divfac 0.5 --verbose files/output_ga/ga_setpack_cosine_outliers.csv --wo
# Silhouette with euclidean
./exact_solution_setpack.py --lang pt --metric euclidean --save_log files/output_ga/exact_setpack_euclidean.csv
./ga_fs_setpack.py --lang pt --metric se --selec bin --save_log --ngen 300 --pop 50 --mincxpb 0.8 --elsize 0.5 --mutpb 0.02 --divfac 0.5 --verbose files/output_ga/ga_setpack_euclidean.csv
./exact_solution_setpack.py --lang pt --metric euclidean --save_log files/output_ga/exact_setpack_euclidean_outliers.csv --wo
./ga_fs_setpack.py --lang pt --metric se --selec bin --save_log --ngen 300 --pop 50 --mincxpb 0.8 --elsize 0.5 --mutpb 0.02 --divfac 0.5 --verbose files/output_ga/ga_setpack_euclidean_outliers.csv --wo
# Variance experiments
./exact_solution_setpack.py --lang pt --metric variance --save_log files/output_ga/exact_setpack_variance.csv
./ga_fs_setpack.py --lang pt --metric v --selec bin --save_log --ngen 300 --pop 50 --mincxpb 0.8 --elsize 0.5 --mutpb 0.02 --divfac 0.5 --verbose files/output_ga/ga_setpack_variance.csv
./exact_solution_setpack.py --lang pt --metric variance --save_log files/output_ga/exact_setpack_variance_outliers.csv --wo
./ga_fs_setpack.py --lang pt --metric v --selec bin --save_log --ngen 300 --pop 50 --mincxpb 0.8 --elsize 0.5 --mutpb 0.02 --divfac 0.5 --verbose files/output_ga/ga_setpack_variance_outliers.csv --wo
# Inertia experiments
./exact_solution_setpack.py --lang pt --metric inertia --save_log files/output_ga/exact_setpack_inertia.csv
./ga_fs_setpack.py --lang pt --metric i --selec bin --save_log --ngen 300 --pop 50 --mincxpb 0.8 --elsize 0.5 --mutpb 0.02 --divfac 0.5 --verbose files/output_ga/ga_setpack_inertia.csv
./exact_solution_setpack.py --lang pt --metric inertia --save_log files/output_ga/exact_setpack_inertia_outliers.csv --wo
./ga_fs_setpack.py --lang pt --metric i --selec bin --save_log --ngen 300 --pop 50 --mincxpb 0.8 --elsize 0.5 --mutpb 0.02 --divfac 0.5 --verbose files/output_ga/ga_setpack_inertia_outliers.csv --wo

