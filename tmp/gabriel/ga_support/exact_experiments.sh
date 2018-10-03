#!/bin/bash
# variance experiments
./exact_solution.py --lang pt --maxs 9 --metric variance files/output_ga/exact_variance.csv
./exact_solution.py --lang pt --maxs 9 --metric variance files/output_ga/exact_variance_outliers.csv --wo
# inertia experiments
./exact_solution.py --lang pt --maxs 9 --metric inertia files/output_ga/exact_inertia.csv
./exact_solution.py --lang pt --maxs 9 --metric inertia files/output_ga/exact_inertia_outliers.csv --wo
# silhouette with euclidean distance experiments
./exact_solution.py --lang pt --maxs 9 --metric euclidean files/output_ga/exact_euclidean.csv
./exact_solution.py --lang pt --maxs 9 --metric euclidean files/output_ga/exact_euclidean_outliers.csv --wo
# silhouette with cosine distance experiments
./exact_solution.py --lang pt --maxs 9 --metric cosine files/output_ga/exact_cosine.csv
./exact_solution.py --lang pt --maxs 9 --metric cosine files/output_ga/exact_cosine_outliers.csv --wo

