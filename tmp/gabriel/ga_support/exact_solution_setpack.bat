REM variance experiments
python exact_solution_setpack.py --lang pt --maxs 9 --metric variance files/output_ga/exact_setpack_variance.csv
python exact_solution_setpack.py --lang pt --maxs 9 --metric variance files/output_ga/exact_setpack_variance_outliers.csv --wo

REM inertia experiments
python exact_solution_setpack.py --lang pt --maxs 9 --metric inertia files/output_ga/exact_setpack_inertia.csv
python exact_solution_setpack.py --lang pt --maxs 9 --metric inertia files/output_ga/exact_setpack_inertia_outliers.csv --wo

REM silhouette with euclidean distance experiments
python exact_solution_setpack.py --lang pt --maxs 9 --metric euclidean files/output_ga/exact_setpack_euclidean.csv
python exact_solution_setpack.py --lang pt --maxs 9 --metric euclidean files/output_ga/exact_setpack_euclidean_outliers.csv --wo

REM silhouette with cosine distance experiments
python exact_solution_setpack.py --lang pt --maxs 9 --metric cosine files/output_ga/exact_setpack_cosine.csv
python exact_solution_setpack.py --lang pt --maxs 9 --metric cosine files/output_ga/exact_setpack_cosine_outliers.csv --wo

