#!/bin/bash

NUMBER_OF_THREADS=6

FAIL=0

echo "Testing KMeans..."
echo "Starting KMeans batch tests."

epochs=(124 124 124 124 124 124)
cl=(10 20 40 100 100 100)
pi=(10 10 10 10 10 10)

timestamp=$(date +%s)
mkdir doc_tests/kmeans_$timestamp

for i in $(seq 0 $NUMBER_OF_THREADS); do
  mkdir doc_tests/kmeans_$timestamp/run_$i
  mkdir doc_tests/kmeans_$timestamp/run_$i/plots
  python run.py --epochs ${epochs[i]} -pi ${pi[i]} --find_centroids_alg KMeans -log doc_tests/kmeans_$timestamp/run_$i/kmeans_loss_$i.csv \
    --centroids_plot_file doc_tests/kmeans_$timestamp/run_$i/plots/kmeans_centroids_$i --clustering_plot_file doc_tests/kmeans_$timestamp/run_$i/plots/clustering_plot.png \
    -cl ${cl[i]} 2>doc_tests/kmeans_$timestamp/run_$i/kmeans_test_$i.txt &
done

for job in $(jobs -p); do
  echo $job
  wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ]; then
  echo "Batch finished. No fails."
else
  echo "# of FAILS: ($FAIL)"
fi
