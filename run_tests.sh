#!/bin/bash

NUMBER_OF_THREADS=6

for alg in KMeans BC PageRank; do
  FAIL=0

  echo "Starting" $alg "batch tests."

  epochs=(1 1 1 1 1 1)
  cl=(20 40 100 40 100 100)
  pi=(10 10 10 5 5 20)

  timestamp=$(date +%s)
  mkdir doc_tests/$alg\_$timestamp

  for i in $(seq 1 $NUMBER_OF_THREADS); do
    mkdir doc_tests/$alg\_$timestamp/run_$i
    mkdir doc_tests/$alg\_$timestamp/run_$i/plots
    python run.py --epochs ${epochs[i - 1]} -pi ${pi[i - 1]} --find_centroids_alg $alg -log doc_tests/$alg\_$timestamp/run_$i/$alg\_loss_$i.csv \
      --centroids_plot_file doc_tests/$alg\_$timestamp/run_$i/plots/$alg\_centroids_$i --clustering_plot_file doc_tests/$alg\_$timestamp/run_$i/plots/clustering_plot.png \
      -cl ${cl[i - 1]} 2>doc_tests/$alg\_$timestamp/run_$i/$alg\_test_$i.txt &
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

done
