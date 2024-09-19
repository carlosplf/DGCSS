#!/bin/bash

FAIL=0

echo "Testing KMeans..."
echo "Starting KMeans batch tests."

timestamp=$(date +%s)
mkdir doc_tests/kmeans_$timestamp

python run.py --epochs 146 --find_centroids_alg KMeans -el doc_tests/kmeans_$timestamp/kmeans_loss_1.csv --centroids_plot_file doc_tests/kmeans_$timestamp/kmeans_centroids_1 -cl 10 2>doc_tests/kmeans_$timestamp/kmeans_test_1.txt &
python run.py --epochs 146 --find_centroids_alg KMeans -el doc_tests/kmeans_$timestamp/kmeans_loss_2.csv --centroids_plot_file doc_tests/kmeans_$timestamp/kmeans_centroids_2 -cl 20 2>doc_tests/kmeans_$timestamp/kmeans_test_2.txt &
python run.py --epochs 146 --find_centroids_alg KMeans -el doc_tests/kmeans_$timestamp/kmeans_loss_3.csv --centroids_plot_file doc_tests/kmeans_$timestamp/kmeans_centroids_3 -cl 40 2>doc_tests/kmeans_$timestamp/kmeans_test_3.txt &
python run.py --epochs 146 --find_centroids_alg KMeans -el doc_tests/kmeans_$timestamp/kmeans_loss_4.csv --centroids_plot_file doc_tests/kmeans_$timestamp/kmeans_centroids_4 -cl 100 2>doc_tests/kmeans_$timestamp/kmeans_test_4.txt &

for job in $(jobs -p); do
  echo $job
  wait $job || let "FAIL+=1"
done

python run.py --epochs 246 --find_centroids_alg KMeans -el doc_tests/kmeans_$timestamp/kmeans_loss_5.csv --centroids_plot_file doc_tests/kmeans_$timestamp/kmeans_centroids_5 -cl 40 2>doc_tests/kmeans_$timestamp/kmeans_test_5.txt &
python run.py --epochs 246 --find_centroids_alg KMeans -el doc_tests/kmeans_$timestamp/kmeans_loss_6.csv --centroids_plot_file doc_tests/kmeans_$timestamp/kmeans_centroids_6 -cl 60 2>doc_tests/kmeans_$timestamp/kmeans_test_6.txt &
python run.py --epochs 126 --find_centroids_alg KMeans -el doc_tests/kmeans_$timestamp/kmeans_loss_7.csv --centroids_plot_file doc_tests/kmeans_$timestamp/kmeans_centroids_7 -cl 40 -pi 5 2>doc_tests/kmeans_$timestamp/kmeans_test_7.txt &
python run.py --epochs 146 --find_centroids_alg KMeans -el doc_tests/kmeans_$timestamp/kmeans_loss_8.csv --centroids_plot_file doc_tests/kmeans_$timestamp/kmeans_centroids_8 -cl 80 -pi 5 2>doc_tests/kmeans_$timestamp/kmeans_test_8.txt &

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

FAIL=0

echo "Testing KCore..."
echo "Starting KCore batch tests."

timestamp=$(date +%s)
mkdir doc_tests/kcore_$timestamp

python run.py --epochs 146 --find_centroids_alg KCore -el doc_tests/kcore_$timestamp/kcore_loss_1.csv --centroids_plot_file doc_tests/kcore_$timestamp/kcore_centroids_1 -cl 10 2>doc_tests/kcore_$timestamp/kcore_test_1.txt &
python run.py --epochs 146 --find_centroids_alg KCore -el doc_tests/kcore_$timestamp/kcore_loss_2.csv --centroids_plot_file doc_tests/kcore_$timestamp/kcore_centroids_2 -cl 20 2>doc_tests/kcore_$timestamp/kcore_test_2.txt &
python run.py --epochs 146 --find_centroids_alg KCore -el doc_tests/kcore_$timestamp/kcore_loss_3.csv --centroids_plot_file doc_tests/kcore_$timestamp/kcore_centroids_3 -cl 40 2>doc_tests/kcore_$timestamp/kcore_test_3.txt &
python run.py --epochs 146 --find_centroids_alg KCore -el doc_tests/kcore_$timestamp/kcore_loss_4.csv --centroids_plot_file doc_tests/kcore_$timestamp/kcore_centroids_4 -cl 100 2>doc_tests/kcore_$timestamp/kcore_test_4.txt &

for job in $(jobs -p); do
  echo $job
  wait $job || let "FAIL+=1"
done

python run.py --epochs 256 --find_centroids_alg KCore -el doc_tests/kcore_$timestamp/kcore_loss_5.csv --centroids_plot_file doc_tests/kcore_$timestamp/kcore_centroids_5 -cl 40 2>doc_tests/kcore_$timestamp/kcore_test_5.txt &
python run.py --epochs 126 --find_centroids_alg KCore -el doc_tests/kcore_$timestamp/kcore_loss_6.csv --centroids_plot_file doc_tests/kcore_$timestamp/kcore_centroids_6 -cl 40 -pi 5 2>doc_tests/kcore_$timestamp/kcore_test_6.txt &
python run.py --epochs 126 --find_centroids_alg KCore -el doc_tests/kcore_$timestamp/kcore_loss_7.csv --centroids_plot_file doc_tests/kcore_$timestamp/kcore_centroids_7 -cl 100 -pi 5 2>doc_tests/kcore_$timestamp/kcore_test_7.txt &
python run.py --epochs 126 --find_centroids_alg KCore -el doc_tests/kcore_$timestamp/kcore_loss_8.csv --centroids_plot_file doc_tests/kcore_$timestamp/kcore_centroids_8 -cl 200 -pi 5 2>doc_tests/kcore_$timestamp/kcore_test_8.txt &

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

echo "Testing Pagerank..."
echo "Starting Pagerank batch tests."

timestamp=$(date +%s)
mkdir doc_tests/pagerank_$timestamp

python run.py --epochs 146 --find_centroids_alg PageRank -el doc_tests/pagerank_$timestamp/pagerank_loss_1.csv --centroids_plot_file doc_tests/pagerank_$timestamp/pagerank_centroids_1 -cl 10 2>doc_tests/pagerank_$timestamp/pagerank_test_1.txt &
python run.py --epochs 146 --find_centroids_alg PageRank -el doc_tests/pagerank_$timestamp/pagerank_loss_2.csv --centroids_plot_file doc_tests/pagerank_$timestamp/pagerank_centroids_2 -cl 20 2>doc_tests/pagerank_$timestamp/pagerank_test_2.txt &
python run.py --epochs 146 --find_centroids_alg PageRank -el doc_tests/pagerank_$timestamp/pagerank_loss_3.csv --centroids_plot_file doc_tests/pagerank_$timestamp/pagerank_centroids_3 -cl 40 2>doc_tests/pagerank_$timestamp/pagerank_test_3.txt &
python run.py --epochs 146 --find_centroids_alg PageRank -el doc_tests/pagerank_$timestamp/pagerank_loss_4.csv --centroids_plot_file doc_tests/pagerank_$timestamp/pagerank_centroids_4 -cl 100 2>doc_tests/pagerank_$timestamp/pagerank_test_4.txt &

for job in $(jobs -p); do
  echo $job
  wait $job || let "FAIL+=1"
done

python run.py --epochs 256 --find_centroids_alg PageRank -el doc_tests/pagerank_$timestamp/pagerank_loss_5.csv --centroids_plot_file doc_tests/pagerank_$timestamp/pagerank_centroids_5 -cl 40 2>doc_tests/pagerank_$timestamp/pagerank_test_5.txt &
python run.py --epochs 126 --find_centroids_alg PageRank -el doc_tests/pagerank_$timestamp/pagerank_loss_6.csv --centroids_plot_file doc_tests/pagerank_$timestamp/pagerank_centroids_6 -cl 40 -pi 5 2>doc_tests/pagerank_$timestamp/pagerank_test_6.txt &
python run.py --epochs 126 --find_centroids_alg PageRank -el doc_tests/pagerank_$timestamp/pagerank_loss_7.csv --centroids_plot_file doc_tests/pagerank_$timestamp/pagerank_centroids_7 -cl 100 -pi 5 2>doc_tests/pagerank_$timestamp/pagerank_test_7.txt &
python run.py --epochs 126 --find_centroids_alg PageRank -el doc_tests/pagerank_$timestamp/pagerank_loss_8.csv --centroids_plot_file doc_tests/pagerank_$timestamp/pagerank_centroids_8 -cl 200 -pi 5 2>doc_tests/pagerank_$timestamp/pagerank_test_8.txt &

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

echo "Testing Random seeds..."
echo "Starting Random seeds batch tests."

timestamp=$(date +%s)
mkdir doc_tests/random_$timestamp

python run.py --epochs 146 --find_centroids_alg Random -el doc_tests/random_$timestamp/random_loss_1.csv --centroids_plot_file doc_tests/random_$timestamp/random_centroids_1 -cl 10 2>doc_tests/random_$timestamp/random_test_1.txt &
python run.py --epochs 146 --find_centroids_alg Random -el doc_tests/random_$timestamp/random_loss_2.csv --centroids_plot_file doc_tests/random_$timestamp/random_centroids_2 -cl 20 2>doc_tests/random_$timestamp/random_test_2.txt &
python run.py --epochs 146 --find_centroids_alg Random -el doc_tests/random_$timestamp/random_loss_3.csv --centroids_plot_file doc_tests/random_$timestamp/random_centroids_3 -cl 40 2>doc_tests/random_$timestamp/random_test_3.txt &
python run.py --epochs 146 --find_centroids_alg Random -el doc_tests/random_$timestamp/random_loss_4.csv --centroids_plot_file doc_tests/random_$timestamp/random_centroids_4 -cl 100 2>doc_tests/random_$timestamp/random_test_4.txt &

for job in $(jobs -p); do
  echo $job
  wait $job || let "FAIL+=1"
done

python run.py --epochs 256 --find_centroids_alg Random -el doc_tests/random_$timestamp/random_loss_5.csv --centroids_plot_file doc_tests/random_$timestamp/random_centroids_5 -cl 40 2>doc_tests/random_$timestamp/random_test_5.txt &
python run.py --epochs 126 --find_centroids_alg Random -el doc_tests/random_$timestamp/random_loss_6.csv --centroids_plot_file doc_tests/random_$timestamp/random_centroids_6 -cl 40 -pi 5 2>doc_tests/random_$timestamp/random_test_6.txt &
python run.py --epochs 126 --find_centroids_alg Random -el doc_tests/random_$timestamp/random_loss_7.csv --centroids_plot_file doc_tests/random_$timestamp/random_centroids_7 -cl 100 -pi 5 2>doc_tests/random_$timestamp/random_test_7.txt &
python run.py --epochs 126 --find_centroids_alg Random -el doc_tests/random_$timestamp/random_loss_8.csv --centroids_plot_file doc_tests/random_$timestamp/random_centroids_8 -cl 200 -pi 5 2>doc_tests/random_$timestamp/random_test_8.txt &

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

echo "Testing Betweenness Centrality..."
echo "Starting Betweenness Centrality batch tests."

timestamp=$(date +%s)
mkdir doc_tests/betweenness_centrality_$timestamp

python run.py --epochs 146 --find_centroids_alg BC -el doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_loss_1.csv --centroids_plot_file doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_centroids_1 -cl 10 2>doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_test_1.txt &
python run.py --epochs 146 --find_centroids_alg BC -el doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_loss_2.csv --centroids_plot_file doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_centroids_2 -cl 20 2>doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_test_2.txt &
python run.py --epochs 146 --find_centroids_alg BC -el doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_loss_3.csv --centroids_plot_file doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_centroids_3 -cl 40 2>doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_test_3.txt &
python run.py --epochs 146 --find_centroids_alg BC -el doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_loss_4.csv --centroids_plot_file doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_centroids_4 -cl 100 2>doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_test_4.txt &

for job in $(jobs -p); do
  echo $job
  wait $job || let "FAIL+=1"
done

python run.py --epochs 256 --find_centroids_alg BC -el doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_loss_5.csv --centroids_plot_file doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_centroids_5 -cl 40 2>doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_test_5.txt &
python run.py --epochs 126 --find_centroids_alg BC -el doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_loss_6.csv --centroids_plot_file doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_centroids_6 -cl 40 -pi 5 2>doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_test_6.txt &
python run.py --epochs 126 --find_centroids_alg BC -el doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_loss_7.csv --centroids_plot_file doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_centroids_7 -cl 100 -pi 5 2>doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_test_7.txt &
python run.py --epochs 126 --find_centroids_alg BC -el doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_loss_8.csv --centroids_plot_file doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_centroids_8 -cl 200 -pi 5 2>doc_tests/betweenness_centrality_$timestamp/betweenness_centrality_test_8.txt &

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
