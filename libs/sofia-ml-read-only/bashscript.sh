#!/bin/bash

fileId=09

# Creating the k means model
nohup ./sofia-kmeans --k 10000 --init_type random --opt_type mini_batch_kmeans --mini_batch_size 100 --iterations 500 --training_file /home/satwik/VisualWord2Vec/data/vis-genome/train/vis_features_${fileId}_libsvm --model_out /home/satwik/VisualWord2Vec/data/vis-genome/train/cluster_model_${fileId}_10k < /dev/null &> out_cluster_${fileId}_10k &

# Obtain the cluster
#nohup ./sofia-kmeans --model_in /home/satwik/VisualWord2Vec/data/vis-genome/train/cluster_model_${fileId}_10k --test_file /home/satwik/VisualWord2Vec/data/vis-genome/train/vis_features_${fileId}_libsvm --cluster_assignments_out /home/satwik/VisualWord2Vec/data/vis-genome/train/cluster_assignment_${fileId}_10k < /dev/null &> out_assign_cluster_${fileId}_10k &
