Inside src/,
javac NegSampler.java
java NegSampler <Dataset> <Batch_size> <Num_neg_per_batch> <#_batches>

Example:
java NegSampler ../../Datasets/Linkedin/network_edgelist.txt 100 5 5000