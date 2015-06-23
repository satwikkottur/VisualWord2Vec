% Script to debug the embedding of the new implementation

diff = abs(Pembed - P_embedding);
sum(diff(:))
diff = abs(Rembed - R_embedding);
sum(diff(:))
diff = abs(Sembed - S_embedding);
sum(diff(:))
