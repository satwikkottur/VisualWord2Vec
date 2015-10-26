function tuples = readImprovedTuples(tupleFile, embedFile)
    fileId = fopen(tupleFile);
    tuples = textscan(fileId, '<%[^<>:]:%[^<>:]:%[^<>:]>(%f,%f)\n');
    fclose(fileId);
end
