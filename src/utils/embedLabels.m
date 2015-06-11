function embedding = embedLabels(labels, model)
    % Function to embed the words using a word2vec
    % Wrapper for the original function embed_str(.)
    % 
    % usage:
    % embedding = embedLabels(labels, model);
    % 
    % Input: 
    % labels = The unique labels for which are to be embedded
    % model = The word2vec modelf used for embedding
    % 
    % embeddings = The embeddings of the labels using the model

    embedding = cell(length(labels), 1);
    for i = 1:length(labels)
        embedding{i} = embed_str(labels{i}, model.tokens, model.fv);
    end
    embedding = cell2mat(embedding);
end
