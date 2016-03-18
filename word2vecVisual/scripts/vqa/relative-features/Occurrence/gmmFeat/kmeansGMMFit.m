function G = kmeansGMMFit(data,NComponents)
    I = kmeans(data,NComponents,...
        'emptyaction','drop','start','uniform',...
        'replicates',5);
    labels = unique(I);
    NComponents = numel(labels);
    mu = zeros(NComponents,2);
    Sigma = zeros(1,2,NComponents);
    PComponents = zeros(1,NComponents);
    for j=1:numel(labels)
        inCluster = I==labels(j);
        mu(j,:) = mean(data(inCluster,:));
        %hack to make the clusters a tiny bit better for
        %encoding features
        Sigma(:,:,j) = var(data(inCluster,:))*2;
        PComponents(j) = mean(inCluster);
    end
    G.mu = mu; 
    G.Sigma = Sigma; 
    G.PComponents = PComponents;
    G.NComponents = numel(labels);    
end
