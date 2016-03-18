function encoding = gmmEncodeF(data,G)
    %Get an encoding of a vector via a Gaussian mixture model
    %This is the fast version which uses no for loops and
    mixing = G.PComponents;
    mu = G.mu;
    Sigma = squeeze(G.Sigma)';    
    NComponents = numel(mixing);    
    likelihoods = eps + mixing' .* ...
                  prod((1 ./ (sqrt(2*pi .* Sigma))) .* ...
                  exp(-0.5*(bsxfun(@minus,data,mu).^2 ./ Sigma)),2);
    encoding = likelihoods / sum(likelihoods);  
end