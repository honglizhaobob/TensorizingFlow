function likes = ftt_eval3(X, C)
    % Authors: Hongli Zhao, Michael Lindsey
    % Evaluates likelihoods from continuous tensor train and samples.
    % on general, non-uniform grid (encoded in C).
    
    % Inputs:
    %       X,                      (2d array) sampled raw data strictly 
    %                               in [-1,1] to evaluate exact likelihoods
    %                               on. Size (d x Ns)
    %       C,                      (tensor_train) coefficient tensor train
    %
    %   
    % Output:
    %       likes,                  (1d array) exact likelihoods for Ns 
    %                               samples.
    
    % get data parameters
    Ns = size(X,2);
    p = C.n(1)-1; 
    d = C.d;
    % preallocate
    likes = zeros(Ns,1);
    Xvec = X(:);
    Phi = permute( reshape(get_legendre(Xvec,p,true),d,Ns,p+1), [3,1,2] );
    for s=1:Ns
      v = einsum(C{1},Phi(:,1,s),2,1);
      for i=2:d
        v = v*einsum(C{i},Phi(:,i,s),2,1);
      end
      likes(s) = v^2;
    end
end