function [logpost, loga, logb] = forward_backward(logemission, tr, seq, prior)
% Compute the log posterior probability for the given HMM, using the forward/backward algorithm
% Parameters:
%   emission(x,z):  Matrix or function that defines probability [density]
%                   for emitting x given state z
%   tr:             Matrix(m,m) of transition probabilities. 
%                   tr(i,j) = probability of transition from state i to state j
%                   IAW: tr(i,j) = p(state at k = j | state at k-1 = i)
% Returns
%   logpost(k, i):  Posterior probability of having state i on measurement k

    error(nargchk(3,4,nargin));

    m = size(tr,1);
    if isscalar(prior)
        prior = (1:m)==prior;
    end
    if nargin<4
        prior=ones(1,m)/m;
    end
    n = size(seq,1);
    logprior = log(prior);
    logtr = log(tr);

    % a(k, l) is the alpha(k) for the value of z=l
    % alpha(k, l) = p(x(1:k), z(k) | model)
    loga = zeros(n, m);

    % Forward algorithm:
    % Goal: compute p(z(k), x(1:k))
    loga(1,:) = logprior + logemission(seq(1,:), 1:m);
    for k=2:n
        for z=1:m % z(k)
            loge = logemission(seq(k,:),z);
            loga(k,z) = loge + logsum(loga(k-1, 1:m) + logtr(1:m, z)');
        end
        if mod(k,max(n/10,1))==0, fprintf('fw(%d/%d)\n', k,n); end
    end
    % b(k, l) is the beta(k) for the value of z=l
    % b(k, l) = p(x(k+1:n) | z(k), model)
    logb = zeros(n, m);
    logb(n, :) = 0;
    for k=n-1:-1:1
        for z=1:m % z(k)
            logb(k,z) = logsum( logb(k+1,1:m) + logemission(seq(k+1,:), 1:m) + logtr(z, 1:m) );
        end
        if mod(k,max(n/10,1))==0, fprintf('bw(%d/%d)\n', n-k,n); end
    end
        
    logpost = loga+logb; % posterior (k, z) is probability of z for measurement k
    for k=1:n
        logpost(k,:) = lognormalize(logpost(k,:));
    end
end

