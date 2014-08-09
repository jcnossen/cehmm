function [seq, logprob] = maximum_likelihood_sequence(samples, tr, logemission, prior)
% Calculate the maximum sequence for an HMM sample, given transition
% matrices, emission probability function and the prior.
% Author: Jelmer Cnossen (j.cnossen@gmail.com)
%
% The function uses the Viterbi algorithm to compute the MLE sequence in
% O(n*m^2) time, where n is the number of samples, and m is the number of
% states.
% Emission is a function or matrix such that:
% emission(x, z) is the probability (density) of emitting x given z, where
% z can be a row vector

    m = size(tr,1);
    n = length(samples);
    if isscalar(prior)
        prior = (1:m)==prior;
    end
    
    logmu = zeros(n,m);
    choice = zeros(n,m); % represents most likely previous state k-1 given x(k) and current state z(k)
    logmu(1,:) = log(prior) + logemission(samples(1,:), 1:m);
    logtr = log(tr);
    
    for k=2:n
        for z=1:m
            x = samples(k,:);
            [v, choice(k,z)] = max( logtr(1:m, z)' + logmu(k-1,1:m) );
            logmu(k,z) = v + logemission(x, z);
        end
    end

    logprob = zeros(1,n);
    seq = zeros(1,n);
    [logprob(n), seq(n)] = max(logmu(n,1:m));
    % backtrack
    for k=n-1:-1:1
        seq(k) = choice(k+1, seq(k+1));
        logprob(k) = logmu(k, seq(k));
    end
end