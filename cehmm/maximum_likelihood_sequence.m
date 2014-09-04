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

    if nargin==0
        tr=[ 10 0.1 0.1 ; 0.3 7 0.2; 0.3 7 0.2];
        tr=normalize_rows(tr);
        prior = [1 0 0];
        emit_means = 1:3;
        emit_sigma = [1 1 1]*.5;
        %emission = @(x, z) ( normpdf(x,emit_means(z),emit_sigma(z)) )
        logemission = @(x,z) lognormal(x,z,emit_means,emit_sigma);
        [emitted, true_seq] = generate_sequence(emit_means, emit_sigma, tr, 10000, prior);
        [seq,logprob] = maximum_likelihood_sequence(true_seq, tr, logemission, prior);
        plot([ emitted true_seq seq' ] );
    else
        [seq,logprob]=maximum_likelihood_sequence_(samples, tr, logemission, prior);
    end
end

function [seq,logprob]=maximum_likelihood_sequence_(samples, tr, logemission, prior)
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