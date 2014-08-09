%   tr:             Matrix(m,m) of transition probabilities. 
%                   tr(i,j) = probability of transition from state i to state j
%                   IAW: tr(i,j) = p(state at k = j | state at k-1 = i)
function [emitted, seq] = generate_sequence(means, stddev, tr, N, prior)

    if nargin<3
        tr=rand(3)+eye(3)*10;
        means = 1:3;
        stddev = [0.1 0.1 0.05];
        tr=norm_rows(tr);
    end
    if nargin<5,
        prior = ones(1,size(tr,1))./size(tr,1);
    end
    if nargin<4
        N=1000;
    end
    
    trc = cumsum(tr,2);
    priorc = cumsum(prior);
    
    seq = zeros(N,1);
    emitted = zeros(N,1);
    seq(1) = pick_state(priorc);
    emitted(1) = randn() * stddev(seq(1)) + means(seq(1));
    for k=2:N
        seq(k) = pick_state( trc(seq(k-1), :) );
        emitted(k) =  randn() * stddev(seq(1)) + means(seq(k));
    end
end


function m=norm_rows(m)
    for k=1:size(m,1)
        m(k,:) = m(k,:) ./ sum(m(k,:));
    end
end

% Choose a discrete random value based on cumulative probability probc
function state=pick_state(probc)
    r = rand();
    for state=1:length(probc)
        if r<probc(state), break; end
    end
end

