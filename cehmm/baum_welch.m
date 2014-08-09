
function [tr_, prior_] = baum_welch(logemission, tr, emitted, prior)
    if nargin == 0
        % test case
        tr=rand(3)+eye(3)*10;
        tr=norm_rows(tr);
        prior = [1 0 0];
        emit_means = 1:3;
        emit_sigma = [.8 .4 .6];
        logemission = @(x, z) normpdf(x,emit_means(z),emit_sigma(z));
        [emitted, seq] = generate_sequence(emit_means, emit_sigma, tr, 1000, prior);
        plot([ emitted seq] );
    end

    [logpost, logalpha, logbeta] = forward_backward(logemission, tr, emitted, prior);
    prior_ = exp(logpost(1,:));
    
    tr_ = zeros(size(tr));
    for i=1:size(tr,1)
        for j=1:size(tr,1)
            % Compute xi(t): probability of being in state i at time t and state j at time t+1
            logalpha(:,i) + tr(i,j) 
            tr_(i,j) = 
        end
    end
end

function m=norm_rows(m)
    for k=1:size(m,1)
        m(k,:) = m(k,:) ./ sum(m(k,:));
    end
end
