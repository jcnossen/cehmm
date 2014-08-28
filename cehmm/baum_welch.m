
function [tr_, prior_] = baum_welch(logemission, tr, emitted, prior)
    if nargin == 0
        % test case
        tr=rand(3)+eye(3)*5;
        tr=normalize_rows(tr);
        prior = [1 0 0];
        emit_means = 1:3;
        emit_sigma = [.1 .1 .1];
        %emission = @(x, z) ( normpdf(x,emit_means(z),emit_sigma(z)) )
        logemission = @(x,z) -(x-emit_means(z)).^2./(2*emit_sigma(z).^2) - log(emit_sigma(z)*sqrt(2*pi));
        [emitted, seq] = generate_sequence(emit_means, emit_sigma, tr, 10000, prior);
        plot([ emitted seq] );
        
        % To test our code, we distort transition matrix and see if it converges to the right
        % values again
        
        tr=normalize_rows(tr.*sqrt(rand(size(tr))));
       
        [tr_, prior_] = baum_welch_iterate(logemission, tr,emitted,prior,1e-4);
        
        tr,tr_
        
    else
        [tr_,prior_]=baum_welch_iterate(logemission,tr,emitted,prior);
    end
end
 
function [tr_, prior_] =  baum_welch_iterate(logemission,tr,emitted,prior, maxdist)
    if nargin<5, maxdist=1e-4;end;
    
    for it=1:60
        tr = normalize_rows(tr);
        
        [logpost, logalpha, logbeta] = forward_backward(logemission, tr, emitted, prior);
        prior_ = exp(logpost(1,:));

        tr_ = zeros(size(tr));
        denum = logsum2(logalpha + logbeta, 2);
        for i=1:size(tr,1)
            for j=1:size(tr,1)
                % Compute xi(t): probability of being in state i at time t and state j at time t+1
                xi = logalpha(1:end-1,i) + log(tr(i,j)) + logbeta(2:end,j) + logemission(emitted(2:end), j) - denum(1:end-1);
            %    tr_(i,j) = 
                tr_(i,j) = exp( logsum(xi) - logsum(logpost(1:end-1,i)) );
            end
        end

        diff=tr-tr_;
        dist=sum(abs(diff(:)));
        fprintf('Dist: %f\n', dist);
        tr=tr_;
        if dist<maxdist, break; end;
    end
end



function s = logsum2(logp, dim)
    b = max (logp,[],dim);
    if isinf(b)
        s = -Inf;
    else
        logp_min_b = logp-repmat(b,size(logp,1)/size(b,1),size(logp,2)/size(b,2));
        s = b + log (sum(exp(logp_min_b),dim));
    end
end


