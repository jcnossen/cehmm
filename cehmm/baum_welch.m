
function [tr_, prior_] = baum_welch(logemission, tr, emitted, prior, maxdist)
    if nargin == 0
        % test case
        tr=rand(3)+eye(3)*5;
        tr=normalize_rows(tr);
        prior = [1 0 0];
        emit_means = 1:3;
        emit_sigma = [.1 .1 .1];
        %emission = @(x, z) ( normpdf(x,emit_means(z),emit_sigma(z)) )
        logemission = @(x,z) lognormal(x,z,emit_means,emit_sigma);
        [emitted, seq] = generate_sequence(emit_means, emit_sigma, tr, 10000, prior);
        plot([ emitted seq] );
        
        % To test our code, we distort transition matrix and see if it converges to the right
        % values again
        
        tr=normalize_rows(tr.*sqrt(rand(size(tr))));
       
        [tr_, prior_] = baum_welch_iterate(logemission, tr,emitted,prior,1e-4);
        
        tr,tr_
        
    else
        if nargin<5, maxdist=1e-4; end;
        
        [tr_,prior_]=baum_welch_iterate(logemission,tr,emitted,prior,maxdist);
    end
end
 
function [tr_, prior_] =  baum_welch_iterate(logemission,tr,emitted,prior, maxdist)
    if nargin<5, maxdist=1e-4;end;
    if ~iscell(emitted), emitted={emitted}; end
    
    for it=1:60
        tr = normalize_rows(tr);
        
        for k=1:length(emitted)
            [logpost{k}, logalpha{k}, logbeta{k}] = forward_backward(logemission, tr, emitted{k}, prior);
            denum{k} = logsum2(logalpha{k} + logbeta{k}, 2);
            % TODO: use other emissions?
            l_post = logpost{k};
            if (k==1), prior_ = exp(l_post(1,:));end;
        end

        tr_ = zeros(size(tr));
        for i=1:size(tr,1)
            for j=1:size(tr,1)
                xi_sum = zeros(1,length(emitted));
                gamma_sum = xi_sum;
                
                % Compute xi(t): probability of being in state i at time t and state j at time t+1
                for k=1:length(emitted)
                    % Copy data from cell matrices for every measurement
                    l_a = logalpha{k}; l_b = logbeta{k};
                    l_post = logpost{k};
                    denum_ = denum{k};
                    emitted_ = emitted{k};
                    
                    % Compute xi
                    xi = l_a(1:end-1,i) + log(tr(i,j)) + l_b(2:end,j) + logemission(emitted_(2:end), j) - denum_(1:end-1);
                    xi_sum(k) = logsum(xi);
                    gamma_sum(k) = logsum(l_post(1:end-1,i));
                end
                
            %    tr_(i,j) = 
                tr_(i,j) = exp( logsum(xi_sum) - logsum(gamma_sum) );
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


