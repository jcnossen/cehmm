function s = logsum(logp)
    b = max (logp);
    if isinf(b)
        s = -Inf;
    else
        s = b + log (sum(exp(logp-b)));
    end
end

