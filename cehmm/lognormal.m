function logprob = lognormal(emission, state, mu, sigma)
    logprob = -(emission-mu(state)).^2./(2*sigma(state).^2) - log(sigma(state)*sqrt(2*pi));
end
