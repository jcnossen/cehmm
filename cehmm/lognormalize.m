function z = lognormalize(z)
    z = z - logsum(z);
end
