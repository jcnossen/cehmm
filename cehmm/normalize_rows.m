function m=normalize_rows(m)
    for k=1:size(m,1)
        m(k,:) = m(k,:) ./ sum(m(k,:));
    end
end
