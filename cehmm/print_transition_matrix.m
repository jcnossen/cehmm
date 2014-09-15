function print_transition_matrix(tr)
    if nargin==0
        tr=normalize_rows(eye(3)+0.1);
    end

    fprintf('to:\t\t');
    fprintf('%d\t\t',1:size(tr,2));
    fprintf('\nfrom:\n');

    for k=1:size(tr,1)
        fprintf('%d\t',k);
        fprintf('%.5f\t',tr(k,:));
        fprintf('\n');
    end
end
