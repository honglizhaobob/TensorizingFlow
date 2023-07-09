function A = checkRLortho(TT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author(s): Yuehaw Khoo
%
% Checks tensor train `TT` is right-left orthogonal.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I = cell(numel(TT),1);
for i=fliplr(2:numel(TT))
    if i==numel(TT)
        I{i} = tns_mult(TT{i},2,TT{i},2);   
    else 
        tmp = tns_mult(TT{i},2,TT{i},2);
        I{i} = permute(tmp,[1 3 2 4]); 
    end
end

A = 1;
for i=2:numel(TT)
    A = A*TT{i};
end
end