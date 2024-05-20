function f = TTfunc_eval(A,basis1,basis2,x)
%Evaluate a tensor for idx vector

val = cell(1,numel(A));

val{1} = eval_basis(basis2,x(1));

for i=2:numel(A)-1
    val{i} = eval_basis(basis1,x(i));
end
val{end} = eval_basis(basis2,x(end));

f = tns_mult(val{1},1,A{1},1);

for i=2:numel(A)
   f = f*squeeze(tns_mult(A{i},2,val{i},1));
end

end

function [val] = eval_basis(basis,x)

val = [];

for i=1:numel(basis)
    val = [val; basis{i}(x)];
    
end


end