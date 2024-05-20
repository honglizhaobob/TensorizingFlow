function f = TTeval(A,idx)
%Evaluate a tensor for idx vector

f = A{1}(idx(1),:);

for i=2:numel(A)
   
   f = f*squeeze(A{i}(:,idx(i),:));
end