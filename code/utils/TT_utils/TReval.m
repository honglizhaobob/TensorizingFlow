function f = TReval(A,idx)
%Evaluate a tensor ring for idx vector

m_f = squeeze(A{1}(:,idx(1),:));

for i=2:numel(A)
   
   m_f = m_f*squeeze(A{i}(:,idx(i),:));
end
f = trace(m_f);
end