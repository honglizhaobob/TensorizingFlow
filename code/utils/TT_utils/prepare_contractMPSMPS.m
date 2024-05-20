function [I,tot] = prepare_contractMPSMPS(S1,S2,idx)


n = numel(S1);
I = cell(1,n);

for i=idx
    
   if i==1
       
       tmp = reshape(tns_mult(S1{i},1,S2{i},1),1,size(S1{i},2)*size(S2{i},2));
       
       I{i} = tmp;
       
   elseif i==n
       
       tmp = reshape(tns_mult(S1{i},2,S2{i},2),size(S1{i},1)*size(S2{i},1),1);
       
       I{i} = tmp;
       
   else
       if ndims(S1{i})<3
           
           
           tmp = reshape(tns_mult(S1{i},2,S2{i},2),size(S1{i},1)*size(S2{i},1),size(S1{i},3)*size(S2{i},3));
           
           I{i} = tmp;
           
           
       else
           
           tmp = reshape(permute(tns_mult(S1{i},2,S2{i},2),[1 3 2 4]),size(S1{i},1)*size(S2{i},1),size(S1{i},3)*size(S2{i},3));
           
           I{i} = tmp;
           
       end
       
   end
   
    
    
end

tot = 1;

for i=idx
    tot = tot*I{i};
    
end