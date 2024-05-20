function [val]= TRnorm_diff(T1,T2)

% n = numel(T1);
% R1 =cell(1,n); 
% R2 =cell(1,n); 
% 
% for i=1:n
%     indices(i) = size(T1{i},2);
% end
% 
% loc_error = 0;
% index = 1;
% 
% 
% for i1 = 1:size(T1,2)
%     R1{1} = squeeze(T1{1}(:,i1,:));
%     R2{1} = squeeze(T2{1}(:,i1,:));
%     for i2 = 1: indices(2)
%         R1{2} = R1{1}*squeeze(T1{2}(:,i2,:));
%         R2{2} = R2{1}*squeeze(T2{2}(:,i2,:));
%         for i3 = 1: indices(3)
%             R1{3} = R1{2}*squeeze(T1{3}(:,i3,:));
%              R2{3} = R2{2}*squeeze(T2{3}(:,i3,:));
%             for i4 = 1: indices(4)
%                 R1{4} = R1{3}*squeeze(T1{4}(:,i4,:));
%                 R2{4} = R2{3}*squeeze(T2{4}(:,i4,:));
%                 for i5 = 1: indices(5)
%                     R1{5} = R1{4}*squeeze(T1{5}(:,i5,:));
%                     R2{5} = R2{4}*squeeze(T2{5}(:,i5,:));
%                     loc_error(index)= trace(R1{5})-trace(R2{5});
%                     index = index +1;
%                 end
%             end
%         end
%     end
% end
% val = norm(loc_error);
val = TRnorm(TRsubtract(T1,T2));
end