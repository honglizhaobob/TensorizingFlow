clear all

%Number of nodes
n = 5;
%Dimension per variable 
d_r = 5;
%Bond dimension
r_b = 2;
%Number of matrices to try
trials=1;
%Number of trials below 10^-4
count_trials = zeros(1,2);
%Average per iteration
iter_trials = zeros(1,2);
for trial=1:trials 
    
%build the tensor ring
Mat=cell(1,n);
for i=1:n
    Mat{i}=abs(rand(r_b,d_r,r_b));
end

for l = 1
R = Mat;
G = TRadd(R,R);

figure()
display('iterative sum -skeleton')
ERR = cell(1,2);
trial
for i=1
    %Iterate for the gauges
    [A_10,ERR{1}] = TRcompress10(G,G,2,1);
    %Optimize the whole tensor
    [A_11,ERR{2}] = TRcompress2(G,A_10);
     
    for j = 1:2
        if ERR{j}(end)<(10^(-4))
            count_trials(j) = count_trials(j) + 1;
        end
        iter_trials(j) = iter_trials(j)+numel(ERR{j});
        loglog(ERR{j},'DisplayName',strcat('skeleton',num2str(j)));
     hold on;
    end
    count_trials
    iter_trials./trial
    hold off
    legend('Location','northeast')
 legend('boxoff')
end

end
end
