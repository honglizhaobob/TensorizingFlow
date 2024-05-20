%Resistor network
Rtop = 2;
n = 15;

r = 1:0.25:Rtop;

for i=1:n
    for k=1:numel(r)
         if i==1
            R{i}(k,:)   = [r(k) 1]; 
         elseif i==n
            R{i}(:,k)   = [1; r(k)];
         elseif i>1&&i<n
            R{i}(:,k,:) = [1 0; r(k) 1];
         end
    end
   
end

%conductance for resistors in series
G = TTreciprocal_newton(R,14,0.2,10^-8);

%log of resistors
G = TTln(R, 15, 12, 10^-8, 0.2, 10^-8);
