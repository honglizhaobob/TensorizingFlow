function [error_T] = TRpwerror_relative(R,approxR, title_f)
index = 1;
R_size = size(R{1});

for i1 = 1:R_size(2)
    for i2 = 1:R_size(2)
        for i3 = 1:R_size(2)
            for i4 = 1:R_size(2)
                for i5 = 1:R_size(2)
                    error(index) =  1-TReval(approxR,[i1 i2 i3 i4 i5])/TReval(R,[i1 i2 i3 i4 i5]);
                    error_tot(index)= abs(TReval(approxR,[i1 i2 i3 i4 i5])-TReval(R,[i1 i2 i3 i4 i5]));
                    %display(['index ' num2str(i1) num2str(i2) num2str(i3) num2str(i4) num2str(i5) ' rval ' num2str(TReval(R,[i1 i2 i3 i4 i5])) ' apval ' num2str(TReval(approxR,[i1 i2 i3 i4 i5]))]);
                    index = index +1;
                end
            end

        end
    end
end
figure()
histogram(error)
title(title_f)
saveas(gcf,strcat(title_f,'histo.png'))
error_T = sum(error_tot);
end