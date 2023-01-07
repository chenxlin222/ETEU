function [PSR2,Fmax] = psr2(response)
    sum = 0;
    Fmax= max(max(response));
    Fmean = mean(mean(response));
    for i = 1: size(response,1)
        for j = 1 :size(response,2)
            sum = sum +( response(i,j) - Fmean)^2;
              
        end
    end
    
    deta2 =  sum / (size(response,1) * size(response,2));  
    PSR2 = (Fmax - Fmean)^2 / deta2;
    %disp(PSR2)
    
end