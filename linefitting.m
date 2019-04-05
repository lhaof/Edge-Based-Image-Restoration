function [ A, B, C ] = linefitting( data1, data2 )

[fit1, gof1] = fit(data1, data2, 'poly1', 'Lower', [-1e6, -inf], 'Upper', [1e6, inf]);
[fit2, gof2] = fit(data2, data1, 'poly1', 'Upper', [-1e6, -inf], 'Upper', [1e6, inf]);
disp(fit1);
disp(gof1);
disp(fit2);
disp(gof2);
end

