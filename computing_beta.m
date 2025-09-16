function [beta_finals]=computing_beta(stocks_ret,index_ret,index_std,sigma,Codes)

n=size(stocks_ret,2);

for i=1:n
    corr_i = corrcoef(stocks_ret(:,i),(index_ret(:,:))); % Correlation coefficient
    standard_deviation=std(stocks_ret(:,i));
    cov_i = corr_i(1,2)*standard_deviation*index_std; % Covariance = Correlation coefficient * StdStockReturns * StdMarket
    b(i) = cov_i/sigma; % Beta = Cov(i,m)/Var(m)
    beta_finals(i,:) = [Codes(i), b(i)];
end