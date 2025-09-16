%Data daily

% data import 
pathname = 'C:\RICERCA\exam 18 25 jan 2024\nuovo ';
filename = 'data for exam 2024.xlsx'; 
[~, Sheet] = xlsfinfo(filename);

[data,textdata]=xlsread(filename,Sheet{1}); 

% data handling 
data_string = textdata(4:end,1); 
name_string = textdata(1,2:end); 
code_string = textdata(2,2:end);

%Daily returns
daily_ret = 100.*tick2ret(data(1:end,1:end),"Method","continuous"); % Daily returns

% Daily statistics
daily_mean = nanmean(daily_ret); % Mean
daily_ann_ret = 252.*daily_mean; % Annualized return
daily_std = nanstd(daily_ret); % Standard deviation
daily_ann_std = sqrt(252).*daily_std; % Annualized volatility
daily_variance = nanvar(daily_ret); % Variance
daily_skew = skewness(daily_ret); % Skweness
daily_kurt= kurtosis(daily_ret); % Kutosis

%Table
Table_daily_ret=table(daily_ann_ret',daily_ann_std',daily_mean',daily_std', daily_variance', daily_skew', daily_kurt', ...
    'VariableNames', {'Ann. return', 'Ann. vol','Mean', 'St Deviation', 'Variance', 'Skewness', 'Kurtosis'}, ...
    'RowNames', code_string);

disp('Statistics for returns in daily terms (%)') 
disp(Table_daily_ret); fprintf('\n')

%From Matlab to Excel
filename = 'daily_ret.xlsx';
writematrix(daily_ret,filename,'Sheet',1);

filename = 'Table_daily_ret.xlsx';
writetable(Table_daily_ret,filename,'Sheet',1,'WriteRowNames',true)

%Daily variance-covariance and correlation matrix

% data import 
pathname = 'C:\RICERCA\exam 18 25 jan 2024\nuovo ';
filename = 'data for exam 2024.xlsx'; 
[~, Sheet] = xlsfinfo(filename);

[data,textdata]=xlsread(filename,Sheet{1}); 

% data handling 
data_string = textdata(4:end,1); 
name_string = textdata(1,2:end); 
code_string = textdata(2,2:end);

%Daily returns
daily_ret = 100.*tick2ret(data(1:end,1:end),"Method","continuous");

%Variance-Covariance matrix
daily_V_C_matrix = cov(daily_ret, 'partialrows'); 

figure(1)
hetamap_var_daily=heatmap(daily_V_C_matrix, "ColorbarVisible","on","XData",code_string, "Ydata",code_string);

%Correlation matrix
daily_corr_matrix= corr(daily_ret, 'rows','complete');

figure(2)
hetamap_corr_matrix_daily=heatmap(daily_corr_matrix, "ColorbarVisible","on","XData",code_string, "Ydata",code_string);

%Data daily

%data import
pathname = 'C:\RICERCA\exam 18 25 jan 2024\nuovo ';
filename = 'data for exam 2024.xlsx'; 
[~, Sheet] = xlsfinfo(filename);

[data,textdata]=xlsread(filename,Sheet{1}); 

%Daily returns
daily_ret = 100.*tick2ret(data(1:end,1:end),"Method","continuous");

vector_securities_chosen=[18 69 31 70 51 52 57 36 78 66 40 60];
vector_code_chosen=transpose({'ARN','VIN','CLT','EDNR','TOD','REC','JUVE','AMP','SOL','ITM','IKG','B'});

daily_mean = nanmean(daily_ret); % Mean
daily_ann_ret = 252.*daily_mean; % Annualized return
daily_std = nanstd(daily_ret); % Standard deviation
daily_ann_std = sqrt(252).*daily_std; % Annualized volatility
daily_variance = nanvar(daily_ret); % Variance
daily_skew = skewness(daily_ret); % Skweness
daily_kurt= kurtosis(daily_ret); % Kutosis

daily_selected=data(:,vector_securities_chosen);
daily_ret_selected=daily_ret(:,vector_securities_chosen);

%Statistics of selected daily stoks
Table_statistics_selected_daily_ret=table(vector_code_chosen, ...
    daily_mean(:,vector_securities_chosen)', ...
    daily_std(:,vector_securities_chosen)', ...
    daily_variance(:,vector_securities_chosen)', ...
    daily_skew(:,vector_securities_chosen)', ...
    daily_kurt(:,vector_securities_chosen)',...
    'VariableNames', {'Asset', 'Mean', 'St Deviation', 'Variance', 'Skewness', 'Kurtosis'})

%From Matlab to Excel

filename = 'Table_statistics_selected_daily_ret.xlsx';
writetable(Table_statistics_selected_daily_ret,filename,'Sheet',1,'WriteRowNames',true)

%%
figure(1)
final_day=length(data(:,1));

m=length(vector_securities_chosen);
for i=1:m
    t=datetime(2015,1,1)+caldays(1:final_day);
    y=data(:,vector_securities_chosen(i));
    plot(t,y);
    xtickformat('MM-dd-yy')
    hold on
end
title('Behaviour of the daily prices');
legend(vector_code_chosen,'Location','southoutside','Orientation','horizontal');
xlabel('Date')
ylabel('daily prices');

%%
v = daily_selected(1,:);
for i=1:m
    daily_comulative_ret_selected(:,i)=(daily_selected(:,i)-v(i))*(1/v(i));
end
figure(2)

for i=1:m
    t=datetime(2015,1,1)+caldays(1:final_day-1);
    z=daily_comulative_ret_selected(2:end,i);
    plot(t,z);
    xtickformat('MM-dd-yy')
    hold on
end
title('Daily comulative returns');
legend(vector_code_chosen,'Location','southoutside','Orientation','horizontal');
xlabel('Date')
ylabel('daily comulative returns');


%%
%Data daily

% data import 
pathname = 'C:\RICERCA\exam 18 25 jan 2024\nuovo ';
filename = 'data for exam 2024.xlsx'; 
[~, Sheet] = xlsfinfo(filename);

[data,textdata]=xlsread(filename,Sheet{1}); 

vector_securities_chosen=[18 69 31 70 51 52 57 36 78 66 40 60];
vector_code_chosen=transpose({'ARN','VIN','CLT','EDNR','TOD','REC','JUVE','AMP','SOL','ITM','IKG','B'});
vector_cap=[48.21, 84.82, 41.44, 166, 1009, 10372, 642,6660, 2412,1160,280,63.29];
vector_cap=vector_cap/sum(vector_cap);

%Daily returns
daily_ret = 100.*tick2ret(data(1:end,1:end),"Method","continuous");

%Daily selected returns
daily_selected=transpose(data(:,vector_securities_chosen));
daily_ret_selected=daily_ret(:,vector_securities_chosen);
%%
%Portfolio with negative weights allowed
portMV_daily_sh = Portfolio('AssetList', vector_code_chosen, 'NumAssets', 12, 'LowerBudget', 1, 'UpperBudget',1, 'LowerBound', -1, 'UpperBound', 1); % MV portfolio with Budget equals to the sum of the weigths
portMV_daily_sh = estimateAssetMoments(portMV_daily_sh,daily_ret_selected);
portMV_daily_std_sh = (sqrt(diag(portMV_daily_sh.AssetCovar)));


weightsMV_fin_daily_sh = portMV_daily_sh.estimateMaxSharpeRatio; % We compute the MV optimal weights that maximise the Sharpe Ratio
[MV_daily_sig_fin_sh, MV_daily_mu_fin_sh]=estimatePortMoments(portMV_daily_sh,weightsMV_fin_daily_sh);
MV_daily_sh_srp=estimatePortSharpeRatio(portMV_daily_sh,weightsMV_fin_daily_sh); % Sharpe Ratio


%%

%Portfolio with negative weights NOT allowed
portMV_daily_lo = Portfolio('AssetList', vector_code_chosen, 'NumAssets', 12, 'LowerBudget', 1, 'UpperBudget',1, 'LowerBound', 0, 'UpperBound', 1); % MV portfolio with Budget equals to the sum of the weigths
portMV_daily_lo = estimateAssetMoments(portMV_daily_lo,daily_ret_selected);
portMV_daily_std_lo = (sqrt(diag(portMV_daily_lo.AssetCovar)));

weightsMV_daily_lo=estimateFrontier(portMV_daily_lo,25); % We estimate the MV frontier for 25 portfolios
[MV_daily_sigma_lo,MV_daily_mu_lo]=estimatePortMoments(portMV_daily_lo,weightsMV_daily_lo);

weightsMV_fin_daily_lo = portMV_daily_lo.estimateMaxSharpeRatio; % We compute the MV optimal weights that maximise the Sharpe Ratio
[MV_daily_sig_fin_lo, MV_daily_mu_fin_lo]=estimatePortMoments(portMV_daily_lo,weightsMV_fin_daily_lo);
MV_daily_lo_srp=estimatePortSharpeRatio(portMV_daily_lo,weightsMV_fin_daily_lo); % Sharpe Ratio

%Table with weights (Unconstr. & Constr.)
Table_weightsMV_fin_daily_sh_and_lo=table(weightsMV_fin_daily_sh,weightsMV_fin_daily_lo, ...
    'VariableNames', {'Daily MV. Port. weights (Unconstr.)','Daily MV. Port. weights (Constr.)'}, ...
    'RowNames', vector_code_chosen);

disp('Mean-Variance Portfolio Weights (Unconstr. & Constr.)') 
disp(Table_weightsMV_fin_daily_sh_and_lo); fprintf('\n')

filename = 'Table_weightsMV_fin_daily_sh_and_lo.xlsx';
writetable(Table_weightsMV_fin_daily_sh_and_lo,filename,'Sheet',1,'WriteRowNames',true)
%%

portMV_daily_ret_selected_sh=daily_ret_selected*weightsMV_fin_daily_sh; % We compute the weighted returns

MV_daily_sh_mu=mean(portMV_daily_ret_selected_sh); % Mean of the optimal portfolio
MV_daily_sh_variance=var(portMV_daily_ret_selected_sh); % Variance of the optimal portfolio
MV_daily_sh_std=sqrt(MV_daily_sh_variance); % Std Deviation of the optimal portfolio
MV_daily_sh_skew=skewness(portMV_daily_ret_selected_sh); % Skewness of the optimal portfolio
MV_daily_sh_kurt=kurtosis(portMV_daily_ret_selected_sh); % Kurtosis of the optimal portfolio

% Optimal MV portfolio with non-negativity constraint on weights
portMV_daily_ret_selected_lo=daily_ret_selected*weightsMV_fin_daily_lo; % We compute the weighted returns

MV_daily_lo_mu=mean(portMV_daily_ret_selected_lo); % Mean of the optimal portfolio
MV_daily_lo_variance=var(portMV_daily_ret_selected_lo); % Variance of the optimal portfolio
MV_daily_lo_std=sqrt(MV_daily_lo_variance); % Std Deviation of the optimal portfolio
MV_daily_lo_skew=skewness(portMV_daily_ret_selected_lo); % Skewness of the optimal portfolio
MV_daily_lo_kurt=kurtosis(portMV_daily_ret_selected_lo); % Kurtosis of the optimal portfolio

%Statistic of daily Unconst. & Constr. Portfolio
Table_statistic_daily_sh_and_lo=table([MV_daily_sh_mu, MV_daily_sh_std, MV_daily_sh_variance, MV_daily_sh_skew, MV_daily_sh_kurt]', ...
    [MV_daily_lo_mu, MV_daily_lo_std, MV_daily_lo_variance, MV_daily_lo_skew, MV_daily_lo_kurt]', ...
    'VariableNames',{'Unconstr. daily portfolio', 'Constr. daily portfolio'},...
    'RowNames', {'Mean', 'St Deviation', 'Variance', 'Skewness', 'Kurtosis'});

disp('Statistic of daily Unconst. & Constr. Portfolio') 
disp(Table_statistic_daily_sh_and_lo); fprintf('\n')

%Table Statistic of daily Unconst. & Constr. Portfolio
filename = 'Table_statistic_daily_sh_and_lo.xlsx';
writetable(Table_statistic_daily_sh_and_lo,filename,'Sheet',1,'WriteRowNames',true)

%%

%Plot the efficient frontiers

% Unconstrainted
figure(3);
portMV_daily_sh.plotFrontier(25); hold on; 
scatter(MV_daily_sig_fin_sh, MV_daily_mu_fin_sh, 'filled', 'green','LineWidth',1); hold on; 
text(MV_daily_sig_fin_sh, MV_daily_mu_fin_sh, 'pMVd', 'FontSize',10); hold on; 

scatter(portMV_daily_std_sh,portMV_daily_sh.AssetMean, 'filled', 'yellow','LineWidth',1); hold on; 
text(portMV_daily_std_sh,portMV_daily_sh.AssetMean,portMV_daily_sh.AssetList, 'FontSize',10); hold on;

%scatter(portGMinV_daily_std,portGMinV_daily_mean, 'filled', 'red','LineWidth',1); hold on; 
%text(portGMinV_daily_std,portGMinV_daily_mean,'GMinV','FontSize',10); hold off;


legend('MV Efficient Frontier - Daily',...
       'Optimal portfolio', ...
       'Assets', ...
       'Location','Northwest');
legend('boxoff')


% Constrainted
figure(4)
portMV_daily_lo.plotFrontier(25); hold on; 
scatter(MV_daily_sig_fin_lo, MV_daily_mu_fin_lo, 'filled', 'green','LineWidth',1); hold on; 
text(MV_daily_sig_fin_lo, MV_daily_mu_fin_lo, 'pMVd', 'FontSize',10); hold on; 
scatter(portMV_daily_std_lo,portMV_daily_lo.AssetMean, 'filled', 'yellow','LineWidth',1); hold on; 
%text(portMV_daily_std_lo,portMV_daily_lo.AssetMean,portMV_daily_lo.AssetList, 'FontSize',10); hold on;
%scatter(portGMinV_daily_std,portGMinV_daily_mean, 'filled', 'red','LineWidth',1); hold on; 
%text(portGMinV_daily_std,portGMinV_daily_mean,'GMinV','FontSize',10); hold off;
legend('MV Efficient Frontier - Daily',...
       'Optimal portfolio', ...
       'Assets', ...
       'Location','Northwest');
legend('boxoff')

%%

%import data
pathname = 'C:\RICERCA\exam 18 25 jan 2024\nuovo ';
filename = 'data for exam 2024.xlsx'; 
[~, Sheet] = xlsfinfo(filename);

[data]=xlsread(filename,Sheet{4},'H:H'); 

%Daily returns
daily_ret_index = 100.*tick2ret(data(1:end,1:end),"Method","continuous");

% Daily statistics
daily_index_mean = mean(daily_ret_index); %standard deviation
daily_index_std = std(daily_ret_index); %standard deviation
daily_index_variance = var(daily_ret_index); %variance
daily_index_skew = skewness(daily_ret_index); %skweness
daily_index_kurt = kurtosis(daily_ret_index); %kurtosis

%Difference between index and Portfolio (Constraint and Unconstraint)
Table_daily_difference_FTSE_MV_lo_and_sh=table([daily_index_mean, daily_index_std, daily_index_variance, daily_index_skew, daily_index_kurt]', ...
    [MV_daily_lo_mu, MV_daily_lo_std, MV_daily_lo_variance, MV_daily_lo_skew, MV_daily_lo_kurt]', ...
    [MV_daily_sh_mu, MV_daily_sh_std, MV_daily_sh_variance, MV_daily_sh_skew, MV_daily_sh_kurt]', ...
    'VariableNames',{'FTSE ITALIA All Share', 'Constr. daily portfolio','Unconstr. daily portfolio'},...
    'RowNames', {'Mean', 'St Deviation', 'Variance', 'Skewness', 'Kurtosis'})

filename = 'Table_daily_diffenrence_FTSE_Port.xlsx';
writetable(Table_daily_difference_FTSE_MV_lo_and_sh,filename,'Sheet',1,'WriteRowNames',true)

%%

%Computing beta is an our function that computes the beta of CAPM theory
[betas_assets_daily] = computing_beta(daily_ret_selected,daily_ret_index,daily_index_std,daily_index_variance,vector_code_chosen)
betas_asset_daily_vector = cell2mat(betas_assets_daily(:,2));

%Unconstraint
beta_MV_daily_sh=betas_asset_daily_vector'*weightsMV_fin_daily_sh;

%Constraint
beta_MV_daily_lo=betas_asset_daily_vector'*weightsMV_fin_daily_lo;

betas_daily_lo= [betas_asset_daily_vector; beta_MV_daily_lo];
betas_daily_sh= [betas_asset_daily_vector; beta_MV_daily_sh];

%Tables:
%Beta assests daily

Table_betas_assets_daily = table(vector_code_chosen(1:12), betas_daily_sh(1:12), 'VariableNames',{'Asset','Daily Beta'})

%Beta Portfolio Unconstrint & Constraint
Table_portMV_betas_daily = table(betas_daily_sh(13),betas_daily_lo(13),'VariableNames',{'Daily beta Unconstraint', 'Daily beta Constraint'})

%From MatLab to Excel
filename = 'Table_betas_assets_daily.xlsx';
writetable(Table_betas_assets_daily,filename,'Sheet',1,'WriteRowNames',true)

filename = 'Table_portMV_betas_daily.xlsx';
writetable(Table_portMV_betas_daily,filename,'Sheet',1,'WriteRowNames',true)


%%

rf=0.03; % Risk-free rate
daily_RiskFree=rf/252; % Daily risk-free rate

%We choose Centrale Del Latte (CLT) & Tod's (TOD)

% Daily frequency
figure(5);
hold on
axis([0 1.1 -0.08 0.15])
beta=0:0.01:4;
title('Security Market Line (Daily)')
xlabel('Beta') 
ylabel('Mean return')
daily_sml=daily_RiskFree+beta*(daily_index_mean-daily_RiskFree);
plot(beta,daily_sml);

%SML
scatter(0, daily_RiskFree, 'filled', 'red','LineWidth',1) % Risk-free rate point
text(0, daily_RiskFree, 'RF', 'FontSize', 10);

%TOD
scatter(betas_asset_daily_vector(5),portMV_daily_lo.AssetMean(5,:),'filled', 'green','LineWidth',1)  % Observed return for TOD
text(betas_asset_daily_vector(5),portMV_daily_lo.AssetMean(5,:),'TOD', 'FontSize', 10)

%CLT
scatter(betas_asset_daily_vector(3),portMV_daily_lo.AssetMean(3,:),'filled', 'green','LineWidth',1) % Observed return for CLT
text(betas_asset_daily_vector(3),portMV_daily_lo.AssetMean(3,:),'CLT', 'FontSize', 10) 

%Long portfolio
scatter(beta_MV_daily_lo, MV_daily_lo_mu,'filled', 'green','LineWidth',1) % Observed return for our Constraint portfolio
text(beta_MV_daily_lo, MV_daily_lo_mu,'PORTlo', 'FontSize', 10) 

%Short portfolio 
scatter(beta_MV_daily_sh, MV_daily_sh_mu,'filled', 'green','LineWidth',1) % Observed return for our Unonstraint portfolio
text(beta_MV_daily_sh, MV_daily_sh_mu,'PORTsh', 'FontSize', 10) 

legend('SML','Risk-Free','Observed values','Location','NorthWest')
hold off 

%With all securities
figure(6);
hold on
axis([0 1.1 -0.08 0.15])
beta=0:0.01:4;
title('Security Market Line (Daily)')
xlabel('Beta') 
ylabel('Mean return')
daily_sml=daily_RiskFree+beta*(daily_index_mean-daily_RiskFree);
plot(beta,daily_sml);

%SML
scatter(0, daily_RiskFree, 'filled', 'red','LineWidth',1) % Risk-free rate point
text(0, daily_RiskFree, 'RF', 'FontSize', 10);

%TOD
scatter(betas_asset_daily_vector(5),portMV_daily_lo.AssetMean(5,:),'filled', 'green','LineWidth',1)  % Observed return for TOD
text(betas_asset_daily_vector(5),portMV_daily_lo.AssetMean(5,:),'TOD', 'FontSize', 10)

%ARN
scatter(betas_asset_daily_vector(1),portMV_daily_lo.AssetMean(1,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(1),portMV_daily_lo.AssetMean(1,:),'ARN', 'FontSize', 10)

%CLT
scatter(betas_asset_daily_vector(3),portMV_daily_lo.AssetMean(3,:),'filled', 'green','LineWidth',1) % Observed return for CLT
text(betas_asset_daily_vector(3),portMV_daily_lo.AssetMean(3,:),'CLT', 'FontSize', 10) 

%Long portfolio
scatter(beta_MV_daily_lo, MV_daily_lo_mu,'filled', 'green','LineWidth',1) % Observed return for our Constraint portfolio
text(beta_MV_daily_lo, MV_daily_lo_mu,'PORTlo', 'FontSize', 10) 

%Short portfolio 
scatter(beta_MV_daily_sh, MV_daily_sh_mu,'filled', 'green','LineWidth',1) % Observed return for our Unonstraint portfolio
text(beta_MV_daily_sh, MV_daily_sh_mu,'PORTsh', 'FontSize', 10) 

%other 

%ARN
scatter(betas_asset_daily_vector(1),portMV_daily_lo.AssetMean(1,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(1),portMV_daily_lo.AssetMean(1,:),'ARN', 'FontSize', 10)

%VIN
scatter(betas_asset_daily_vector(2),portMV_daily_lo.AssetMean(2,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(2),portMV_daily_lo.AssetMean(2,:),'VIN', 'FontSize', 10)

%EDNR
scatter(betas_asset_daily_vector(4),portMV_daily_lo.AssetMean(4,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(4),portMV_daily_lo.AssetMean(4,:),'EDNR', 'FontSize', 10)

%REC
scatter(betas_asset_daily_vector(6),portMV_daily_lo.AssetMean(6,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(6),portMV_daily_lo.AssetMean(6,:),'REC', 'FontSize', 10)

%JUVE
scatter(betas_asset_daily_vector(7),portMV_daily_lo.AssetMean(7,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(7),portMV_daily_lo.AssetMean(7,:),'JUVE', 'FontSize', 10)

%AMP
scatter(betas_asset_daily_vector(8),portMV_daily_lo.AssetMean(8,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(8),portMV_daily_lo.AssetMean(8,:),'AMP', 'FontSize', 10)

%SOL
scatter(betas_asset_daily_vector(9),portMV_daily_lo.AssetMean(9,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(9),portMV_daily_lo.AssetMean(9,:),'SOL', 'FontSize', 10)

%ITM
scatter(betas_asset_daily_vector(10),portMV_daily_lo.AssetMean(10,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(10),portMV_daily_lo.AssetMean(10,:),'ITM', 'FontSize', 10)

%IKG
scatter(betas_asset_daily_vector(11),portMV_daily_lo.AssetMean(11,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(11),portMV_daily_lo.AssetMean(11,:),'IKG', 'FontSize', 10)

%B
scatter(betas_asset_daily_vector(12),portMV_daily_lo.AssetMean(12,:),'filled', 'green','LineWidth',1)
text(betas_asset_daily_vector(12),portMV_daily_lo.AssetMean(12,:),'B', 'FontSize', 10)

legend('SML','Risk-Free','Observed values','Location','NorthWest')
hold off 

%% 
n_views=4; % Number of views
daily_pick=zeros(n_views,size(daily_ret_selected',1)); % Pick matrix
daily_views_vect=zeros(n_views,1); % Vectors of views

% View 1
daily_pick(1,2)=1; % Absolute view Vianini
daily_views_vect(1)=0.006; % Vianini is going to have a 0.6%

% View 2 
daily_pick(2,3)=1; % Absolute view for Centrale del latte
daily_views_vect(2)=-0.008; % Centrale del latte is going to have a -0.8%

% View 3
daily_pick(3,9)=1; % Relative Sol
daily_pick(3,6)=-1; % Related to Recordati
daily_views_vect(3)=0.0025; 

% View 4 
daily_pick(4,4)=1; % Relative Edison
daily_pick(4,5)=-1; % Related to Tod's
daily_views_vect(4)=0.008;

%weights of the market
W_market=vector_cap;

daily_sigma=portMV_daily_sh.AssetCovar;
n_days=size(daily_ret_selected,1); % Number of days
n_assets=size(daily_ret_selected,2); % Number of assets
daily_tau=1/n_days; 

daily_omega_diagonale = diag(daily_pick*daily_tau*daily_sigma*daily_pick');
daily_omega = diag(daily_omega_diagonale);

daily_srp=0.5/sqrt(252); % We decided to calibrate with a fixed annualised Sharpe Ratio of 0.5 
daily_delta=daily_srp/daily_index_std;
daily_pi=daily_delta*daily_sigma*W_market';

daily_BL_mu=inv((inv((daily_tau*daily_sigma))+transpose(daily_pick)*inv(daily_omega)*daily_pick))*((inv((daily_tau*daily_sigma)))*daily_pi+transpose(daily_pick)*inv(daily_omega)*daily_views_vect);
daily_BL_cov=inv((inv((daily_tau*daily_sigma))+transpose(daily_pick)*inv(daily_omega)*daily_pick));

%Portfolio Unconstreint
daily_portBL_sh=Portfolio('AssetList', vector_code_chosen, 'NumAssets', 12, 'LowerBound',-1,'UpperBound',1,'LowerBudget',1,'UpperBudget',1);
daily_portBL_sh=setAssetMoments(daily_portBL_sh,daily_BL_mu,daily_BL_cov+daily_sigma);
daily_weights_BL_sh=estimateFrontier(daily_portBL_sh,25);
[Bayes_daily_mu_sh]=estimatePortMoments(daily_portBL_sh,daily_weights_BL_sh);
daily_weights_BL_final_sh=daily_portBL_sh.estimateMaxSharpeRatio;
[BL_daily_sigma_srp_sh, BL_daily_mu_srp_sh]=estimatePortMoments(daily_portBL_sh,daily_weights_BL_final_sh);

%Portfolio Constreint
daily_portBL_lo=Portfolio('AssetList', vector_code_chosen, 'NumAssets', 12, 'LowerBound',0,'UpperBound',1,'LowerBudget',1,'UpperBudget',1);
daily_portBL_lo=setAssetMoments(daily_portBL_lo,daily_BL_mu,daily_BL_cov+daily_sigma);
daily_weights_BL_lo=estimateFrontier(daily_portBL_lo,25);
[Bayes_daily_mu_lo]=estimatePortMoments(daily_portBL_sh,daily_weights_BL_lo);
daily_weights_BL_final_lo=daily_portBL_lo.estimateMaxSharpeRatio;
[BL_daily_sigma_srp_lo, BL_daily_mu_srp_lo]=estimatePortMoments(daily_portBL_sh,daily_weights_BL_final_lo);

clf
figure(7);
ax1=subplot(2,2,1);
idx=weightsMV_fin_daily_lo>0.001;
pie(ax1,weightsMV_fin_daily_lo(idx));
legend(vector_code_chosen(idx),'Location','southeastoutside');
title(ax1,'Mean Variance (Constr.)');

ax2=subplot(2,2,2);
idx=daily_weights_BL_final_lo>0.001;
pie(ax2,daily_weights_BL_final_lo);
legend(vector_code_chosen(idx),'Location','southeastoutside');
title(ax2,'Black-Litterman (Constr.)');

ax1=subplot(2,2,3);
idx=weightsMV_fin_daily_sh>0.001;
pie(ax1,weightsMV_fin_daily_sh(idx));
legend(vector_code_chosen(idx),'Location','southeastoutside');
title(ax1,'Mean Variance (Unonstr.)');

ax2=subplot(2,2,4);
idx=daily_weights_BL_final_sh>0.001;
pie(ax2,daily_weights_BL_final_sh(idx));
legend(vector_code_chosen(idx),'Location','southeastoutside');
title(ax2,'Black-Litterman (Unonstr.)');

% Statistics of Black-Litterman Model Constraint
portBL_daily_ret_selected_lo=daily_ret_selected*daily_weights_BL_final_lo; % We compute the weighted returns
portBL_daily_mean_lo=mean(portBL_daily_ret_selected_lo);
portBL_daily_variance_lo=var(portBL_daily_ret_selected_lo);
portBL_daily_std_lo=sqrt(portBL_daily_variance_lo);
portBL_daily_skew_lo=skewness(portBL_daily_ret_selected_lo);
portBL_daily_kurt_lo=kurtosis(portBL_daily_ret_selected_lo);
portBL_daily_srp_lo=estimatePortSharpeRatio(daily_portBL_lo,daily_weights_BL_final_lo); % Sharpe Ratio

% Statistics of Black-Litterman Model Unconstraint
portBL_daily_ret_selected_sh=daily_ret_selected*daily_weights_BL_final_sh; % We compute the weighted returns
portBL_daily_mean_sh=mean(portBL_daily_ret_selected_sh);
portBL_daily_variance_sh=var(portBL_daily_ret_selected_sh);
portBL_daily_std_sh=sqrt(portBL_daily_variance_sh);
portBL_daily_skew_sh=skewness(portBL_daily_ret_selected_sh);
portBL_daily_kurt_sh=kurtosis(portBL_daily_ret_selected_sh);
portBL_daily_srp_sh=estimatePortSharpeRatio(daily_portBL_lo,daily_weights_BL_final_sh); % Sharpe Ratio

Table_statatistics_daily_MV_and_BL = table([MV_daily_mu_fin_lo,MV_daily_lo_std,MV_daily_lo_variance,MV_daily_lo_skew,MV_daily_lo_kurt,MV_daily_lo_srp]',...
    [portBL_daily_mean_lo,portBL_daily_std_lo,portBL_daily_variance_lo,portBL_daily_skew_lo,portBL_daily_kurt_lo,portBL_daily_srp_lo]',...
    [MV_daily_mu_fin_sh,MV_daily_sh_std,MV_daily_sh_variance,MV_daily_sh_skew,MV_daily_sh_kurt,MV_daily_sh_srp]',...
    [portBL_daily_mean_sh,portBL_daily_std_sh,portBL_daily_variance_sh,portBL_daily_skew_sh,portBL_daily_kurt_sh,portBL_daily_srp_sh]',...
    'VariableNames',{'MV daily portfolio (Con.)', 'BL daily portfolio (Con.)','MV daily portfolio (Unc.)', 'BL daily portfolio (Unc.)'},...
    'RowNames', {'Mean', 'St Deviation', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe Ratio'})

%From Matlab to Excel

filename = 'Table_statistics_daily_MV_and_BL.xlsx';
writetable(Table_statatistics_daily_MV_and_BL,filename,'Sheet',1,'WriteRowNames',true)

%%

daily_mu_prior=portMV_daily_lo.AssetMean+1*(sqrt(diag(daily_sigma))); % mu_0
daily_lambda=2*daily_sigma; %New Varaince_Covariance_Matrix

daily_Bayes_mu=(inv(n_assets*inv(daily_sigma)+inv(daily_lambda)))*(n_assets*inv(daily_sigma)*portMV_daily_lo.AssetMean+inv(daily_lambda)*daily_mu_prior); % Mean of the standard Bayesian model (page:46)
daily_Bayes_sigma=inv(n_assets*inv(daily_sigma)+inv(daily_lambda)); % Standard deviation of the standard Bayesian  model


daily_portBayes=Portfolio('AssetList', vector_code_chosen, 'NumAssets', 12,'LowerBudget',1,'UpperBudget',1,'LowerBound',0,'UpperBound',1); % We create the portfolio 
daily_portBayes.AssetMean=daily_Bayes_mu;
daily_portBayes.AssetCovar=daily_Bayes_sigma;
daily_weights_Bayes=estimateFrontier(daily_portBayes,25);
[Bayes_daily_sigma, Bayes_daily_mu]=estimatePortMoments(daily_portBayes, daily_weights_Bayes);
daily_weights_Bayes_srp=daily_portBayes.estimateMaxSharpeRatio;
[Bayes_daily_sigma_srp, Bayes_daily_mu_srp]=estimatePortMoments(daily_portBayes,daily_weights_Bayes_srp);

clf
figure(9)
ax1=subplot(1,3,1);
idx=weightsMV_fin_daily_lo>0.001;
pie(ax1,weightsMV_fin_daily_lo(idx));
legend(vector_code_chosen(idx),'Location','southoutside');
title(ax1,'Mean Variance (Constr.)');

ax2=subplot(1,3,2);
idx=daily_weights_BL_final_lo>0.001;
pie(ax2,daily_weights_BL_final_lo);
legend(vector_code_chosen(idx),'Location','southoutside');
title(ax2,'Black-Litterman (Constr.)');

ax3=subplot(1,3,3);
idx=daily_weights_Bayes_srp>0.001;
pie(ax3,daily_weights_Bayes_srp(idx))
legend(vector_code_chosen(idx),'Location','southoutside');
title(ax3,'Bayesian approach', 'FontSize', 15);

% Statistics of the optimal portfolio (Bayesian approach)
portBayes_daily_ret_selected=daily_ret_selected*daily_weights_Bayes_srp; % We compute the weighted returns
portBayes_daily_mean=mean(portBayes_daily_ret_selected);
portBayes_daily_variance=portvar(portBayes_daily_ret_selected);
portBayes_daily_std=sqrt(portBayes_daily_variance);
portBayes_daily_skew=skewness(portBayes_daily_ret_selected);
portBayes_daily_kurt=kurtosis(portBayes_daily_ret_selected);
portBayes_daily_srp=estimatePortSharpeRatio(daily_portBayes,daily_weights_Bayes_srp); % Sharpe Ratio

Table_statatistics_daily_MV_BL_and_Bayes = table([MV_daily_mu_fin_lo,MV_daily_lo_std,MV_daily_lo_variance,MV_daily_lo_skew,MV_daily_lo_kurt,MV_daily_lo_srp]',...
    [portBL_daily_mean_lo,portBL_daily_std_lo,portBL_daily_variance_lo,portBL_daily_skew_lo,portBL_daily_kurt_lo,portBL_daily_srp_lo]',...
    [portBayes_daily_mean,portBayes_daily_std,portBayes_daily_variance,portBayes_daily_skew,portBayes_daily_kurt,portBayes_daily_srp]',...
    'VariableNames',{'MV daily portfolio (Con.)', 'BL daily portfolio (Con.)','Bayes portfolio'},...
    'RowNames', {'Mean', 'St Deviation', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe Ratio'});

%From MatLab to Excell

filename = 'Table_statatistics_daily_MV_BL_and_Bayes.xlsx';
writetable(Table_statatistics_daily_MV_BL_and_Bayes,filename,'Sheet',1,'WriteRowNames',true)

%% 

daily_portGMinV = Portfolio('AssetList', vector_code_chosen, 'NumAssets', 12);
daily_portGMinV= daily_portGMinV.estimateAssetMoments(daily_ret_selected);
daily_portGMinV = setDefaultConstraints(daily_portGMinV); % We impose non_negative weights that sum up to 1

% To find minimum variance portfolio we use the function:
% "estimateFrontierLimits"

daily_weights_GMinV = daily_portGMinV.estimateFrontierLimits('min');

daily_weights_GMinV_srp=daily_portGMinV.estimateMaxSharpeRatio;
[GMinV_daily_sigma_final, GMinV_daily_mu_final]=estimatePortMoments(daily_portGMinV,daily_weights_GMinV_srp);

% Statistics of the GMV portfolio
portGMinV_daily_ret_selected=daily_ret_selected*daily_weights_GMinV; % We compute the weighted returns
portGMinV_daily_mean=mean(portGMinV_daily_ret_selected);
portGMinV_daily_variance=var(portGMinV_daily_ret_selected);
portGMinV_daily_std=sqrt(portGMinV_daily_variance);
portGMinV_daily_skew=skewness(portGMinV_daily_ret_selected);
portGMinV_daily_kurt=kurtosis(portGMinV_daily_ret_selected);
portGMinV_daily_srp=estimatePortSharpeRatio(daily_portGMinV,daily_weights_GMinV_srp); %Sharpe Ratio

figure(10);
ax1=subplot(2,2,1);
idx=weightsMV_fin_daily_lo>0.001;
pie(ax1,weightsMV_fin_daily_lo(idx));
legend(vector_code_chosen(idx),'Location','eastoutside');
title(ax1,'Mean Variance (Constr.)');

ax2=subplot(2,2,2);
idx=daily_weights_BL_final_lo>0.001;
pie(ax2,daily_weights_BL_final_lo);
legend(vector_code_chosen(idx),'Location','eastoutside');
title(ax2,'Black-Litterman (Constr.)');

ax3=subplot(2,2,3);
idx = daily_weights_Bayes_srp>0.001;
pie(ax3,daily_weights_Bayes_srp(idx))
legend(vector_code_chosen(idx),'Location','eastoutside');
title(ax3,'Bayesian approach', 'FontSize', 15);

ax4=subplot(2,2,4);
idx=daily_weights_GMinV>0.001;
pie(ax4,daily_weights_GMinV(idx))
legend(vector_code_chosen(idx), 'Location','eastoutside');
title(ax4,'Global Min-Variance', 'FontSize',15);

%Table statistics
Table_statistics_daily_MV_BL_Bayes_and_GMV=table([MV_daily_lo_mu, MV_daily_lo_std, MV_daily_lo_variance, MV_daily_lo_skew, MV_daily_lo_kurt, MV_daily_lo_srp]', ...
    [portBL_daily_mean_lo, portBL_daily_std_lo, portBL_daily_variance_lo, portBL_daily_skew_lo, portBL_daily_kurt_lo, portBL_daily_srp_lo]', ...
    [portBayes_daily_mean, portBayes_daily_std, portBayes_daily_variance,  portBayes_daily_skew, portBayes_daily_kurt, portBayes_daily_srp]', ...
    [portGMinV_daily_mean, portGMinV_daily_std, portGMinV_daily_variance, portGMinV_daily_skew, portGMinV_daily_kurt, portGMinV_daily_srp]',...
    'VariableNames',{'MV daily portfolio (Cons.)', 'BL daily portfolio (Cons.)', 'Bayes daily portfolio', 'MGV daily portfolio'},...
    'RowNames', {'Mean', 'St Deviation', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe Ratio'});

%Table weights GMinV
Table_weights_GMinv_daily=table(daily_weights_GMinV, ...
    'VariableNames', {'Wheights of daily Global Min. Variance Portfolio'},'RowNames', vector_code_chosen);

%From MatLab to Excell

filename='Table_weights_GMinv_daily.xlsx';
writetable(Table_weights_GMinv_daily,filename,'Sheet',1,'WriteRowNames',true)

filename = 'Table_statistics_daily_MV_BL_Bayes_and_GMV.xlsx';
writetable(Table_statistics_daily_MV_BL_Bayes_and_GMV,filename,'Sheet',1,'WriteRowNames',true)

%%

vector_portfolio_combination=[0.20,0.18,0.32,0.30];
portMixed_daily_ret_selected=[portMV_daily_ret_selected_lo,portBL_daily_ret_selected_lo,portBayes_daily_ret_selected,portGMinV_daily_ret_selected]*vector_portfolio_combination';

portMixed_daily_mean=mean(portMixed_daily_ret_selected);
portMixed_daily_variance=var(portMixed_daily_ret_selected);
portMixed_daily_std=sqrt(portMixed_daily_variance);
portMixed_daily_skew=skewness(portMixed_daily_ret_selected);
portMixed_daily_kurt=kurtosis(portMixed_daily_ret_selected);
portMixed_daily_srp=sharpe(portMixed_daily_ret_selected);


%Table of statistics
Table_statistics_portMixed_daily=table([portMixed_daily_mean, portMixed_daily_std, portMixed_daily_variance, portMixed_daily_skew, portMixed_daily_kurt,portMixed_daily_srp]', ...
    'VariableNames',{'Mixed portfolio'},...
    'RowNames', {'Mean', 'St Deviation', 'Variance', 'Skewness', 'Kurtosis', 'Sharpe Ratio'})

%From MatLab to Excell

filename='Table_statistics_portMixed_daily.xlsx';
writetable(Table_statistics_portMixed_daily,filename,'Sheet',1,'WriteRowNames',true)
