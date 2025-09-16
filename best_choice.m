function [securties_chosen,K] = best_choice(C,N,name_string)
%Choice of securities
n=size(C,1);
for i=1:n
    norma(i)=norm(C(i,:),1);
end

[H,K]=sort(norma);

for i=1:N
    securties_chosen(i)=name_string(K(i));
end
K=K(1:N);
end