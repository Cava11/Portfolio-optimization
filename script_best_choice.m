%Script for best_choice

[Names,Ind]=best_choice(monthly_corr_matrix,25,code_string);

Names_N=code_string(Ind);

%tabel_25_asset=table(Names',Names_N','VariableNames',{'Name','Position in the Dataset'})

vector_securities_chosen=[18 69 31 70 51 52 57 36 78 66 40 60];
vector_code_chosen=transpose({'ARN','VIN','CLT','EDNR','TOD','REC','JUVE','AMP','SOL','ITM','IKG','B'});

nomi_scelti=name_string(vector_securities_chosen);

Table_securites_chosen = table([nomi_scelti]',[vector_code_chosen],'VariableNames' ,{'Name','Code'})

%From MatLab to Excell
filename = 'Table_securites_chosen.xlsx';
writetable(Table_securites_chosen,filename,'Sheet',1,'WriteRowNames',true)