clc;
clear all;

% butun veri setlerini 2 matriste toplama(D ve E matrisleri)
%--------------------------------------------------------------------
% butun veri setlerini 2 matriste toplama(D ve E matrisleri)
%--------------------------------------------------------------------
path1 = '/Users/demo/Documents/MATLAB/machine learning in matlab/d/';
% Bu directory deki butun dosyalari indexler
my_files = dir(fullfile(path1, '*.txt'));

% 100 dosyanin uzunlugunu al
N_files = length(my_files);
A = zeros(4097,N_files); % 100 sutun olacak. 4099
for k = 1:9 % bir ile dokuz arasindaki dosyalar icin islem Yap
  
    % ilgili txt dosyasindaki verileri oku ve A matrisine yaz.
    % Sprintf string icerisindeki datayi formatlar.(format data into string)
    % dlmread ascii kodlarini dosya icinden okur.
    A(:,k) = dlmread(sprintf(fullfile(path1,'F00%d.txt'),k));
end

for k = 10:99
   
    A(:,k) = dlmread(sprintf(fullfile(path1,'F0%d.txt'),k));
    % asagidaki sekilde de yazilabilir.
    % A(:,k) = dlmread(sprintf('/Users/demo/Documents/MATLAB/machine learning in matlab/d/F0%d.txt',k));

end

    A(:,100) = dlmread(sprintf(fullfile(path1,'F%d.txt'),100));
    
    
    
% e klasoru icindeki txt verilerini okuyup bir matriste topluyoruz.    
path2 = '/Users/demo/Documents/MATLAB/machine learning in matlab/e/';

my_files2 = dir(fullfile(path2, '*.txt'));

N_files2 = length(my_files2);
B = zeros(4097,N_files2); % Matrixi 0 lar ile baslatiyoruz
for k = 1:9
  
    B(:,k) = dlmread(sprintf(fullfile(path2,'S00%d.txt'),k));
    
end

for k = 10:99
    
    B(:,k) = dlmread(sprintf(fullfile(path2,'S0%d.txt'),k));
   
end

B(:,100) = dlmread(sprintf(fullfile(path2,'S%d.txt'),100));
%---Normalize Ediliyor------------------------------------


% ANorm = A-min(A(:));
% ANorm = ANorm./max(ANorm(:));
% 
% BNorm = B-min(B(:));
% BNorm = BNorm./max(BNorm(:));

for k=1:100
    
        
        A(:,k) = (A(:,k)-mean(A(:,k)))/std(A(:,k));
        
        B(:,k) = (B(:,k)-mean(B(:,k)))/std(B(:,k));
       
 
end


D = A';
E = B';

% Feature extraction(Ozellik Cikarimi Bolumu)
%--------------------------------------------------------------

n = 200;
X = zeros(n,9);

% D Matrisindeki her bir kisiye ait verilerin ortalama, standard sapma,
% entropi vs. lerini alip bunlari X matrisinde sakliyoruz.(Matrisin ilk
% 100 indexinde sakliyoruz)
for idx = 1:100
    
    X(idx,1) = mean(abs(D(idx,:)));
    X(idx,2) = max(abs(D(idx,:)));
    X(idx,3) = meanfreq(abs(D(idx,:)));
    X(idx,4) = std(abs(D(idx,:)));
    X(idx,5) = median(abs(D(idx,:)));
    X(idx,6) = kurtosis(abs(D(idx,:)));
    X(idx,7) = skewness(abs(D(idx,:)));
    X(idx,8) = entropy(abs(D(idx,:)));
    
   
end

% E Matrisindeki her bir kisiye ait verilerin ortalama, standard sapma,
% entropi vs. lerini alip bunlari X matrisinde sakliyoruz.(Matrisin
% indexinin 100-200 kismini kullaniyoruz.
for idx = 1:100
    
    X(idx+100,1) = mean(abs(E(idx,:)));
    X(idx+100,2) = max(abs(E(idx,:)));
    X(idx+100,3) = meanfreq(abs(E(idx,:)));
    X(idx+100,4) = std(abs(E(idx,:)));
    X(idx+100,5) = median(abs(E(idx,:)));
    X(idx+100,6) = kurtosis(abs(E(idx,:)));
    X(idx+100,7) = skewness(abs(E(idx,:)));
    X(idx+100,8) = min(abs(E(idx,:)));
    X(idx+100,8) = entropy(abs(E(idx,:)));
    
   
end


% X in 9 uncu sutununa 0 ve 1 siniflarini atiyoruz(sinif etiketlerimiz)
% Not: siniflar 1 ve -1 ile etiketlendirildiginde hata aliyoruz bundan
% oturu 0(kriz bolgesinde) ve 1(Kriz esnas?nda) olarak etiketlendirildi.
X(1:100,9) = 0; % Normal 
X(101:200,9) = 1; % Hasta

pred = X(50:149,1:8); % D'nin son 50, E'nin ilk 50 hastasi egitime dahil oluyor



% Yukarida egitime tabi tutacagimiz(pred) verilerin sinif etiketlerini giriyoruz 
resp = zeros(100:1);
resp(1:50,1) = X(50:99,9);
resp(51:100,1) = X(100:149,9);

% logic veri turune donusturmemiz gerekiyor. yoksa islem yapmiyor. 
resp = logical(resp);


% KNN, Naive Bayes, ANN, Logistic Regression ve SVM ile Siniflandirma bolumu   
%-------------------------------------------------------------------------   
% Egitimde kullanilmayan D'nin ilk 50 ve E'nin son 50 verisi Test objesine
% aktariliyor.
Test = zeros(100,8);
Test(1:49,1:8) = X(1:49,1:8);
Test(50:100,1:8) = X(150:200,1:8);



% Support Vector Machines egitim asamasi
%--------------------------------------------------------------------------
Mdl_svm = fitcsvm(Test,resp);
% cross validation yapiliyor
crossModel_svm = crossval(Mdl_svm, 'KFold', 10);
[y_svm,score_svm] = kfoldPredict(crossModel_svm);
% roc egrimiz icin X_knn ve Y_knn degerlerini almamiz gerekiyor.Bundan
% oturu perfcurve komutuna score parametresini girmemiz gerekiyor.
[Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT_svm] = ...
    perfcurve(resp,score_svm(:,Mdl_svm.ClassNames),'true');


   actual_labels = resp(1:end);
   predicted_labels_svm = y_svm(1:end);
   
conf_mat_svm = confusionmat(actual_labels, predicted_labels_svm)
precision_svm = conf_mat_svm(1,1)/(conf_mat_svm(1,1)+conf_mat_svm(2,1))
accuracy_svm = (conf_mat_svm(1,1)+conf_mat_svm(2,2))/(conf_mat_svm(1,1)+...
conf_mat_svm(1,2)+conf_mat_svm(2,1)+conf_mat_svm(2,2))
recall_svm = conf_mat_svm(1,1)/(conf_mat_svm(1,1)+conf_mat_svm(1,2))
f_measure_svm = (2*precision_svm*recall_svm)/(precision_svm+recall_svm)
C_svm = classperf(actual_labels,predicted_labels_svm);
C_svm_sen = C_svm.Sensitivity
C_svm_spe = C_svm.Specificity

%--------------------------------------------------------------------------



% Logistic Regression egitim asamasi
%--------------------------------------------------------------------------
Mdl_log = GeneralizedLinearModel.fit(pred,resp,'linear','Distribution','binomial','link','logit');
score_log = Mdl_log.Fitted.Probability; 
Y_lg = Mdl_log.predict(Test);
Y_lg = round(Y_lg);


actual_labels = resp(1:end);
predicted_labels_lg = Y_lg(1:end);
conf_mat_lg = confusionmat(double(actual_labels), predicted_labels_lg)
precision_lg = conf_mat_lg(1,1)/(conf_mat_lg(1,1)+conf_mat_lg(2,1))
accuracy_lg = (conf_mat_lg(1,1)+conf_mat_lg(2,2))/(conf_mat_lg(1,1)+...
conf_mat_lg(1,2)+conf_mat_lg(2,1)+conf_mat_lg(2,2))
recall_lg = conf_mat_lg(1,1)/(conf_mat_lg(1,1)+conf_mat_lg(1,2))
f_measure_lg = (2*precision_lg*recall_lg)/(precision_lg+recall_lg)
C_lg = classperf(actual_labels,predicted_labels_lg);
C_lg_sen = C_lg.Sensitivity
C_lg_spe = C_lg.Specificity


[Xlog,Ylog,Tlog,AUClog] = perfcurve(resp,score_log,'true');



%--------------------------------------------------------------------------


% Naive Bayes egitim asamasi
%-------------------------------------------------------------------------
Mdl_nb = fitcnb(Test,resp);
% cross validation yapiliyor
crossModel_nb = crossval(Mdl_nb, 'KFold', 10);
[y_nb,score_nb] = kfoldPredict(crossModel_nb);
% roc egrimiz icin X_knn ve Y_knn degerlerini almamiz gerekiyor.
[Xnb,Ynb,Tnb,AUCnb,OPTROCPT_nb] =...
    perfcurve(resp,score_nb(:,Mdl_nb.ClassNames),'true');



   actual_labels = resp(1:end);
   predicted_labels_nb = y_nb(1:end);
   
conf_mat_nb = confusionmat(actual_labels, predicted_labels_nb)
accuracy_nb = (conf_mat_nb(1,1)+conf_mat_nb(2,2))/(conf_mat_nb(1,1)+...
conf_mat_nb(1,2)+conf_mat_nb(2,1)+conf_mat_nb(2,2))
precision_nb = conf_mat_nb(1,1)/(conf_mat_nb(1,1)+conf_mat_nb(2,1))
recall_nb = conf_mat_nb(1,1)/(conf_mat_nb(1,1)+conf_mat_nb(1,2))
f_measure_nb = (2*precision_nb*recall_nb)/(precision_nb+recall_nb)
C_nb = classperf(actual_labels,predicted_labels_nb);
C_nb_sen = C_nb.Sensitivity
C_nb_spe = C_nb.Specificity

%--------------------------------------------------------------------------

% k-NN egitim asamasi
% -------------------------------------------------------------------------
Mdl_knn = fitcknn(Test,resp);
% cross validation yapiliyor
crossModel_knn = crossval(Mdl_knn, 'KFold', 10);
[y_knn,score_knn] = kfoldPredict(crossModel_knn);
% roc egrimiz icin X_knn ve Y_knn degerlerini almamiz gerekiyor.
[Xknn,Yknn,Tknn,AUCknn,OPTROCPT_knn] = ...
    perfcurve(resp,score_knn(:,Mdl_knn.ClassNames),'true');


%table(Y(1:10),y_knn(1:10),score_knn(1:10,2),'VariableNames',...
    %{'TrueLabel','PredictedLabel','Score'})
   actual_labels = resp(1:end);
   predicted_labels_knn = y_knn(1:end);
   
conf_mat_knn = confusionmat(actual_labels, predicted_labels_knn)
accuracy_knn = (conf_mat_knn(1,1)+conf_mat_knn(2,2))/(conf_mat_knn(1,1)+...
conf_mat_knn(1,2)+conf_mat_knn(2,1)+conf_mat_knn(2,2))
precision_knn = conf_mat_knn(1,1)/(conf_mat_knn(1,1)+conf_mat_knn(2,1))
recall_knn = conf_mat_knn(1,1)/(conf_mat_knn(1,1)+conf_mat_knn(1,2))
f_measure_knn = (2*precision_knn*recall_knn)/(precision_knn+recall_knn)
C_knn = classperf(actual_labels,predicted_labels_knn);
C_knn_sen = C_knn.Sensitivity
C_knn_spe = C_knn.Specificity
%------------------------------------------------------------------ 

%--------- ANN Kismi burdan basliyor-----------------

% X matrisimizin 9 ve 10 uncu satirlarini sinif olarak kullanacagiz. 
% ilk sinifimiz [1 0] ikinci sinifimiz [0 1]
X(1:100,10) = 0;
X(101:200,9) = 0;
X(101:200,10) = 1;


resp2 = zeros(100:2);
% response(resp) matrisimizin ilk 50 satirlik kismina [1 0], son 50
% satirlik kismina [0 1] sinif verilerinin atamasini yapiyoruz. 
% Predictor(pred) matrisinde D matrisinin son 50, E matisinin 
% ilk 50 (toplamda 100 giris noronu) matrisi bizim noron girislerimiz
% olacak. bunlara karsilik sirasiyla [1 0] ve [0 1] siniflarimizi egitime
% dahil edecegiz.
resp2(1:50,1) = X(50:99,9);
resp2(1:50,2) = X(50:99,10);
resp2(51:100,1) = X(100:149,9);
resp2(51:100,2) = X(100:149,10);

P = pred';
T = resp2';

PR = minmax(P);

% gizli(hidden) noron sayimiz 60 cikis noron sayimiz 2...
n1 = 60;
n2 = 2;

% activasyon fonkisyonlari olarak tansig ve logsig secildi. train
% fonksiyonlarimizi degistirerek performanslarini karsilastiracagiz.
% simdilik trainrp fonksiyonu kullanilacak
net=newff(PR,[n1 n2],{'tansig','logsig'},'trainrp');

test1=sim(net,P);   % egitilmemis network simule ediliyor
fprintf('Before training\n');

disp([T test1]);

% bu kisimda girilen test verileri hedef(T) verileri ile karsilastiriliyor
% ve dogru tahmin edilmis ve edilmemis veriler ekrana yansitiliyor.
iterasyon1 = 0;
iterasyon2 = 0;
for k=1:100
    
    if round(T(:,k)) == round(test1(:,k))
        iterasyon1 = iterasyon1 + 1;
    else 
        iterasyon2 = iterasyon2 + 1;
        
    end 
    
end 

egitilmemis_dogru_tahmin = iterasyon1
egitilmemis_yanlis_tahmin = iterasyon2

egitim=train(net,P,T);
test2=sim(egitim,P);  % egitilmis network
fprintf('After training\n');

disp([T test2]);

iterasyon3 = 0;
iterasyon4 = 0;
% train islemi yapildiktan sonra dogrulama islemi yapiliyor.
% yanlis ve dogru tahmin sayilari ekrana aktariliyor.
for k=1:100
    
    if round(T(:,k)) == round(test2(:,k))
        iterasyon3 = iterasyon3 + 1;
    else 
        iterasyon4 = iterasyon4 + 1;
        
    end 
    
end 

egitilmis_dogru_tahmin = iterasyon3
egitilmis_yanlis_tahmin = iterasyon4    
        

% ikinci test asamasi... Burada X matrisinin train kisminda kullanilmayan
% ilk 50 kisisi(D Normal) ve yine train kisminda kullanilmayan son 50
% verisi(E Normal) noron girisine verilip siniflandirma yapilacak.
pred2= zeros(100:8);
pred2(1:50,1:8) = X(1:50,1:8);
pred2(50:100,1:8) = X(150:200,1:8);



P2 = pred2';

% egitilmis network'e P2 test verileri giriliyor.
test3=sim(egitim,P2);  % egitilmis network
fprintf('After last training\n');

iterasyon5 = 0;
iterasyon6 = 0;
% train islemi yapildiktan sonra dogrulama islemi yapiliyor.
% yanlis ve dogru tahmin sayilari ekrana aktariliyor.
for k=1:100
    
    if round(T(:,k)) == round(test3(:,k))
        iterasyon5 = iterasyon5 + 1;
    else 
        iterasyon6 = iterasyon6 + 1;
        
    end 
    
end 

egitilmis_dogru_tahmin2 = iterasyon5
egitilmis_yanlis_tahmin2 = iterasyon6

% yanlis yapildi P2 bizim ham verimiz.


% y 2 li konfusion matrisi oluyor.
% round yap?lmadiginda, Target cikisla eslesmediginden butun hepsi yanlis
% cikiyor.


[x,y,z,t] = confusion(T,round(test3));


conf_mat_nn = y
accuracy_nn = (conf_mat_nn(1,1)+conf_mat_nn(2,2))/(conf_mat_nn(1,1)+...
conf_mat_nn(1,2)+conf_mat_nn(2,1)+conf_mat_nn(2,2))
precision_nn = conf_mat_nn(1,1)/(conf_mat_nn(1,1)+conf_mat_nn(2,1))
recall_nn = conf_mat_nn(1,1)/(conf_mat_nn(1,1)+conf_mat_nn(1,2))
f_measure_nn = (2*precision_nn*recall_nn)/(precision_nn+recall_nn)
sensitivity_nn = conf_mat_nn(1,1)/(conf_mat_nn(1,1)+conf_mat_nn(1,2))
specificity_nn = conf_mat_nn(2,2)/(conf_mat_nn(2,1)+conf_mat_nn(2,2))

% ANN'i ROC egrisinde gostermede problem ?ikiyor
%---------------
% plotroc(T,round(test3))
% hold on

plot(Xknn,Yknn)
hold on
plot(Xnb,Ynb)
hold on
plot(Xsvm,Ysvm)
hold on
plot(Xlog,Ylog)

legend('K-NN Metodu','NB Metodu','Support Vector Machines',...
    'Logistic Regression','Location','Best')
xlabel('False pozitif orani'); ylabel('True pozitif orani');
title('Siniflandirma Algoritmalari ROC Egrileri')
hold off


class = {'Accuracy';'Sensitivity';'Specificity';'Precision';'Recall';'F-Measure'};


KNN = [accuracy_knn;C_knn_sen;C_knn_spe;precision_knn;recall_knn;f_measure_knn];
LR = [accuracy_lg;C_lg_sen;C_lg_spe;precision_lg;recall_lg;f_measure_lg];
Naive_Bayes = [accuracy_nb;C_nb_sen;C_nb_spe;precision_nb;recall_nb;f_measure_nb];
SVM = [accuracy_svm;C_svm_sen;C_svm_spe;precision_svm;recall_svm;f_measure_svm];
YSA = [accuracy_nn;sensitivity_nn;specificity_nn;precision_nn;recall_nn;f_measure_nn];



  
Table = table(KNN, LR, Naive_Bayes, SVM, YSA, 'RowNames',class)


% sonrasinda roc egrisini cizdir.
% normalize de edip oyle cizdir. 
%--------------------------------------------------------------







