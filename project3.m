clc;                                                         %Clearing the data in command window
clear;                                                       %Clearing the variables in workspace
generes={'classical';'jazz';'country';'pop';'rock';'metal'}; %Initializing the variable with all the gener names as strings
fft_count=1;                                                 %Initializing the variable for length of fft matrix
MFCC_count=1;                                                %Initializing the variable for lenght of MFCC matrix
for i=1:1:6
for j=0:1:99
    if j<10                                                  %Intentionally keeping this variable less than 10 since string name below and above 10 varies
x=strcat('D:\Spring 2015\Machine Learning\Project3\genres\',generes(i),'\',generes(i),'.0000');%Initializing the variable with path of folder
    else
x=strcat('D:\Spring 2015\Machine Learning\Project3\genres\',generes(i),'\',generes(i),'.000');%Initializing the variable with path of folder
    end
%strcat(strcat(x,num2str(j)),'.AU')                          %printing the path of the songs
[y,fs(fft_count)] = audioread(char(strcat(strcat(x,num2str(j)),'.AU')));%performing audioread operation on the song
fourier(:,fft_count)=abs(fft(y,1000));                       %Computing fourier transform on the song components
[ MFCC, FBEs, frames ] = mfcc(y,fs(fft_count), 25, 7.25, .97, @hamming, [300 3700], 20, 13, 22 );%Computing MFCC on the song components using 3rd party file published in mathworks
[row column]=size(MFCC);                                     %computing size of the MFCC values obtained
z=MFCC(:,round(column/10):round(column*8/10));               %Taking only 80% of total values, leaving first and last 10% components 
z=z';                                                        %Performing tranpose to get the songs as rows and features as columns
for k=1:1:13
MFCC_final(MFCC_count,k)=mean(z(:,k));                       %Calculating the mean of the MFCC components obtained for a song to get 1x13 matrix
end
MFCC_count=MFCC_count+1;                                     %updating MFCC_count so that the MFCC_final matrix is built
fft_count=fft_count+1;                                       %updating fft_count so that the fft_final matrix is built
end
end
fourier=fourier';                                            %performing transform on the fourier matrix
for i=1:1:1000
    normalised_fourier(:,i)=fourier(:,i)/max(fourier(:,i));  %Normalizing the fourier matrix by dividing valus with max column value
end

for i=0:1:5;    
label(i*100+1:(i+1)*100)=i+1;                                %Creating the COLUMN MATRIX- lable with song categories as array values
end
label=label';                                                %Transposing the label matrix to get a ROW MATRIX

for i=1:1:10
    k=1;        
    l=1;
for j=1:1:600                          
    if ((rem(j,10)==i)|| (i==10 && rem(j,10)==0))            %K fold methodology implemented to prepare the train matrix and test matrix 
        test(k,i)=j;                                         %    as 540 and 60 songs respectively to perform subsequent operations.
        k=k+1;
    else
        train(l,i)=j;
        l=l+1;
    end
end
end


fft_confusion_matrix=zeros(6,6);                            %confusion matrix initialization with zeros
for loop=1:1:10                                             %loop or performing the test with 10 column values obtained with fold logic
for i=1:1:540
train_fourier(i,:)= [1 normalised_fourier(train(i,loop),:)];%Appending 1 for probability calculation       
end
weight=zeros(6,1001);                                       %Weight Initialozation to zeros
train_matrix=exp(weight*train_fourier');                    %Computing the product of weights and tranpose of train_fourier matrix
train_matrix(6,:)=1;                                        %Hardcoding the last row of the matrix to 1 for simplifying the formula
for i=1:1:540
train_matrix(:,i)=train_matrix(:,i)/sum(train_matrix(:,i)); %First Logostic regression implementation ,Methodology to simplify the formula  that is dividing the
end                                                         %  values by sum of the respective columns
delta=zeros(6,540);                                         %Initializig the delta values to zeros
for i=1:1:540
delta(label(train(i,loop),1),i)=1;                          %Updating the delta values with 1 as per the train data category
end
lamda=.001;                                                 %Initializing the lamda to 0.001
n_not=.01;                                                  %Initializing the n_not to 0.01
for i=1:1:1000                                              %Epoc count hardcoded to 1000
n=n_not/(1+i/1000);                                         %Computing n for 1000 Epocs
weight=weight+ (n*(((delta-train_matrix)*train_fourier)-(lamda*weight)));%Weight updation for each Epoc which is Gradient Descent Implementation
train_matrix=exp(weight*train_fourier');                    %train_matrix updation for each Epoc
train_matrix(6,:)=1;                                        %Hardcoding the last row of the matrix to 1 for simplifying the formula
for j=1:1:540 
train_matrix(:,j)=train_matrix(:,j)/sum(train_matrix(:,j)); %Logistic regression implementation on train data,Methodology to simplify the formula  that is dividing the
end                                                         %  values by sum of the respective columns
end

for i=1:1:60
test_fourier(i,:)= [1 normalised_fourier(test(i,loop),:)];  %Appending 1 for probability calculation
end
test_matrix=exp(weight*test_fourier');                      %Computing the product of weights and tranpose of test_fourier matrix to obtain test matrix
test_matrix(6,:)=1;                                         %Hardcoding the last row of the matrix to 1 for simplifying the formula
for i=1:1:60
test_matrix(:,i)=test_matrix(:,i)/sum(test_matrix(:,i));    %Obtaining the final test matrix
end
for i=1:1:60
[maximum(i,1) maximum(i,2)]=max(test_matrix(:,i));          %Finding the maximum value using max function to get the categorised song value
end
correct=0;
wrong=0;
for i=1:1:60
if(label(test(i,1))==maximum(i,2))                          %Comparing the label values and the cateorised song values to determine accuracy
    correct=correct+1;                                      %Updation of the correct variable when the label values matched
else
    wrong=wrong+1;                                          %Updation of the wrong variable when the label values mismatched
end
end
fft_confusion_matrix=confusionmat(label(test(:,1)),maximum(:,2)) +fft_confusion_matrix;
                                                            %Confusion matrix generation with the Label values and the categorised values
fft_accuracy(loop)=correct*100/(correct+wrong);             %Accuracy calculation
end
final_fft_accuracy=sum(fft_accuracy)/10;                    %Computing the average of the Accuracies obtained for 10 fold values



unique_values=[];                                           %Iniializing a blank matrix
for count=1:100:600                     
varience=var(normalised_fourier(count:count-1+100,:));      %Computing varience to find the best songs in the gener that is songs of the same category having almost similar features
[value index]=sort(varience);                               %Sorting the varience values to obtain the indexes of song
unique_values=[unique_values index(1:20)];                  %Appending the Indexes to itself for every iteration
end
unique_values=unique(unique_values);                        %Removing duplicates and collecing only unique values
[row column]=size(unique_values);                           %Obtaining the size of unique_values with size function
for i=1:1:column
featured_matrix(:,i)=normalised_fourier(:,unique_values(1,i));%Copying all the important features to a matrix 
end
feature_confusion_matrix=zeros(6,6);                        %confusion matrix initialization with zeros
for loop=1:1:10                                             %loop or performing the test with 10 column values obtained with fold logic
for i=1:1:540
train_feature(i,:)= [1 featured_matrix(train(i,loop),:)];   %Appending 1 for probability calculation  
end
weight=zeros(6,column+1);                                   %Weight Initialozation to zeros
train_matrix=exp(weight*train_feature');                    %Computing the product of weights and tranpose of train_fourier matrix
train_matrix(6,:)=1;                                        %Hardcoding the last row of the matrix to 1 for simplifying the formula
for i=1:1:540
train_matrix(:,i)=train_matrix(:,i)/sum(train_matrix(:,i)); %First Logostic regression implementation ,Methodology to simplify the formula  that is dividing the
end                                                         %  values by sum of the respective columns
delta=zeros(6,540);                                         %Initializig the delta values to zeros
for i=1:1:540
delta(label(train(i,loop),1),i)=1;                          %Updating the delta values with 1 as per the train data category
end
lamda=.001;                                                 %Initializing the lamda to 0.001
n_not=.01;                                                  %Initializing the n_not to 0.01
for i=1:1:1000                                              %Epoc count hardcoded to 1000
n=n_not/(1+i/1000);                                         %Computing n for 1000 Epocs
weight=weight+ (n*(((delta-train_matrix)*train_feature)-(lamda*weight)));%Weight updation for each Epoc which is Gradient Descent Implementation
train_matrix=exp(weight*train_feature');                    %train_matrix updation for each Epoc
train_matrix(6,:)=1;                                        %Hardcoding the last row of the matrix to 1 for simplifying the formula
for j=1:1:540
train_matrix(:,j)=train_matrix(:,j)/sum(train_matrix(:,j)); %Logistic regression implementation on train data,Methodology to simplify the formula  that is dividing the
end
end

for i=1:1:60
test_feature(i,:)= [1 featured_matrix(test(i,loop),:)];     %Appending 1 for probability calculation
end
test_matrix=exp(weight*test_feature');                      %Computing the product of weights and tranpose of test_feature matrix to obtain test matrix
test_matrix(6,:)=1;                                         %Hardcoding the last row of the matrix to 1 for simplifying the formula
for i=1:1:60
test_matrix(:,i)=test_matrix(:,i)/sum(test_matrix(:,i));    %Obtaining the final test matrix
end
for i=1:1:60
[maximum(i,1) maximum(i,2)]=max(test_matrix(:,i));          %Finding the maximum value using max function to get the categorised song value
end
correct=0;
wrong=0;
for i=1:1:60
if(label(test(i,1))==maximum(i,2))                          %Comparing the label values and the cateorised song values to determine accuracy
    correct=correct+1;                                      %Updation of the correct variable when the label values matched
else
    wrong=wrong+1;                                          %Updation of the wrong variable when the label values mismatched
end
end
feature_confusion_matrix=confusionmat(label(test(:,1)),maximum(:,2)) +feature_confusion_matrix;
                                                            %Confusion matrix generation with the Label values and the categorised values
feature_accuracy(loop)=correct*100/(correct+wrong);         %Accuracy calculation
end
final_feature_accuracy=sum(feature_accuracy)/10;            %Computing the average of the Accuracies obtained for 10 fold values



for i=1:1:13
    normalised_MFCC(:,i)=MFCC_final(:,i)/max(MFCC_final(:,i));%Normalizing the MFFC matrix by dividing valus with max column value
end

MFCC_confusion_matrix=zeros(6,6);                            %confusion matrix initialization with zeros
for loop=1:1:10                                              %loop or performing the test with 10 column values obtained with fold logic
for i=1:1:540
train_MFCC(i,:)= [1 normalised_MFCC(train(i,loop),:)];       %Computing the product of weights and tranpose of train_MFFC matrix to obtain test matrix
end
weight=zeros(6,14);                                          %Weight Initialozation to zeros
train_matrix=exp(weight*train_MFCC');                        %Computing the product of weights and tranpose of train_MFCC matrix
train_matrix(6,:)=1;                                         %Hardcoding the last row of the matrix to 1 for simplifying the formula
for i=1:1:540
train_matrix(:,i)=train_matrix(:,i)/sum(train_matrix(:,i));  %First Logostic regression implementation ,Methodology to simplify the formula  that is dividing the
end                                                          %  values by sum of the respective columns
delta=zeros(6,540);                                          %Initializig the delta values to zeros
for i=1:1:540
delta(label(train(i,loop),1),i)=1;                           %Updating the delta values with 1 as per the train data category
end
lamda=.001;                                                  %Initializing the lamda to 0.001
n_not=.01;                                                   %Initializing the n_not to 0.01
for i=1:1:1000                                               %Epoc count hardcoded to 1000
n=n_not/(1+i/1000);                                          %Computing n for 1000 Epocs                                          
weight=weight+ (n*(((delta-train_matrix)*train_MFCC)-(lamda*weight)));%Weight updation for each Epoc which is Gradient Descent Implementation
train_matrix=exp(weight*train_MFCC');                        %train_matrix updation for each Epoc
train_matrix(6,:)=1;                                         %Hardcoding the last row of the matrix to 1 for simplifying the formula
for j=1:1:540
train_matrix(:,j)=train_matrix(:,j)/sum(train_matrix(:,j));  %Logistic regression implementation on train data,Methodology to simplify the formula  that is dividing the
end
end

for i=1:1:60
test_MFCC(i,:)= [1 normalised_MFCC(test(i,loop),:)];         %Appending 1 for probability calculation
end
test_matrix=exp(weight*test_MFCC');                          %Computing the product of weights and tranpose of train_fourier matrix to obtain test matrix
test_matrix(6,:)=1;                                          %Hardcoding the last row of the matrix to 1 for simplifying the formula
for i=1:1:60
test_matrix(:,i)=test_matrix(:,i)/sum(test_matrix(:,i));     %Obtaining the final test matrix
end
for i=1:1:60
[maximum(i,1) maximum(i,2)]=max(test_matrix(:,i));           %Finding the maximum value using max function to get the categorised song value
end
correct=0;
wrong=0;
for i=1:1:60
if(label(test(i,1))==maximum(i,2))                           %Comparing the label values and the cateorised song values to determine accuracy
    correct=correct+1;                                       %Updation of the correct variable when the label values matched
else
    wrong=wrong+1;                                           %Updation of the wrong variable when the label values mismatched
end
end
MFCC_confusion_matrix=confusionmat(label(test(:,1)),maximum(:,2))+MFCC_confusion_matrix;
                                                             %Confusion matrix generation with the Label values and the categorised values 
MFCC_accuracy(loop)=correct*100/(correct+wrong);             %Accuracy calculation
end
final_MFCC_accuracy=sum(MFCC_accuracy)/10;                   %Computing the average of the Accuracies obtained for 10 fold values

