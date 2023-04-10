% Prepare the datasets to train and test the DNN
clc
clearvars
close all
warning('off','all')

% Current directory (must be the same as the one with the path Python_codes)
currentPath = pwd;
path = erase(currentPath,'Matlab_codes');

% Generating testing and training indices
N_CH                      = 10000;
% idx = randperm(N_CH);
% training_indices          = idx(1:floor(0.8*N_CH));               % number of channel realizations for training
% testing_indices           = idx(length(training_indices)+1:end);  % number of channel realizations for testing
% save('indices','testing_indices','training_indices')
load('indices');
Testing_Data_set_size     = size(testing_indices,2);
Training_Data_set_size    = size(training_indices,2);

% Define Simulation parameters
SNR                       = 0:5:30;
nUSC                      = 52;
nSym                      = 50;
algo                      = 'DPA';

mod = 'QPSK';
ChType = 'RTV_UC';
IBO = 2;
scenario = 'Memory_woCompensation'; % Memoryless, Memory_woCompensation, Memory_Compensated, 
pathdata = ['10000_50Symbols_' mod '_' ChType '_50kmh_' scenario '_IBO' num2str(IBO)];

ppositions                = [7,21, 32,46];                           % Pilots positions in Kset

% Define the sampling rate
rate = 2;
% X_sub will define the quantity of active subcarriers to be used,
% maintaining all the pilots
X_Sub_aux = [1:rate:52 ppositions];
X_Sub_aux = sort(X_Sub_aux, 'ascend');
X_Sub = unique(X_Sub_aux);

Training_DatasetX         = zeros(length(X_Sub),nSym, Training_Data_set_size);
Training_DatasetY         = zeros(nUSC,nSym, Training_Data_set_size);
Testing_DatasetX          = zeros(length(X_Sub),nSym, Testing_Data_set_size);
Testing_DatasetY          = zeros(nUSC,nSym, Testing_Data_set_size);

Train_X                   = zeros(length(X_Sub)*2, Training_Data_set_size * nSym);
Train_Y                   = zeros(nUSC*2, Training_Data_set_size * nSym);
Test_X                    = zeros(length(X_Sub)*2, Testing_Data_set_size * nSym);
Test_Y                    = zeros(nUSC*2, Testing_Data_set_size * nSym);

for n_snr = 1:size(SNR,2)
    % Load simulation data according to the defined configurations (Ch, mod, algorithm) 
    load(['data_' pathdata '\Simulation_' num2str(n_snr),'.mat'], 'True_Channels_Structure', [algo '_Structure']);
    Algo_Channels_Structure = eval([algo '_Structure']);
    
    Training_DatasetX =  Algo_Channels_Structure(X_Sub,:,training_indices);
    Training_DatasetY =  True_Channels_Structure(:,:,training_indices);
    Testing_DatasetX  =  Algo_Channels_Structure(X_Sub,:,testing_indices);
    Testing_DatasetY  =  True_Channels_Structure(:,:,testing_indices);
   
    % Expend Testing and Training Datasets
    Training_DatasetX_expended = reshape(Training_DatasetX, length(X_Sub), nSym * Training_Data_set_size);
    Training_DatasetY_expended = reshape(Training_DatasetY, nUSC, nSym * Training_Data_set_size);
    Testing_DatasetX_expended  = reshape(Testing_DatasetX, length(X_Sub), nSym * Testing_Data_set_size);
    Testing_DatasetY_expended  = reshape(Testing_DatasetY, nUSC, nSym * Testing_Data_set_size);

    % Complex to Real domain conversion
    Train_X(1:length(X_Sub),:)                    = real(Training_DatasetX_expended);
    Train_X(length(X_Sub)+1:2*length(X_Sub),:)    = imag(Training_DatasetX_expended);
    Train_Y(1:nUSC,:)                             = real(Training_DatasetY_expended);
    Train_Y(nUSC+1:2*nUSC,:)                      = imag(Training_DatasetY_expended);

    Test_X(1:length(X_Sub),:)                       = real(Testing_DatasetX_expended);
    Test_X(length(X_Sub)+1:2*length(X_Sub),:)       = imag(Testing_DatasetX_expended);
    Test_Y(1:nUSC,:)                                = real(Testing_DatasetY_expended);
    Test_Y(nUSC+1:2*nUSC,:)                         = imag(Testing_DatasetY_expended);
    
    % Save training and testing datasets to the DNN_Datasets structure
    DNN_Datasets.('Train_X') =  Train_X;
    DNN_Datasets.('Train_Y') =  Train_Y;
    DNN_Datasets.('Test_X')  =  Test_X;
    DNN_Datasets.('Test_Y')  =  Test_Y;

    % Save the DNN_Datasets structure to the specified folder in order to be used later in the Python code 
    save([path '\Python_Codes\data\' scenario '\' algo '_LSTM_Less_Dataset_' num2str(n_snr)],  'DNN_Datasets');
    disp(['Data generated for ' algo ', SNR = ', num2str(SNR(n_snr))]);
end