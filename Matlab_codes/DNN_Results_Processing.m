% Process the results from the testing done on Python

clc
clearvars
% close all
warning('off','all')

% Current directory (must be the same as the one with the path Python_codes)
currentPath = pwd;
path = erase(currentPath,'Matlab_codes');

mod = 'QPSK';
ChType = 'RTV_UC';
IBO = 2;
scenario = 'Memory_woCompensation'; % Memoryless, Memory_woCompensation, Memory_Compensated, 
pathdata = ['10000_50Symbols_' mod '_' ChType '_50kmh_' scenario '_IBO' num2str(IBO)];
 
% Loading Simulation Data
load(['data_' pathdata '\Simulation_variables.mat']);

%% ------ Bits Modulation Technique------------------------------------------
if(strcmp(mod,'QPSK') == 1)
     nBitPerSym            = 2; 
elseif (strcmp(mod,'16QAM') == 1)
     nBitPerSym            = 4; 
elseif (strcmp(mod,'64QAM') == 1)
     nBitPerSym            = 6; 
end
M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
load('indices.mat');
N_Test_Frames = length(testing_indices);
EbN0dB                    = (0:5:30)';
Pow                       = mean(abs(qammod(0:(M-1),M)).^2); 
nSym                      = 50;
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
scramInit                 = 93;
nDSC                      = 48;
nUSC                      = 52;
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
ppositions                = [7,21,32,46].';                           % Pilots positions in Kset

N_SNR                      = size(EbN0dB,1);
Phf                        = zeros(N_SNR,1);
 
Err_DPA_LSTM_DNN           = zeros(N_SNR,1);

Ber_DPA_LSTM_DNN           = zeros(N_SNR,1);

dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';    % Data positions in the set of allocated subcarriers Kset 

nsub = 28;
for i = 1:N_SNR 
    
    % Loading Simulation Parameters Results
    load(['data_' pathdata '\Simulation_' num2str(i) '.mat']);

    % Loading DPA-LSTM-DNN Results
    load([path '\Python_Codes\data\' scenario '\DPA_LSTM_Less_DNN_' num2str(nsub) '15_Results_' num2str(i),'.mat']);
    DPA_LSTM_DNN = eval(['DPA_LSTM_Less_DNN_' num2str(nsub) '15_corrected_y_',num2str(i)]);
    DPA_LSTM_DNN = reshape(DPA_LSTM_DNN(1:52,:) + 1i*DPA_LSTM_DNN(53:104,:), nUSC, nSym, N_Test_Frames);  
    
    tic;
    for u = 1:N_Test_Frames
        
        c = testing_indices(1,u);        
        Phf(i)  = Phf(i)  + norm(True_Channels_Structure(:,:,c))^ 2;

        % DPA-LSTM-DNN
        H_DPA_LSTM_DNN = DPA_LSTM_DNN(:,:,u);
        Err_DPA_LSTM_DNN (i) =  Err_DPA_LSTM_DNN (i) +  norm(H_DPA_LSTM_DNN - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_DPA_LSTM_DNN = Received_Symbols_FFT_Structure(dpositions,:,c) ./ H_DPA_LSTM_DNN(dpositions,:);
        
        % QAM - DeMapping
        De_Mapped_DPA_LSTM_DNN      = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_DPA_LSTM_DNN,M);

        % Bits Extraction
        Bits_DPA_LSTM_DNN       = de2bi(De_Mapped_DPA_LSTM_DNN,nBitPerSym);
       
        % De-Interleaving
        % General Block De-Interleaving
        General_Block_De_Interleaved_Data_DPA_LSTM_DNN      = deintrlv(Bits_DPA_LSTM_DNN(:),Random_permutation_Vector);

        % Matrix De-Interleaving
        Matrix_De_Interleaved_Data_DPA_LSTM_DNN     = matintrlv(General_Block_De_Interleaved_Data_DPA_LSTM_DNN.',Interleaver_Columns,Interleaver_Rows).';

        % Viterbi decoder
        Decoded_Bits_DPA_LSTM_DNN    = vitdec(Matrix_De_Interleaved_Data_DPA_LSTM_DNN,trellis,tbl,'trunc','hard');

        % De-scrambler Data
        Bits_DPA_LSTM_DNN_Final      = wlanScramble(Decoded_Bits_DPA_LSTM_DNN,scramInit);

        % BER Calculation
        ber_DPA_LSTM_DNN    = biterr(Bits_DPA_LSTM_DNN_Final,TX_Bits_Stream_Structure(:,c));

        Ber_DPA_LSTM_DNN(i)        = Ber_DPA_LSTM_DNN(i) + ber_DPA_LSTM_DNN;  

    end
    toc;
end

%% Bit Error Rate (BER)
BER_DPA_LSTM_DNN      = Ber_DPA_LSTM_DNN / (N_Test_Frames * nSym * nDSC * nBitPerSym);

%% Normalized Mean Square Error
Phf = Phf ./ N_Test_Frames;
ERR_DPA_LSTM_DNN     = Err_DPA_LSTM_DNN / (N_Test_Frames * Phf);

save(['data_' pathdata '\DNNs_Results_DPA_batches_Less' num2str(nsub) '15'],'BER_DPA_LSTM_DNN','ERR_DPA_LSTM_DNN');
