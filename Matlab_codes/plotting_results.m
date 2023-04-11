clc
clearvars
% close all
warning('off','all')

EbN0dB = 0:5:30;
mod = 'QPSK';
ChType = 'RTV_UC';
IBO = 2;

scenario = 'Memoryless'; 
pathdata = ['10000_50Symbols_' mod '_' ChType '_50kmh_' scenario '_IBO' num2str(IBO)];
load(['data_' pathdata '\Classical_Results'],'BER_Ideal');
memoryless_ideal = BER_Ideal;
load(['data_' pathdata '\DNNs_Results_DPA_batches_Less2815'],'BER_DPA_LSTM_DNN','ERR_DPA_LSTM_DNN');
memoryless_DPA_LSTM_NN_BER = BER_DPA_LSTM_DNN;
memoryless_DPA_LSTM_NN_ERR = ERR_DPA_LSTM_DNN;
load(['data_' pathdata '\DNNs_Results_LSTM_NN_DPA5215'],'BER_LSTM_NN_DPA','ERR_LSTM_NN_DPA');
memoryless_LSTM_DPA_NN_BER = BER_LSTM_NN_DPA;
memoryless_LSTM_DPA_NN_ERR = ERR_LSTM_NN_DPA;
load(['data_' pathdata '\DNNs_Results_LSTM_DPA_TA52'],'BER_LSTM_DPA_TA','ERR_LSTM_DPA_TA');
memoryless_LSTM_DPA_TA_BER = BER_LSTM_DPA_TA;
memoryless_LSTM_DPA_TA_ERR = ERR_LSTM_DPA_TA;
clearvars  scenario pathdata BER_Ideal BER_DPA_LSTM_DNN ERR_DPA_LSTM_DNN...
    BER_LSTM_NN_DPA ERR_LSTM_NN_DPA BER_LSTM_DPA_TA ERR_LSTM_DPA_TA

scenario = 'Memory_woCompensation';
pathdata = ['10000_50Symbols_' mod '_' ChType '_50kmh_' scenario '_IBO' num2str(IBO)];
load(['data_' pathdata '\Classical_Results'],'BER_Ideal');
load(['data_' pathdata '\DNNs_Results_DPA_batches_Less2815'],'BER_DPA_LSTM_DNN','ERR_DPA_LSTM_DNN');
memory_woCompensation_ideal = BER_Ideal;
memory_woCompensation_DPA_LSTM_NN_BER = BER_DPA_LSTM_DNN;
memory_woCompensation_DPA_LSTM_NN_ERR = ERR_DPA_LSTM_DNN;
load(['data_' pathdata '\DNNs_Results_LSTM_NN_DPA5215'],'BER_LSTM_NN_DPA','ERR_LSTM_NN_DPA');
memory_woCompensation_LSTM_DPA_NN_BER = BER_LSTM_NN_DPA;
memory_woCompensation_LSTM_DPA_NN_ERR = ERR_LSTM_NN_DPA;
load(['data_' pathdata '\DNNs_Results_LSTM_DPA_TA52'],'BER_LSTM_DPA_TA','ERR_LSTM_DPA_TA');
memory_woCompensation_LSTM_DPA_TA_BER = BER_LSTM_DPA_TA;
memory_woCompensation_LSTM_DPA_TA_ERR = ERR_LSTM_DPA_TA;
clearvars  scenario pathdata BER_Ideal BER_DPA_LSTM_DNN ERR_DPA_LSTM_DNN...
    BER_LSTM_NN_DPA ERR_LSTM_NN_DPA BER_LSTM_DPA_TA ERR_LSTM_DPA_TA

scenario = 'Memory_Compensated'; % Memoryless, Memory_woCompensation, Memory_Compensated, 
pathdata = ['10000_50Symbols_' mod '_' ChType '_50kmh_' scenario '_IBO' num2str(IBO)];
load(['data_' pathdata '\Classical_Results'],'BER_Ideal');
load(['data_' pathdata '\DNNs_Results_DPA_batches_Less2815'],'BER_DPA_LSTM_DNN','ERR_DPA_LSTM_DNN');
memory_Compensated_ideal = BER_Ideal;
memory_Compensated_DPA_LSTM_NN_BER = BER_DPA_LSTM_DNN;
memory_Compensated_DPA_LSTM_NN_ERR = ERR_DPA_LSTM_DNN;
load(['data_' pathdata '\DNNs_Results_LSTM_NN_DPA5215'],'BER_LSTM_NN_DPA','ERR_LSTM_NN_DPA');
memory_Compensated_LSTM_DPA_NN_BER = BER_LSTM_NN_DPA;
memory_Compensated_LSTM_DPA_NN_ERR = ERR_LSTM_NN_DPA;
load(['data_' pathdata '\DNNs_Results_LSTM_DPA_TA52'],'BER_LSTM_DPA_TA','ERR_LSTM_DPA_TA');
memory_Compensated_LSTM_DPA_TA_BER = BER_LSTM_DPA_TA;
memory_Compensated_LSTM_DPA_TA_ERR = ERR_LSTM_DPA_TA;
clearvars  scenario pathdata BER_Ideal BER_DPA_LSTM_DNN ERR_DPA_LSTM_DNN...
    BER_LSTM_NN_DPA ERR_LSTM_NN_DPA BER_LSTM_DPA_TA ERR_LSTM_DPA_TA

%% Normalized Mean Squared Error
figure(1)
colorOrder = get(gca, 'ColorOrder');
LSTMDPANN_nonCompensated = semilogy(EbN0dB,memory_woCompensation_LSTM_DPA_NN_ERR ,'-o','MarkerFaceColor',colorOrder(1,:),'color',colorOrder(1,:),'MarkerSize',8,'LineWidth',2);
hold on;
LSTMDPATA_nonCompensated = semilogy(EbN0dB,memory_woCompensation_LSTM_DPA_TA_ERR ,'-^','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',8,'LineWidth',2);
DPALSTMNN_nonCompensated = semilogy(EbN0dB,memory_woCompensation_DPA_LSTM_NN_ERR,'-h','MarkerFaceColor',colorOrder(3,:),'color',colorOrder(3,:),'MarkerSize',8,'LineWidth',2);
grid on
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
hAxes.TickLabelInterprete = 'latex';
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
xlabel('SNR ($\xi$) [dB]');
ylabel('NMSE');
set(gca,'FontSize',18)
if strcmp(mod,'16QAM') && IBO == 2 
    axis([min(EbN0dB) max(EbN0dB) 10^-3 1])
    yticks([10^-3 10^-2 10^-1 10^0])
else
    axis([min(EbN0dB) max(EbN0dB) 10^-3 1])
    yticks([10^-3 10^-2 10^-1 10^0])
end

%% Bit Error Rate
figure(2)
subplot(1,3,1)
colorOrder = get(gca, 'ColorOrder');
semilogy(EbN0dB, memoryless_ideal,'k--','LineWidth',2);
hold on;
semilogy(EbN0dB, memoryless_LSTM_DPA_NN_BER,'-o','MarkerFaceColor',colorOrder(1,:),'color',colorOrder(1,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, memoryless_LSTM_DPA_TA_BER,'-^','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, memoryless_DPA_LSTM_NN_BER,'-h','MarkerFaceColor',colorOrder(3,:),'color',colorOrder(3,:),'MarkerSize',8,'LineWidth',2);
grid on
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
hAxes.TickLabelInterprete = 'latex';
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
xlabel('SNR ($\xi$) [dB]');
ylabel('BER');
set(gca,'FontSize',18)
if strcmp(mod,'16QAM') && IBO == 2 
    axis([min(EbN0dB) max(EbN0dB) 10^-3 1])
    yticks([10^-3 10^-2 10^-1 10^0])
else
    axis([min(EbN0dB) max(EbN0dB) 10^-4 1])
    yticks([10^-4 10^-3 10^-2 10^-1 10^0])
end
title('Memoryless','FontSize',16)

subplot(1,3,2)
colorOrder = get(gca, 'ColorOrder');
semilogy(EbN0dB, memory_woCompensation_ideal,'k--','LineWidth',2);
hold on;
semilogy(EbN0dB,memory_woCompensation_LSTM_DPA_NN_BER ,'-o','MarkerFaceColor',colorOrder(1,:),'color',colorOrder(1,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB,memory_woCompensation_LSTM_DPA_TA_BER ,'-^','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB,memory_woCompensation_DPA_LSTM_NN_BER,'-h','MarkerFaceColor',colorOrder(3,:),'color',colorOrder(3,:),'MarkerSize',8,'LineWidth',2);
grid on
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
hAxes.TickLabelInterprete = 'latex';
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
xlabel('SNR ($\xi$) [dB]');
ylabel('BER');
set(gca,'FontSize',18)
if strcmp(mod,'16QAM') && IBO == 2 
    axis([min(EbN0dB) max(EbN0dB) 10^-3 1])
    yticks([10^-3 10^-2 10^-1 10^0])
else
    axis([min(EbN0dB) max(EbN0dB) 10^-4 1])
    yticks([10^-4 10^-3 10^-2 10^-1 10^0])
end
title('Memory','FontSize',16)

subplot(1,3,3)
colorOrder = get(gca, 'ColorOrder');
semilogy(EbN0dB, memory_Compensated_ideal,'k--','LineWidth',2);
hold on;
semilogy(EbN0dB,memory_Compensated_LSTM_DPA_NN_BER ,'-o','MarkerFaceColor',colorOrder(1,:),'color',colorOrder(1,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB,memory_Compensated_LSTM_DPA_TA_BER ,'-^','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB,memory_Compensated_DPA_LSTM_NN_BER,'-h','MarkerFaceColor',colorOrder(3,:),'color',colorOrder(3,:),'MarkerSize',8,'LineWidth',2);
grid on
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
hAxes.TickLabelInterprete = 'latex';
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
xlabel('SNR ($\xi$) [dB]');
ylabel('BER');
set(gca,'FontSize',18)
if strcmp(mod,'16QAM') && IBO == 2 
    axis([min(EbN0dB) max(EbN0dB) 10^-3 1])
    yticks([10^-3 10^-2 10^-1 10^0])
else
    axis([min(EbN0dB) max(EbN0dB) 10^-4 1])
    yticks([10^-4 10^-3 10^-2 10^-1 10^0])
end
title('Memory Compensated','FontSize',16)
