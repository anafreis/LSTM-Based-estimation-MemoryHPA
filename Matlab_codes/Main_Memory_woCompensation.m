clc
clearvars
% close all
warning('off','all')
ch_func = Channel_functions();

mod = '16QAM';
ChType = 'RTV_UC';
IBO = 2;

% pathdata = ['10000_50Symbols_' mod '_' ChType '_50kmh_Memory_woCompensation_IBO' num2str(IBO)];

%% Physical Layer Specifications for IEEE 802.11p / OFDM, described by Table 1
ofdmBW                 = 10 * 10^6 ;     % OFDM bandwidth (Hz)
nFFT                   = 64;             % FFT size 
nDSC                   = 48;             % Number of data subcarriers
nPSC                   = 4;              % Number of pilot subcarriers
nZSC                   = 12;             % Number of zeros subcarriers
nUSC                   = nDSC + nPSC;    % Number of total used subcarriers
K                      = nUSC + nZSC;    % Number of total subcarriers
nSym                   = 50;             % Number of OFDM symbols within one frame
deltaF                 = ofdmBW/nFFT;    % Bandwidth for each subcarrier - include all used and unused subcarriers 
Tfft                   = 1/deltaF;       % IFFT or FFT period = 6.4us
Tgi                    = Tfft/4;         % Guard interval duration - duration of cyclic prefix - 1/4th portion of OFDM symbols = 1.6us
Tsignal                = Tgi+Tfft;       % Total duration of BPSK-OFDM symbol = Guard time + FFT period = 8us
K_cp                   = nFFT*Tgi/Tfft;  % Number of symbols allocated to cyclic prefix 
pilots_locations       = [8,22,44,58].'; % Pilot subcarriers positions
pilots                 = [1 1 1 -1].';
data_locations         = [2:7, 9:21, 23:27, 39:43, 45:57, 59:64].'; % Data subcarriers positions
ppositions             = [7,21, 32,46].';                           % Pilots positions in Kset
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';        % Data positions in Kset
% Pre-defined preamble in frequency domain
dp = [ 0 0 0 0 0 0 +1 +1 -1 -1 +1  +1 -1  +1 -1 +1 +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 +1 +1 +1 0 +1 -1 -1 +1 +1 -1 +1 -1 +1 -1 -1 -1 -1 -1 +1 +1 -1 -1 +1 -1 +1 -1 +1 +1 +1 +1 0 0 0 0 0];
Ep                     = 1;              % preamble power per sample
dp                     = fftshift(dp);   % Shift zero-frequency component to center of spectrum    
predefined_preamble    = dp;
Kset                   = find(dp~=0);               % set of allocated subcarriers                  
Kon                    = length(Kset);              % Number of active subcarriers
dp                     = sqrt(Ep)*dp.';
%%%%%%%%%
xp                     = sqrt(K)*ifft(dp);
%%%%%%%%%
xp_cp                  = [xp(end-K_cp+1:end); xp];  % Adding CP to the time domain preamble
preamble_80211p        = repmat(xp_cp,1,2);         % IEEE 802.11p preamble symbols (tow symbols)
%% ------ Bits Modulation Technique------------------------------------------
Mod_Type                  = 1;              % 0 for BPSK and 1 for QAM 
if(Mod_Type == 0)
    nBitPerSym            = 1;
    Pow                   = 1;
    %BPSK Modulation Objects
    bpskModulator         = comm.BPSKModulator;
    bpskDemodulator       = comm.BPSKDemodulator;
    M                     = 1;
elseif(Mod_Type == 1)
    if(strcmp(mod,'QPSK') == 1)
         nBitPerSym            = 2; 
    elseif (strcmp(mod,'16QAM') == 1)
         nBitPerSym            = 4; 
    elseif (strcmp(mod,'64QAM') == 1)
         nBitPerSym            = 6; 
    end
    M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
    Pow                   = mean(abs(qammod(0:(M-1),M)).^2); % Normalization factor for QAM        
end
%% --------- Scrambler Parameters ---------------------------------------------
scramInit                 = 93; % As specidied in IEEE 802.11p Standard [1011101] in binary representation
%% --------- Convolutional Coder Parameters -----------------------------------
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
rate                      = 1/2;
%% -------Interleaver Parameters---------------------------------------------
% Matrix Interleaver
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
% General Block Interleaver
Random_permutation_Vector = randperm(nBitPerSym*nDSC*nSym); % Permutation vector
%% -----------------Vehicular Channel Model Parameters--------------------------
fs                        = K*deltaF;               % Sampling frequency in Hz, here case of 802.11p with 64 subcarriers and 156250 Hz subcarrier spacing
fc                        = 5.9e9;                  % Carrier Frequecy in Hz.
v                         = 50;                     % Moving speed of user in km/h
c                         = 3e8;                    % Speed of Light in m/s
fD                        = (v/3.6)/c*fc;           % Doppler freq in Hz
plotFlag                  = 0;                      % 1 to display the channel frequency response
[rchan,~,avgPathGains]    = ch_func.GenFadingChannel(ChType, fD, fs);
init_seed = 22;
%% ---------Bit to Noise Ratio------------------%
EbN0dB                    = 0:5:30;         % bit to noise ratio
SNR_p                     = EbN0dB + 10*log10(K/nDSC) + 10*log10(K/(K + K_cp)) + 10*log10(nBitPerSym) + 10*log10(rate); % converting to symbol to noise ratio
SNR_p                     = SNR_p.';
EbN0Lin                   = 10.^(SNR_p/10);
%snr_p = Ep/KN0 => N0 = Ep/(K*snr_p)
%%%%%%
N0 = Ep*10.^(-SNR_p/10);
%%%%%%
%% ---------HPA variables: 3GPP_POLY ---------
fig = 0;         % Plot curves if fig == 1
[s_AMAM_3GPP, s_AMPM_3GPP, val_m1dB_hpa] = function_HPA_WONG5_Poly(fig);
%% EFFECTS OF MEMORY
h_fir = [0.7692, 0.1538, 0.0769]; % FIR coefficients for memory effects from the HPA
%% Simulation Parameters 
N_CH                    = 1; % number of channel realizations
N_SNR                   = length(SNR_p); % SNR length

% Normalized mean square error (NMSE) vectors
Err_LS_Preamble         = zeros(N_SNR,1);
Err_Initial             = zeros(N_SNR,1);

% Bit error rate (BER) vectors
Ber_Ideal               = zeros(N_SNR,1);
Ber_LS                  = zeros(N_SNR,1);
Ber_Initial             = zeros(N_SNR,1);

% average channel power E(|hf|^2)
Phf_H_Total             = zeros(N_SNR,1);
%% Simulation Loop
for n_snr = 1:N_SNR
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(n_snr))]);
    tic;    
    TX_Bits_Stream_Structure                = zeros(nDSC * nSym  * nBitPerSym * rate, N_CH);
    Received_Symbols_FFT_Structure          = zeros(Kon,nSym, N_CH);
    True_Channels_Structure                 = zeros(Kon, nSym, N_CH);
    LS_Structure                            = zeros(Kon, nSym, N_CH);
    DPA_Structure                           = zeros(Kon, nSym, N_CH);

    for n_ch = 1:N_CH % loop over channel realizations
        
        % Bits Stream Generation 
        Bits_Stream_Coded = randi(2, nDSC * nSym  * nBitPerSym * rate,1)-1;
        % Data Scrambler 
        scrambledData = wlanScramble(Bits_Stream_Coded,scramInit);
        % Convolutional Encoder
        dataEnc = convenc(scrambledData,trellis);
        % Interleaving
        % Matrix Interleaving
        codedata = dataEnc.';
        Matrix_Interleaved_Data = matintrlv(codedata,Interleaver_Rows,Interleaver_Columns).';
        % General Block Interleaving
        General_Block_Interleaved_Data = intrlv(Matrix_Interleaved_Data,Random_permutation_Vector);
        % Bits Mapping: M-QAM Modulation
        TxBits_Coded = reshape(General_Block_Interleaved_Data,nDSC , nSym  , nBitPerSym);
        % Gray coding goes here
        TxData_Coded = zeros(nDSC ,nSym);
        for m = 1 : nBitPerSym
           TxData_Coded = TxData_Coded + TxBits_Coded(:,:,m)*2^(m-1);
        end
        % M-QAM Modulation
        Modulated_Bits_Coded  =1/sqrt(Pow) * qammod(TxData_Coded,M);
        % OFDM Frame Generation
        OFDM_Frame_Coded = zeros(K,nSym);
        OFDM_Frame_Coded(data_locations,:) = Modulated_Bits_Coded;
        OFDM_Frame_Coded(pilots_locations,:) = repmat(pilots,1,nSym);
        % Taking FFT and normalizing (power of transmit symbol needs to be 1)
        %%%%%%%%%%
        IFFT_Data_Coded = ifft(OFDM_Frame_Coded);
        %%%%%%%%%%
        norm_factor = sqrt(sum(abs(IFFT_Data_Coded(:).^2))./length(IFFT_Data_Coded(:)));
        IFFT_Data_Coded = IFFT_Data_Coded/norm_factor;
        power_Coded_frame = sqrt(sum(abs(IFFT_Data_Coded(:).^2))./length(IFFT_Data_Coded(:)));
        % Appending cylic prefix
        CP_Coded = IFFT_Data_Coded((K - K_cp +1):K,:);
        IFFT_Data_CP_Coded = [CP_Coded; IFFT_Data_Coded];
        % Appending preamble symbol 
        IFFT_Data_CP_Preamble_Coded = [ preamble_80211p IFFT_Data_CP_Coded];
       
        % NL HPA: Polynomial Model
        % Before amplifying the signal by the HPA it is necessary to scale
        % it by the coefficient alpha (Equation 2) 
        coeff_IBO_m1dB = val_m1dB_hpa*sqrt(10^(-IBO/10));
        IFFT_Data_CP_Preamble_Coded_HPA = coeff_IBO_m1dB*IFFT_Data_CP_Preamble_Coded;
        % The conversion function from the Polynomial model: Document R4-163314
        output_HPA = polyval(s_AMAM_3GPP,abs(IFFT_Data_CP_Preamble_Coded_HPA)).*exp(1i*angle(IFFT_Data_CP_Preamble_Coded_HPA)+ 1i*polyval(s_AMPM_3GPP,abs(IFFT_Data_CP_Preamble_Coded_HPA))*2*pi/360);

        % Estimation of K0 and SIGMAD2: modeling the non-linear distortions
        inhpa = linspace(0.001, 0.7, 10000);
        vout1 = polyval(s_AMAM_3GPP,abs(inhpa)).*exp(1i*angle(inhpa)+ 1i*polyval(s_AMPM_3GPP,abs(inhpa))*2*pi/360);
        % Computation of polynomial model of the HPA with order Np
        %   Theoretical computation of K0 using the polynomial model: K0theo
        %   Theoretical computation of Sigmad2 = sigmad2theo
        Nphpa = 7; 
        [spolynom, K0theo, sigmad2theo] = charac_hpa(coeff_IBO_m1dB, inhpa, vout1, Nphpa);

        % Phase correction due to K0
        input_OFDM = reshape(IFFT_Data_CP_Preamble_Coded,1,size(IFFT_Data_CP_Preamble_Coded,1)*size(IFFT_Data_CP_Preamble_Coded,2));
        output_HPA_reshape = reshape(output_HPA,1,size(output_HPA,1)*size(output_HPA,2));    
        output_HPA_final = exp(-1i*angle(K0theo))*sqrt(var(input_OFDM))*output_HPA/sqrt(var(output_HPA_reshape)); 
                
        % Memory effects
        output_HPA_final_memory = filter(h_fir,1,output_HPA_final); % Filter construction

        % Applying filter FIR and normalizing the signal
        figure
        Sf = mean(abs(fft(output_HPA_final_memory,10*nFFT).^2),2);
        findex = linspace(1, 64, 640);
        plot(findex,mag2db(normalize(Sf,'range')),'k-','LineWidth',1.5);
        hold on
        H_fir = abs(fft(h_fir,10*nFFT).^2);
        plot(findex,mag2db(H_fir),'LineWidth',2)
        legend({'$\left| {\mathbf{U}_{i,k}}^2 \right|$','$\left| {\mathbf{H}_{\mathrm{FIR}_{k}}}^2 \right|$'},'FontSize',16,'Interpreter','latex')
        xlim([0 nFFT])
        grid on
        xlabel('Subcarrier index')
        ylabel('Magnitude (dB)')
        set(gca,'FontSize',16)
        set(0,'defaulttextinterpreter','latex')
        set(gca,'TickLabelInterpreter','latex')
        return
        % ideal estimation
        release(rchan);
        rchan.Seed = rchan.Seed+1;
        [ h, y ] = ch_func.ApplyChannel(rchan, output_HPA_final_memory, K_cp);

        yp = y((K_cp+1):end,1:2);
        y  = y((K_cp+1):end,3:end);
        
        %%%%%
        yFD = sqrt(1/K)*fft(y);
        yfp = sqrt(1/K)*fft(yp); % FD preamble
        %%%%%

        h = h((K_cp+1):end,:);
        hf = fft(h); % Fd channel
        hf  = hf(:,3:end);

        Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + norm(hf(Kset))^2;
        %add noise
	    noise_preamble = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,2], 1);	
        yfp_r = yfp +  noise_preamble;	
        noise_OFDM_Symbols = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,size(yFD,2)], 1);	
        y_r   = yFD + noise_OFDM_Symbols;   
       %% Channel Estimation
       % IEEE 802.11p LS Estimate at Preambles
       he_LS_Preamble = ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
       H_LS = repmat(he_LS_Preamble,1,nSym);
       err_LS_Preamble = norm(H_LS - hf(Kset,:))^2;
       Err_LS_Preamble (n_snr) = Err_LS_Preamble (n_snr) + err_LS_Preamble;
           
       % Initial Channel Estimation
       [H_Initial, Equalized_OFDM_Symbols_Initial] = Initial_Channel_Estimation(he_LS_Preamble ,y_r, Kset, ppositions, mod, nUSC, nSym);
       err_H_Initial = norm(H_Initial - hf(Kset,:))^2;
       Err_Initial(n_snr) = Err_Initial(n_snr) + err_H_Initial;
            
       % Equalization
       y_Ideal = y_r(data_locations ,:) ./ hf(data_locations,:); %Ideal
       y_LS = y_r(data_locations ,:)./ H_LS(dpositions,:); % LS

       % QAM - DeMapping
       De_Mapped_Ideal     = qamdemod(sqrt(Pow) * y_Ideal,M);
       De_Mapped_LS        = qamdemod(sqrt(Pow) * y_LS,M);
       De_Mapped_Initial   = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_Initial(dpositions,:),M); 
        
       % Bits Extraction
       Bits_Ideal       = zeros(nDSC,nSym,log2(M));
       Bits_LS          = zeros(nDSC,nSym,log2(M));
       Bits_Initial     = zeros(nDSC,nSym,log2(M));

       for b = 1:nSym
           Bits_Ideal(:,b,:)     = de2bi(De_Mapped_Ideal(:,b),nBitPerSym);
           Bits_LS(:,b,:)        = de2bi(De_Mapped_LS(:,b),nBitPerSym); 
           Bits_Initial(:,b,:)   = de2bi(De_Mapped_Initial(:,b),nBitPerSym);  
       end

       % De-Interleaving
       % General Block De-Interleaving
       General_Block_De_Interleaved_Data_Ideal     = deintrlv(Bits_Ideal(:),Random_permutation_Vector);
       General_Block_De_Interleaved_Data_LS        = deintrlv(Bits_LS(:),Random_permutation_Vector);    
       General_Block_De_Interleaved_Data_Initial   = deintrlv(Bits_Initial(:),Random_permutation_Vector); 

       % Matrix De-Interleaving
       Matrix_De_Interleaved_Data_Ideal     = matintrlv(General_Block_De_Interleaved_Data_Ideal.',Interleaver_Columns,Interleaver_Rows).';
       Matrix_De_Interleaved_Data_LS        = matintrlv(General_Block_De_Interleaved_Data_LS.',Interleaver_Columns,Interleaver_Rows).';
       Matrix_De_Interleaved_Data_Initial   = matintrlv(General_Block_De_Interleaved_Data_Initial.',Interleaver_Columns,Interleaver_Rows).';
       
       % Viterbi decoder
       Decoded_Bits_Ideal     = vitdec(Matrix_De_Interleaved_Data_Ideal,trellis,tbl,'trunc','hard');
       Decoded_Bits_LS        = vitdec(Matrix_De_Interleaved_Data_LS,trellis,tbl,'trunc','hard');
       Decoded_Bits_Initial   = vitdec(Matrix_De_Interleaved_Data_Initial,trellis,tbl,'trunc','hard');

       % De-scrambler Data
       Bits_Ideal_Final     = wlanScramble(Decoded_Bits_Ideal,scramInit);
       Bits_LS_Final        = wlanScramble(Decoded_Bits_LS,scramInit);
       Bits_Initial_Final   = wlanScramble(Decoded_Bits_Initial,scramInit);

       % BER Calculation
       ber_Ideal   = biterr(Bits_Ideal_Final,Bits_Stream_Coded);
       ber_LS      = biterr(Bits_LS_Final,Bits_Stream_Coded);
       ber_Initial = biterr(Bits_Initial_Final,Bits_Stream_Coded);        

       Ber_Ideal (n_snr)    = Ber_Ideal (n_snr) + ber_Ideal;
       Ber_LS (n_snr)       = Ber_LS (n_snr) + ber_LS;
       Ber_Initial(n_snr)   = Ber_Initial(n_snr) + ber_Initial;

       TX_Bits_Stream_Structure(:, n_ch) = Bits_Stream_Coded;
       Received_Symbols_FFT_Structure(:,:,n_ch) = y_r(Kset,:);
       True_Channels_Structure(:,:,n_ch) = hf(Kset,:);
       LS_Structure(:,n_ch)     = he_LS_Preamble;
       DPA_Structure(:,:,n_ch)  = H_Initial;        

    end   
    save(['data_' pathdata '\Simulation_' num2str(n_snr)],...
           'TX_Bits_Stream_Structure',...
           'Received_Symbols_FFT_Structure',...
           'True_Channels_Structure',...
           'LS_Structure','DPA_Structure');
    toc;
end
%% Bit Error Rate (BER)
BER_Ideal             = Ber_Ideal /(N_CH * nSym * nDSC * nBitPerSym);
figure
semilogy(EbN0dB,BER_Ideal,'k--')
BER_LS                = Ber_LS / (N_CH * nSym * nDSC * nBitPerSym);
BER_Initial           = Ber_Initial / (N_CH * nSym * nDSC * nBitPerSym);

%% Normalized Mean Square Error
Phf_H       = Phf_H_Total/(N_CH);
ERR_LS      = Err_LS_Preamble / (Phf_H * N_CH * nSym);  
ERR_Initial = Err_Initial / (Phf_H * N_CH * nSym);

save(['data_' pathdata '\Simulation_variables'],'mod','Kset','Random_permutation_Vector','fD','ChType','avgPathGains');
save(['data_' pathdata '\Classical_Results'],'ERR_LS','ERR_Initial','BER_Ideal','BER_LS','BER_Initial');