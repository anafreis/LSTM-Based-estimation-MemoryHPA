# LSTM-Based_estimation_MemoryHPA
This project presents codes for the DPA-LSTM-NN channel estimation affected by HPA impairments proposed in "Memory Effects of High-Power Amplifiers in
LSTM-based Vehicular Channel Estimation".

The project is divided into two paths named Matlab_Codes and Python_Codes:

Matlab_codes folder includes all the communication scenarios implementations.
Python_codes folder includes the LSTM-NN network, including training and testing phases. 

The following instructions will guide the execution:
1) Matlab_codes/[Main_Memoryless;Main_Memory_woCompensation;Main_Memory_Compensation]: Present the main simulation file. Here the user needs to define the simulation parameters (Speed, channel model, modulation order, HPA IBO, [...]). Note that each main is used to generate the data in a specific scenario according to the HPA model. 
2) Matlab_codes/DNN_Datasets_Generation_LSTM_lessSubcarriers: This file generates the actual dataset used for training and testing the LSTM-NN network (As default, 80% of data is for training and 20% of data is for testing). Since 10000 data samples were generated in the main file,  this code selects 8000 and 2000 random indices and saves them, according to the scenario defined.  Here the user can define the sampling rate for downsampling the information of the active subcarriers considered during training of the LSTM-NN network.
3) Python_codes/DNN_Training_LSTM_batches_Less: The LSTM-NN training is performed by employing the training dataset. 500 models are saved and considered as average (in order to obtain reproducible results) in the next step.
4) Python_codes/DNN_Avg_Model_LSTM: The latest 50 trained models are averaged.
5) Python_codes/DNN_Testing_LSTM_batches_Less: The LSTM-NN model is tested in considering the testing datasets and the results are saved in .mat files.
6) Matlab_codes/DNN_Results_Processing: Process the testing datasets and calculates the BER and NMSE results of the LS-LSTM-NN estimator.
	 
Additional files:
- Matlab_codes/Channel_functions: Includes the vehicular channel models based on [1].
- Matlab_codes/[function_HPA_WONG5_Poly;charac_hpa]: Are related to the Memoryless Polynomial HPA described in [2].
- Matlab_codes/Initial_Channel_Estimation: Presents the implementation of the data-pilot aided (DPA) procedure, used as initial step in the DPA-LSTM-NN estimator.
- Matlab_codes/plotting_results: Plot the results (the results for the R2V-UC channel model with QPSK modulation, 50km/h and IBO = 2 dB is available on this repository).

For the benchmark estimators used as comparison, please follow the same methodology for including the HPA models on the codes described in [3].

[1] G. Acosta-Marum and M. A. Ingram, ‘‘Six time- and frequency-selective empirical channel models for vehicular wireless LANs,’’ IEEE Veh. Technol. Mag., vol. 2, no. 4, pp. 4–11, Dec. 2007.

[2] H. Shaiek, R. Zayani, Y. Medjahdi, and D. Roviras, “Analytical analysis of SER for beyond 5G post-OFDM waveforms in presence of high power amplifiers,” IEEE Access, vol. 7, pp. 29 441–29 452, 201.

[3] A. K. Gizzini, M. Chafii, S. Ehsanfar and R. M. Shubair, "Temporal Averaging LSTM-based Channel Estimation Scheme for IEEE 802.11p Standard," 2021 IEEE Global Communications Conference (GLOBECOM), 2021, pp. 01-07, doi: 10.1109/GLOBECOM46510.2021.9685409. Codes available at: https://github.com/abdulkarimgizzini/Temporal-Averaging-LSTM-based-Channel-Estimation-Scheme-for-IEEE-802.11p-Standard.

_______________________

If you use any of these codes for research that results in publications, please cite our reference:
A. F. Dos Reis, Y. Medjahdi, B. S. Chang, G. Brante, and C. F. Bader, “LSTM-Based estimation with Memory HPA.” [Online]. Available: https://github.com/anafreis/LSTM-Based-estimation-MemoryHPA
