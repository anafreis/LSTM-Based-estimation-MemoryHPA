import pickle
import numpy as np
from scipy.io import loadmat
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import scipy.io
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.backend import squeeze

scenario = 'Memory_woCompensation'
DNN_Model = 30
scheme = 'DPA_LSTM_Less'  
LSTM_size = 28
MLP_size = 15

SNR_index = np.arange(1, 8)
SNR_array = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for j in SNR_index:
    mat = loadmat('data\{}\{}_Dataset_{}.mat'.format(scenario,scheme, j))
    Testing_Dataset = mat['DNN_Datasets']
    Testing_Dataset = Testing_Dataset[0, 0]
    X = Testing_Dataset['Test_X']
    Y = Testing_Dataset['Test_Y']
    print('Loaded Dataset Inputs: ', X.shape)
    print('Loaded Dataset Outputs: ', Y.shape)
    # Normalizing Datasets
    scalerx = StandardScaler()
    scalerx.fit(X)
    scalery = StandardScaler()
    scalery.fit(Y)
    XS = scalerx.transform(X)
    YS = scalery.transform(Y)
    XS = XS.transpose()
    YS = YS.transpose()

    # To use LSTM networks, the input needs to be reshaped to be [samples, time steps, features]
    XS = np.reshape(XS,(2000, 50, 56))
    print(XS.shape)

    # Loading trained DNN
    model = load_model('data\{}\{}_DNN_{}{}_{}.h5'.format(scenario,scheme, LSTM_size, MLP_size, DNN_Model))
    print('Model Loaded: ', DNN_Model)

    # Testing the model
    Y_pred = model.predict(XS)
   
    XS = np.reshape(XS,(100000, 56))
    Y_pred = np.reshape(Y_pred,(100000, 104))

    XS = XS.transpose()
    YS = YS.transpose()
    Y_pred = Y_pred.transpose()

    # Calculation of Mean Squared Error (MSE)

    Original_Testing_X = scalerx.inverse_transform(XS)
    Original_Testing_Y = scalery.inverse_transform(YS)
    Prediction_Y = scalery.inverse_transform(Y_pred)

    Error = mean_squared_error(Original_Testing_Y, Prediction_Y)
    print('MSE: ', Error)
    
    # Saving the results and converting to .mat
    result_path = 'data\{}\{}_DNN_{}{}_Results_{}.pickle'.format(scenario,scheme,  LSTM_size, MLP_size, j)
    with open(result_path, 'wb') as f:
        pickle.dump([Original_Testing_X, Original_Testing_Y, Prediction_Y], f)

    dest_name = 'data\{}\{}_DNN_{}{}_Results_{}.mat'.format(scenario,scheme,  LSTM_size, MLP_size, j)
    a = pickle.load(open(result_path, "rb"))
    scipy.io.savemat(dest_name, {
        '{}_DNN_{}{}_test_x_{}'.format(scheme, LSTM_size, MLP_size, j): a[0],
        '{}_DNN_{}{}_test_y_{}'.format(scheme,  LSTM_size, MLP_size, j): a[1],
        '{}_DNN_{}{}_corrected_y_{}'.format(scheme,  LSTM_size, MLP_size, j): a[2]
    })
    
    print("Data successfully converted to .mat file ")
