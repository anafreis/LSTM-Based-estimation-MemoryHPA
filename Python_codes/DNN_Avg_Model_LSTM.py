from keras.models import load_model
from keras.models import clone_model
from numpy import average
from numpy import array
from tcn import TCN

scenario = 'Memory_woCompensation'
DNN_Model = 30
scheme = 'DPA_LSTM_Less' 
LSTM_size = 28
MLP_size = 15
epoch = 500


# load models from file
def load_all_models(n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
		# load model from file
		model = load_model('data\{}\models\{}_DNN_{}{}_{}_{}.h5'.format(scenario,scheme, LSTM_size, MLP_size, DNN_Model,epoch))
		# add to list of members
		all_models.append(model)
	return all_models

# create a model from the weights of multiple models
def model_weight_ensemble(members, weights):
	# determine how many layers need to be averaged
	n_layers = len(members[0].get_weights())
	# create an set of average model weights
	avg_model_weights = list()
	for layer in range(n_layers):
		# collect this layer from each model
		layer_weights = array([model.get_weights()[layer] for model in members])
		# weighted average of weights for this layer
		avg_layer_weights = average(layer_weights, axis=0, weights=weights)
		# store average layer weights
		avg_model_weights.append(avg_layer_weights)
	# create a new model with the same structure
	model = clone_model(members[0])
	# set the weights in the new
	model.set_weights(avg_model_weights)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

# load all models into memory
members = load_all_models(450,epoch)
print('Loaded %d models' % len(members))
# prepare an array of equal weights
n_models = len(members)
weights = [1/n_models for i in range(1, n_models+1)]
# create a new model with the weighted average of all model weights
model_avg = model_weight_ensemble(members, weights)
# summarize the created model
model_avg.summary()
model_avg.save('data\{}\{}_DNN_{}{}_{}.h5'.format(scenario,scheme, LSTM_size, MLP_size, DNN_Model))