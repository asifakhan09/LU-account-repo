from jonathan_nn import FeedForwardNN, ActivationFunctions, LossFunctions
from jonathan_data import load_mnist, convert_label

############ TASK 5-6 #############

def prepare_data(n_input):

    #Load mnits
    train_set, valid_set, test_set = load_mnist("mnist.pkl.gz")
    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    x_test, y_test = test_set

    #Normalize inputs (already in [0,1], but reshape for network)
    x_train = [x.reshape(n_input,1) for x in x_train]
    x_valid = [x.reshape(n_input,1) for x in x_valid]
    x_test  = [x.reshape(n_input,1) for x in x_test]

    #Convert labels to one-hot vectors
    y_train = convert_label(y_train)
    y_valid = convert_label(y_valid)
    y_test  = convert_label(y_test) 

    #Zip inputs and labels
    training_data = list(zip(x_train, [y.reshape(10,1) for y in y_train]))
    validation_data = list(zip(x_valid, [y.reshape(10,1) for y in y_valid]))
    test_data = list(zip(x_test, [y.reshape(10,1) for y in y_test]))

    return training_data, validation_data, test_data


def main():

    #Set number of layers and neurons.
    #sizes = [784, 30, 10]
    sizes = [784, 30, 50,10]

    #Initialize network 
    nn = FeedForwardNN(sizes, seed=42)
    
    #Prepare data, set activation and loss functions
    training_data, validation_data, test_data = prepare_data(sizes[0])
    act_func = ActivationFunctions.sigmoid
    act_func_deriv = ActivationFunctions.sigmoid_deriv
    loss_func = LossFunctions.quad_deriv

    #Train using SGD
    nn.SGD(training_data, epochs=10, mini_batch_size=10,lr = 0.05, act_func = act_func,
           act_func_deriv = act_func_deriv, loss_func = loss_func, test_data=test_data)


if __name__ == "__main__":
    main()