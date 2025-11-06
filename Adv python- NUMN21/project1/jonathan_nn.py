import numpy as np

class ActivationFunctions:

     #clipping avoids overflow for large z
    def sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    
    #Returns derivative of sigmoid function, needed for backpropagation.
    def sigmoid_deriv(z):
        z = np.clip(z, -50, 50)
        sig = 1.0 / (1.0 + np.exp(-z))
        return sig*(1-sig)

class LossFunctions:
    
   #For quadratic loss and sigmoid: l = 0.5 * ||a - y||^2 => dl/da = (a - y) => delta = (a - y) * sigma'(z)
    def quad_deriv(a,y,act_func_deriv):
        return (a-y)*act_func_deriv



class FeedForwardNN:

    ############## TASK 1 ############
    """
    Feed Forward Neural Network with singmoid activaton function. Supports any numner of hidden layers.
    """

    def __init__(self, sizes, seed = None):

        self.width = len(sizes)

        assert self.width >= 2 #input, hidden (any number), output. Else raiseError
        self.sizes = sizes
    
        rng = np.random.default_rng(seed)

        #initialize biases and weights. Weights scaled to kep signal stable, donw want variance too high. 
        self.biases = [np.zeros((s, 1)) for s in sizes[1:]]
        self.weights = [
            rng.normal(0.0, 1.0/np.sqrt(sizes[i]), (sizes[i+1], sizes[i]))
            for i in range(self.width - 1)
        ]


        ######### OLD CODE FROM BEFORE GENERALIZATION TO ANY NUMBER OF LAYERS ETC. SAVING FOR LOGIC ##########
        #Scaling weights to kep signals stable. Dont want variance of drawn weights too high. Biases start at 0
        # n_in, n_hidden, n_out = sizes

        # self.W1 = rng.normal(0.0, 1.0/np.sqrt(n_in), (n_hidden, n_in))
        # self.b1 = np.zeros((n_hidden, 1))
        # self.W2 = rng.normal(0.0, 1.0/np.sqrt(n_hidden), (n_out,n_hidden))
        # self.b2 = np.zeros((n_out, 1))
    
    
    def forward(self, x, act_func):
        """
        Reshapes input to column vector, computes z1, z2 and returns output activations. 

        x: input column vector of shape (n_input,) or (n_input,1)
        returns: output activations (n_output,1)
        """
        a = np.asarray(x, dtype=float).reshape(self.sizes[0], 1)

        for W, b in zip(self.weights, self.biases):
            a = ActivationFunctions.sigmoid(W @ a + b)
        return a

        ######### OLD CODE FROM BEFORE GENERALIZATION TO ANY NUMBER OF LAYERS ETC. SAVING FOR LOGIC ##########
        # z1 = self.W1 @ a + self.b1
        # a1 = act_func(z1)
        # z2 = self.W2 @ a1 + self.b2
        # a2 = act_func(z2)

        # return a2

       
    
    def predict_class(self, x, act_func):
        """
        Helper for classification, returns index of largest output activation.
        """
        return int(np.argmax(self.forward(x, act_func)))
    


    ############# TASK 3-4 ################


    ######### BACK PROPAGATION ###########

    def backprop (self,x,y, act_func, act_func_deriv, loss_func):
        """
        Backpropagation for one training example
        x: input xolumn vector (n_input,1)
        y: expected output column vector (n_output,1)
        returns: gradients (dW1, db1, dW2, db2)
        """


        #Forward pass
        a = np.asarray(x, dtype=float).reshape(self.sizes[0], 1)

        activations = [a]
        zs = []
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            zs.append(z)
            a = act_func(z)
            activations.append(a)

        #Backward pass
        grads_w = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        #Output error, obs averaging (by dividing with minibactch size) is done in update_mini_batch.
        delta = loss_func(activations[-1],y,act_func_deriv(zs[-1]))
        grads_w[-1] = delta @ activations[-2].T
        grads_b[-1] = delta

        #Backpropagate through hidden layers
        for l in range(2, self.width):
            z = zs[-l]
            sp = act_func_deriv(z)
            delta = (self.weights[-l+1].T @ delta) * sp
            grads_w[-l] = delta @ activations[-l-1].T
            grads_b[-l] = delta

        return grads_w, grads_b


        ######### OLD CODE FROM BEFORE GENERALIZATION TO ANY NUMBER OF LAYERS ETC. SAVING FOR LOGIC ##########
        # z1 = self.W1 @ a0 + self.b1
        # a1 = act_func(z1)
        # z2 = self.W2 @ a1 + self.b2
        # a2 = act_func(z2)

        #Output error by square loss function. Obs: averaging (by dividing by minibatch size) done in update_mini_batch.
        # delta2 = loss_func(a2,y, act_func_deriv(z2))
        # dW2 = delta2 @ np.transpose(a1)
        # db2 = delta2

        # #Hidden layer error
        # delta1 = (np.transpose(self.W2) @ delta2) * act_func_deriv(z1)
        # dW1 = delta1 @ np.transpose(a0)
        # db1 = delta1

        # return dW1, db1, dW2, db2
    

######### SGD TRAINING #########

    def update_mini_batch(self, mini_batch, lr, act_func, act_func_deriv, loss_func):
        """
        Update network weights and biases using one mini-batch.
        mini_batch: list of (x,y) pairs
        lr: learning rate
        """

        grads_w = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        for x, y in mini_batch:
            dW, db = self.backprop(x, y, act_func, act_func_deriv, loss_func)
            grads_w = [gw + dw for gw, dw in zip(grads_w, dW)]
            grads_b = [gb + db for gb, db in zip(grads_b, db)]

        m = len(mini_batch)
        self.weights = [W - (lr/m) * dW for W, dW in zip(self.weights, grads_w)]
        self.biases  = [b - (lr/m) * db for b, db in zip(self.biases, grads_b)]



        ######### OLD CODE FROM BEFORE GENERALIZATION TO ANY NUMBER OF LAYERS ETC. SAVING FOR LOGIC ##########
        # dW1 = np.zeros_like(self.W1)
        # db1 = np.zeros_like(self.b1) 
        # dW2 = np.zeros_like(self.W2)
        # db2 = np.zeros_like(self.b2)

        # batch_size = len(mini_batch)
        # for x, y in mini_batch:
        #     grad_W1, grad_b1, grad_W2, grad_b2 = self.backprop(x, y, act_func, act_func_deriv, loss_func)
        #     dW1 += grad_W1
        #     db1 += grad_b1
        #     dW2 += grad_W2
        #     db2 += grad_b2

        # #Update
        # self.W1 -= lr * (dW1/batch_size)
        # self.b1 -= lr * (db1/batch_size)
        # self.W2 -= lr * (dW2/batch_size)
        # self.b2 -= lr * (db2/batch_size)
 

    def SGD(self, training_data, epochs, mini_batch_size, lr, act_func, act_func_deriv, loss_func, test_data=None):
        """
        Train the network with stochastic gradient descent.
        training_data: list of (x,y) pairs
        test_data: optional, list of (x,y) pairs for evaluation
        """

        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)

            #Split into mini-batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for batch in mini_batches:
                self.update_mini_batch(batch, lr, act_func, act_func_deriv, loss_func)

            #Evaluation
            if test_data:
                acc = self.evaluate(act_func, test_data)
                print(f"Epoch {epoch+1}: {acc} / {len(test_data)} correct")
            else:
                print(f"Epoch {epoch+1} complete")


######### EVALUATION ###########
    def evaluate(self, act_func, test_data):
        """
        Help function for evaluateing accuracy on test data.
        """
        results = [(self.predict_class(x, act_func), np.argmax(y)) for (x,y) in test_data]
        return sum(int(p == y) for (p, y) in results) 
    

    



    



