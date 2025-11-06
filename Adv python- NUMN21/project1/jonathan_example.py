from jonathan_nn import FeedForwardNN, ActivationFunctions
from jonathan_data import load_mnist
import matplotlib.pyplot as plt

def main():
    #Load data
    train_set, valid_set, test_set = load_mnist("mnist.pkl.gz")
    x_train, y_train = train_set
    print("Training set:", x_train.shape, y_train.shape)

    #initalize nn
    #nn = FeedForwardNN([784, 30, 10], seed=42)
    #nn = FeedForwardNN([784, 30, 40, 10], seed=42) #To test for 2 hidden layers
    nn = FeedForwardNN([784, 30, 40, 50, 10, 20, 10], seed=42) #To test more hidden layers
    act_func = ActivationFunctions.sigmoid


    #Take the first training sample
    x = x_train[0]   #shape (784,)
    y = y_train[0]   #integer label
    print("First sample label:", y)

    #Forward pass
    output = nn.forward(x,act_func)  #shape(10,1)
    predicted = nn.predict_class(x, act_func)

    print("Network output activations:\n", output.ravel())
    print("Predicted class:", predicted)


def visualize_digit_with_prediction(index=0): 
    #Load MNIST
    train_set, _, _ = load_mnist("mnist.pkl.gz")
    x_train, y_train = train_set

    #Build the network (untrained, so predictions random)
    #nn = FeedForwardNN([784, 30, 10], seed=42)
    #nn = FeedForwardNN([784, 30, 40, 10], seed=42) #To test for 2 hidden layers
    nn = FeedForwardNN([784, 30, 40, 50, 10, 20, 10], seed=42) #To test more hidden layers
    act_func = ActivationFunctions.sigmoid

    #Pick one sample
    x = x_train[index]
    label = y_train[index]

    #forward pass
    output = nn.forward(x, act_func).ravel()
    predicted = nn.predict_class(x, act_func) #index of max output

    #Visualization
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    #Left: the digit image
    axes[0].imshow(x.reshape(28, 28), cmap="gray")
    axes[0].set_title(f"Label: {label}")
    axes[0].axis("off")

    #Right: bar chart of network outputs
    axes[1].bar(range(10), output)
    axes[1].set_xticks(range(10))
    axes[1].set_title(f"Prediction: {predicted}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    visualize_digit_with_prediction(0)
