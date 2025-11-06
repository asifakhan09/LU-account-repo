import abc
import random
from time import time
from typing import Callable, Iterable, Literal, Type

import numpy as np

import io_util


class ActivationFunction:
    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""

    def forward(self, x):
        """Sigmoid. numerically stable?"""
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """numerically stable?"""
        return np.exp(-x) / (1 + np.exp(-x)) ** 2


class ReLU(ActivationFunction):
    """ReLu activation function"""

    def forward(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return x > 0


class Ident(ActivationFunction):
    """No-op activation function"""

    def forward(self, x) -> np.ndarray:
        return x

    def derivative(self, x) -> np.ndarray:
        return np.ones_like(x)


class MSELoss:
    diff: np.ndarray | None = None

    def forward(self, yhat: np.ndarray, ytrue: np.ndarray):
        self.diff = yhat - ytrue
        return 0.5 * (self.diff**2).mean()

    def backward(self):
        """Get deltaT, for a network with this lossfunction"""
        assert self.diff is not None, "run forward first"
        return self.diff


class CrossEntropyLoss:
    """Combined softmax with CE Loss.
    - NOTE is this correct?
    - Asifa's verion looks a bit simpler?
    - Seems to work, about the same as MSE
    """

    def forward(self, logits: np.ndarray, y_true: np.ndarray):
        """
        logits: (bs, num_classes)
        y_true: (bs, num_classes) one-hot encoded
        """
        self.y_true = y_true
        # stability
        self.logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_probs = self.logits_shifted - np.log(
            np.sum(np.exp(self.logits_shifted), axis=1, keepdims=True)
        )
        loss = -np.sum(y_true * log_probs) / logits.shape[0]
        return loss

    def backward(self):
        """
        Gradient wrt logits
        """
        exp_shifted = np.exp(self.logits_shifted)
        softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        grad = (softmax - self.y_true) / self.logits_shifted.shape[0]
        return grad


class FFLayerSimple:
    """A single Feed Forward layer of size dim with initialized weights, biases and activation function. 

    - Has separate weight and bias arrays
    - Should not contain backward logic, handled in backprop function on network
    """

    def __init__(
        self, dim_pre: int, dim: int, act_fun: ActivationFunction, seed: int | None
    ) -> None:
        self.dim_pre = dim_pre
        self.dim = dim
        # init random weights 
        sigma = 1 / np.sqrt(self.dim)
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, sigma, (self.dim, self.dim_pre))
        self.b = np.zeros(self.dim)
        self.act_fun = act_fun

    def forward(self, x: np.ndarray):
        """
        #Runs one forward pass. Returns preactivations and output (used in backprop). 
        """
        if x.ndim != 2:
            raise ValueError(f"unexpected input {x.ndim=}")

        pre_act = x @ self.W.T + self.b  #(bs,dimpre)(dimpre,dim)->(bs,dim)
        out = self.act_fun(pre_act)

        return pre_act, out

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}({self.dim} nodes) f={type(self.act_fun).__name__}"
        )

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    @property
    def n_param(self):
        return self.W.size + self.b.size


class FeedForwardNNWithLayers:
    """
    Feed Forward Neural Network with singmoid activaton function. Supports any numner of hidden layers.
    """

    layers: list[FFLayerSimple]
    loss_fn: MSELoss | CrossEntropyLoss

    def __init__(
        self,
        hidden_sizes: Iterable[int],
        d_in: int = 784,
        d_out: int = 10,
        loss_fn: MSELoss | CrossEntropyLoss = MSELoss(),
        act_fn_hidden: Type[ActivationFunction] = Sigmoid,
        act_fn_out: Type[ActivationFunction] = Ident,
        seed: int | None = None,
    ) -> None:
        
        #Initialize 
        layers = []
        d = d_in

        #Hidden layers. Creates and appends single layer, updates previous layer dim for next loop.
        for h in hidden_sizes:
            layers.append(FFLayerSimple(d, h, act_fn_hidden(), seed=seed))
            d = h

        #Output layer
        layers.append(FFLayerSimple(d, d_out, act_fn_out(), seed=seed))
        self.layers = layers
        self.loss_fn = loss_fn

    @property
    def n_param(self):
        return sum(la.n_param for la in self.layers)

    def forward(self, x: np.ndarray):
        """pass through layers"""
        for la in self.layers:
            #Retrieves output from network. Since FFLayerSimple callable, 
            # la(x) = la.forward(x)
            _, x = la(x)

        return x

    def predict_class(self, x):
        """
        Helper for classification, returns index of largest output activation.
        """
        return int(np.argmax(self.forward(x)))

    def backprop(self, x: np.ndarray, y: np.ndarray):
        """
        Backpropagation for one training example
        x: input (bs,n_input)
        y: expected output column vector (bs, n_output)
        returns: gradients (dW, db) for each layer
        """

        assert x.ndim == 2 #(batch_size, input size)
        assert x.shape[1] == self.layers[0].dim_pre

        bs = x.shape[0] #batch size

        # Forward pass
        activations = [x]
        zs = []
        for la in self.layers:
            #Store pre-activation and layer output, from la.forward() since callable.
            z, a = la(activations[-1])
            zs.append(z)
            activations.append(a)

        #Prepare to collect gradients
        grads = [(np.empty_like(la.W), np.empty_like(la.b)) for la in self.layers]

        loss = self.loss_fn.forward(activations[-1], y)
        delta = self.loss_fn.backward() #(a-y) for mse
        assert delta.shape == (bs, self.layers[-1].dim)

        #Backpropagate through hidden layers
        for step_back in range(1, len(self.layers) + 1):
            la = self.layers[-step_back] #Current layer
            z = zs[-step_back] #Current pre-activation
            assert z.shape == (bs, la.dim), f"expected {(bs, la.dim)} got {z.shape=}"

            act_diff = la.act_fun.derivative(z) #retrieve sig'
            delta_weighted = delta * act_diff #(a-y)*sig'
            assert delta_weighted.shape == (bs, la.dim)

            delta = delta_weighted @ la.W #update delta (a-y) for next layer in loop.

            assert delta.shape == (bs, la.dim_pre), (
                f"expected {(bs, la.dim_pre)} got {delta.shape=}"
            )

            #Reshape: 
            #    delta_weighted (bs, la.dim) -> (bs, la.dim, 1)
            #    activations (bs,  la.dim_pre) -> (bs,1 , la.dim_pre)
            # => product (bs, la.dim, la.dim_pre)
            # Each batch sample dW[k] has stored gradients (la.dim, la.dim_pre)
            dW = delta_weighted[:, :, None] * activations[-step_back - 1][:, None]

            #Store averaged gradients 
            grads[-step_back] = (
                dW.mean(0),  #dW
                delta_weighted.mean(0),  #db
            )

        return grads, loss

    def update_mini_batch(self, x, y, lr):
        """
        Update network weights and biases using one mini-batch.
        lr: learning rate
        """
        #Retrieve gradients and loss
        grads, loss = self.backprop(x, y)

        #Zip through each layer and corresponding gradient touple to update weights and biases. 
        for la, (dW, db) in zip(self.layers, grads):
            la.W = la.W - lr * dW
            la.b = la.b - lr * db

        return loss

    def __str__(self) -> str:
        return (
            f"FFN ({self.n_param} params)\n  input: {self.layers[0].dim_pre}\n"
            + "\n".join(f"  {la}" for la in self.layers)
        )


class DataLoader:
    """Keep features X, and labels Y. Allows getting a batch"""

    def __init__(self, X: np.ndarray, Y: np.ndarray, bs: int, C: int = 10) -> None:
        assert X.ndim == 2
        assert Y.ndim == 1
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        self.bs = bs
        self.C = C  # for one-hot labels

    def __str__(self) -> str:
        return f"DL: {len(self.X)} samples = {len(self)} batches x {self.bs}"

    def __len__(self):
        return len(self.X) // self.bs

    def _one_hot(self, ix: np.ndarray):
        v = np.zeros((len(ix), self.C))
        v[np.arange(len(ix)), ix] = 1
        return v

    def get_batch(self, idx):
        """
        ## returns
        - xb: (bs, n_features) array
        - yb: (bs, n_classes) array (one-hot)
        """
        xb = self.X[idx : idx + self.bs, :]
        yb = self.Y[idx : idx + self.bs]
        return xb, self._one_hot(yb)

    def get_randomized_batch(self):
        """Construct a batch from random data samples"""
        idxs = random.sample(range(len(self.X)), k=self.bs)
        xb = self.X[idxs, :]
        yb = self.Y[idxs]
        return xb, self._one_hot(yb)

    def random_pseudo_iter(self):
        """get an iterator of full size, that samples randomly"""
        return iter(self.get_randomized_batch() for _ in range(len(self)))

    def shuffled(self):
        idxs = list(range(len(self.X)))
        random.shuffle(idxs)
        self.X = self.X[idxs]
        self.Y = self.Y[idxs]
        return iter(self.get_batch(i) for i in range(len(self)))


class LRScheduler:
    since_step: int = 0
    min_lr: float

    @abc.abstractmethod
    def update(self, current_lr: float, metric: float) -> float:
        pass


class Training:
    def __init__(
        self,
        model: FeedForwardNNWithLayers,
        data: io_util.MnistDataSet,
        bs_train: int = 2,
        start_lr: float = 1e-1,
        stop_patience: int | None = None,
        lr_scheduler: LRScheduler | None = None,
    ) -> None:
        self.model = model
        self.dl_train = DataLoader(data.train.X, data.train.Y, bs_train)
        self.dl_val = DataLoader(
            data.val.X, data.val.Y, len(data.val)
        )  # all at once for validation

        self.lr = start_lr
        self.lrs = lr_scheduler
        self.stop_patience = stop_patience

        print(f"Train {self.dl_train}")
        print(f"Val   {self.dl_val}")

    def run(
        self,
        max_iter=100000,
        verbose=False,
        epoch_callback: Callable | None = None,
        sample_mode: Literal["shuffle", "random", "none"] = "shuffle",
    ):
        # track metrics over epochs
        metrics = {"loss_train": [], "loss_val": [], "lr": [], "acc_val": []}

        best = {"epoch": 0, "loss_val": float("inf")}

        for epoch in range(max_iter):
            # train
            loss_mean_train = 0
            ts = time()

            if sample_mode == "none":
                it = iter(self.dl_train.get_batch(i) for i in range(len(self.dl_train)))
            elif sample_mode == "shuffle":
                it = self.dl_train.shuffled()
            elif sample_mode == "random":
                it = self.dl_train.random_pseudo_iter()
            else:
                raise ValueError("unknown mode")

            for xb, yb in it:
                # Optimization
                loss = self.model.update_mini_batch(xb, yb, self.lr)
                loss_mean_train += loss

            # Epoch average loss
            loss_mean_train /= len(self.dl_train)
            samp_sec = (len(self.dl_train) * self.dl_train.bs) / (time() - ts)

            # validate
            xb_val, yb_val = self.dl_val.get_batch(0)
            yhat_val = self.model.forward(xb_val)
            loss_mean_val = self.model.loss_fn.forward(yhat_val, yb_val) / len(
                self.dl_val
            )
            # accuracy?
            pred = yhat_val.argmax(axis=-1)  # (bs_val,)
            acc_val = (pred == self.dl_val.Y).mean()

            # track metrics
            metrics["loss_train"].append(loss_mean_train)
            metrics["loss_val"].append(loss_mean_val)
            metrics["lr"].append(self.lr)
            metrics["acc_val"].append(acc_val)

            if epoch_callback is not None:
                epoch_callback({"epoch": epoch, "acc_val": acc_val})

            if verbose or epoch == max_iter - 1:
                print(
                    f"{epoch=:4d} | {loss_mean_train=:.4f} | {loss_mean_val=:.4f} | samples/second {samp_sec:.0f} | {acc_val=:.1%} | lr: {self.lr}"
                )

            if self.lrs:
                self.lr = self.lrs.update(self.lr, metric=loss_mean_val)

            if loss_mean_val < best["loss_val"]:
                best["epoch"] = epoch
                best["loss_val"] = loss_mean_val
            # EARLY STOPPING
            if (
                self.stop_patience is not None
                and epoch > best["epoch"] + self.stop_patience
            ):
                print(f"EARLY STOPPING: {epoch=}")
                print(
                    f"{epoch=:4d} | {loss_mean_train=:.4f} | {loss_mean_val=:.4f} | samples/second {samp_sec:.0f} | {acc_val=:.1%} | lr: {self.lr}"
                )
                break

        return metrics
