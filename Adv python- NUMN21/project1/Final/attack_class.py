import numpy as np
import matplotlib.pyplot as plt


class NetworkAttack:
    def __init__(self, model):
        self.model = model

    def fgsm_attack(self, x: np.ndarray, y: np.ndarray, epsilon: float = 0.1):
        """Fast Gradient Sign Method - fool network by changing input slightly"""

        # Get input gradient (how loss changes with input pixels)
        input_grad = self._input_gradient(x, y)

        # Create attack: move input in direction that increases loss
        attack = epsilon * np.sign(input_grad)
        adversarial_x = np.clip(x + attack, 0, 1)

        return adversarial_x

    def _input_gradient(self, x: np.ndarray, y: np.ndarray):
        """Calculate gradient of loss w.r.t input (modified backprop)"""

        bs = x.shape[0]
        assert x.shape == (bs, 784)
        # Forward pass
        activations = [x]
        zs = []
        for layer in self.model.layers:
            z, a = layer(activations[-1])
            assert z.shape == (bs, layer.dim), (
                f"expected {(bs, layer.dim)} got {z.shape=}"
            )

            zs.append(z)
            activations.append(a)

        # Loss and initial gradient
        self.model.loss_fn.forward(activations[-1], y)
        delta = self.model.loss_fn.backward()

        # Backprop to input
        for i in range(len(self.model.layers) - 1, -1, -1):
            layer = self.model.layers[i]
            z = zs[i]
            delta = delta * layer.act_fun.derivative(z)

            delta = delta @ layer.W
            assert delta.shape == (bs, layer.dim_pre), (
                f"expected {(bs, layer.dim_pre)} got {delta.shape=}"
            )

        return delta

    def test_robustness(self, x: np.ndarray, y: np.ndarray, epsilon: float = 0.1):
        """Test how many examples can be fooled"""

        original_pred = np.argmax(self.model.forward(x), axis=1)
        adversarial_x = self.fgsm_attack(x, y, epsilon)
        attacked_pred = np.argmax(self.model.forward(adversarial_x), axis=1)

        success_rate = np.mean(original_pred != attacked_pred)
        print(f"Attack success rate: {success_rate:.1%}")

        return success_rate

    def plot_attack_examples(
        self, x: np.ndarray, y: np.ndarray, epsilon: float = 0.1, n_examples: int = 5
    ):
        """Plot original vs adversarial examples"""

        x_sample = x[:n_examples]
        y_sample = y[:n_examples]

        # Create adversarial samples
        adversarial_x = self.fgsm_attack(x_sample, y_sample, epsilon)

        # Get predictions
        original_pred = np.argmax(self.model.forward(x_sample), axis=1)
        attacked_pred = np.argmax(self.model.forward(adversarial_x), axis=1)
        true_labels = np.argmax(y_sample, axis=1)

        fig, axes = plt.subplots(3, n_examples, figsize=(2 * n_examples, 6))

        for i in range(n_examples):
            # Original image
            axes[0, i].imshow(x_sample[i].reshape(28, 28), cmap="gray")
            axes[0, i].set_title(
                f"Original\nTrue: {true_labels[i]}, Pred: {original_pred[i]}"
            )
            axes[0, i].axis("off")

            # Adversarial image
            axes[1, i].imshow(adversarial_x[i].reshape(28, 28), cmap="gray")
            axes[1, i].set_title(f"Adversarial\nPred: {attacked_pred[i]}")
            axes[1, i].axis("off")

            # Difference (perturbation)
            diff = adversarial_x[i] - x_sample[i]
            axes[2, i].imshow(
                diff.reshape(28, 28), cmap="RdBu", vmin=-epsilon, vmax=epsilon
            )
            axes[2, i].set_title(f"Perturbation\n(ε={epsilon})")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.show()
