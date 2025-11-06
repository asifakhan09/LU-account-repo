from Final.Final_nn_classes import FeedForwardNNWithLayers, Training
import io_util


class HyperparamStudy:
    def __init__(self, data: io_util.MnistDataSet, max_iter=15):
        self.data = data
        self.max_iter = max_iter

    def test_architecture(self, architectures: list):
        """Test different network sizes"""

        results = []

        for arch in architectures:
            print(f"\nTesting architecture: {arch}")

            # Create and train model
            model = FeedForwardNNWithLayers(arch, seed=42)
            training = Training(model, self.data, bs_train=32, start_lr=0.1)
            metrics = training.run(max_iter=self.max_iter, verbose=False)

            # Record results
            best_acc = max(metrics["acc_val"])
            results.append({"arch": arch, "accuracy": best_acc})
            print(f"Best accuracy: {best_acc:.3f}")

        # Find best
        best = max(results, key=lambda x: x["accuracy"])
        print(f"\nBest architecture: {best['arch']} (acc: {best['accuracy']:.3f})")

        return results

    def test_learning_rates(self, learning_rates: list):
        """Test different learning rates"""

        results = []

        for lr in learning_rates:
            print(f"\nTesting learning rate: {lr}")

            model = FeedForwardNNWithLayers([30], seed=42)
            training = Training(model, self.data, bs_train=32, start_lr=lr)
            metrics = training.run(max_iter=self.max_iter, verbose=False)

            best_acc = max(metrics["acc_val"])
            results.append({"lr": lr, "accuracy": best_acc})
            print(f"Best accuracy: {best_acc:.3f}")

        best = max(results, key=lambda x: x["accuracy"])
        print(f"\nBest learning rate: {best['lr']} (acc: {best['accuracy']:.3f})")

        return results
