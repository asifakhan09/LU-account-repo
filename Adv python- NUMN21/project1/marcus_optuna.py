from typing import Any
import optuna
import sys

sys.path.append("Final")
from Final import io_util
import Final.Final_nn_classes as nn
import marcus_nn_util as nn_util

# For speed, hopefully results generalize
MAX_TRAIN = 5_000
data = io_util.load_mnist("data", maximum=(MAX_TRAIN, 10_000, 10_000))
print(data)

# separate study for different depths
N_HIDDEN = 1

actfuns_hidden = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
}
actfuns_out = {
    "sigmoid": nn.Sigmoid,
    "ident": nn.Ident,
}
lossfuns = {
    "mse": nn.MSELoss,
    "ce": nn.CrossEntropyLoss,
}

n_trials = int(sys.argv[1])


def epoch_callback(trial: optuna.Trial, data: dict[str, Any]):
    trial.report(value=data["acc_val"], step=data["epoch"])

    if data["acc_val"] < 0.5 and data["epoch"] > 3:
        raise optuna.TrialPruned()

    if trial.should_prune():
        raise optuna.TrialPruned()


def objective(trial: optuna.Trial):
    """Maximize validation accuracy"""

    if MAX_TRAIN == 50_000:
        loss_mode = "mse"
        act_h_name = "relu"
        trial.set_user_attr("loss_mode", "mse")
        trial.set_user_attr("act_fn_hidden", "relu")
    else:
        loss_mode = trial.suggest_categorical("loss_mode", list(lossfuns.keys()))
        act_h_name = trial.suggest_categorical(
            "act_fn_hidden", list(actfuns_hidden.keys())
        )

    hidden_sizes = [
        trial.suggest_int(f"hidden_size{k}", 16, 128 if N_HIDDEN == 1 else 48)
        for k in range(N_HIDDEN)
    ]

    model = nn.FeedForwardNNWithLayers(
        hidden_sizes,
        act_fn_hidden=actfuns_hidden[act_h_name],
        # appropriate output activation for lossfunction
        act_fn_out=nn.Sigmoid if loss_mode == "mse" else nn.Ident,
        loss_fn=lossfuns[loss_mode](),
    )

    print(model)

    trainer = nn.Training(
        model,
        data,
        bs_train=trial.suggest_int("bs", 32, 128, step=16),
        start_lr=trial.suggest_float("start_lr", 0.1, 2),
        lr_scheduler=nn_util.PlateauLrs(10, 0.5, min_lr=0.05),
        stop_patience=15,  # not needed with pruning... but good for first trials
    )

    metrics = trainer.run(
        max_iter=30 if MAX_TRAIN > 10_000 else 100,
        epoch_callback=lambda x: epoch_callback(trial, x),
        sample_mode="random",
    )
    assert isinstance(trainer.lrs, nn_util.PlateauLrs)
    trial.set_user_attr("n_lr_steps", trainer.lrs.nsteps)

    dl_test = nn.DataLoader(data.test.X, data.test.Y, len(data.test))
    xtest, ytest = dl_test.get_batch(0)
    yhat_test = model.forward(xtest)
    loss_test = model.loss_fn.forward(yhat_test, ytest)
    # accuracy?
    pred = yhat_test.argmax(axis=-1)  # (bs_val,)
    acc_test = (pred == dl_test.Y).mean()
    trial.set_user_attr("loss_test", loss_test)
    trial.set_user_attr("acc_test", acc_test)
    return max(metrics["acc_val"])


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///data/optuna.db",
        study_name=f"FFN_{N_HIDDEN}hidden_{MAX_TRAIN}tr",
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=15,
            interval_steps=3,
        ),
    )
    study.optimize(
        objective,
        n_trials,  # very slow with more depth
    )
