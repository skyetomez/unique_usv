import matplotlib.pyplot as plt
from pathlib import Path


def plot_crossentropy(history, save: bool = True, name="") -> None:
    if save:
        assert name != "", "Name cannot be empty"
    epochs = history.params["epochs"]
    cate_entropy = history.history["categorical_crossentropy"]
    val_cate_entropy = history.history["val_categorical_crossentropy"]
    plt.figure(figsize=[7, 5], dpi=200)
    plt.style.use("ggplot")
    plt.plot(range(1, epochs + 1), cate_entropy, label="crossentropy")
    plt.plot(range(1, epochs + 1), val_cate_entropy, label=" validation crossentropy")
    plt.ylabel("categorical crossentropy")
    plt.xlabel("epochs")
    plt.legend()

    if save:
        plt.savefig(
            name, dpi="figure", format="jpg", pad_inches=0.1, orientation="landscape"
        )
    else:
        plt.show()
    return None


def plot_loss(history, save: bool = True, name: str = "") -> None:
    if save:
        assert name != "", "Name cannot be empty"

    epochs = history.params["epochs"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure(figsize=[7, 5], dpi=200)
    plt.style.use("ggplot")
    plt.plot(range(1, epochs + 1), loss, label="loss")
    plt.plot(range(1, epochs + 1), val_loss, label=" validation loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    if save:
        plt.savefig(
            name, dpi="figure", format="jpg", pad_inches=0.1, orientation="landscape"
        )
    else:
        plt.show()
    return None
