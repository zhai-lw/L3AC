def train():
    import model.exp.train

    model.exp.train.train()


if __name__ == '__main__':
    # init_mnist_dataset()

    train()
    print("done")
