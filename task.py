import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Task:
    def __init__(self, model, training_dataloader, testing_dataloader,
                 labels_map,
                 batch_size,
                 loss_func,
                 training_epochs,
                 device,
                 learning_rate=1e-3):
        self.model = model
        self.training_dataloader = training_dataloader
        self.testing_dataloader = testing_dataloader
        self.labels_map = labels_map
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.training_epochs = training_epochs
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self):
        for t in range(self.training_epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            size = len(self.training_dataloader.dataset)
            self.model.train()
            for batch, (X, y) in enumerate(self.training_dataloader):
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction error
                pred = self.model(X)
                loss = self.loss_func(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            self.test()
        print("Done with train()!")
        torch.save(self.model.state_dict(), "./checkpoints/test.pth")

    def test(self):
        size = len(self.testing_dataloader.dataset)
        num_batches = len(self.testing_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.testing_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_func(pred, y).item()
                pred_res = pred.argmax(1)
                correct += (pred_res == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def visualize_incorrect(self):
        self.model.eval()
        with torch.no_grad():
            batch_c = 0
            incorrect_c = np.zeros((10, 10))
            for X, y in self.testing_dataloader:
                batch_c += 1
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                pred_res = pred.argmax(1)
                incorrect = pred_res != y
                index = 0
                if len(X[incorrect]) != 0:
                    for i in X[incorrect]:
                        plt.imshow(i.cpu().numpy().squeeze(), cmap="gray")
                        ol_index = int(y[incorrect][index].cpu().numpy())
                        pl_index = int(pred_res[incorrect][index].cpu().numpy())
                        incorrect_c[ol_index][pl_index] += 1
                        plt.title("Original label: " +
                                  str(self.labels_map[ol_index]) +
                                  ", Prediction label: " +
                                  str(self.labels_map[pl_index]))
                        # plt.show()
                        index += 1
                print("%d incorrect predictions in batch %d." % (index, batch_c))
        fx = np.arange(0, 10, 1)
        fy = np.arange(0, 10, 1)
        fx, fy = np.meshgrid(fx, fy)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(fx, fy, incorrect_c)
        plt.show()
        print("Done with visualize_incorrect()!")
