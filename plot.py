import matplotlib.pyplot as plt
from main import epoch_loss, epoch_train_acc, epoch_valid_acc


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))

ax1.plot(range(len(epoch_loss)), epoch_loss)
ax1.set_ylabel("MeanSquareError(MSE)")
ax1.set_xlabel("Epochs")
ax1.set_title("MSE от количества эпох обучения")

ax2.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Learning')
ax2.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation')
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Epochs")
ax2.legend(loc='lower right')

fig.suptitle('Оценка производительности нейронной сети')
plt.tight_layout()
plt.show()


