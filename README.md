## What is NeurAI?

NeurAI is a flexible tool built on top of JAX designed for spiking neural network and machine learning research

Documentation on NeurAI can be found at https://neurai.readthedocs.io/.

## How to simulate
```python
import neurai
from neurai import nn
import neurai.numpy as nnp
from neurai.nn.conn import One2One
from neurai.runner import Runner


class SimNet(neurai.SNet):

  def __init__(self, sim_t: float, dt: float):
    super().__init__(sim_t=sim_t, dt=dt)
    self.n1 = nn.LIF(size=4, V_rest=-60., V_th=-50., V_reset=-60., tau=20., I_ext=20.0)
    self.n1.spike = nnp.array([0, 1, 1, 0])
    self.n2 = nn.LIF(size=4, V_rest=-60., V_th=-50., V_reset=-60., tau=20.)
    self.synapse = nn.VoltageJump(
      pre=self.n1, post=self.n2, conn=One2One(), weight=nnp.array([1.0, 2.0, 3.0, 4.0]), delay_step=2)


net = SimNet(10.0, 0.1)
runner = Runner(net)
runner.run()
```

## How to train?


```python
from neurai.datasets.mnist import MNIST
from neurai.datasets.dataloader import DataLoader
from neurai.opt.optimizers import SGD
from neurai.grad.autograd import BP
from neurai.nn.layer.loss import softmax_cross_entropy, softmax
import neurai.numpy as nnp
from neurai.nn.layer.linear import Linear
from neurai.nn.module import Module
from jax import value_and_grad
from neurai.nn.layer.activate import Relu

# downLoad datasets
train_data = MNIST("./datasets", train=True)
test_data = MNIST("./datasets", train=False)

# need add ont_hot operation
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)


def accuracy(predict, target):
  return nnp.mean(nnp.argmax(predict, axis=1) == nnp.argmax(target, axis=1))


# compute gradient operation has some questions.
bp = BP(loss_f=softmax_cross_entropy, acc_f=accuracy)
optim = SGD(lr=0.001)


class MLP(Module):

  def __init__(self) -> None:
    super().__init__()
    self.fc1 = Linear(784, 256)
    self.fc2 = Linear(256, 128)
    self.fc3 = Linear(128, 10)
    self.relu = Relu()

  def __call__(self, params, inputs):
    y = self.relu(self.fc1(params, inputs))
    y = self.relu(self.fc2(params, y))
    y = self.fc3(params, y)
    return softmax(y.value)
    # return softmax(self.fc3(self.fc2(self.fc1(inputs))))


mlp = MLP()


def loss_fn(params, inputs):
  data, label = inputs
  outputs = mlp(params, data)
  loss = nnp.mean(-nnp.sum(label * nnp.log(outputs + 1e-7), axis=-1))
  acc = accuracy(outputs, label)
  return loss, acc


def accuracy(predict, target):
  return nnp.mean(nnp.argmax(predict, axis=1) == nnp.argmax(target, axis=1))


def _one_hot(x, k, dtype=nnp.float32):
  """Create a one-hot encoding of x of size k."""
  return nnp.array(x[:, None] == nnp.arange(k), dtype)


ps = mlp.var_manager.vs

for ep in range(10):
  for st, (data, label) in enumerate(train_loader):
    data = data.reshape(data.shape[0], -1) / 255.
    label = _one_hot(label, 10)
    (loss, acc), grads = value_and_grad(loss_fn, has_aux=True)(ps, (data, label))
    ps = optim.update(ps, grads)
    if st % 100 == 0:
      print("{}/{}: loss = {}, acc = {}".format(ep, st, loss, acc))
```

