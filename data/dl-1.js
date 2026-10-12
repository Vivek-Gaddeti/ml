/* Deep Learning, part 1: basics (neurons, MLPs, backprop, autograd) and training (loops, optimizers, schedules, init, normalization, regularization)
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/dl-1.js */
DL.push(
{
  id: "what-is-dl", group: "Basics", title: "What is Deep Learning?",
  sub: "Neural networks with many layers that learn their own features from raw data, instead of relying on features a human designed.",
  tags: ["intro"],
  blocks: [
    ["p", "In [classical machine learning](ml.html#what-is-ml) you usually hand-craft **features**: for an image classifier you might compute edge counts and colour histograms, then feed them to a model. **Deep learning** removes that step. You give the model raw pixels (or raw text, or raw audio) and a stack of simple **layers** learns the useful features by itself, from simple ones to complex ones."],
    ["diagram", "dl1_hier", [], "A deep network builds understanding in stages. Early layers find edges, middle layers combine them into parts, late layers recognise whole objects. Nobody programs these stages: they emerge from training."],
    ["p", "Each layer is just a weighted sum followed by a simple non-linear function. One layer is weak, but dozens stacked together can represent extremely complicated functions. The price is that the model has millions or billions of numbers (**parameters**) to fit, so it needs a lot of data, a lot of compute, and some care in how it is trained. That is what the rest of this page is about."],
    ["h", "Why does 'deep' help?"],
    ["list", [
      "**Non-linearity.** A single [linear model](ml.html#linear-regression) can only draw straight boundaries. Stacking layers with non-linear activations lets the network bend the boundary to fit almost any shape.",
      "**Reusable building blocks.** A layer's output is a new set of features that the next layer builds on. Features like 'edge' are shared by thousands of higher-level concepts.",
      "**Learning from raw data.** The same recipe (data + loss + gradient descent) works for images, text, speech and games. You change the architecture, not the whole approach.",
    ]],
    ["h", "A first taste: a straight line vs a small network"],
    ["p", "The 'two moons' data cannot be split by a straight line. A [logistic regression](ml.html#logistic-regression) is limited to a line; a small neural network with two hidden layers is not. (scikit-learn's `MLPClassifier` is a quick way to try a network before we build one by hand.)"],
    ["code", R`
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

line = LogisticRegression().fit(X_tr, y_tr)                      # one straight boundary
net = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=2000, random_state=0).fit(X_tr, y_tr)

print("logistic regression test accuracy:", round(line.score(X_te, y_te), 3))
print("neural network      test accuracy:", round(net.score(X_te, y_te), 3))
print("network parameters:", sum(w.size for w in net.coefs_) + sum(b.size for b in net.intercepts_))`, "logistic regression test accuracy: 0.908\nneural network      test accuracy: 0.942\nnetwork parameters: 337"],
    ["h", "When to use deep learning (and when not to)"],
    ["table", ["Situation", "Good fit?", "Why"], [
      ["Images, audio, video", "Yes", "Raw data with spatial or temporal structure; [CNNs](dl.html#cnn) and transformers dominate."],
      ["Text and language", "Yes", "[Transformers](transformers.html#why-tokens) and [LLMs](llm.html#pretraining) are deep networks."],
      ["Small table of numbers (a few thousand rows)", "Usually no", "[Gradient boosting](ml.html#boosting) or a linear model is faster, simpler and often more accurate."],
      ["You must explain every decision", "Careful", "Deep models are hard to interpret; a tree or linear model may be required."],
      ["Very little data, no pretrained model", "Careful", "Deep nets are data hungry. Use [transfer learning](dl.html#transfer-learning) or classical ML."],
    ]],
    ["note", "Deep learning is a toolbox, not a magic wand. For tabular data, strong tree models (see [Boosting](ml.html#boosting)) are still the first thing to try."],
    ["h", "The learning map for this page"],
    ["steps", [
      "**Basics:** one neuron, then layers ([MLP](dl.html#mlp)), the non-linearities ([activations](dl.html#activations)), what we minimise ([losses](dl.html#dl-losses)), and how gradients reach every weight ([backprop](dl.html#backprop), [autograd](dl.html#autograd)).",
      "**Training:** the [mini-batch loop](dl.html#training-loop), [optimizers](dl.html#optimizers), [learning-rate schedules](dl.html#lr-schedules), [initialization](dl.html#initialization), [vanishing gradients](dl.html#vanishing-gradients), [normalization](dl.html#normalization) and [regularization](dl.html#dl-regularization).",
      "**Architectures:** [CNNs](dl.html#cnn), [RNNs](dl.html#rnn), generative models; then [PyTorch in practice](dl.html#pytorch-basics).",
    ]],
  ],
},
{
  id: "perceptron", group: "Basics", title: "The Neuron & Perceptron",
  sub: "One artificial neuron: multiply inputs by weights, add them up, add a bias, and decide. The seed of every neural network.",
  tags: ["neuron"],
  blocks: [
    ["p", "A **neuron** is a tiny function. It looks at a few numbers (its inputs), decides how much each one matters (its **weights**), adds them up with a constant (its **bias**), and passes the total through an **activation function** to produce one output."],
    ["diagram", "dl1_neuron", [], "One neuron. The weights and the bias are the numbers that get learned; the activation f decides how the weighted sum turns into an output."],
    ["math", R`z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \mathbf w^\top \mathbf x + b \qquad a = f(z)`, "x: inputs · w: weights · b: bias · f: activation · a: the neuron's output"],
    ["p", "Think of it as a voter. Each input casts a vote, weighted by how much the neuron trusts it; the bias is the neuron's built-in mood (how high the total must be before it says yes). If $f$ is a step function that returns 1 when $z > 0$ and 0 otherwise, you have the original **perceptron** (Rosenblatt, 1958). If $f$ is the [sigmoid](dl.html#activations), you have exactly [logistic regression](ml.html#logistic-regression): a neuron is a small linear model with an activation on top."],
    ["h", "What a perceptron can draw"],
    ["p", "The perceptron says 'yes' when $\\mathbf w^\\top\\mathbf x + b > 0$. The set of points where $\\mathbf w^\\top\\mathbf x + b = 0$ is a straight line (a plane in 3-D), so a perceptron splits the space into two halves with a flat cut. The weights set the line's direction, the bias slides it."],
    ["h", "The perceptron learning rule"],
    ["p", "Show the neuron an example. If it is right, do nothing. If it is wrong, nudge the weights toward the correct answer:"],
    ["math", R`\mathbf w \leftarrow \mathbf w + \eta\,(y - \hat y)\,\mathbf x \qquad b \leftarrow b + \eta\,(y - \hat y)`, "y: the true label (0 or 1) · ŷ: the neuron's output · η: step size (1 here)"],
    ["p", "If the neuron said 0 but the answer is 1 ($y-\\hat y = +1$) we **add** the input to the weights, which makes the score for that input higher next time. If it said 1 but the answer is 0 we **subtract** it. Step through it below, first on AND (which a line can solve), then on XOR (which it cannot)."],
    ["viz", "dl1_perceptron"],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def train_perceptron(X, y, epochs=25):
    w, b = np.zeros(2), 0.0
    for epoch in range(1, epochs + 1):
        mistakes = 0
        for x_i, y_i in zip(X, y):
            pred = int(w @ x_i + b > 0)          # step activation
            err = y_i - pred                     # -1, 0 or +1
            if err != 0:
                w = w + err * x_i                # the perceptron rule
                b = b + err
                mistakes += 1
        if mistakes == 0:                        # a clean pass: done
            return w, b, epoch
    return w, b, None

for name, y in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]:
    w, b, epochs = train_perceptron(X, np.array(y))
    msg = f"converged in {epochs} epochs" if epochs else "never converged"
    print(f"{name}: w = {w.tolist()}, b = {b}  ->  {msg}")`, "AND: w = [2.0, 1.0], b = -2.0  ->  converged in 6 epochs\nOR: w = [1.0, 1.0], b = 0.0  ->  converged in 4 epochs\nXOR: w = [-1.0, 0.0], b = 1.0  ->  never converged"],
    ["h", "Why XOR is impossible for one neuron"],
    ["p", "XOR wants (0,0) and (1,1) in one class and (0,1) and (1,0) in the other. Those two groups sit on opposite diagonals of a square; no single straight line can have one group on each side. The famous 1969 book *Perceptrons* (Minsky and Papert) pointed this out and helped freeze neural-network research for years. The fix, known long before it was practical, is a **hidden layer**: the first layer re-describes the points so that a second neuron can separate them. That is the [multi-layer perceptron](dl.html#mlp)."],
    ["table", ["", "Perceptron", "Sigmoid neuron (logistic regression)", "Modern neuron (inside a deep net)"], [
      ["Activation", "Step (0 or 1)", "Sigmoid", "ReLU, GELU, tanh…"],
      ["Output", "A hard decision", "A probability", "Any real number"],
      ["Learned with", "The perceptron rule", "Gradient descent on log loss", "[Backpropagation](dl.html#backprop) + gradient descent"],
      ["Why it matters", "Historical seed", "Shows a neuron is a linear model", "The building block of everything else"],
    ]],
    ["warn", "The step function is flat everywhere except at 0, so its slope is zero: gradient descent cannot train it. Modern networks use smooth (or at least sloped) activations so that gradients can flow. The perceptron rule only works because it is a special hand-made update, not a gradient."],
    ["tip", "Practice: build these pieces yourself in [Activations and Their Derivatives](practice.html#p/activations-derivatives) and [A Dense Layer: Forward and Backward](practice.html#p/dense-layer-forward-backward)."],
  ],
},
{
  id: "mlp", group: "Basics", title: "Multi-Layer Perceptrons & the Forward Pass",
  sub: "Stack layers of neurons, with a non-linearity between them, and push data through: that is a neural network.",
  tags: ["architecture"],
  blocks: [
    ["p", "A **multi-layer perceptron (MLP)**, also called a feed-forward or fully connected network, is neurons arranged in layers. Every neuron in one layer looks at **all** the outputs of the previous layer. The first layer is the input (just your data), the last is the output (the prediction), and everything between is a **hidden layer**."],
    ["diagram", "dl1_mlp", [[3, 4, 4, 2]], "A 3-4-4-2 network: 3 input features, two hidden layers of 4 neurons, 2 outputs. Each line is one weight. Counting weights + biases: 3·4+4 + 4·4+4 + 4·2+2 = 46 parameters."],
    ["h", "One layer in matrix form"],
    ["p", "Computing a whole layer for a whole batch is one matrix multiplication. If a batch $X$ has shape (n, d_in) and the layer has $d_{out}$ neurons, its weight matrix $W$ has shape (d_in, d_out) and its bias $\\mathbf b$ has shape (d_out,):"],
    ["math", R`H = f(XW + \mathbf b)`, "X: (n, d_in) · W: (d_in, d_out) · b: (d_out,), broadcast over rows · H: (n, d_out)"],
    ["p", "Column $j$ of $W$ holds the weights of neuron $j$, so $XW$ computes every neuron's weighted sum for every example at once. The **forward pass** is just this repeated: the output of one layer is the input of the next, until the last layer gives the prediction."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))                      # a batch: 5 examples, 3 features

W1, b1 = rng.normal(size=(3, 4)) * 0.5, np.zeros(4)   # layer 1: 3 -> 4
W2, b2 = rng.normal(size=(4, 2)) * 0.5, np.zeros(2)   # layer 2: 4 -> 2

relu = lambda z: np.maximum(0, z)

Z1 = X @ W1 + b1          # (5, 3) @ (3, 4) -> (5, 4)
H = relu(Z1)              # hidden activations
out = H @ W2 + b2         # (5, 4) @ (4, 2) -> (5, 2): raw scores ("logits")

print("shapes:", X.shape, Z1.shape, H.shape, out.shape)
print("parameters:", W1.size + b1.size + W2.size + b2.size)
print("output for example 0:", out[0].round(3).tolist())`, "shapes: (5, 3) (5, 4) (5, 4) (5, 2)\nparameters: 26\noutput for example 0: [0.029, -0.133]"],
    ["h", "Why the non-linearity is not optional"],
    ["p", "If you stack layers without an activation between them, the whole network collapses into **one** linear layer, because a product of matrices is just another matrix. All the depth is wasted. The activation function is what makes depth useful."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(4, 3))
W1, W2 = rng.normal(size=(3, 5)), rng.normal(size=(5, 2))

two_layers = (X @ W1) @ W2             # two "layers", no activation
one_layer = X @ (W1 @ W2)              # a single matrix W1 @ W2 does the same job
print("identical:", np.allclose(two_layers, one_layer))

relu = lambda z: np.maximum(0, z)
with_relu = relu(X @ W1) @ W2
print("with a ReLU in between, still identical?", np.allclose(with_relu, one_layer))`, "identical: True\nwith a ReLU in between, still identical? False"],
    ["h", "An MLP solves XOR"],
    ["p", "Remember that one neuron cannot do XOR. With two hidden ReLU neurons it is easy. Hidden neuron 1 computes $\\max(0, x_1+x_2)$ (how many inputs are on), hidden neuron 2 computes $\\max(0, x_1+x_2-1)$ (are both on?), and the output is $h_1 - 2h_2$. Check all four inputs:"],
    ["code", R`
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

W1 = np.array([[1.0, 1.0],      # column 0: h1 = relu(x1 + x2)
               [1.0, 1.0]])     # column 1: h2 = relu(x1 + x2 - 1)
b1 = np.array([0.0, -1.0])
W2 = np.array([[1.0], [-2.0]])  # output = h1 - 2 * h2

H = np.maximum(0, X @ W1 + b1)
out = H @ W2

for x, h, o in zip(X, H, out):
    print(f"x = {x.astype(int).tolist()}  hidden = {h.tolist()}  output = {o[0]}")`, "x = [0, 0]  hidden = [0.0, 0.0]  output = 0.0\nx = [0, 1]  hidden = [1.0, 0.0]  output = 1.0\nx = [1, 0]  hidden = [1.0, 0.0]  output = 1.0\nx = [1, 1]  hidden = [2.0, 1.0]  output = 0.0"],
    ["p", "Nobody wrote those weights by hand in a real project: a network **learns** them. We do exactly that with [backpropagation](dl.html#backprop). Before that, two more ingredients: the choice of [activation](dl.html#activations) and the [loss](dl.html#dl-losses)."],
    ["h", "How big is a network?"],
    ["p", "A layer from $m$ inputs to $n$ outputs has $m\\cdot n$ weights and $n$ biases. Parameter counts add up fast: a layer from 784 inputs (a 28×28 image) to 512 neurons already has $784\\cdot512+512 = 401{,}920$ parameters."],
    ["h", "The same in PyTorch"],
    ["code", R`
import torch
from torch import nn

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(3, 4), nn.ReLU(),
    nn.Linear(4, 4), nn.ReLU(),
    nn.Linear(4, 2),
)
x = torch.randn(5, 3)
out = model(x)
print("output shape:", tuple(out.shape))
for name, p in model.named_parameters():
    print(f"{name:10s} shape {tuple(p.shape)}")
print("total parameters:", sum(p.numel() for p in model.parameters()))`, "output shape: (5, 2)\n0.weight   shape (4, 3)\n0.bias     shape (4,)\n2.weight   shape (4, 4)\n2.bias     shape (4,)\n4.weight   shape (2, 4)\n4.bias     shape (2,)\ntotal parameters: 46", "PyTorch", "local"],
    ["note", "PyTorch's `nn.Linear(3, 4)` stores its weight as shape (4, 3), i.e. (out, in), and computes $x W^\\top + b$. The NumPy code above uses (in, out) so that it reads as `X @ W`. Same maths, transposed storage."],
    ["qa", "Going deeper: how much can an MLP represent?", [
      "The **universal approximation theorem** says a network with just one hidden layer and enough neurons can approximate any reasonable continuous function on a bounded region as closely as you like. That sounds like depth is unnecessary, but the theorem says nothing about how many neurons you need (it can be astronomically many) or whether gradient descent can find the weights.",
      "In practice, several layers of moderate width represent useful functions far more efficiently than one enormous layer, and are easier to train. That is the practical argument for **depth**.",
    ]],
    ["tip", "Next: [Activation Functions](dl.html#activations) explains what to put between the layers."],
  ],
},
{
  id: "activations", group: "Basics", title: "Activation Functions",
  sub: "The small non-linear function after each layer's weighted sum. Without it, depth is pointless; the wrong one stops learning.",
  tags: ["activation"],
  blocks: [
    ["p", "After a neuron computes $z = \\mathbf w^\\top\\mathbf x + b$, the **activation function** $f$ turns $z$ into the neuron's output $a = f(z)$. It does two jobs: it adds the **non-linearity** that makes deep networks more powerful than a single linear model (see [MLPs](dl.html#mlp)), and its **derivative** decides how much gradient flows back during [backpropagation](dl.html#backprop). Pick a function below and slide $x$. The legend in the picture tells you which curve is $f(x)$ and which (red) is its slope $f'(x)$."],
    ["viz", "dl1_act"],
    ["h", "The classic squashing functions"],
    ["math", R`\sigma(x)=\frac{1}{1+e^{-x}} \quad \sigma'(x)=\sigma(x)\big(1-\sigma(x)\big) \qquad \tanh'(x)=1-\tanh^2(x)`],
    ["p", "**Sigmoid** maps to (0, 1), handy for probabilities. **Tanh** maps to (−1, 1) and is centred on zero, which makes learning a bit easier. Both **saturate**: for large $|x|$ the curve is almost flat, so the slope is nearly 0. Sigmoid's slope never exceeds 0.25, so stacking many sigmoid layers shrinks gradients layer after layer (see [Vanishing Gradients](dl.html#vanishing-gradients))."],
    ["h", "ReLU and its relatives"],
    ["math", R`\mathrm{ReLU}(x)=\max(0,x) \qquad \mathrm{ReLU}'(x)=\begin{cases}1 & x>0\\ 0 & x<0\end{cases}`],
    ["p", "**ReLU** (rectified linear unit) is the workhorse. It is trivial to compute, its slope is exactly 1 for positive inputs, so it does not shrink gradients, and it makes many neurons output exactly 0 (a sparse, efficient representation). The catch: a neuron whose input is negative for every example has slope 0 forever and can never recover. This is a **dead ReLU**. **Leaky ReLU** keeps a small slope (e.g. 0.01 or 0.1) on the negative side to avoid that."],
    ["p", "Modern transformers mostly use smooth variants: **GELU** $x\\,\\Phi(x)$ (where $\\Phi$ is the normal CDF) and **SiLU / Swish** $x\\,\\sigma(x)$. They look like a ReLU with a rounded corner and a small dip below zero. [Gated variants such as SwiGLU](transformers.html#transformer-block) multiply two projections together and are common in current LLMs."],
    ["code", R`
import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
acts = {
    "sigmoid":    (sigmoid,                        lambda x: sigmoid(x) * (1 - sigmoid(x))),
    "tanh":       (np.tanh,                        lambda x: 1 - np.tanh(x) ** 2),
    "ReLU":       (lambda x: np.maximum(0, x),     lambda x: (x > 0).astype(float)),
    "LeakyReLU":  (lambda x: np.where(x > 0, x, 0.1 * x), lambda x: np.where(x > 0, 1.0, 0.1)),
}

xs = np.array([-4.0, -1.0, 0.5, 4.0])
print("x           ", xs.tolist())
for name, (f, df) in acts.items():
    print(f"{name:10s} f:", f(xs).round(3).tolist())
    print(f"{'':10s} f':", df(xs).round(3).tolist())

# check one derivative numerically
h = 1e-6
x0 = 0.5
num = (sigmoid(x0 + h) - sigmoid(x0 - h)) / (2 * h)
print("sigmoid'(0.5) analytic vs numerical:", round(float(sigmoid(x0) * (1 - sigmoid(x0))), 6), round(float(num), 6))`, "x            [-4.0, -1.0, 0.5, 4.0]\nsigmoid    f: [0.018, 0.269, 0.622, 0.982]\n           f': [0.018, 0.197, 0.235, 0.018]\ntanh       f: [-0.999, -0.762, 0.462, 0.999]\n           f': [0.001, 0.42, 0.786, 0.001]\nReLU       f: [0.0, 0.0, 0.5, 4.0]\n           f': [0.0, 0.0, 1.0, 1.0]\nLeakyReLU  f: [-0.4, -0.1, 0.5, 4.0]\n           f': [0.1, 0.1, 1.0, 1.0]\nsigmoid'(0.5) analytic vs numerical: 0.235004 0.235004"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 400)
sigmoid = 1 / (1 + np.exp(-x))
fns = {"sigmoid": (sigmoid, sigmoid * (1 - sigmoid)),
       "tanh": (np.tanh(x), 1 - np.tanh(x) ** 2),
       "ReLU": (np.maximum(0, x), (x > 0) * 1.0)}

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
for name, (f, d) in fns.items():
    axes[0].plot(x, f, label=name)
    axes[1].plot(x, d, label=name)
axes[0].set_title("f(x)"); axes[1].set_title("slope f'(x)")
axes[0].set_ylim(-1.5, 2); axes[1].set_ylim(-0.1, 1.2)
axes[0].legend(); plt.tight_layout()
plt.show()
print("max slope of sigmoid:", round(float((sigmoid * (1 - sigmoid)).max()), 3))`, "max slope of sigmoid: 0.25"],
    ["h", "Which one should I use?"],
    ["table", ["Where", "Default choice", "Notes"], [
      ["Hidden layers (MLP, CNN)", "ReLU", "Fast and works well. Try Leaky ReLU if many neurons die."],
      ["Hidden layers (transformers)", "GELU or SwiGLU", "Smooth; standard in BERT, GPT and most LLMs as of 2026."],
      ["RNN / LSTM gates", "sigmoid and tanh", "Sigmoid gives a 0-1 'gate'; tanh squashes the state ([LSTM](dl.html#lstm))."],
      ["Output: binary classification", "sigmoid", "One probability. Pair with binary cross-entropy ([losses](dl.html#dl-losses))."],
      ["Output: multi-class", "softmax", "Probabilities that sum to 1 ([softmax](math.html#entropy)). Pair with cross-entropy."],
      ["Output: regression", "none (linear)", "Let the output be any real number."],
    ]],
    ["danger", "Putting a ReLU or sigmoid on the output layer of a regression model, or a softmax before `nn.CrossEntropyLoss` in PyTorch (which applies it for you). The output activation must match the loss; see [Loss Functions](dl.html#dl-losses)."],
    ["h", "The same in PyTorch"],
    ["p", "PyTorch lets you ask for the derivative with autograd and confirms our formulas:"],
    ["code", R`
import torch
import torch.nn.functional as F

x = torch.tensor([-4.0, -1.0, 0.5, 4.0], requires_grad=True)
for name, f in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh), ("relu", F.relu), ("gelu", F.gelu), ("silu", F.silu)]:
    y = f(x)
    (grad,) = torch.autograd.grad(y.sum(), x)          # sums of independent elements: grad = f'(x)
    print(f"{name:8s} f(x) = {[round(v, 3) for v in y.tolist()]}  f'(x) = {[round(v, 3) for v in grad.tolist()]}")`, "sigmoid  f(x) = [0.018, 0.269, 0.622, 0.982]  f'(x) = [0.018, 0.197, 0.235, 0.018]\ntanh     f(x) = [-0.999, -0.762, 0.462, 0.999]  f'(x) = [0.001, 0.42, 0.786, 0.001]\nrelu     f(x) = [0.0, 0.0, 0.5, 4.0]  f'(x) = [0.0, 0.0, 1.0, 1.0]\ngelu     f(x) = [-0.0, -0.159, 0.346, 4.0]  f'(x) = [-0.001, -0.083, 0.867, 1.001]\nsilu     f(x) = [-0.072, -0.269, 0.311, 3.928]  f'(x) = [-0.053, 0.072, 0.74, 1.053]", "PyTorch", "local"],
    ["tip", "If training is stuck at a high loss and many ReLU outputs are exactly 0, suspect dead neurons (learning rate too high, or bad [initialization](dl.html#initialization)). If early layers do not learn in a deep sigmoid/tanh network, see [vanishing gradients](dl.html#vanishing-gradients)."],
  ],
},
{
  id: "dl-losses", group: "Basics", title: "Loss Functions for Neural Networks",
  sub: "The single number training tries to shrink. Pick the output layer and the loss as a matched pair.",
  tags: ["loss"],
  blocks: [
    ["p", "A network's weights are tuned to make one number small: the **loss**. It compares the network's prediction $\\hat y$ with the true answer $y$. The general idea is covered in [Loss Functions & Empirical Risk](ml.html#loss-functions); here is what matters for neural networks. The key rule: **the output layer and the loss come as a pair**, and together they should make the gradient simple and stable."],
    ["table", ["Task", "Output layer", "Loss", "Formula (one example)"], [
      ["Regression", "linear (no activation)", "Mean squared error (MSE)", "$(\\hat y - y)^2$"],
      ["Regression with outliers", "linear", "Huber / L1", "$|\\hat y - y|$ for large errors"],
      ["Binary classification", "1 unit + sigmoid", "Binary cross-entropy", "$-[y\\ln p + (1-y)\\ln(1-p)]$"],
      ["Multi-class (one answer)", "K units + softmax", "Categorical cross-entropy", "$-\\ln p_{\\text{true class}}$"],
      ["Multi-label (several can be true)", "K units + sigmoid each", "BCE per label", "sum of K binary losses"],
    ]],
    ["h", "Softmax and cross-entropy"],
    ["p", "For K classes the network outputs K raw scores called **logits** $\\mathbf z$. **Softmax** turns them into probabilities, and **cross-entropy** is the negative log of the probability given to the correct class:"],
    ["math", R`p_k=\frac{e^{z_k}}{\sum_j e^{z_j}} \qquad L=-\ln p_{c}\ \ (c=\text{true class})`, "If the true class gets probability 1 the loss is 0; if it gets 0.01 the loss is 4.6"],
    ["p", "Why log? Probabilities that are confidently wrong should hurt a lot, and the log gives that: loss $\\to\\infty$ as $p_c \\to 0$. It also comes from [maximum likelihood](math.html#mle), see [Entropy & Cross-Entropy](math.html#entropy). And the gradient is beautifully simple. Combining softmax and cross-entropy, the derivative of the loss with respect to the logits is:"],
    ["math", R`\frac{\partial L}{\partial z_k}=p_k-y_k \qquad\text{(prediction minus one-hot target)}`],
    ["p", "That is the same 'prediction minus truth' shape as [logistic regression](ml.html#logistic-regression) and linear regression. It is the first number [backprop](dl.html#backprop) will propagate."],
    ["code", R`
import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # subtract the max: stops exp() overflowing
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(logits, y):                      # y: integer class labels
    p = softmax(logits)
    return -np.mean(np.log(p[np.arange(len(y)), y]))

logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3]])
y = np.array([0, 1])

p = softmax(logits)
print("probabilities:\n", p.round(3))
print("loss:", round(cross_entropy(logits, y), 4))

# gradient: (p - one_hot) / batch size
onehot = np.eye(3)[y]
grad = (p - onehot) / len(y)
print("gradient wrt logits:\n", grad.round(3))

# verify one entry numerically
h = 1e-6
bump = logits.copy(); bump[0, 2] += h
dip = logits.copy();  dip[0, 2] -= h
print("numerical d loss/d logit[0,2]:", round((cross_entropy(bump, y) - cross_entropy(dip, y)) / (2 * h), 5), " analytic:", round(grad[0, 2], 5))`, "probabilities:\n [[0.659 0.242 0.099]\n [0.109 0.802 0.089]]\nloss: 0.3185\ngradient wrt logits:\n [[-0.17   0.121  0.049]\n [ 0.054 -0.099  0.044]]\nnumerical d loss/d logit[0,2]: 0.04928  analytic: 0.04928"],
    ["warn", "Never compute `log(softmax(z))` naively: for large logits `exp` overflows, and small probabilities underflow to 0 giving `log(0) = -inf`. Libraries use the **log-sum-exp trick**, $\\ln p_c = z_c - \\ln\\sum_j e^{z_j}$, and fuse softmax with the loss (`nn.CrossEntropyLoss`, `F.cross_entropy`)."],
    ["code", R`
import numpy as np

z = np.array([[1000.0, 0.0, -1000.0]])
naive = np.log(np.exp(z) / np.exp(z).sum())                        # overflow -> nan
stable = z - (z.max() + np.log(np.exp(z - z.max()).sum()))         # log-sum-exp
print("naive :", naive.tolist())
print("stable:", stable.tolist())`, "naive : [[nan, -inf, -inf]]\nstable: [[0.0, -1000.0, -2000.0]]"],
    ["h", "Binary cross-entropy and MSE"],
    ["p", "For a single yes/no output, $p=\\sigma(z)$ and BCE $=-[y\\ln p+(1-y)\\ln(1-p)]$; its gradient w.r.t. $z$ is again $p - y$. For regression, MSE $\\tfrac12(\\hat y - y)^2$ has gradient $\\hat y - y$ with respect to the output. Do not use MSE for classification with a sigmoid: when the neuron is confidently wrong, the sigmoid slope is tiny and MSE gives almost no gradient, while cross-entropy keeps the gradient at $p-y$."],
    ["h", "In PyTorch"],
    ["code", R`
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
y = torch.tensor([0, 1])

loss = nn.CrossEntropyLoss()(logits, y)      # takes RAW logits and integer labels
loss.backward()
print("loss:", round(loss.item(), 4))
print("grad wrt logits:\n", logits.grad.round(decimals=3))

# the same thing built from pieces
print("manual:", round(F.nll_loss(F.log_softmax(logits.detach(), dim=1), y).item(), 4))

# binary: BCEWithLogitsLoss fuses sigmoid + BCE (stable)
b = nn.BCEWithLogitsLoss()(torch.tensor([2.0, -1.0]), torch.tensor([1.0, 0.0]))
print("BCE with logits:", round(b.item(), 4))

# label smoothing softens the one-hot target a little
print("with label smoothing 0.1:", round(nn.CrossEntropyLoss(label_smoothing=0.1)(logits.detach(), y).item(), 4))`, "loss: 0.3185\ngrad wrt logits:\n tensor([[-0.1700,  0.1210,  0.0490],\n        [ 0.0540, -0.0990,  0.0440]])\nmanual: 0.3185\nBCE with logits: 0.2201\nwith label smoothing 0.1: 0.4369", "PyTorch", "local"],
    ["danger", "Applying `softmax` (or `sigmoid`) yourself and then using `nn.CrossEntropyLoss` / `BCEWithLogitsLoss`. They already apply it internally, so you would squash twice, which trains slowly and gives wrong probabilities. Feed them raw logits, and only apply softmax/sigmoid at prediction time."],
    ["tip", "Other losses you will meet: **label smoothing** (soften one-hot targets so the model is not forced to be 100% sure), **focal loss** (down-weight easy examples for [imbalanced data](ml.html#imbalanced)), **contrastive losses** for embeddings, and the language-model loss, which is just cross-entropy over the vocabulary ([next-token prediction](llm.html#pretraining))."],
  ],
},
{
  id: "backprop", group: "Basics", title: "Backpropagation by Hand",
  sub: "How a network finds the gradient of the loss for every weight: one forward pass, then the chain rule run backwards. Worked with real numbers.",
  tags: ["backprop", "calculus"],
  blocks: [
    ["p", "To train a network with [gradient descent](ml.html#gradient-descent) we need $\\partial L/\\partial w$ for **every** weight. A network may have millions of them. **Backpropagation** computes all of them in one sweep, by applying the [chain rule](math.html#chain-rule) from the loss backwards and **reusing** work: the gradient at a layer is built from the gradient at the layer after it."],
    ["p", "The recipe is always the same two passes:"],
    ["steps", [
      "**Forward pass.** Compute and **remember** every intermediate value, from the inputs to the loss.",
      "**Backward pass.** Start at the loss with $\\partial L/\\partial L = 1$. Walk back one operation at a time, multiplying the incoming gradient by that operation's **local derivative**. Where a value is used in several places, add the gradients from each path.",
    ]],
    ["h", "A tiny network we can do by hand"],
    ["p", "Two inputs, two hidden neurons, one output, all with sigmoid activation, and the loss $L=\\tfrac12(y-t)^2$. One training example: $\\mathbf x=(1,\\,0.5)$ with target $t=1$."],
    ["table", ["Parameter", "Value"], [
      ["Hidden weights $W_1$ (row j = neuron j)", "$\\begin{pmatrix}0.5 & -0.4\\\\ 0.3 & 0.8\\end{pmatrix}$, biases $b_1=(0.1,\\,-0.1)$"],
      ["Output weights $W_2$", "$(0.7,\\,-0.6)$, bias $b_2=0.2$"],
    ]],
    ["p", "Step through it below. Every circle and every edge shows the real number; forward values are black, gradients are red. Change the inputs, target or learning rate with the sliders and the numbers recompute."],
    ["viz", "dl1_netpass"],
    ["h", "Forward pass, with numbers"],
    ["ol", [
      "Hidden pre-activations: $z_1 = 0.5\\cdot1 + (-0.4)\\cdot0.5 + 0.1 = 0.4$ and $z_2 = 0.3\\cdot1 + 0.8\\cdot0.5 + (-0.1) = 0.6$.",
      "Hidden activations: $h_1=\\sigma(0.4)=0.5987$, $h_2=\\sigma(0.6)=0.6457$.",
      "Output pre-activation: $z_o = 0.7\\cdot0.5987 + (-0.6)\\cdot0.6457 + 0.2 = 0.2317$.",
      "Prediction: $y=\\sigma(0.2317)=0.5577$.",
      "Loss: $L=\\tfrac12(0.5577-1)^2 = 0.0978$.",
    ]],
    ["h", "Backward pass, with numbers"],
    ["p", "We need two small facts: the sigmoid's slope $\\sigma'(z)=\\sigma(z)(1-\\sigma(z))$, and that for $z=\\sum w\\,a + b$ the derivative of $z$ with respect to a weight is the input $a$ that the weight multiplies."],
    ["ol", [
      "Loss to prediction: $\\partial L/\\partial y = y - t = 0.5577 - 1 = -0.4423$.",
      "Through the output sigmoid: $\\partial L/\\partial z_o = -0.4423\\cdot y(1-y) = -0.4423\\cdot0.2467 = -0.1091$. (This is the output neuron's **error signal**, often called $\\delta_o$.)",
      "Output weights: $\\partial L/\\partial w_{o1} = \\delta_o h_1 = -0.1091\\cdot0.5987 = -0.0653$; $\\partial L/\\partial w_{o2} = -0.1091\\cdot0.6457 = -0.0704$; $\\partial L/\\partial b_2=\\delta_o=-0.1091$.",
      "Back into the hidden layer: $\\partial L/\\partial h_1 = \\delta_o w_{o1} = -0.1091\\cdot0.7=-0.0764$ and $\\partial L/\\partial h_2=-0.1091\\cdot(-0.6)=0.0655$.",
      "Through the hidden sigmoids: $\\delta_1=-0.0764\\cdot h_1(1-h_1)=-0.0764\\cdot0.2403=-0.0184$ and $\\delta_2=0.0655\\cdot0.2288=0.0150$.",
      "Hidden weights: $\\partial L/\\partial w_{ji}=\\delta_j\\,x_i$. So row 1 is $(-0.0184\\cdot1,\\ -0.0184\\cdot0.5)=(-0.0184,\\,-0.0092)$ and row 2 is $(0.0150,\\ 0.0075)$. The biases get $\\delta_1,\\delta_2$.",
    ]],
    ["p", "Notice the pattern: at every layer there is an error signal $\\delta$ (the gradient with respect to that layer's pre-activation). The weights' gradient is $\\delta$ times the layer's input, and the error signal for the layer below is $\\delta$ times the weights, times the activation's slope."],
    ["h", "Confirm with NumPy (and with finite differences)"],
    ["p", "Here is the same computation in code. The last lines check every gradient against a **numerical** estimate: nudge one parameter by a tiny $\\epsilon$, re-run the forward pass, and see how much the loss changes. If backprop is right the two agree to many decimals."],
    ["code", R`
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
x, t = np.array([1.0, 0.5]), 1.0
W1 = np.array([[0.5, -0.4], [0.3, 0.8]]); b1 = np.array([0.1, -0.1])
W2 = np.array([0.7, -0.6]);               b2 = 0.2

def forward(W1, b1, W2, b2):
    z1 = W1 @ x + b1; h = sigmoid(z1)
    z2 = W2 @ h + b2; y = sigmoid(z2)
    return z1, h, z2, y, 0.5 * (y - t) ** 2

z1, h, z2, y, L = forward(W1, b1, W2, b2)
print("z1 =", z1.round(4).tolist(), " h =", h.round(4).tolist())
print("z2 =", round(z2, 4), " y =", round(y, 4), " L =", round(L, 4))

# ---- backward pass ----
dy  = y - t                          # dL/dy
dz2 = dy * y * (1 - y)               # through the output sigmoid  (delta of the output)
dW2 = dz2 * h                        # dL/dW2 = delta * input
db2 = dz2
dh  = dz2 * W2                       # send the error back through the weights
dz1 = dh * h * (1 - h)               # through the hidden sigmoids (deltas of the hidden layer)
dW1 = np.outer(dz1, x)               # dL/dW1[j, i] = delta_j * x_i
db1 = dz1

print("dL/dW2 =", dW2.round(4).tolist(), " dL/db2 =", round(db2, 4))
print("dL/dW1 =", dW1.round(4).tolist(), " dL/db1 =", db1.round(4).tolist())

# ---- numerical check of all 9 gradients ----
params = [W1, b1, W2, np.array(b2)]
grads = [dW1, db1, dW2, np.array(db2)]
eps, worst = 1e-6, 0.0
for P, G in zip(params, grads):
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps; up = forward(params[0], params[1], params[2], float(params[3]))[4]
        P[idx] = old - eps; dn = forward(params[0], params[1], params[2], float(params[3]))[4]
        P[idx] = old
        worst = max(worst, abs((up - dn) / (2 * eps) - G[idx]))
print("largest |numerical - backprop| over all weights:", f"{worst:.1e}")

# ---- one gradient-descent step with learning rate 0.5 ----
lr = 0.5
new = forward(W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2)
print("loss before:", round(L, 4), " after one step:", round(new[4], 4))`, "z1 = [0.4, 0.6]  h = [0.5987, 0.6457]\nz2 = 0.2317  y = 0.5577  L = 0.0978\ndL/dW2 = [-0.0653, -0.0704]  dL/db2 = -0.1091\ndL/dW1 = [[-0.0184, -0.0092], [0.015, 0.0075]]  dL/db1 = [-0.0184, 0.015]\nlargest |numerical - backprop| over all weights: 3.1e-11\nloss before: 0.0978  after one step: 0.087"],
    ["h", "The general formulas (for a whole batch)"],
    ["p", "For a batch with $n$ examples and layer $l$ with pre-activation $Z_l = A_{l-1}W_l + \\mathbf b_l$ and activation $A_l = f(Z_l)$, the error signal $\\Delta_l = \\partial L/\\partial Z_l$ obeys:"],
    ["math", R`\Delta_L = \frac{\partial L}{\partial A_L}\odot f'(Z_L)\qquad
\Delta_l = \big(\Delta_{l+1}W_{l+1}^{\top}\big)\odot f'(Z_l)`, "⊙ is element-wise multiplication. For softmax + cross-entropy the first line simplifies to Δ_L = (P − Y)/n"],
    ["math", R`\frac{\partial L}{\partial W_l} = A_{l-1}^{\top}\,\Delta_l \qquad \frac{\partial L}{\partial \mathbf b_l} = \sum_{\text{rows}}\Delta_l`],
    ["p", "Check the shapes: $A_{l-1}$ is (n, d_in) so $A_{l-1}^\\top$ is (d_in, n); $\\Delta_l$ is (n, d_out); their product is (d_in, d_out), exactly the shape of $W_l$. The gradient always has the shape of the thing it is the gradient of (see [Jacobians & shapes](math.html#jacobians))."],
    ["code", R`
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def loss_and_grads(params, X, y):
    W1, b1, W2, b2 = params
    Z1 = X @ W1 + b1; H = np.tanh(Z1)              # hidden layer (tanh)
    P = softmax(H @ W2 + b2)                       # output layer (softmax)
    n = len(y)
    loss = -np.mean(np.log(P[np.arange(n), y]))
    D2 = P.copy(); D2[np.arange(n), y] -= 1; D2 /= n   # (P - onehot) / n
    dW2, db2 = H.T @ D2, D2.sum(axis=0)
    D1 = (D2 @ W2.T) * (1 - H ** 2)                # tanh'(z) = 1 - tanh(z)^2
    dW1, db1 = X.T @ D1, D1.sum(axis=0)
    return loss, [dW1, db1, dW2, db2]

rng = np.random.default_rng(0)
X = rng.normal(size=(6, 3)); y = rng.integers(0, 4, size=6)
params = [rng.normal(size=(3, 5)) * 0.5, np.zeros(5), rng.normal(size=(5, 4)) * 0.5, np.zeros(4)]

loss, grads = loss_and_grads(params, X, y)
print("loss:", round(loss, 4))
print("gradient shapes:", [g.shape for g in grads])

# finite-difference check on a few random entries of each parameter
eps = 1e-6
for k, (P_, G_) in enumerate(zip(params, grads)):
    idx = tuple(rng.integers(0, s) for s in P_.shape)
    old = P_[idx]
    P_[idx] = old + eps; up = loss_and_grads(params, X, y)[0]
    P_[idx] = old - eps; dn = loss_and_grads(params, X, y)[0]
    P_[idx] = old
    print(f"param {k}: backprop {G_[idx]: .6f}   numerical {(up - dn) / (2 * eps): .6f}")`, "loss: 1.4326\ngradient shapes: [(3, 5), (5,), (5, 4), (4,)]\nparam 0: backprop -0.070367   numerical -0.070367\nparam 1: backprop -0.202733   numerical -0.202733\nparam 2: backprop -0.048231   numerical -0.048231\nparam 3: backprop -0.119281   numerical -0.119281"],
    ["h", "Why run it backwards?"],
    ["diagram", "dl1_graph", [], "The chain rule on a computation graph: z = w·x + b, a = σ(z), L = (a−t)². Forward values on top, gradients below, each arrow multiplying by one local derivative. [Build a Tiny Autograd Engine](dl.html#autograd) turns this picture into code."],
    ["p", "We could try to get gradients without backprop by **nudging each weight** and re-running the network (the numerical check above). That costs one extra forward pass per parameter. With a billion parameters that is hopeless. Backprop gets **all** gradients for roughly the cost of two or three forward passes, because the backward sweep reuses the stored forward values and the shared error signals. That efficiency is the reason deep learning is practical at all."],
    ["qa", "Going deeper: why backwards, not forwards?", [
      "The chain rule is a product of Jacobian matrices. You can multiply them from the input side (forward mode) or from the loss side (reverse mode); the result is the same, but the cost differs. A network has **many inputs (parameters) and one output (the loss)**. Reverse mode starts from that single output, so one sweep gives the derivative with respect to all inputs. Forward mode would need one sweep per parameter.",
      "The price of reverse mode is **memory**: the forward values must be kept until the backward pass uses them. That is why large models run out of GPU memory during training, and why tricks like gradient checkpointing (recompute instead of store) exist.",
    ]],
    ["danger", "Typical backprop bugs: (1) forgetting to multiply by the activation's derivative; (2) mixing up $W$ and $W^\\top$, which shows up as a shape error, or worse, silently when the matrix is square; (3) forgetting to divide by the batch size, so the effective learning rate changes with batch size; (4) reusing stale gradients: gradients must be **reset** before each backward pass. Always run a finite-difference **gradient check** on a tiny example, exactly as above."],
    ["tip", "Next you will see that none of this has to be written by hand: [Build a Tiny Autograd Engine](dl.html#autograd) records the operations and derives the backward pass automatically, the way PyTorch does. Practice: [Softmax + Cross-Entropy Gradient](practice.html#p/softmax-ce-gradient), [A Dense Layer: Forward and Backward](practice.html#p/dense-layer-forward-backward), [Train a 2-Layer Net on XOR](practice.html#p/xor-backprop)."],
  ],
},
{
  id: "autograd", group: "Basics", title: "Build a Tiny Autograd Engine",
  sub: "Around 60 lines of pure Python that record a calculation and run backprop on it: the idea inside PyTorch, in a form you can step through.",
  tags: ["autograd", "from scratch"],
  blocks: [
    ["p", "[Backprop by hand](dl.html#backprop) works, but writing the backward formulas for every new layer would be exhausting. **Automatic differentiation (autograd)** does it for you. The trick is simple. Wrap every number in an object that remembers (1) the numbers it was computed from and (2) a tiny function that knows its own local derivative. A calculation then builds a **graph** as it runs; calling `.backward()` walks that graph in reverse and applies the chain rule."],
    ["diagram", "dl1_graph", [], "The graph our engine will build for L = (σ(w·x + b) − t)². Each node stores its value; each arrow knows its local derivative."],
    ["h", "The Value class"],
    ["p", "Each `Value` holds `data` (the number), `grad` (the gradient of the final output with respect to this value, filled in by `backward`), and `_parents` (what it was computed from). Every operation creates a new `Value` and attaches a `_backward` closure with the local derivative. Note the `+=`: if a value is used twice, **both** paths must contribute to its gradient."],
    ["code", R`
import math

class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = data
        self.grad = 0.0                       # d(final output) / d(this value)
        self._parents = parents
        self._backward = lambda: None         # how to push gradient to the parents
        self.op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad             # d(a+b)/da = 1
            other.grad += out.grad            # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad    # d(a*b)/da = b
            other.grad += self.data * out.grad    # d(a*b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad   # tanh'(x) = 1 - tanh(x)^2
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    def backward(self):
        order, seen = [], set()
        def visit(v):                         # topological order: parents before children
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0                       # d(output)/d(output) = 1
        for v in reversed(order):             # children before parents
            v._backward()

# try it: L = (a*b + c) * f
a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
L = (a * b + c) * f
L.backward()
print("L =", L.data)
print("dL/da =", a.grad, " dL/db =", b.grad, " dL/dc =", c.grad, " dL/df =", f.grad)

# a single tanh neuron with two inputs (the bias is chosen so that the output is tanh(0.8814) = 0.7071)
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(6.8813735870195432)
o = (x1 * w1 + x2 * w2 + b).tanh()        # 2*(-3) + 0*1 + 6.88 = 0.8814
o.backward()
print("neuron output:", round(o.data, 4))
for name, v in [("x1", x1), ("w1", w1), ("x2", x2), ("w2", w2), ("b", b)]:
    print(f"d o / d {name:2s} = {v.grad: .4f}")`, "L = -8.0\ndL/da = 6.0  dL/db = -4.0  dL/dc = -2.0  dL/df = 4.0\nneuron output: 0.7071\nd o / d x1 = -1.5000\nd o / d w1 =  1.0000\nd o / d x2 =  0.5000\nd o / d w2 =  0.0000\nd o / d b  =  0.5000"],
    ["p", "Check by hand: with $d = ab + c = 4$ and $L = d\\cdot f = -8$, we get $\\partial L/\\partial f = d = 4$, $\\partial L/\\partial d = f = -2$, so $\\partial L/\\partial c=-2$, $\\partial L/\\partial a = -2\\cdot b = 6$ and $\\partial L/\\partial b=-2\\cdot a=-4$. The engine agrees. The **topological order** step guarantees a node's `_backward` runs only after every node that uses it has finished adding to its `grad`."],
    ["p", "The second test is one tanh neuron. The gradient with respect to $w_2$ is exactly 0 because its input $x_2$ is 0: changing $w_2$ cannot change the output. The gradient with respect to $x_1$ is $w_1\\cdot(1-o^2) = -3\\cdot0.5=-1.5$."],
    ["h", "A neural network out of Values"],
    ["p", "A neuron is a weighted sum plus bias and a tanh; a layer is a list of neurons; an MLP is a list of layers. With `Value` objects, building the network **is** building the graph, and training is the loop from the [previous lessons](dl.html#backprop): forward, loss, zero grads, backward, update."],
    ["code", R`
import math, random

class Value:                                  # the same engine as above, condensed so this block runs on its own
    def __init__(self, data, parents=()):
        self.data, self.grad, self._parents, self._backward = data, 0.0, parents, (lambda: None)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad; other.grad += out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def backward(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents: visit(p)
                order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v._backward()

class Neuron:
    def __init__(self, nin, act=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act
    def __call__(self, x):
        s = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # w . x + b
        return s.tanh() if self.act else s
    def parameters(self):
        return self.w + [self.b]

class MLP:
    def __init__(self, nin, sizes):
        dims = [nin] + sizes
        self.layers = [[Neuron(dims[i], act=(i < len(sizes) - 1)) for _ in range(dims[i + 1])]
                       for i in range(len(sizes))]
    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0] if len(x) == 1 else x
    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]

random.seed(0)
model = MLP(2, [4, 1])                       # 2 inputs, 4 hidden tanh neurons, 1 linear output
print("parameters:", len(model.parameters()))

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]                          # XOR, with targets -1 / +1 to suit tanh

for step in range(201):
    loss = sum((model(x) - y) ** 2 for x, y in zip(xs, ys))   # forward: builds the graph
    for p in model.parameters():
        p.grad = 0.0                                            # zero the gradients
    loss.backward()                                             # backward
    for p in model.parameters():
        p.data -= 0.05 * p.grad                                 # gradient-descent update
    if step % 40 == 0:
        print(f"step {step:3d}  loss {loss.data:.4f}")

print("predictions:", [round(model(x).data, 3) for x in xs])`, "parameters: 17\nstep   0  loss 4.1718\nstep  40  loss 2.6882\nstep  80  loss 0.2105\nstep 120  loss 0.0013\nstep 160  loss 0.0000\nstep 200  loss 0.0000\npredictions: [-1.0, 1.0, 1.0, -1.0]"],
    ["p", "A network built from scratch, with every gradient derived automatically, has learned XOR, the function a single neuron could not. Everything a deep-learning library does is this idea, made fast: PyTorch stores **tensors** instead of single numbers, runs the operations on a GPU, and has hundreds of operations with hand-written backward rules."],
    ["h", "The same graph in PyTorch"],
    ["p", "Here is the first example again with PyTorch's autograd. The gradients match ours exactly:"],
    ["code", R`
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)
f = torch.tensor(-2.0, requires_grad=True)
L = (a * b + c) * f
L.backward()                                      # same algorithm as Value.backward()
print("L =", L.item())
print("dL/da =", a.grad.item(), " dL/db =", b.grad.item(), " dL/dc =", c.grad.item(), " dL/df =", f.grad.item())

x1, x2 = torch.tensor(2.0), torch.tensor(0.0)
w1 = torch.tensor(-3.0, requires_grad=True); w2 = torch.tensor(1.0, requires_grad=True)
bias = torch.tensor(6.8813735870195432, requires_grad=True)
o = torch.tanh(x1 * w1 + x2 * w2 + bias)
o.backward()
print("neuron:", round(o.item(), 4), " dw1 =", round(w1.grad.item(), 4), " dw2 =", round(w2.grad.item(), 4), " db =", round(bias.grad.item(), 4))
print("the graph PyTorch recorded for o:", type(o.grad_fn).__name__)`, "L = -8.0\ndL/da = 6.0  dL/db = -4.0  dL/dc = -2.0  dL/df = 4.0\nneuron: 0.7071  dw1 = 1.0  dw2 = 0.0  db = 0.5\nthe graph PyTorch recorded for o: TanhBackward0", "PyTorch", "local"],
    ["h", "Things this tiny engine does not do"],
    ["list", [
      "**Tensors.** One `Value` is a single number. Real engines use arrays, so a matrix multiply is one node instead of thousands.",
      "**Speed and GPUs.** Pure Python is thousands of times slower; libraries run vectorized kernels.",
      "**More operations** (exp, log, matmul, convolution, softmax…), each with a backward rule. Adding one is just another `_backward` closure.",
      "**Memory tricks** such as freeing intermediate values or recomputing them (checkpointing).",
    ]],
    ["warn", "Gradients **accumulate** (`+=`): calling `backward()` twice adds to the old values. That is why training loops must zero the gradients every step (`optimizer.zero_grad()` in PyTorch). Forgetting it is one of the most common deep-learning bugs."],
    ["tip", "Practice: [A Tiny Autograd Value](practice.html#p/tiny-autograd) asks you to write `+`, `*` and `tanh` with their backward passes yourself. For the PyTorch side, see [PyTorch Basics](dl.html#pytorch-basics)."],
  ],
},
{
  id: "training-loop", group: "Training", title: "The Mini-Batch Training Loop",
  sub: "Shuffle, take a small batch, run forward, compute the loss, backpropagate, update. Repeat until the validation loss stops improving.",
  tags: ["training"],
  blocks: [
    ["p", "Training a network is one loop, and every deep-learning project (from a toy classifier to an LLM) is a variation of it. Each pass of the loop takes a small group of examples (a **mini-batch**), measures the loss on it, finds the gradients with [backprop](dl.html#backprop), and nudges all weights a little."],
    ["diagram", "dl1_loop", [], "The training loop. One run through the data is an epoch; one pass around the cycle is an iteration (a step)."],
    ["h", "Vocabulary you will see everywhere"],
    ["table", ["Term", "Meaning", "Example"], [
      ["Batch size $B$", "Examples used for one gradient step", "32"],
      ["Iteration (step)", "One forward + backward + update", "—"],
      ["Epoch", "One full pass over the training data", "400 examples / 32 per batch = 13 iterations (the last batch is smaller)"],
      ["Training loss", "Loss on the batches the model learns from", "Falls steadily, mostly"],
      ["Validation loss", "Loss on held-out data, never trained on", "Tells you about overfitting"],
    ]],
    ["p", "Why mini-batches and not the whole dataset? A full pass per step is slow and wasteful: the first thousand examples already point in nearly the same direction as all million. A single example is cheap but noisy. A batch of 32 to a few hundred is the sweet spot: the gradient is a good estimate, and GPUs process a batch almost as fast as one example. The noise even helps a little in escaping bad spots. This is **stochastic gradient descent (SGD)** in its usual meaning (see [Gradient Descent](ml.html#gradient-descent))."],
    ["h", "From scratch: a 2-layer net on the 'moons' data"],
    ["p", "Everything in one place: He-initialised weights ([why](dl.html#initialization)), a ReLU hidden layer, a sigmoid output with binary cross-entropy, manual backprop (the formulas from the [previous section](dl.html#backprop)), shuffled mini-batches, and a validation check after each epoch."],
    ["code", R`
import numpy as np
from sklearn.datasets import make_moons

rng = np.random.default_rng(0)
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X_tr, y_tr, X_va, y_va = X[:400], y[:400], X[400:], y[400:]

sigmoid = lambda z: 1 / (1 + np.exp(-z))
H = 16
W1, b1 = rng.normal(size=(2, H)) * np.sqrt(2 / 2), np.zeros(H)      # He init for ReLU
W2, b2 = rng.normal(size=(H, 1)) * np.sqrt(2 / H), np.zeros(1)

def forward(X):
    Z1 = X @ W1 + b1; A1 = np.maximum(0, Z1)
    return Z1, A1, sigmoid(A1 @ W2 + b2).ravel()

def bce(p, y):
    return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

lr, B, epochs = 0.1, 32, 60
for epoch in range(1, epochs + 1):
    order = rng.permutation(len(X_tr))                  # shuffle every epoch
    for i in range(0, len(order), B):
        idx = order[i:i + B]
        xb, yb = X_tr[idx], y_tr[idx]
        Z1, A1, p = forward(xb)                         # forward
        d2 = ((p - yb) / len(yb))[:, None]              # dL/d(logit) for sigmoid + BCE
        dW2, db2 = A1.T @ d2, d2.sum(axis=0)            # backward
        d1 = (d2 @ W2.T) * (Z1 > 0)
        dW1, db1 = xb.T @ d1, d1.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1                  # update
        W2 -= lr * dW2; b2 -= lr * db2
    if epoch % 10 == 0:
        p_tr, p_va = forward(X_tr)[2], forward(X_va)[2]
        print(f"epoch {epoch:2d}  train loss {bce(p_tr, y_tr):.3f}  val loss {bce(p_va, y_va):.3f}  val acc {np.mean((p_va > 0.5) == y_va):.3f}")`, "epoch 10  train loss 0.328  val loss 0.296  val acc 0.895\nepoch 20  train loss 0.299  val loss 0.272  val acc 0.895\nepoch 30  train loss 0.287  val loss 0.261  val acc 0.895\nepoch 40  train loss 0.277  val loss 0.252  val acc 0.905\nepoch 50  train loss 0.266  val loss 0.245  val acc 0.895\nepoch 60  train loss 0.255  val loss 0.237  val acc 0.900"],
    ["h", "The same loop in PyTorch"],
    ["p", "Libraries do the three hard parts for you: batching (`DataLoader`), backprop (`loss.backward()`), and the update rule (`optimizer.step()`). The structure is identical, and the order of the three calls inside the loop matters."],
    ["code", R`
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons

torch.manual_seed(0)
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X = torch.tensor(X, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32)
train = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))   # outputs a logit
loss_fn = nn.BCEWithLogitsLoss()                                        # sigmoid + BCE, fused
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 61):
    model.train()
    for xb, yb in train:
        opt.zero_grad()                       # 1. clear old gradients
        loss = loss_fn(model(xb).squeeze(1), yb)
        loss.backward()                       # 2. backprop
        opt.step()                            # 3. update the weights
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():                 # no graph needed for evaluation
            logits = model(X[400:]).squeeze(1)
            val_loss = loss_fn(logits, y[400:]).item()
            val_acc = ((logits > 0).float() == y[400:]).float().mean().item()
        print(f"epoch {epoch:2d}  val loss {val_loss:.3f}  val acc {val_acc:.3f}")`, "epoch 10  val loss 0.327  val acc 0.865\nepoch 20  val loss 0.293  val acc 0.895\nepoch 30  val loss 0.283  val acc 0.895\nepoch 40  val loss 0.275  val acc 0.900\nepoch 50  val loss 0.269  val acc 0.900\nepoch 60  val loss 0.265  val acc 0.900", "PyTorch", "local"],
    ["h", "Choosing the batch size"],
    ["table", ["Batch size", "Effect"], [
      ["Small (8-32)", "Noisy gradients, which can regularise a little; many updates per epoch; poor use of a GPU."],
      ["Medium (64-256)", "A good default for many problems."],
      ["Large (1024+)", "Smooth gradients, great hardware use, but fewer updates per epoch; often needs a larger learning rate (and warmup, see [Learning-Rate Schedules](dl.html#lr-schedules)) and can generalise slightly worse."],
    ]],
    ["warn", "Shuffle the training data every epoch. If the data is ordered (all class 0 first), unshuffled batches give the model one-sided gradients and it keeps lurching toward whichever class it just saw."],
    ["danger", "Evaluating with dropout or batch-norm still in **training mode** (`model.eval()` forgotten), or computing validation metrics with gradients on (wasted memory; use `torch.no_grad()`). Also never train on the validation or test data; see [train/validation/test](ml.html#splits)."],
    ["tip", "Before a long run, try to **overfit one small batch** (say 16 examples) to near-zero loss. If the loop, loss, and gradients are right, a net can always memorise 16 examples; if it cannot, you have a bug. More in [Debugging Neural Network Training](dl.html#debugging-training). The full PyTorch recipe, with evaluation and saving, is in [Training a Model in PyTorch](dl.html#pytorch-training)."],
  ],
},
{
  id: "optimizers", group: "Training", title: "Optimizers: SGD, Momentum, RMSProp & Adam",
  sub: "All of them walk downhill using gradients; they differ in how they turn a gradient into a step.",
  tags: ["optimizer"],
  blocks: [
    ["p", "The gradient says which way is downhill. The **optimizer** decides how far to step and how to smooth the steps. Plain [gradient descent](ml.html#gradient-descent) uses one fixed step size for every parameter. That struggles on **ravines**: loss surfaces that are steep in one direction and nearly flat in another, which is common in real networks. The fancier optimizers fix this in two ways: by **averaging** gradients over time (momentum) and by **rescaling** each parameter's step (RMSProp, Adam)."],
    ["viz", "dl1_optim"],
    ["h", "SGD"],
    ["math", R`\theta \leftarrow \theta - \eta\, g_t`, "g_t: the (mini-batch) gradient · η: learning rate"],
    ["p", "Simple and memory-free, but it has the ravine problem: the step size is limited by the steepest direction, so it crawls along the flat one."],
    ["h", "Momentum"],
    ["math", R`v \leftarrow \beta v + g_t \qquad \theta \leftarrow \theta - \eta\, v`, "β ≈ 0.9: how much of the previous velocity to keep"],
    ["p", "Think of a heavy ball rolling downhill. Gradients that point the same way keep adding to the velocity; gradients that flip sign from step to step (the zig-zag across a ravine) cancel out. Effectively the step grows by up to $1/(1-\\beta)$, so $\\beta=0.9$ is like a 10x larger step along consistent directions."],
    ["h", "RMSProp"],
    ["math", R`s \leftarrow \rho\, s + (1-\rho)\, g_t^2 \qquad \theta \leftarrow \theta - \eta\,\frac{g_t}{\sqrt{s}+\epsilon}`, "s: running average of the squared gradient, per parameter · ρ ≈ 0.9 · ε ≈ 1e-8 avoids dividing by zero"],
    ["p", "Divide each parameter's gradient by the typical size of its recent gradients. Parameters with large gradients get smaller steps, parameters with tiny gradients get bigger ones. Every direction then moves at roughly the same pace, which is what a ravine needs."],
    ["h", "Adam"],
    ["math", R`m \leftarrow \beta_1 m + (1-\beta_1) g_t \qquad s \leftarrow \beta_2 s + (1-\beta_2) g_t^2`],
    ["math", R`\hat m = \frac{m}{1-\beta_1^{\,t}} \qquad \hat s = \frac{s}{1-\beta_2^{\,t}} \qquad \theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat s}+\epsilon}`, "Typical: β₁ = 0.9, β₂ = 0.999, ε = 1e-8"],
    ["p", "Adam = momentum ($m$) + RMSProp scaling ($s$). The division by $1-\\beta^t$ is **bias correction**: $m$ and $s$ start at 0, so in the first steps they are too small; the correction rescales them. A neat consequence: on the very first step, $\\hat m/\\sqrt{\\hat s} = g/|g|$, so every parameter moves by about $\\eta$, whatever the gradient's size."],
    ["h", "All four from scratch"],
    ["p", "Same ravine as in the picture: $L=\\tfrac12(0.05x^2+20y^2)$, same start and the same learning rate 0.05 for everyone, 100 steps. Compare the final losses with the visualizer."],
    ["code", R`
import numpy as np

A, B = 0.05, 20.0
loss = lambda p: 0.5 * (A * p[0] ** 2 + B * p[1] ** 2)
grad = lambda p: np.array([A * p[0], B * p[1]])

def sgd(lr=0.05):
    return lambda p, g, st: p - lr * g

def momentum(lr=0.05, beta=0.9):
    def step(p, g, st):
        st["v"] = beta * st.get("v", 0) + g
        return p - lr * st["v"]
    return step

def rmsprop(lr=0.05, rho=0.9, eps=1e-8):
    def step(p, g, st):
        st["s"] = rho * st.get("s", 0) + (1 - rho) * g ** 2
        return p - lr * g / (np.sqrt(st["s"]) + eps)
    return step

def adam(lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    def step(p, g, st):
        st["t"] = st.get("t", 0) + 1
        st["m"] = b1 * st.get("m", 0) + (1 - b1) * g
        st["s"] = b2 * st.get("s", 0) + (1 - b2) * g ** 2
        m_hat = st["m"] / (1 - b1 ** st["t"])              # bias correction
        s_hat = st["s"] / (1 - b2 ** st["t"])
        return p - lr * m_hat / (np.sqrt(s_hat) + eps)
    return step

for name, opt in [("SGD", sgd()), ("Momentum", momentum()), ("RMSProp", rmsprop()), ("Adam", adam())]:
    p, st = np.array([-8.0, 1.5]), {}
    for _ in range(100):
        p = opt(p, grad(p), st)
    print(f"{name:9s} final loss {loss(p):.3f}   position {p.round(2).tolist()}")

# Adam's first step has size ~ lr for every parameter, whatever the gradient size
p, st = np.array([-8.0, 1.5]), {}
p1 = adam()(p, grad(p), st)
print("Adam first step:", (p1 - p).round(3).tolist(), "  (gradient was", grad(p).round(2).tolist(), ")")`, "SGD       final loss 0.970   position [-6.23, 0.0]\nMomentum  final loss 0.003   position [-0.34, 0.0]\nRMSProp   final loss 0.214   position [-2.93, -0.0]\nAdam      final loss 0.332   position [-3.64, 0.01]\nAdam first step: [0.05, -0.05]   (gradient was [-0.4, 30.0] )"],
    ["h", "Which one to use?"],
    ["table", ["Optimizer", "Strengths", "Weaknesses / notes"], [
      ["SGD + momentum", "Simple; often the best final generalisation for CNNs when well tuned", "Needs a tuned learning rate **and** schedule; slower to get going"],
      ["RMSProp", "Good on noisy problems, RNNs", "Largely replaced by Adam"],
      ["Adam", "Works out of the box (lr 1e-3, or 3e-4 for transformers); fast start", "Stores two extra values per weight; can generalise slightly worse than tuned SGD"],
      ["AdamW", "Adam with **decoupled weight decay**; the standard for transformers and LLMs", "Use it instead of Adam + L2 (see [Dropout, Weight Decay & Early Stopping](dl.html#dl-regularization))"],
    ]],
    ["tip", "A solid default: **AdamW, lr 1e-3 (small nets) to 3e-4 (transformers), betas (0.9, 0.999), weight decay 0.01-0.1**, with [warmup and cosine decay](dl.html#lr-schedules). The learning rate matters more than any other choice here."],
    ["h", "The same in PyTorch"],
    ["p", "PyTorch's built-in optimizers use exactly the update rules above (we pass `alpha=0.9` to RMSprop to match our $\\rho$). On the same ravine they land on the same losses as our NumPy versions:"],
    ["code", R`
import torch

def run(make_opt):
    p = torch.tensor([-8.0, 1.5], dtype=torch.float64, requires_grad=True)
    opt = make_opt([p])
    for _ in range(100):
        opt.zero_grad()
        loss = 0.5 * (0.05 * p[0] ** 2 + 20.0 * p[1] ** 2)
        loss.backward()
        opt.step()
    return round((0.5 * (0.05 * p[0] ** 2 + 20.0 * p[1] ** 2)).item(), 3)

print("SGD      ", run(lambda ps: torch.optim.SGD(ps, lr=0.05)))
print("Momentum ", run(lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)))
print("RMSProp  ", run(lambda ps: torch.optim.RMSprop(ps, lr=0.05, alpha=0.9)))
print("Adam     ", run(lambda ps: torch.optim.Adam(ps, lr=0.05)))`, "SGD       0.97\nMomentum  0.003\nRMSProp   0.214\nAdam      0.332", "PyTorch", "local"],
    ["warn", "Adam is not magic. If the loss explodes, lower the learning rate before blaming the optimizer. And an optimizer that wins on one problem can lose on another: the picture above changes if you change the learning rate or momentum."],
    ["danger", "Using `Adam(weight_decay=...)` and believing it is proper weight decay. In Adam the L2 term gets divided by the adaptive scaling, so it is not the same as shrinking the weights. Use `AdamW`."],
    ["tip", "Practice: [SGD with Momentum and Adam Steps](practice.html#p/sgd-momentum-adam)."],
  ],
},
{
  id: "lr-schedules", group: "Training", title: "Learning-Rate Schedules",
  sub: "The best learning rate changes during training: warm up gently, run fast in the middle, finish with small careful steps.",
  tags: ["learning rate"],
  blocks: [
    ["p", "The [learning rate](ml.html#gradient-descent) is the most important knob in training. But a single fixed value is rarely ideal for a whole run. Early on the weights are random and the gradients are large and unreliable, so big steps can cause a blow-up. Later we want to settle into a minimum, and the **noise** in mini-batch gradients keeps kicking us around unless the steps shrink. A **schedule** changes the learning rate as training goes on. Try the four common ones:"],
    ["viz", "dl1_lrsched"],
    ["table", ["Schedule", "Shape", "When it is used"], [
      ["Constant", "flat", "Quick experiments, baselines, some Adam fine-tuning"],
      ["Step decay", "staircase: multiply by γ every k steps", "Classic CNN training (e.g. divide by 10 at fixed epochs)"],
      ["Cosine decay", "smooth half-cosine from peak to near 0", "Very common default; no extra knobs"],
      ["Linear warmup + decay", "ramp up, then cosine or linear decay", "The standard for transformers and LLMs"],
      ["Reduce on plateau", "cut the rate when the validation loss stops improving", "Adaptive; useful when you do not know the training length"],
      ["One-cycle", "up to a big peak, then down far below the start", "Fast training ('super-convergence') with SGD"],
    ]],
    ["h", "Why warmup?"],
    ["p", "At step 0 the weights are random, and adaptive optimizers like [Adam](dl.html#optimizers) have not yet collected good statistics about the gradients (their running averages start at zero). A big first step can throw the weights into a bad region from which the network never recovers (the loss spikes or becomes NaN). Starting from a tiny rate and ramping up over the first few hundred or thousand steps avoids this. Large batch sizes and deep transformers need it most."],
    ["h", "Why decay?"],
    ["p", "Each mini-batch gradient is a noisy estimate. With a fixed step size, the weights keep jittering around the minimum at a distance proportional to $\\eta\\cdot$(noise). Shrinking $\\eta$ averages the noise out and lets the weights settle. A toy demo: minimise $\\tfrac12w^2$ when every gradient has random noise added."],
    ["code", R`
import numpy as np

def final_loss(schedule, steps=500, seed=0):
    rng = np.random.default_rng(seed)
    w, losses = 5.0, []
    for t in range(steps):
        g = w + rng.normal(0, 2.0)            # true gradient of 0.5*w^2, plus mini-batch noise
        w -= schedule(t) * g
        losses.append(0.5 * w * w)
    return np.mean(losses[-100:])             # average loss over the last 100 steps

constant = lambda t: 0.1
cosine = lambda t: 0.1 * 0.5 * (1 + np.cos(np.pi * t / 500))

print("constant lr 0.1   : final loss", round(final_loss(constant), 3))
print("cosine 0.1 -> 0   : final loss", round(final_loss(cosine), 3))`, "constant lr 0.1   : final loss 0.109\ncosine 0.1 -> 0   : final loss 0.002"],
    ["h", "The formulas"],
    ["math", R`\text{step: }\ \eta_t=\eta_0\,\gamma^{\lfloor t/k\rfloor} \qquad \text{cosine: }\ \eta_t=\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\Big(1+\cos\frac{\pi t}{T}\Big)`],
    ["math", R`\text{warmup: }\ \eta_t=\eta_0\,\frac{t+1}{T_w}\quad(t<T_w), \text{ then cosine over the remaining steps}`],
    ["code", R`
import numpy as np

T, base = 1000, 0.03

def step_decay(t, gamma=0.5, k=250):
    return base * gamma ** (t // k)

def cosine(t, lr_min=0.0):
    return lr_min + 0.5 * (base - lr_min) * (1 + np.cos(np.pi * t / T))

def warmup_cosine(t, warm=100, lr_min=0.0):
    if t < warm:
        return base * (t + 1) / warm                   # linear ramp up
    return lr_min + 0.5 * (base - lr_min) * (1 + np.cos(np.pi * (t - warm) / (T - warm)))

print(" step | constant | step decay | cosine | warmup+cosine")
for t in [0, 50, 100, 250, 500, 750, 999]:
    print(f"{t:5d} | {base:8.4f} | {step_decay(t):10.4f} | {cosine(t):6.4f} | {warmup_cosine(t):.4f}")`, " step | constant | step decay | cosine | warmup+cosine\n    0 |   0.0300 |     0.0300 | 0.0300 | 0.0003\n   50 |   0.0300 |     0.0300 | 0.0298 | 0.0153\n  100 |   0.0300 |     0.0300 | 0.0293 | 0.0300\n  250 |   0.0300 |     0.0150 | 0.0256 | 0.0280\n  500 |   0.0300 |     0.0075 | 0.0150 | 0.0176\n  750 |   0.0300 |     0.0037 | 0.0044 | 0.0054\n  999 |   0.0300 |     0.0037 | 0.0000 | 0.0000"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

T, base, warm = 1000, 0.03, 100
t = np.arange(T)
constant = np.full(T, base)
step = base * 0.5 ** (t // 250)
cos = 0.5 * base * (1 + np.cos(np.pi * t / T))
wc = np.where(t < warm, base * (t + 1) / warm, 0.5 * base * (1 + np.cos(np.pi * (t - warm) / (T - warm))))

plt.figure(figsize=(6, 3.4))
for name, y in [("constant", constant), ("step decay", step), ("cosine", cos), ("warmup + cosine", wc)]:
    plt.plot(t, y, label=name)
plt.xlabel("training step"); plt.ylabel("learning rate"); plt.title("Four learning-rate schedules")
plt.legend(); plt.tight_layout(); plt.show()
print("peak of warmup+cosine at step", int(wc.argmax()), "value", round(float(wc.max()), 4))`, "peak of warmup+cosine at step 99 value 0.03"],
    ["h", "In PyTorch"],
    ["p", "Schedulers wrap an optimizer. Call `scheduler.step()` once per step (or per epoch, depending on the scheduler), after `optimizer.step()`. Here a linear warmup is chained with cosine decay:"],
    ["code", R`
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

model = torch.nn.Linear(4, 1)
opt = torch.optim.AdamW(model.parameters(), lr=0.03)
warm, total = 100, 1000
sched = SequentialLR(
    opt,
    schedulers=[LinearLR(opt, start_factor=0.01, total_iters=warm),
                CosineAnnealingLR(opt, T_max=total - warm)],
    milestones=[warm],
)

for step in range(total):
    opt.step()                    # (a real loop computes a loss and calls backward() first)
    sched.step()
    if step in (0, 49, 99, 100, 500, 999):
        print(f"after step {step:4d}: lr = {opt.param_groups[0]['lr']:.5f}")`, "after step    0: lr = 0.00060\nafter step   49: lr = 0.01515\nafter step   99: lr = 0.03000\nafter step  100: lr = 0.03000\nafter step  500: lr = 0.01755\nafter step  999: lr = 0.00000", "PyTorch", "local"],
    ["tip", "A learning-rate **range test**: run a short training while the learning rate grows exponentially from tiny to huge, plot the loss, and pick a value a bit below where the loss starts to rise steeply. It takes a minute and often saves hours of guessing."],
    ["warn", "Match the schedule to the **length** of training. A cosine schedule needs to know the total number of steps $T$; if you stop early the rate has not decayed yet, and if you train longer it hits the floor and stays there."],
  ],
},
{
  id: "initialization", group: "Training", title: "Weight Initialization",
  sub: "How you pick the starting weights decides whether signals and gradients survive the trip through many layers.",
  tags: ["initialization"],
  blocks: [
    ["p", "Before training, every weight needs a starting value. This sounds like a detail, but it is not: **bad initial weights can make a deep network untrainable** before it has seen a single example. Two things must hold at the start: the neurons must be **different** from each other, and the **size** of the signals (and gradients) must stay roughly constant from layer to layer."],
    ["h", "Rule 1: do not start with all zeros (or any constant)"],
    ["p", "If every weight in a layer starts equal, every neuron computes exactly the same thing, receives exactly the same gradient, and gets exactly the same update. They stay clones forever, no matter how many neurons you add. Random values **break the symmetry**."],
    ["code", R`
import numpy as np

x = np.array([1.0, 2.0]); t = 1.0
W1 = np.full((2, 3), 0.5)                       # 3 hidden neurons, all weights identical
w2 = np.full(3, 0.5)

h = np.tanh(x @ W1)                             # all three hidden values are equal
y = h @ w2
dy = y - t                                      # gradient of 0.5*(y-t)^2
dh = dy * w2
dz = dh * (1 - h ** 2)
dW1 = np.outer(x, dz)                           # gradient for every first-layer weight
print("hidden values:", h.round(3).tolist())
print("gradient of W1 (columns = neurons):\n", dW1.round(3))
print("all three neurons receive the same gradient:", np.allclose(dW1[:, 0], dW1[:, 1]) and np.allclose(dW1[:, 1], dW1[:, 2]))`, "hidden values: [0.905, 0.905, 0.905]\ngradient of W1 (columns = neurons):\n [[0.032 0.032 0.032]\n [0.065 0.065 0.065]]\nall three neurons receive the same gradient: True"],
    ["h", "Rule 2: keep the signal size steady"],
    ["p", "A neuron adds up $n$ (fan-in) products. If each weight has variance $\\sigma^2$ and inputs have variance 1, the sum has variance $n\\sigma^2$. Stack layers and the variance gets multiplied by $n\\sigma^2$ each time: if $n\\sigma^2<1$ the signal **shrinks to zero**, if $>1$ it **explodes**. The fix is to choose $\\sigma^2\\approx 1/n$. The same argument applied to the backward pass shows why gradients vanish or explode (see [Vanishing & Exploding Gradients](dl.html#vanishing-gradients))."],
    ["table", ["Scheme", "Weight variance", "Designed for"], [
      ["Xavier / Glorot", "$\\dfrac{2}{n_{in}+n_{out}}$ (or $1/n_{in}$)", "tanh, sigmoid, linear layers"],
      ["He / Kaiming", "$\\dfrac{2}{n_{in}}$", "ReLU and its relatives (a ReLU zeroes half the signal, so you need twice the variance)"],
      ["Small normal (e.g. std 0.01)", "$10^{-4}$", "Only for shallow nets; shrinks the signal in deep ones"],
      ["Zero / constant", "0", "Biases only. Never weights"],
    ]],
    ["p", "Watch what happens to the activations of a 10-layer, 100-wide network under different choices. The number printed is the standard deviation of the activations at the output of layers 1, 5 and 10; ideal is roughly constant."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
width, depth = 100, 10
x0 = rng.normal(size=(256, width))

def activation_std(act, std_fn):
    a, out = x0, []
    for _ in range(depth):
        W = rng.normal(size=(width, width)) * std_fn(width)
        a = act(a @ W)
        out.append(a.std())
    return [round(float(out[i]), 4) for i in (0, 4, 9)]

relu = lambda z: np.maximum(0, z)
print("tanh, std 0.01       :", activation_std(np.tanh, lambda n: 0.01))
print("tanh, std 1.0        :", activation_std(np.tanh, lambda n: 1.0))
print("tanh, Xavier 1/sqrt n:", activation_std(np.tanh, lambda n: np.sqrt(1 / n)))
print("ReLU, Xavier         :", activation_std(relu, lambda n: np.sqrt(1 / n)))
print("ReLU, He sqrt(2/n)   :", activation_std(relu, lambda n: np.sqrt(2 / n)))`, "tanh, std 0.01       : [0.0989, 0.0, 0.0]\ntanh, std 1.0        : [0.9585, 0.9575, 0.9574]\ntanh, Xavier 1/sqrt n: [0.6259, 0.3218, 0.2197]\nReLU, Xavier         : [0.5762, 0.1136, 0.018]\nReLU, He sqrt(2/n)   : [0.8335, 0.8835, 1.4168]"],
    ["p", "Tiny weights make the signal vanish; large weights saturate the tanh (outputs stuck at ±1, where the slope is 0); Xavier keeps tanh healthy; for ReLU, only He initialization holds the scale steady."],
    ["h", "In PyTorch"],
    ["p", "`nn.Linear` already uses a sensible default (a uniform distribution scaled by $1/\\sqrt{n_{in}}$), so for simple models you do nothing. For ReLU networks or special cases, call the `nn.init` functions yourself:"],
    ["code", R`
import torch
from torch import nn

torch.manual_seed(0)
fan_in, fan_out = 256, 256
layer = nn.Linear(fan_in, fan_out)
print("default init std       :", round(layer.weight.std().item(), 4), "(theory 1/sqrt(3*fan_in) =", round((1 / (3 * fan_in)) ** 0.5, 4), ")")

nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
print("kaiming_normal_ std    :", round(layer.weight.std().item(), 4), "(theory sqrt(2/fan_in) =", round((2 / fan_in) ** 0.5, 4), ")")

nn.init.xavier_uniform_(layer.weight)
print("xavier_uniform_ std    :", round(layer.weight.std().item(), 4), "(theory sqrt(2/(in+out)) =", round((2 / (fan_in + fan_out)) ** 0.5, 4), ")")

nn.init.zeros_(layer.bias)                    # biases: zeros are fine
print("bias all zero:", bool((layer.bias == 0).all()))

def init_weights(m):                          # apply to a whole model with model.apply(...)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
model.apply(init_weights)
print("applied to", sum(isinstance(m, nn.Linear) for m in model.modules()), "linear layers")`, "default init std       : 0.036 (theory 1/sqrt(3*fan_in) = 0.0361 )\nkaiming_normal_ std    : 0.0888 (theory sqrt(2/fan_in) = 0.0884 )\nxavier_uniform_ std    : 0.0626 (theory sqrt(2/(in+out)) = 0.0625 )\nbias all zero: True\napplied to 2 linear layers", "PyTorch", "local"],
    ["tip", "Rule of thumb: ReLU / GELU hidden layers → He (Kaiming); tanh / sigmoid → Xavier. Biases → 0. And [normalization layers](dl.html#normalization) make the network far less sensitive to the choice. For transformers a small normal (std about 0.02) with scaled residual layers is the usual recipe."],
    ["note", "Fine-tuning a [pretrained model](dl.html#transfer-learning) skips initialization: it starts from weights that already work. Only newly added layers (for example a fresh classification head) are initialised from scratch."],
  ],
},
{
  id: "vanishing-gradients", group: "Training", title: "Vanishing & Exploding Gradients",
  sub: "In a deep network the gradient is a long product of factors. Product of numbers below 1 → zero. Above 1 → infinity.",
  tags: ["gradients"],
  blocks: [
    ["p", "[Backpropagation](dl.html#backprop) multiplies one local factor per layer (the layer's weights, and the activation's slope). The gradient that arrives at layer 1 of a 50-layer network is a product of about 50 such factors. A product of many numbers a bit smaller than 1 collapses toward 0 (**vanishing gradient**); a product of numbers bigger than 1 grows without bound (**exploding gradient**). We saw the arithmetic in the [chain rule](math.html#chain-rule) section: $0.25^{20}\\approx10^{-12}$."],
    ["p", "A vanishing gradient means the early layers get almost no learning signal: the loss stops improving and those layers stay near their random start. An exploding gradient makes weights jump wildly, the loss shoots up, and finally becomes `NaN`. The picture below measures the gradient size at each layer of a real random network."],
    ["viz", "dl1_vanish"],
    ["p", "Things to try: with the default weight scale 1.4, ReLU holds steady, while sigmoid collapses by many orders of magnitude within 10-15 layers. Slide the weight scale down to 0.5 and ReLU vanishes too; slide it up to 3 and everything explodes. **Both depth and weight scale matter.**"],
    ["h", "Measure it in NumPy"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
width, depth = 50, 20
sigmoid = lambda z: 1 / (1 + np.exp(-z))

def grad_norms(act, dact, gain):
    Ws = [rng.normal(size=(width, width)) * gain / np.sqrt(width) for _ in range(depth)]
    a, zs, As = rng.normal(size=width), [], []
    As.append(a)
    for W in Ws:                                   # forward
        z = W @ As[-1]; zs.append(z); As.append(act(z))
    g = np.ones(width) / np.sqrt(width)            # gradient arriving at the last layer
    norms = [0.0] * depth
    for l in reversed(range(depth)):               # backward
        norms[l] = np.linalg.norm(g)
        g = Ws[l].T @ (g * dact(zs[l], As[l + 1]))
    return norms

setups = {
    "sigmoid, gain 1.0": (sigmoid, lambda z, a: a * (1 - a), 1.0),
    "tanh,    gain 1.0": (np.tanh, lambda z, a: 1 - a ** 2, 1.0),
    "ReLU,    gain 1.0": (lambda z: np.maximum(0, z), lambda z, a: (z > 0) * 1.0, 1.0),
    "ReLU,    gain 1.41 (He)": (lambda z: np.maximum(0, z), lambda z, a: (z > 0) * 1.0, 1.41),
    "ReLU,    gain 3.0": (lambda z: np.maximum(0, z), lambda z, a: (z > 0) * 1.0, 3.0),
}
print("gradient size at layer 1, 5, 10, 20 (relative to the last layer = 1):")
for name, (f, df, gain) in setups.items():
    n = grad_norms(f, df, gain)
    print(f"{name:24s}", "  ".join(f"{n[i - 1] / n[-1]:.1e}" for i in (1, 5, 10, 20)))`, "gradient size at layer 1, 5, 10, 20 (relative to the last layer = 1):\nsigmoid, gain 1.0        7.3e-13  2.8e-10  4.1e-07  1.0e+00\ntanh,    gain 1.0        1.6e-01  3.0e-01  4.8e-01  1.0e+00\nReLU,    gain 1.0        2.5e-03  8.5e-03  4.2e-02  1.0e+00\nReLU,    gain 1.41 (He)  3.5e+00  2.1e+00  2.3e+00  1.0e+00\nReLU,    gain 3.0        2.1e+06  1.2e+05  2.9e+03  1.0e+00"],
    ["h", "Cures"],
    ["table", ["Remedy", "How it helps"], [
      ["ReLU-family activations", "Slope is 1 for positive inputs, so there is no repeated shrinking ([Activations](dl.html#activations))."],
      ["Proper [initialization](dl.html#initialization) (He / Xavier)", "Starts each layer's factor near 1 (variance preserved)."],
      ["[BatchNorm / LayerNorm](dl.html#normalization)", "Re-centres and re-scales activations at every layer, keeping gradients in range."],
      ["Residual (skip) connections", "Add a path where the gradient flows straight through: $y = x + F(x)$ has derivative $1 + F'$. This is what makes 100+ layer [ResNets](dl.html#pooling-resnet) and transformers trainable."],
      ["Gated cells (LSTM, GRU)", "For sequences: additive paths through time ([LSTM](dl.html#lstm))."],
      ["Gradient clipping", "Cap the gradient norm: a cure for **exploding** gradients (below)."],
      ["Smaller learning rate, warmup", "Soften the damage while the network is fragile ([Schedules](dl.html#lr-schedules))."],
    ]],
    ["h", "Gradient clipping"],
    ["p", "If the total gradient norm exceeds a threshold, scale the whole gradient down so its norm equals the threshold. The direction is unchanged; only a freak huge step is tamed. It is standard in RNN and transformer training."],
    ["code", R`
import numpy as np

def clip_by_global_norm(grads, max_norm):
    total = np.sqrt(sum((g ** 2).sum() for g in grads))          # norm of all gradients together
    scale = min(1.0, max_norm / (total + 1e-6))
    return [g * scale for g in grads], total

grads = [np.array([30.0, 40.0]), np.array([[0.0, 0.0]])]          # a huge gradient (norm 50)
clipped, before = clip_by_global_norm(grads, max_norm=1.0)
after = np.sqrt(sum((g ** 2).sum() for g in clipped))
print("norm before:", round(float(before), 3), " after:", round(float(after), 3))
print("direction unchanged:", clipped[0].round(3).tolist())`, "norm before: 50.0  after: 1.0\ndirection unchanged: [0.6, 0.8]"],
    ["h", "In PyTorch"],
    ["code", R`
import torch
from torch import nn

torch.manual_seed(0)

def make(act):
    layers = []
    for _ in range(15):
        layers += [nn.Linear(50, 50), act()]
    return nn.Sequential(*layers, nn.Linear(50, 1))

x = torch.randn(64, 50)
for name, act in [("sigmoid", nn.Sigmoid), ("ReLU", nn.ReLU)]:
    model = make(act)                                       # PyTorch's default initialization
    model(x).pow(2).mean().backward()
    first = model[0].weight.grad.norm().item()
    last = model[-1].weight.grad.norm().item()
    print(f"{name:8s} grad norm  first layer {first:.2e}   last layer {last:.2e}   ratio {first / last:.1e}")

# clipping: returns the total norm BEFORE clipping and rescales the gradients in place
total = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
print(f"clip_grad_norm_(max_norm=0.001): norm before {total.item():.5f}, after {after.item():.5f}")`, "sigmoid  grad norm  first layer 1.41e-13   last layer 3.65e+00   ratio 3.9e-14\nReLU     grad norm  first layer 1.69e-09   last layer 5.45e-04   ratio 3.1e-06\nclip_grad_norm_(max_norm=0.001): norm before 0.00148, after 0.00100", "PyTorch", "local"],
    ["tip", "How to spot the problem: log the gradient norm per layer (or just the total). Vanishing: early layers' norms are orders of magnitude smaller than late layers'. Exploding: norms spike, or the loss turns `NaN`. See [Debugging Neural Network Training](dl.html#debugging-training)."],
  ],
},
{
  id: "normalization", group: "Training", title: "BatchNorm & LayerNorm",
  sub: "Re-centre and re-scale the numbers flowing through the network at every layer, so each layer sees well-behaved inputs.",
  tags: ["normalization"],
  blocks: [
    ["p", "Remember that features should be [standardised](ml.html#feature-engineering) (mean 0, standard deviation 1) before training a model. A deep network has the same problem **inside**: after each layer the numbers can drift to a new scale, so the next layer keeps chasing a moving target. **Normalization layers** standardise the activations at every layer, which makes training faster, allows bigger learning rates, and reduces the sensitivity to [initialization](dl.html#initialization)."],
    ["h", "Batch normalization"],
    ["p", "**BatchNorm** (Ioffe & Szegedy, 2015) normalises each **feature** using the mean and variance computed **across the mini-batch**, then lets the network undo it if it wants, using two learned parameters per feature, a scale $\\gamma$ and a shift $\\beta$:"],
    ["math", R`\mu_B=\frac1m\sum_{i=1}^m x_i \qquad \sigma_B^2=\frac1m\sum_{i=1}^m (x_i-\mu_B)^2 \qquad \hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}} \qquad y_i=\gamma\,\hat x_i+\beta`],
    ["p", "A subtle point is **training vs inference**. In training, the statistics come from the current batch. At inference you may have a single example, so the layer instead uses **running averages** of the mean and variance collected during training. That is why you must call `model.eval()` before predicting."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
# 8 examples, 3 features with very different scales
x = rng.normal(loc=[5.0, -2.0, 0.0], scale=[10.0, 0.1, 1.0], size=(8, 3))

class BatchNorm:
    def __init__(self, n, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(n), np.zeros(n)        # learned
        self.run_mean, self.run_var = np.zeros(n), np.ones(n)  # running statistics for inference
        self.momentum, self.eps = momentum, eps

    def __call__(self, x, training):
        if training:
            mean, var = x.mean(axis=0), x.var(axis=0)          # statistics over the batch
            m = len(x)
            self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * mean
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * var * m / (m - 1)
        else:
            mean, var = self.run_mean, self.run_var            # frozen statistics
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

bn = BatchNorm(3)
y = bn(x, training=True)
print("input mean :", x.mean(axis=0).round(2).tolist(), " std:", x.std(axis=0).round(2).tolist())
print("output mean:", y.mean(axis=0).round(3).tolist(), " std:", y.std(axis=0).round(3).tolist())
print("running mean after 1 step:", bn.run_mean.round(3).tolist())
for _ in range(60):                                            # many batches -> running stats converge
    bn(rng.normal(loc=[5.0, -2.0, 0.0], scale=[10.0, 0.1, 1.0], size=(8, 3)), training=True)
print("running mean after 61 steps:", bn.run_mean.round(2).tolist(), " (true: [5, -2, 0])")
print("eval-mode output for one example:", bn(x[:1], training=False).round(3).tolist())`, "input mean : [3.74, -2.01, -0.12]  std: [11.82, 0.07, 0.58]\noutput mean: [0.0, 0.0, 0.0]  std: [1.0, 0.999, 1.0]\nrunning mean after 1 step: [0.374, -0.201, -0.012]\nrunning mean after 61 steps: [4.41, -1.97, -0.05]  (true: [5, -2, 0])\neval-mode output for one example: [[0.189, -0.471, 0.599]]"],
    ["h", "Layer normalization"],
    ["p", "**LayerNorm** normalises across the **features of each single example** instead of across the batch. It does not depend on the other examples, has no running statistics, and behaves the same in training and inference, so it works for any batch size (even 1) and for sequences of varying length. That is why [Transformers](transformers.html#transformer-block) and LLMs use it (many modern LLMs use the cheaper **RMSNorm**, which skips the mean-subtraction and divides by the root-mean-square only)."],
    ["math", R`\hat x_j=\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}},\quad \mu=\frac1d\sum_j x_j,\ \ \sigma^2=\frac1d\sum_j (x_j-\mu)^2 \qquad y_j=\gamma_j\hat x_j+\beta_j`],
    ["table", ["", "BatchNorm", "LayerNorm"], [
      ["Normalises over", "the batch (for each feature)", "the features (for each example)"],
      ["Depends on other examples?", "Yes: batch statistics", "No"],
      ["Training vs inference", "Different: batch stats vs running stats", "Identical"],
      ["Works with batch size 1?", "Poorly", "Yes"],
      ["Typical home", "CNNs (images)", "Transformers, RNNs, LLMs"],
    ]],
    ["code", R`
import numpy as np

x = np.array([[1.0, 2.0, 3.0, 4.0],
              [10.0, 20.0, 30.0, 40.0]])        # 2 examples, 4 features

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)       # one mean per example
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

y = layer_norm(x)
print(y.round(3))
print("per-example mean:", y.mean(axis=-1).round(3).tolist(), " std:", y.std(axis=-1).round(3).tolist())

# the effect on a deep net: ReLU layers with small weights, without and with normalization
rng = np.random.default_rng(0)
x0 = rng.normal(size=(64, 100))
def run(norm):
    a, stds = x0, []
    for _ in range(10):
        a = a @ (rng.normal(size=(100, 100)) * 0.05)       # too-small weights: signal would shrink
        if norm:
            a = layer_norm(a)
        a = np.maximum(0, a)
        stds.append(round(float(a.std()), 4))
    return [stds[0], stds[4], stds[9]]
print("activation std at layers 1, 5, 10 without norm:", run(False))
print("activation std at layers 1, 5, 10 with norm   :", run(True))`, "[[-1.342 -0.447  0.447  1.342]\n [-1.342 -0.447  0.447  1.342]]\nper-example mean: [0.0, 0.0]  std: [1.0, 1.0]\nactivation std at layers 1, 5, 10 without norm: [0.2845, 0.005, 0.0]\nactivation std at layers 1, 5, 10 with norm   : [0.5813, 0.5665, 0.5893]"],
    ["h", "In PyTorch"],
    ["code", R`
import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(8, 3) * torch.tensor([10.0, 0.1, 1.0]) + torch.tensor([5.0, -2.0, 0.0])
r = lambda t: [round(v, 3) for v in t.tolist()]          # tidy printing

bn = nn.BatchNorm1d(3)
y = bn(x)                                               # training mode: uses batch statistics
print("train output mean:", r(y.mean(0).detach()), " std:", r(y.std(0, unbiased=False).detach()))
print("running mean after one step:", r(bn.running_mean))

bn.eval()                                               # inference: uses the running statistics instead
print("eval output mean :", r(bn(x).mean(0).detach()), "(not 0: the running stats are still catching up)")

ln = nn.LayerNorm(3)
z = ln(x)
manual = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
print("LayerNorm == manual:", torch.allclose(z, manual, atol=1e-5))`, "train output mean: [-0.0, -0.0, 0.0]  std: [1.0, 1.0, 1.0]\nrunning mean after one step: [0.452, -0.2, 0.007]\neval output mean : [1.342, -1.895, 0.066] (not 0: the running stats are still catching up)\nLayerNorm == manual: True", "PyTorch", "local"],
    ["warn", "BatchNorm quirks: it needs a reasonably large batch (8+; ideally 32+) for stable statistics; training and inference behave differently, so always call `model.eval()` / `model.train()`; and when fine-tuning with a tiny batch it is common to keep BatchNorm layers in eval mode."],
    ["tip", "Where does the layer go? Usually after the linear (or convolution) and before the activation in classic CNNs; in transformers the 'pre-norm' layout puts LayerNorm **before** the attention and feed-forward blocks. Practice: [Batch Normalization Forward Pass](practice.html#p/batchnorm-forward)."],
  ],
},
{
  id: "dl-regularization", group: "Training", title: "Dropout, Weight Decay & Early Stopping",
  sub: "Three simple ways to stop a network with millions of parameters from memorising its training data.",
  tags: ["regularization"],
  blocks: [
    ["p", "Neural networks have so many parameters that they can memorise the training set, noise included. This is [overfitting](ml.html#bias-variance): the training loss keeps falling while the validation loss starts to rise. **Regularization** is anything that makes memorising harder. For deep nets there are three workhorses (plus [data augmentation](dl.html#cnn) and simply more data)."],
    ["h", "1. Dropout"],
    ["p", "During training, randomly **switch off** each neuron with probability $p$ (set its output to 0) on every step. The network can no longer rely on any single neuron or on fixed co-operations between neurons, so it has to spread what it knows around. It is like training a huge number of different thinned-out networks that share weights, and averaging them at test time."],
    ["p", "The standard trick is **inverted dropout**: scale the survivors up by $1/(1-p)$ during training, so the expected value of each activation is unchanged, and at test time simply do nothing."],
    ["viz", "dl1_dropout"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)

def dropout(x, p, training):
    if not training or p == 0:
        return x                                      # inference: identity, no scaling needed
    mask = rng.random(x.shape) >= p                   # keep each unit with probability 1 - p
    return x * mask / (1 - p)                         # scale survivors so the expectation is unchanged

x = np.full(100000, 2.0)
y = dropout(x, p=0.5, training=True)
print("fraction dropped        :", round(float(np.mean(y == 0)), 3))
print("mean before / after     :", x.mean(), round(float(y.mean()), 3))
print("values that survive     :", sorted(set(y.tolist())))
print("inference mode returns x:", np.array_equal(dropout(x, 0.5, training=False), x))`, "fraction dropped        : 0.501\nmean before / after     : 2.0 1.996\nvalues that survive     : [0.0, 4.0]\ninference mode returns x: True"],
    ["h", "2. Weight decay (L2 regularization)"],
    ["p", "Add a penalty for large weights, $L_{total}=L+\\tfrac{\\lambda}{2}\\lVert\\mathbf w\\rVert^2$, which pulls every weight a little toward zero on each step. Small weights give smoother functions that are less able to fit noise. It is the same idea as [ridge regression](ml.html#regularization). Its gradient is simply $\\lambda w$, so the SGD update becomes:"],
    ["math", R`w \leftarrow w - \eta\,(g + \lambda w) \;=\; (1-\eta\lambda)\,w - \eta\, g`, "each step first shrinks the weight by the factor (1 − ηλ), then applies the usual gradient step: hence 'decay'"],
    ["code", R`
import numpy as np

# two equivalent views of weight decay with plain SGD
w_a = w_b = 3.0
lr, lam = 0.1, 0.5
g = 0.2                                              # pretend gradient of the data loss
for _ in range(5):
    w_a = w_a - lr * (g + lam * w_a)                 # L2 penalty gradient added to g
    w_b = (1 - lr * lam) * w_b - lr * g              # shrink, then step
print("L2 view     :", round(w_a, 6))
print("decay view  :", round(w_b, 6), "-> identical:", abs(w_a - w_b) < 1e-12)`, "L2 view     : 2.230855\ndecay view  : 2.230855 -> identical: True"],
    ["warn", "With adaptive optimizers like Adam, adding the L2 term to the loss is **not** the same as weight decay, because Adam rescales the gradient by its running size (including the penalty part). **AdamW** applies the shrinking step directly to the weights, separate from the gradient. Use `AdamW` with weight decay around 0.01-0.1; do not apply decay to biases and normalization parameters."],
    ["h", "3. Early stopping"],
    ["p", "Watch the validation loss. When it stops improving, stop training and go back to the weights from the best epoch. It costs nothing, and it works because networks usually fit the general pattern first and the noise later. The **patience** setting says how many epochs without improvement to tolerate, since the validation loss is itself noisy. Here, deliberately, a big network on only 60 training points:"],
    ["code", R`
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

X, y = make_moons(n_samples=400, noise=0.35, random_state=1)
X_tr, y_tr, X_va, y_va = X[:60], y[:60], X[60:], y[60:]            # tiny training set

net = MLPClassifier(hidden_layer_sizes=(64, 64), learning_rate_init=0.01, random_state=0)
best, best_epoch, wait, patience = np.inf, 0, 0, 25
for epoch in range(1, 301):
    net.partial_fit(X_tr, y_tr, classes=[0, 1])                     # one epoch of training
    val = log_loss(y_va, net.predict_proba(X_va))
    if val < best - 1e-4:                                           # improved: remember these weights
        best, best_epoch, wait = val, epoch, 0
        best_coefs = [w.copy() for w in net.coefs_]
        best_icpts = [b.copy() for b in net.intercepts_]
    else:
        wait += 1
        if wait >= patience:                                        # no improvement for 'patience' epochs
            break

print(f"stopped at epoch {epoch}; best validation loss {best:.3f} was at epoch {best_epoch}")
print(f"validation loss when stopped: {val:.3f}   accuracy {net.score(X_va, y_va):.3f}")
net.coefs_, net.intercepts_ = best_coefs, best_icpts                # restore the best weights
print(f"after restoring the best weights: loss {log_loss(y_va, net.predict_proba(X_va)):.3f}   accuracy {net.score(X_va, y_va):.3f}")`, "stopped at epoch 66; best validation loss 0.334 was at epoch 41\nvalidation loss when stopped: 0.422   accuracy 0.838\nafter restoring the best weights: loss 0.334   accuracy 0.859"],
    ["h", "In PyTorch"],
    ["code", R`
import torch
from torch import nn

torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
x = torch.ones(1, 10)

drop.train()
print("train mode:", drop(x).tolist()[0])           # about half zeros, the rest scaled to 2.0
drop.eval()
print("eval mode :", drop(x).tolist()[0])           # unchanged

model = nn.Linear(10, 1)
decay, no_decay = [model.weight], [model.bias]      # no decay on the bias
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                         {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
print("AdamW groups:", [g["weight_decay"] for g in opt.param_groups])`, "train mode: [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0]\neval mode : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\nAdamW groups: [0.05, 0.0]", "PyTorch", "local"],
    ["table", ["Method", "Strength", "Typical setting", "Watch out"], [
      ["Dropout", "Strong for big fully connected layers", "p = 0.1-0.5", "Slows training; less used in convolutional and modern transformer layers (small p)"],
      ["Weight decay", "Mild but nearly free", "0.01-0.1 with AdamW", "Do not decay biases / norm parameters"],
      ["Early stopping", "Free, and a safety net", "patience 5-20 epochs", "Needs a validation set; restore the best weights"],
      ["More data / augmentation", "Often the best of all", "—", "Data may be expensive"],
      ["Smaller model", "Cuts capacity directly", "—", "May underfit"],
    ]],
    ["danger", "Leaving dropout on at evaluation time (`model.train()` instead of `model.eval()`): predictions become random and noisy. Also, stacking too much regularization (heavy dropout and big weight decay together) makes the model **underfit**; add one thing at a time and watch the train and validation curves."],
    ["tip", "Order of attack for an overfitting network: get more data or augment, add weight decay, add dropout, shrink the model, and use early stopping throughout. Practice: [Inverted Dropout](practice.html#p/inverted-dropout)."],
  ],
},
);
