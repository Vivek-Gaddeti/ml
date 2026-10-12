/* Deep Learning, part 2: architectures (CNNs, RNNs, LSTMs, seq2seq, embeddings, generative models) and in practice (PyTorch, transfer learning, debugging)
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/dl-2.js */
DL.push({
  id: "cnn", group: "Architectures", title: "Convolutional Neural Networks",
  sub: "Slide a small shared filter over an image to find patterns wherever they appear, using a tiny fraction of the weights of a dense layer.",
  tags: ["images", "convolution"],
  blocks: [
    ["p", "Take a modest photo, 224 × 224 pixels with 3 colour channels. That is 150,528 numbers. Feed it into an ordinary [MLP](dl.html#mlp) with just 1,000 hidden units and the first layer alone has 150 million weights. Worse, a dense layer treats pixel (0, 0) and pixel (0, 1) as two unrelated inputs, so it has no idea that neighbouring pixels belong together, and a cat in the top-left corner looks to it like a completely different input from a cat in the bottom-right."],
    ["p", "A **convolutional neural network (CNN)** fixes both problems with two ideas:"],
    ["list", [
      "**Local connectivity.** Each output looks at a small neighbourhood (say 3 × 3 pixels), not the whole image. Edges, corners and textures are local things.",
      "**Weight sharing.** The *same* small grid of weights, called a **kernel** or **filter**, is slid over every position. A pattern detector that works in one corner works everywhere, and the layer needs only $3\\times3 = 9$ weights per filter instead of millions.",
    ]],
    ["p", "Think of the kernel as a little stamp with a pattern printed on it. You press the stamp at every position of the image; wherever the picture under the stamp looks like the pattern, you write down a big number. The result, a map of \"where was this pattern found\", is called a **feature map**."],
    ["viz", "dl2_conv"],
    ["diagram", "dl2_cnn", [], "A small CNN: convolution + ReLU and pooling repeated, then a dense classifier. Each block of overlapping squares is a stack of feature maps (one per filter)."],
    ["h", "The operation"],
    ["math", R`\text{out}[i,j] \;=\; b + \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K[u,v]\; x[\,i\cdot s+u,\; j\cdot s+v\,]`, "K: the k×k kernel · s: stride · b: a bias added to every output of the filter"],
    ["p", "Multiply the patch by the kernel element by element, then add up: one number per position. Strictly this is called *cross-correlation* (a true mathematical convolution flips the kernel first), but since the kernel is learned, the flip makes no difference and every deep learning library just calls it convolution."],
    ["p", "Two knobs control the output size:"],
    ["list", [
      "**Stride** $s$: how many pixels the kernel jumps each time. Stride 2 skips every other position and halves the output.",
      "**Padding** $p$: zeros added around the border. Without it the output shrinks and the edge pixels are used less often than the middle ones. With $p=(k-1)/2$ (e.g. 1 for a 3 × 3 kernel) and stride 1, the output has the **same** size as the input (\"same\" padding).",
    ]],
    ["math", R`n_{\text{out}} \;=\; \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1`, "n: input size · k: kernel size · p: padding · s: stride (the same formula applies to height and width)"],
    ["h", "Convolution from scratch"],
    ["p", "Here is the whole thing in a few lines of NumPy, on the same 6 × 6 \"handwritten 2\" as in the picture. The vertical-edge kernel gives a positive number where the image gets brighter to the right, and a negative number where it gets darker:"],
    ["code", R`
import numpy as np

def conv2d(x, k, stride=1, pad=0):
    """x: (H, W) image, k: (kh, kw) kernel -> feature map."""
    x = np.pad(x, pad)                              # zeros around the border
    kh, kw = k.shape
    oh = (x.shape[0] - kh) // stride + 1
    ow = (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = x[i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[i, j] = (patch * k).sum()           # multiply, then add
    return out

img = np.array([[1, 1, 1, 1, 1, 1],
                [1, 9, 9, 9, 9, 1],
                [1, 1, 1, 9, 9, 1],
                [1, 1, 9, 9, 1, 1],
                [1, 9, 9, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]], dtype=float)
edge = np.array([[-1, 0, 1]] * 3, dtype=float)      # vertical-edge detector

print(conv2d(img, edge))
print("pad=1   ->", conv2d(img, edge, pad=1).shape)
print("stride=2->", conv2d(img, edge, stride=2).shape)
print("pad=1, stride=2 ->", conv2d(img, edge, stride=2, pad=1).shape)`, "[[  8.   8.   8. -16.]\n [ 16.  16.   0. -24.]\n [ 16.   8.  -8. -16.]\n [ 16.   0. -16.  -8.]]\npad=1   -> (6, 6)\nstride=2-> (2, 2)\npad=1, stride=2 -> (3, 3)"],
    ["h", "Sizes and parameter counts"],
    ["p", "Real layers have many **channels**. A colour image has 3 input channels; a layer with $C_{out}$ filters produces $C_{out}$ feature maps. Each filter has a $k\\times k$ slice for *every* input channel, so one conv layer has"],
    ["math", R`\#\text{params} \;=\; C_{out}\,(C_{in}\,k^2 + 1)`, "the +1 is each filter's bias"],
    ["p", "Compare that with a dense layer producing the same number of numbers:"],
    ["code", R`
def out_size(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1

for n, k, s, p in [(32, 3, 1, 0), (32, 3, 1, 1), (32, 3, 2, 1), (224, 7, 2, 3), (28, 5, 1, 0)]:
    print(f"input {n:3d}  kernel {k}  stride {s}  pad {p}  ->  output {out_size(n, k, s, p)}")

# one conv layer vs one dense layer on a 224x224 RGB image, both producing 64 feature maps
conv_params = 64 * (3 * 3 * 3 + 1)                    # 64 filters of 3x3x3, plus biases
dense_in = 224 * 224 * 3
dense_params = dense_in * (224 * 224 * 64) + 224 * 224 * 64   # connect every input to every output number
print("conv 3x3, 3 -> 64 channels :", conv_params, "parameters")
print("dense layer, same output   :", f"{dense_params:,}", "parameters")`, "input  32  kernel 3  stride 1  pad 0  ->  output 30\ninput  32  kernel 3  stride 1  pad 1  ->  output 32\ninput  32  kernel 3  stride 2  pad 1  ->  output 16\ninput 224  kernel 7  stride 2  pad 3  ->  output 112\ninput  28  kernel 5  stride 1  pad 0  ->  output 24\nconv 3x3, 3 -> 64 channels : 1792 parameters\ndense layer, same output   : 483,388,358,656 parameters"],
    ["h", "Seeing more with depth: the receptive field"],
    ["p", "One 3 × 3 filter only sees a 3 × 3 patch. But the next layer looks at 3 × 3 *units* of the first, and each of those saw 3 pixels, so together they cover 5 × 5 pixels. The area of the input that can affect one unit is its **receptive field**, and it grows with every layer. That is how a stack of tiny filters can eventually respond to an entire face: early layers see edges, middle layers combine them into corners and textures, late layers see parts and whole objects."],
    ["viz", "dl2_receptive"],
    ["tip", "Stride 2 (or [pooling](dl.html#pooling-resnet)) makes the receptive field grow much faster, because every later unit then covers twice the ground per step. This is why CNNs shrink the picture as they get deeper while adding more channels."],
    ["h", "Fast convolution: im2col"],
    ["p", "Python double loops are far too slow. Real libraries use a trick called **im2col**: copy every patch into one column of a big matrix, so the whole convolution becomes a single matrix multiplication, which GPUs do extremely well. Below also handles several channels and several filters at once. We check it against a slow, obviously-correct version."],
    ["code", R`
import numpy as np

def conv_naive(x, W, b, stride=1):
    """x: (C, H, W)  W: (F, C, kh, kw)  b: (F,)  ->  (F, oh, ow)"""
    F, C, kh, kw = W.shape
    oh = (x.shape[1] - kh) // stride + 1
    ow = (x.shape[2] - kw) // stride + 1
    out = np.zeros((F, oh, ow))
    for f in range(F):
        for i in range(oh):
            for j in range(ow):
                patch = x[:, i * stride : i * stride + kh, j * stride : j * stride + kw]
                out[f, i, j] = (patch * W[f]).sum() + b[f]
    return out

def conv_im2col(x, W, b, stride=1):
    F, C, kh, kw = W.shape
    oh = (x.shape[1] - kh) // stride + 1
    ow = (x.shape[2] - kw) // stride + 1
    cols = np.empty((C * kh * kw, oh * ow))           # one column per output position
    n = 0
    for i in range(oh):
        for j in range(ow):
            cols[:, n] = x[:, i * stride : i * stride + kh, j * stride : j * stride + kw].ravel()
            n += 1
    out = W.reshape(F, -1) @ cols + b[:, None]        # ONE matrix multiply does all the filters
    return out.reshape(F, oh, ow)

rng = np.random.default_rng(0)
x = rng.normal(size=(3, 8, 8))                        # 3 channels
W = rng.normal(size=(4, 3, 3, 3))                     # 4 filters of 3x3x3
b = rng.normal(size=4)
a, c = conv_naive(x, W, b, stride=2), conv_im2col(x, W, b, stride=2)
print("output shape:", a.shape, " same result:", np.allclose(a, c))
print("im2col matrix:", (3 * 3 * 3, a.shape[1] * a.shape[2]), "x filter matrix:", W.reshape(4, -1).shape)`, "output shape: (4, 3, 3)  same result: True\nim2col matrix: (27, 9) x filter matrix: (4, 27)"],
    ["h", "The same in PyTorch"],
    ["p", "`nn.Conv2d` does exactly this (with a much faster implementation). Images are laid out as **(batch, channels, height, width)**. Here we take its random weights, run our own NumPy version with them, and compare:"],
    ["code", R`
import numpy as np, torch, torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

torch.manual_seed(0)
conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1)
x = np.random.default_rng(0).normal(size=(3, 8, 8)).astype(np.float32)

def mine(x, W, b, stride, pad):
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (W.shape[1], W.shape[2], W.shape[3]), axis=(0, 1, 2))[0, ::stride, ::stride]
    return np.einsum("ijcuv,fcuv->fij", win, W) + b[:, None, None]

W, b = conv.weight.detach().numpy(), conv.bias.detach().numpy()
ours = mine(x, W, b, stride=2, pad=1)
theirs = conv(torch.from_numpy(x)[None])[0].detach().numpy()
print("PyTorch output shape:", tuple(theirs.shape), " matches our NumPy:", np.allclose(ours, theirs, atol=1e-5))
print("parameters:", sum(p.numel() for p in conv.parameters()), " (= 4 * (3*3*3 + 1))")`, "PyTorch output shape: (4, 4, 4)  matches our NumPy: True\nparameters: 112  (= 4 * (3*3*3 + 1))", "PyTorch", "local"],
    ["table", ["Term", "Meaning", "Typical values"], [
      ["kernel size $k$", "Height and width of the filter", "3 (the default choice), sometimes 5 or 7 in the first layer"],
      ["stride $s$", "Jump between positions", "1 inside blocks, 2 to downsample"],
      ["padding $p$", "Zeros around the border", "$(k-1)/2$ to keep the size"],
      ["channels", "Feature maps in and out", "3 → 32 → 64 → 128 → 256 … (more as the image shrinks)"],
      ["1 × 1 convolution", "Mixes channels at each pixel, no spatial look", "Cheaply changes the number of channels"],
    ]],
    ["warn", "Output-size mistakes are the number one CNN bug. Always compute $\\lfloor (n+2p-k)/s\\rfloor+1$ by hand for the first few layers, and print `x.shape` after each layer while building a model. A wrong size usually shows up as a shape error at the first dense layer."],
    ["danger", "Putting the channels last. PyTorch expects `(N, C, H, W)`, while image libraries and TensorFlow usually give `(N, H, W, C)`. If your 28 × 28 greyscale batch has shape `(64, 28, 28)`, add the channel with `x.unsqueeze(1)` first."],
    ["tip", "Practice: [Naive 2-D Convolution with Stride and Padding](practice.html#p/conv2d-naive) and [Convolution as a Matrix Multiply (im2col)](practice.html#p/conv-im2col). Next: [pooling and the classic CNNs](dl.html#pooling-resnet)."],
  ],
},
{
  id: "pooling-resnet", group: "Architectures", title: "Pooling, Classic CNNs & ResNets",
  sub: "How CNNs shrink the picture, how the famous ones were built, and the skip connection that made very deep networks trainable.",
  tags: ["images", "resnet"],
  blocks: [
    ["p", "A [convolution layer](dl.html#cnn) finds patterns but keeps the image size. To build up from edges to whole objects, a CNN needs to **shrink** the picture as it goes deeper, so that later filters see more of the scene and the computation stays affordable. The classic way is **pooling**."],
    ["viz", "dl2_pool"],
    ["h", "Pooling"],
    ["p", "Pooling summarises each small window by one number and moves on. It has **no weights**. The most common version is **max pooling** with a 2 × 2 window and stride 2, which halves the height and width and keeps only the strongest response in each window. That also gives a little **translation tolerance**: if a feature moves by a pixel inside its window, the output does not change."],
    ["list", [
      "**Max pooling**: \"was this feature detected anywhere in the window?\" Used between conv blocks.",
      "**Average pooling**: the mean of the window. Smoother, keeps more of the overall level.",
      "**Global average pooling**: average each whole feature map to a single number, so $H\\times W\\times C$ becomes a vector of $C$ numbers. Modern CNNs use it right before the classifier, which removes the huge dense layers older networks needed.",
    ]],
    ["code", R`
import numpy as np

def pool2d(x, size=2, mode="max"):
    """x: (H, W). Non-overlapping windows (stride = size)."""
    H, W = x.shape
    x = x[: H // size * size, : W // size * size]                  # drop leftover rows/cols
    win = x.reshape(H // size, size, W // size, size)            # (rows, in-window row, cols, in-window col)
    return win.max(axis=(1, 3)) if mode == "max" else win.mean(axis=(1, 3))

fmap = np.array([[1, 3, 2, 1, 0, 2],
                 [4, 6, 5, 2, 1, 1],
                 [2, 1, 7, 8, 3, 0],
                 [0, 2, 9, 4, 2, 5],
                 [1, 0, 3, 3, 6, 7],
                 [5, 2, 1, 2, 8, 4]], dtype=float)
print(pool2d(fmap, 2, "max"))
print(pool2d(fmap, 2, "mean").round(2))
nudged = fmap.copy()
nudged[0, 1], nudged[1, 1] = nudged[1, 1], nudged[0, 1]           # move the strongest value (6) one pixel up, inside its window
print("feature map changed:", not np.array_equal(fmap, nudged), "  max-pool output unchanged:", np.array_equal(pool2d(fmap), pool2d(nudged)))
gap = fmap.mean()                                                 # global average pooling of one channel
print("global average pooling ->", round(float(gap), 3))`, "[[6. 5. 2.]\n [2. 9. 5.]\n [5. 3. 8.]]\n[[3.5  2.5  1.  ]\n [1.25 7.   2.5 ]\n [2.   2.25 6.25]]\nfeature map changed: True   max-pool output unchanged: True\nglobal average pooling -> 3.139"],
    ["note", "Many modern networks skip pooling layers and downsample with a **stride-2 convolution** instead, letting the network learn how to shrink. Both work; the stride-2 version has a few extra weights."],
    ["h", "A short history of CNNs"],
    ["table", ["Network", "Year", "What it showed"], [
      ["LeNet-5", "1998", "Conv → pool → conv → pool → dense, about 60 thousand parameters. Read handwritten digits (the ancestor of everything below)."],
      ["AlexNet", "2012", "Won ImageNet by a wide margin using ReLU, dropout and GPUs. Started the deep learning boom."],
      ["VGG", "2014", "Just stacks of 3 × 3 convolutions and max pooling. Simple and deep (16-19 layers), but about 138 million parameters in VGG-16, mostly in the dense layers."],
      ["GoogLeNet / Inception", "2014", "Several filter sizes in parallel inside a block, and 1 × 1 convolutions to keep the cost down. Far fewer parameters than VGG."],
      ["ResNet", "2015", "Skip connections: 50, 101, even 152 layers train fine. The backbone of most vision systems for years."],
      ["EfficientNet, ConvNeXt", "2019 / 2022", "Careful scaling of depth, width and resolution; CNNs modernised with ideas from transformers."],
    ]],
    ["p", "Since about 2020, [Vision Transformers](transformers.html#vit-multimodal) compete with and often replace CNNs for large-scale image work, but CNNs (and ResNets in particular) remain the default for small and medium problems, for speed, and as the starting point in [transfer learning](dl.html#transfer-learning)."],
    ["h", "Why deeper plain networks stopped working"],
    ["p", "After VGG it seemed obvious that more layers means better results. But when researchers stacked 56 plain layers they found something strange: the deeper network had **higher training error** than the 20-layer one. That is not overfitting (overfitting means low training error, high test error). It is an optimisation failure: a deeper net could in principle copy the shallow one and set the extra layers to \"do nothing\", yet plain layers find it hard to learn the identity function, and gradients shrink on the way back ([vanishing gradients](dl.html#vanishing-gradients))."],
    ["h", "The residual (skip) connection"],
    ["p", "ResNet's fix is almost embarrassingly simple: let each block **add its input back to its output**."],
    ["diagram", "dl2_resblock", [], "A residual block. The layers compute a correction F(x); the skip connection adds the unchanged input x back."],
    ["math", R`y \;=\; \operatorname{ReLU}\big(F(x) + x\big)`, "F(x): two conv layers · x: the shortcut"],
    ["p", "Two things happen. First, \"do nothing\" is now easy: just push $F(x)$ toward 0 and the block passes $x$ straight through. Second, look at the gradient:"],
    ["math", R`\frac{\partial y}{\partial x} \;=\; I + \frac{\partial F}{\partial x}`, "the identity I gives gradients a direct highway to earlier layers"],
    ["p", "The \"+ I\" means that even if $\\partial F/\\partial x$ is tiny, the gradient still reaches the early layers. Let us watch it happen. We send a gradient back through 30 random layers, once for a plain stack and once with a skip connection around each layer (the weights are slightly small, a common situation):"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
n, depth = 100, 30
Ws = [rng.normal(0, 0.5 * np.sqrt(2 / n), (n, n)) for _ in range(depth)]   # half the He scale: a bit too small

def grad_norms(residual):
    h = rng.normal(size=n)
    masks = []
    for W in Ws:                                       # forward
        z = W @ h
        masks.append(z > 0)
        h = h + np.maximum(z, 0) if residual else np.maximum(z, 0)
    g = np.ones(n) / np.sqrt(n)                        # gradient arriving at the output (norm 1)
    norms = [np.linalg.norm(g)]
    for W, m in zip(Ws[::-1], masks[::-1]):            # backward
        g = (g + W.T @ (m * g)) if residual else W.T @ (m * g)
        norms.append(np.linalg.norm(g))
    return norms

plain, res = grad_norms(False), grad_norms(True)
for L in [0, 5, 10, 20, 30]:
    print(f"gradient size after going back {L:2d} layers:  plain {plain[L]:.2e}   residual {res[L]:.2e}")`, "gradient size after going back  0 layers:  plain 1.00e+00   residual 1.00e+00\ngradient size after going back  5 layers:  plain 5.74e-02   residual 4.47e+00\ngradient size after going back 10 layers:  plain 2.11e-03   residual 1.89e+01\ngradient size after going back 20 layers:  plain 1.34e-06   residual 2.80e+02\ngradient size after going back 30 layers:  plain 1.26e-09   residual 3.81e+03"],
    ["p", "The plain stack's gradient dies off exponentially; the residual one stays alive (here it even grows, because the identity path adds a full copy of the gradient at every layer; real ResNets tame this with [BatchNorm](dl.html#normalization) and a small initial branch). In a real ResNet block each conv is followed by BatchNorm, and when the shape changes (more channels or stride 2) the shortcut uses a 1 × 1 convolution to match shapes. Deeper ResNets (50 and up) use a **bottleneck** block: 1 × 1 conv to shrink the channels, 3 × 3 conv, 1 × 1 conv to expand again, which is much cheaper."],
    ["h", "A residual block in PyTorch"],
    ["code", R`
import torch, torch.nn as nn
import torchvision

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        # when the shape changes, the shortcut needs a 1x1 conv to match it
        self.skip = nn.Identity() if (stride == 1 and c_in == c_out) else \
            nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
    def forward(self, x):
        return torch.relu(self.f(x) + self.skip(x))     # F(x) + x

torch.manual_seed(0)
x = torch.randn(2, 16, 32, 32)
print(BasicBlock(16, 16)(x).shape, BasicBlock(16, 32, stride=2)(x).shape)

net = torchvision.models.resnet18(weights=None)          # the architecture only: no download, random weights
print("ResNet-18 parameters:", f"{sum(p.numel() for p in net.parameters()):,}")
print("output for a 224x224 image:", tuple(net(torch.randn(1, 3, 224, 224)).shape))`, "torch.Size([2, 16, 32, 32]) torch.Size([2, 32, 16, 16])\nResNet-18 parameters: 11,689,512\noutput for a 224x224 image: (1, 1000)", "PyTorch", "local"],
    ["tip", "Skip connections are not just a vision trick: every [transformer block](transformers.html#transformer-block) is wrapped in them too, and so is nearly every deep architecture today."],
    ["danger", "Using pooling too early or too often: after five 2 × 2 poolings a 64 × 64 image is only 2 × 2, and the fine detail you needed is gone. Also, do not forget that max pooling throws away three quarters of the values, so it is lossy by design."],
    ["tip", "Practice: [Max Pooling](practice.html#p/max-pool-2d). Next: sequences instead of images, with [recurrent networks](dl.html#rnn)."],
  ],
},
{
  id: "rnn", group: "Architectures", title: "Recurrent Neural Networks",
  sub: "A network that reads a sequence one step at a time and carries a memory (the hidden state) from step to step.",
  tags: ["sequences"],
  blocks: [
    ["p", "Text, speech, sensor readings, stock prices: these are **sequences**. They have two awkward properties for the networks we have built so far. They come in **different lengths** (a tweet versus a book chapter), and **order matters** (\"dog bites man\" is not \"man bites dog\"). An MLP wants a fixed-size input and has no notion of order."],
    ["p", "A **recurrent neural network (RNN)** reads the sequence one item at a time while keeping a small vector called the **hidden state** $h$, its memory of everything read so far. At each step it looks at the new item *and* its old memory and produces a new memory. It is a loop."],
    ["viz", "dl2_rnn"],
    ["h", "The equations"],
    ["math", R`h_t \;=\; \tanh\!\big(W_x\,x_t + W_h\,h_{t-1} + b\big)`, "x_t: the input at step t · h_{t-1}: the previous memory · W_x, W_h, b: the weights (the same at every step)"],
    ["math", R`y_t \;=\; W_y\,h_t + b_y`, "optional output at each step (or only at the last one)"],
    ["p", "The crucial detail: **the same weights are used at every time step**. That is why one small RNN can process a sequence of any length, and why it can recognise a pattern wherever it occurs in time. It is the time-version of the weight sharing in a [CNN](dl.html#cnn). Drawn \"unrolled\" through time, an RNN looks like a very deep network with one layer per step, all sharing the same weights."],
    ["table", ["Pattern", "Example", "What you read out"], [
      ["many → one", "Sentiment of a review; classify a sensor window", "Only the last hidden state $h_T$"],
      ["one → many", "Image captioning (image in, words out)", "An output $y_t$ at every step"],
      ["many → many (aligned)", "Tagging each word with its part of speech", "An output $y_t$ for each input"],
      ["many → many (different lengths)", "Translation, summarisation", "[Encoder-decoder](dl.html#seq2seq): read all, then write"],
    ]],
    ["h", "An RNN cell from scratch"],
    ["p", "Everything above is three lines of NumPy per step. Let us run a made-up sequence of 4 inputs (each a vector of 3 numbers) through an RNN with 2 hidden units:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
n_in, n_hid, T = 3, 2, 4
Wx = rng.normal(0, 0.5, (n_hid, n_in))
Wh = rng.normal(0, 0.5, (n_hid, n_hid))
b = np.zeros(n_hid)

def rnn_forward(xs, h0=None):
    h = np.zeros(n_hid) if h0 is None else h0
    states = []
    for x in xs:                                   # one step per element
        h = np.tanh(Wx @ x + Wh @ h + b)           # new memory from new input + old memory
        states.append(h)
    return np.array(states)

xs = rng.normal(size=(T, n_in))
H = rnn_forward(xs)
print("hidden states, one row per time step:")
print(H.round(3))
print("parameters:", Wx.size + Wh.size + b.size, " (does not depend on the sequence length)")
H2 = rnn_forward(np.vstack([xs, rng.normal(size=(6, n_in))]))
print("a sequence of length 10 works with the same weights:", H2.shape)`, "hidden states, one row per time step:\n[[-0.656 -0.433]\n [-0.664  0.601]\n [-0.029 -0.016]\n [ 0.45   0.342]]\nparameters: 12  (does not depend on the sequence length)\na sequence of length 10 works with the same weights: (10, 2)"],
    ["h", "Backpropagation through time and the memory problem"],
    ["p", "Training an RNN means unrolling it and running [backpropagation](dl.html#backprop) through all the steps: **backpropagation through time (BPTT)**. The gradient of a late loss with respect to an early hidden state is a product of one Jacobian per step:"],
    ["math", R`\frac{\partial h_T}{\partial h_t} \;=\; \prod_{k=t+1}^{T} \operatorname{diag}\!\big(1 - h_k^2\big)\, W_h`, "each factor is the tanh slope times W_h"],
    ["p", "A product of many matrices tends to either **shrink to zero** or **blow up**, depending on whether $W_h$ is smaller or larger than 1 in size (its largest eigenvalue, or spectral radius). This is the [vanishing and exploding gradient](dl.html#vanishing-gradients) problem in its purest form. Watch it:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(1)
n = 20

def gradient_size(spectral_radius, T):
    """Norm of the product of T Jacobians (diag(1 - h^2) @ Wh) pushed back through time."""
    Wh = rng.normal(size=(n, n))
    Wh *= spectral_radius / np.max(np.abs(np.linalg.eigvals(Wh)))      # set the largest eigenvalue
    h = rng.normal(0, 0.5, n)
    g = np.ones(n) / np.sqrt(n)
    for _ in range(T):
        h = np.tanh(Wh @ h)
        g = Wh.T @ ((1 - h ** 2) * g)                                  # one step back in time
    return np.linalg.norm(g)

print("steps back |  Wh radius 0.8 |  radius 1.0 |  radius 1.5")
for T in [1, 5, 10, 20, 50]:
    print(f"{T:10d} |  {gradient_size(0.8, T):13.2e} |  {gradient_size(1.0, T):10.2e} |  {gradient_size(1.5, T):10.2e}")`, "steps back |  Wh radius 0.8 |  radius 1.0 |  radius 1.5\n         1 |       7.55e-01 |    8.06e-01 |    1.06e+00\n         5 |       1.27e-01 |    6.72e-01 |    1.12e+00\n        10 |       7.56e-02 |    2.43e-01 |    9.52e-02\n        20 |       5.02e-03 |    2.61e-01 |    5.25e+00\n        50 |       3.41e-06 |    1.10e-02 |    2.02e+00"],
    ["p", "With a radius of 0.8 the signal from 20 steps ago is essentially gone, so the network cannot learn that something at the start of a sentence matters at the end. At 1.0 it fades slowly. At 1.5 it no longer dies, and can grow in bursts (the exact numbers depend on the random matrix, and the saturating tanh limits the growth, but exploding gradients are a real problem, especially with other activations). Exploding gradients have a cheap fix, **gradient clipping**: if the gradient's norm is above a threshold (say 1 or 5), rescale it down. Vanishing is harder, and it is the reason for the [LSTM](dl.html#lstm) in the next section."],
    ["code", R`
import numpy as np

def clip_by_norm(grads, max_norm):
    total = np.sqrt(sum((g ** 2).sum() for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))     # only ever shrinks
    return [g * scale for g in grads], total

grads = [np.array([30.0, -40.0]), np.array([[0.0, 120.0]])]
clipped, before = clip_by_norm(grads, max_norm=5.0)
print("norm before:", round(float(before), 2))
print("norm after :", round(float(np.sqrt(sum((g ** 2).sum() for g in clipped))), 3))
print("direction kept:", [c.round(3).tolist() for c in clipped])`, "norm before: 130.0\nnorm after : 5.0\ndirection kept: [[1.154, -1.538], [[0.0, 4.615]]]"],
    ["h", "In PyTorch"],
    ["p", "`nn.RNN` implements the same formula (it has two bias vectors, which just add up). By default the input shape is **(sequence, batch, features)**; pass `batch_first=True` for (batch, sequence, features). It returns the hidden state at every step and the final one. Below it reproduces our NumPy loop exactly, and then learns a small memory task: the label is the **first** element of a 12-step sequence of noisy numbers, so the network has to carry one bit across 12 steps."],
    ["code", R`
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0)
rnn = nn.RNN(input_size=3, hidden_size=2, batch_first=True)
x = torch.randn(1, 4, 3)
out, h_last = rnn(x)

# our loop with the same weights
Wx, Wh = rnn.weight_ih_l0.detach().numpy(), rnn.weight_hh_l0.detach().numpy()
b = (rnn.bias_ih_l0 + rnn.bias_hh_l0).detach().numpy()
h = np.zeros(2); mine = []
for xt in x[0].numpy():
    h = np.tanh(Wx @ xt + Wh @ h + b); mine.append(h)
print("matches nn.RNN:", np.allclose(np.array(mine), out[0].detach().numpy(), atol=1e-5))

# memory task: predict the sign of the FIRST step of a 12-step sequence
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 16, batch_first=True); self.out = nn.Linear(16, 1)
    def forward(self, x):
        _, h = self.rnn(x)                      # h: (1, batch, 16), the last hidden state
        return self.out(h[0]).squeeze(-1)

def batch(n, T=12):
    x = torch.randn(n, T, 1)
    return x, (x[:, 0, 0] > 0).float()

net = Net(); opt = torch.optim.Adam(net.parameters(), lr=0.01)
for step in range(300):
    xb, yb = batch(64)
    loss = nn.functional.binary_cross_entropy_with_logits(net(xb), yb)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
xt, yt = batch(2000)
print("accuracy on fresh sequences:", round(((net(xt) > 0).float() == yt).float().mean().item(), 3))`, "matches nn.RNN: True\naccuracy on fresh sequences: 0.983", "PyTorch", "local"],
    ["warn", "Hidden states must be **reset between unrelated sequences**. If you keep $h$ from one batch to the next by accident, the memory of a different example leaks into this one. Also remember to pad variable-length batches (and use `pack_padded_sequence` or a mask) so the padding does not count as input."],
    ["tip", "Practice: [RNN Cell Forward Pass](practice.html#p/rnn-forward). Next: the fix for the memory problem, [LSTM and GRU](dl.html#lstm)."],
  ],
},
{
  id: "lstm", group: "Architectures", title: "LSTM & GRU",
  sub: "RNN cells with gates that decide what to forget, what to write and what to reveal, so memories survive across many steps.",
  tags: ["sequences", "gates"],
  blocks: [
    ["p", "A plain [RNN](dl.html#rnn) rewrites its entire memory at every step: $h_t$ is a brand-new tanh of everything. After many steps, whatever it learned at the beginning has been overwritten or has faded from the gradient. The **Long Short-Term Memory (LSTM)**, introduced by Hochreiter and Schmidhuber in 1997, adds a second, protected memory lane and learns **gates** that control what goes in and out of it."],
    ["p", "A gate is a number between 0 and 1 (a sigmoid output) that is multiplied onto a signal: 1 lets everything through, 0 blocks it. The LSTM has three of them:"],
    ["list", [
      "**Forget gate** $f$: how much of the old memory to keep.",
      "**Input gate** $i$: how much of the new candidate information to write.",
      "**Output gate** $o$: how much of the memory to reveal as the output $h_t$.",
    ]],
    ["viz", "dl2_lstm"],
    ["h", "The equations"],
    ["math", R`\begin{aligned} f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \\ i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \\ \tilde c_t &= \tanh(W_g [h_{t-1}, x_t] + b_g) \\ o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \end{aligned}`, "[h, x]: the previous hidden state and the input joined into one vector"],
    ["math", R`c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t \qquad\qquad h_t = o_t \odot \tanh(c_t)`, "⊙: element-wise product · c_t: the cell state (long-term memory) · h_t: the hidden state (what the outside sees)"],
    ["p", "The magic is in the left equation. The cell state is updated by **addition** of a gated old value and a gated new value. There is no squashing matrix multiplication between $c_{t-1}$ and $c_t$, so gradients flow back along the cell state almost untouched whenever $f\\approx1$. It is the same idea as the [skip connection in ResNets](dl.html#pooling-resnet), found 18 years earlier."],
    ["h", "One LSTM step from scratch"],
    ["p", "With vectors instead of single numbers, the four gates can be computed with one matrix multiplication and then split. Hidden size 3, input size 2:"],
    ["code", R`
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def lstm_step(x, h, c, W, b):
    """W: (4H, H+I) stacked as [forget; input; candidate; output]; b: (4H,)."""
    H = h.size
    z = W @ np.concatenate([h, x]) + b
    f = sigmoid(z[0:H])                  # forget gate
    i = sigmoid(z[H:2 * H])              # input gate
    g = np.tanh(z[2 * H:3 * H])          # candidate memory
    o = sigmoid(z[3 * H:4 * H])          # output gate
    c_new = f * c + i * g                # keep some of the old, write some of the new
    h_new = o * np.tanh(c_new)
    return h_new, c_new, (f, i, g, o)

rng = np.random.default_rng(0)
I, H = 2, 3
W = rng.normal(0, 0.5, (4 * H, H + I))
b = np.zeros(4 * H); b[:H] = 1.0         # forget-gate bias = 1: remember by default
h, c = np.zeros(H), np.zeros(H)
for t, x in enumerate(rng.normal(size=(3, I))):
    h, c, (f, i, g, o) = lstm_step(x, h, c, W, b)
    print(f"t={t}  forget {f.round(2)}  input {i.round(2)}  output {o.round(2)}")
    print(f"     cell c {c.round(3)}   hidden h {h.round(3)}")
print("parameters:", W.size + b.size, " = 4 * (H*(H+I) + H) =", 4 * (H * (H + I) + H))`, "t=0  forget [0.78 0.87 0.86]  input [0.33 0.35 0.49]  output [0.24 0.65 0.66]\n     cell c [-0.079 -0.275 -0.155]   hidden h [-0.019 -0.174 -0.101]\nt=1  forget [0.77 0.63 0.78]  input [0.54 0.5  0.42]  output [0.73 0.73 0.28]\n     cell c [0.169 0.004 0.047]   hidden h [0.123 0.003 0.013]\nt=2  forget [0.75 0.75 0.74]  input [0.47 0.48 0.47]  output [0.53 0.57 0.48]\n     cell c [ 0.126 -0.018  0.048]   hidden h [ 0.066 -0.011  0.023]\nparameters: 72  = 4 * (H*(H+I) + H) = 72"],
    ["h", "GRU: a lighter alternative"],
    ["p", "The **Gated Recurrent Unit** (Cho et al., 2014) keeps the idea but merges the memory lanes: there is only a hidden state $h$ and two gates, an *update* gate $z$ (how much to replace) and a *reset* gate $r$ (how much of the past to use when proposing new content):"],
    ["math", R`z_t = \sigma(W_z[h_{t-1},x_t]) \quad r_t = \sigma(W_r[h_{t-1},x_t]) \quad \tilde h_t = \tanh(W[\,r_t\odot h_{t-1},\,x_t]) \quad h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t`],
    ["table", ["", "Vanilla RNN", "LSTM", "GRU"], [
      ["State", "$h$", "$h$ and $c$", "$h$"],
      ["Gates", "none", "3 (forget, input, output)", "2 (update, reset)"],
      ["Weight matrices", "1", "4", "3"],
      ["Parameters (hidden $H$, input $I$)", "$H(H+I)+H$", "$4\\,[H(H+I)+H]$", "$3\\,[H(H+I)+H]$"],
      ["Long-range memory", "poor", "good", "good, usually about the same as LSTM"],
    ]],
    ["p", "In practice LSTM and GRU perform similarly on most tasks; GRU is faster and smaller, LSTM is the safer default when the sequences are long."],
    ["h", "Does it really remember? A test in PyTorch"],
    ["p", "First we check that our NumPy step matches PyTorch's `nn.LSTMCell` (PyTorch stacks the gates in the order input, forget, candidate, output). Then a direct measurement of memory: how much of a loss that depends only on the **last** hidden state still reaches the **first** input? For a plain RNN this gradient collapses as the sequence grows (the problem from the [RNN section](dl.html#rnn)); the LSTM with forget bias 1 keeps it orders of magnitude larger. Values are averages over 5 random initialisations, before any training."],
    ["code", R`
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0)
# 1) our lstm_step against nn.LSTMCell
cell = nn.LSTMCell(2, 3)
x, h, c = torch.randn(1, 2), torch.randn(1, 3), torch.randn(1, 3)
h2, c2 = cell(x, (h, c))
W = torch.cat([cell.weight_hh, cell.weight_ih], dim=1).detach().numpy()     # (4H, H+I), order i, f, g, o
b = (cell.bias_ih + cell.bias_hh).detach().numpy()
sig = lambda z: 1 / (1 + np.exp(-z))
z = W @ np.concatenate([h[0].numpy(), x[0].numpy()]) + b
i, f, g, o = sig(z[0:3]), sig(z[3:6]), np.tanh(z[6:9]), sig(z[9:12])
c_new = f * c[0].numpy() + i * g
print("matches nn.LSTMCell:", np.allclose(o * np.tanh(c_new), h2[0].detach().numpy(), atol=1e-5))

# 2) how much gradient from the LAST hidden state reaches the FIRST input?
def first_input_grad(kind, T, seed, forget_bias):
    torch.manual_seed(seed)
    H = 32
    rnn = (nn.RNN if kind == "rnn" else nn.LSTM)(1, H, batch_first=True)
    if kind == "lstm":
        rnn.bias_hh_l0.data[H:2 * H] = forget_bias          # the forget-gate slice of the bias
    x = torch.randn(64, T, 1, requires_grad=True)
    out, _ = rnn(x)
    out[:, -1].sum().backward()
    return x.grad[:, 0].abs().mean().item()

print("steps |    RNN    | LSTM (forget bias 0) | LSTM (forget bias 1)")
for T in [5, 20, 50, 100, 200]:
    r = np.mean([first_input_grad("rnn", T, s, 0) for s in range(5)])
    l0 = np.mean([first_input_grad("lstm", T, s, 0) for s in range(5)])
    l1 = np.mean([first_input_grad("lstm", T, s, 1) for s in range(5)])
    print(f"{T:5d} | {r:.2e} | {l0:.2e}  | {l1:.2e}")`, "matches nn.LSTMCell: True\nsteps |    RNN    | LSTM (forget bias 0) | LSTM (forget bias 1)\n    5 | 3.98e-02 | 1.41e-02  | 5.11e-02\n   20 | 1.65e-05 | 1.41e-05  | 4.37e-03\n   50 | 1.15e-11 | 3.14e-11  | 2.75e-05\n  100 | 2.61e-21 | 5.59e-20  | 1.11e-07\n  200 | 3.48e-40 | 2.47e-37  | 6.65e-13", "PyTorch", "local"],
    ["p", "Everything decays at long range (this is an untrained network, and 200 steps is very long), but the LSTM with a positive forget bias keeps the signal many orders of magnitude stronger. During training it can also learn to keep $f$ close to 1 for the information that matters."],
    ["tip", "Initialise the forget-gate bias to about 1 (as we did). Then $f\\approx0.73$ at the start and the cell remembers by default, which makes long-range learning much easier. PyTorch does not do this for you."],
    ["note", "Useful variants: **bidirectional** RNNs read the sequence both forwards and backwards and join the two states (great for tagging, impossible for generating text left to right); **stacked** RNNs feed one layer's $h_t$ sequence into the next layer. As of 2026, [transformers](transformers.html#attention-intuition) have replaced LSTMs for most language work, because attention connects any two positions directly and trains in parallel, but LSTMs and GRUs are still used for small, streaming and on-device sequence models and appear constantly in interviews."],
    ["tip", "Practice: [LSTM Cell Step](practice.html#p/lstm-step). Next: two RNNs working together, in [Seq2Seq](dl.html#seq2seq)."],
  ],
},
{
  id: "seq2seq", group: "Architectures", title: "Seq2Seq & the Birth of Attention",
  sub: "Two RNNs, one to read and one to write, and the idea that let the writer look back at everything the reader saw.",
  tags: ["sequences", "attention"],
  blocks: [
    ["p", "Translation, summarisation, speech recognition, chat: the input is a sequence and the output is a *different* sequence, often of a different length. \"I am a student\" is 4 words; \"je suis étudiant\" is 3. The classic solution, from 2014, is **sequence-to-sequence (seq2seq)**: two recurrent networks, joined in the middle."],
    ["list", [
      "The **encoder** (an [LSTM or GRU](dl.html#lstm)) reads the input one token at a time. Its final hidden state is a summary of the whole input, the **context vector**.",
      "The **decoder** is another RNN. It starts from the context vector and writes the output one token at a time, feeding each token it produces back in as the next input, until it writes a special `<end>` token.",
    ]],
    ["diagram", "dl2_encdec", [], "Seq2seq without attention: everything the decoder knows about the input has to fit in one fixed-size vector."],
    ["h", "Teacher forcing"],
    ["p", "During training we know the right answer, so we do not feed the decoder its own (possibly wrong) previous guesses. We feed it the **true** previous word. This is called **teacher forcing**. It trains fast and stably, but there is a catch: at test time the decoder must use its own predictions, which it never practised on, so one mistake can snowball (\"exposure bias\"). Generating with the decoder's own outputs is the job of [greedy and beam search](llm.html#greedy-beam)."],
    ["h", "The bottleneck"],
    ["p", "A single vector of, say, 512 numbers has to carry the whole input: every word, its order, its meaning. For a 5-word sentence that works. For a 50-word paragraph it does not: the encoder's final state mostly remembers the last few words and forgets the start. Translation quality in the 2014 papers visibly dropped as sentences got longer."],
    ["p", "A human translator does not memorise the whole sentence and then write from memory. They keep glancing back at the source, at *the relevant word* for the one they are writing now. In 2014 Bahdanau, Cho and Bengio gave the decoder exactly that ability."],
    ["h", "Attention"],
    ["p", "Keep **all** the encoder states $h_1,\\dots,h_n$, not just the last one. At every decoding step, the decoder state $s$ asks \"which input positions matter right now?\", scores each one, turns the scores into weights with a [softmax](math.html#entropy), and takes a weighted average:"],
    ["math", R`e_j = \operatorname{score}(s,\,h_j) \qquad \alpha_j = \frac{e^{e_j}}{\sum_k e^{e_k}} \qquad c = \sum_j \alpha_j\, h_j`, "score: a dot product (Luong) or a tiny neural net (Bahdanau) · α: attention weights, positive, summing to 1 · c: the context for this step"],
    ["diagram", "dl2_attnseq", [], "Attention: the decoder builds a fresh context vector at each step as a weighted average of all encoder states."],
    ["p", "The context $c$ is different at every step, so the decoder no longer depends on one overloaded vector. As a bonus, the weights $\\alpha$ form a table of \"which input word was used for which output word\" that often lines up with word alignments, a rare window into what a network is doing."],
    ["code", R`
import numpy as np

def softmax(e):
    e = np.exp(e - e.max())                   # shift for stability
    return e / e.sum()

rng = np.random.default_rng(0)
H = rng.normal(size=(4, 3))                   # 4 encoder states (one per input word), 3 numbers each
words = ["I", "am", "a", "student"]
s = H[3] * 1.5 + 0.1 * rng.normal(size=3)     # decoder state that "resembles" the last input word

scores = H @ s                                # dot-product score for each encoder state
alpha = softmax(scores)                       # attention weights
context = alpha @ H                           # weighted average of encoder states
for w, a in zip(words, alpha):
    print(f"{w:8s} weight {a:.3f}")
print("weights sum to", alpha.sum().round(3))
print("context vector:", context.round(3).tolist())
print("closest single state:", H[3].round(3).tolist())`, "I        weight 0.029\nam       weight 0.045\na        weight 0.001\nstudent  weight 0.925\nweights sum to 1.0\ncontext vector: [-1.161, -0.604, 0.072]\nclosest single state: [-1.265, -0.623, 0.041]"],
    ["p", "This is exactly the mechanism that [transformers](transformers.html#attention-intuition) are built on. Their only change: instead of having an RNN produce the states $h_j$, let *every position attend to every other position* (self-attention), and drop the recurrence altogether. That makes training parallel across the sequence and shortens every path between two words to a single step. The paper that did this (2017) was called *Attention Is All You Need*."],
    ["h", "Seeing the difference in PyTorch"],
    ["p", "Task: reverse a sequence of 10 random digits (`3 1 4 … → … 4 1 3`). The first word the decoder must write is the *last* input digit, so the model has to carry the whole sequence. Both models use the same GRU encoder and decoder; the second one adds a three-line attention step (dot product → softmax → weighted sum). Training both takes about half a minute on a CPU."],
    ["code", R`
import torch, torch.nn as nn

V, L, H = 10, 10, 64                         # digits 0-9, sequence length 10; token 10 = <start>
START = V

class Seq2Seq(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.emb = nn.Embedding(V + 1, 32)
        self.enc = nn.GRU(32, H, batch_first=True)
        self.dec = nn.GRU(32, H, batch_first=True)
        self.attn = attn
        self.out = nn.Linear(H * (2 if attn else 1), V)
    def forward(self, src, tgt_in):
        enc_out, h = self.enc(self.emb(src))            # h = the single context vector
        dec_out, _ = self.dec(self.emb(tgt_in), h)
        if self.attn:
            w = torch.softmax(dec_out @ enc_out.transpose(1, 2), dim=-1)   # (batch, out steps, in steps)
            dec_out = torch.cat([dec_out, w @ enc_out], dim=-1)            # glue the context onto the decoder state
        return self.out(dec_out)

def data(n):
    src = torch.randint(0, V, (n, L)); tgt = src.flip(1)
    tgt_in = torch.cat([torch.full((n, 1), START), tgt[:, :-1]], dim=1)    # teacher forcing: shifted true answer
    return src, tgt, tgt_in

for attn in [False, True]:
    torch.manual_seed(0)
    model = Seq2Seq(attn); opt = torch.optim.Adam(model.parameters(), 3e-3)
    for step in range(600):
        s, y, yi = data(96)
        loss = nn.functional.cross_entropy(model(s, yi).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                               # test: decode using the model's OWN previous outputs
        s, y, _ = data(2000)
        ys = torch.full((2000, 1), START)
        for _ in range(L):
            nxt = model(s, ys)[:, -1].argmax(-1)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
        pred = ys[:, 1:]
    print("with attention   " if attn else "without attention", "digit accuracy", round((pred == y).float().mean().item(), 3),
          " whole sequence correct", round((pred == y).all(1).float().mean().item(), 3))`, "without attention digit accuracy 0.655  whole sequence correct 0.178\nwith attention    digit accuracy 0.987  whole sequence correct 0.962", "PyTorch", "local"],
    ["warn", "Do not read too much into one toy run, but the pattern is real and was the central finding of the 2014-2015 papers: attention helps most on **long** inputs, because it removes the single-vector bottleneck."],
    ["tip", "Next: how words become vectors in the first place, in [Word Embeddings](dl.html#word-embeddings)."],
  ],
},
{
  id: "word-embeddings", group: "Architectures", title: "Word Embeddings: word2vec & GloVe",
  sub: "Give every word a short vector learned from the company it keeps, so that similar words get similar vectors.",
  tags: ["nlp", "embeddings"],
  blocks: [
    ["p", "A network needs numbers, not words. The simplest idea is **one-hot encoding**: with a vocabulary of 50,000 words, \"cat\" is a vector of 50,000 numbers with a single 1 at position 1,234. That is wasteful and, worse, **meaningless**: every pair of different words is exactly as far apart as every other pair, so \"cat\" is no closer to \"dog\" than to \"carburettor\"."],
    ["p", "An **embedding** replaces it with a short dense vector (50 to 1,000 numbers) in which *distance means similarity*. Mathematically it is a lookup: a matrix $E$ of shape (vocabulary × dimension), and the vector for word $w$ is simply row $w$ of $E$. The question is how to learn good rows."],
    ["h", "The distributional hypothesis"],
    ["p", "\"You shall know a word by the company it keeps\" (the linguist J. R. Firth, 1957). Words that appear in similar contexts have similar meanings: \"cat\" and \"dog\" are both found near \"pet\", \"fed\", \"fur\" and \"vet\". So we can learn the vectors from raw text, with no labels, by training a model to predict a word from its neighbours, or the neighbours from the word."],
    ["h", "word2vec"],
    ["p", "word2vec (Mikolov et al., 2013) comes in two flavours. **CBOW** predicts the centre word from its surrounding words. **Skip-gram** does the reverse: given the centre word, predict each word in a window around it."],
    ["diagram", "dl2_skipgram", [], "Skip-gram: training pairs (centre → neighbour) are read off the text with a sliding window."],
    ["p", "Predicting a word out of a 50,000-word vocabulary with a full softmax is expensive. **Negative sampling** is the trick: for each real pair (centre $c$, true neighbour $o$) also draw a few random words $n_1\\dots n_K$ that were *not* neighbours, and train a tiny logistic classifier to say \"real\" for the true pair and \"fake\" for the random ones:"],
    ["math", R`L \;=\; -\log \sigma\!\big(v_c\cdot u_o\big) \;-\; \sum_{k=1}^{K} \log \sigma\!\big(-\,v_c\cdot u_{n_k}\big)`, "v: the 'centre' vector of a word · u: its 'context' vector · σ: sigmoid · K ≈ 5-20 negatives"],
    ["p", "Pull the vectors of words that occur together closer (dot product up), push random pairs apart. After training, the rows of $V$ are the embeddings. Words that share contexts end up with similar vectors."],
    ["h", "Skip-gram with negative sampling from scratch"],
    ["p", "A toy corpus with some structure: animals eat foods and run in cities, people cook foods and visit cities. A real corpus has billions of words; this one has about 800 short sentences, 20 words and an 8-number embedding. If the method works, animals should cluster, and so should foods and cities."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
animals, foods, places = ["cat", "dog", "horse", "cow"], ["bread", "rice", "cake", "pasta"], ["paris", "rome", "london", "madrid"]
sents = []
for _ in range(800):
    a, f, p = rng.choice(animals), rng.choice(foods), rng.choice(places)
    sents.append([f"the {a} sleeps", f"the {a} eats {f}", f"people cook {f}", f"people visit {p}", f"the {a} runs in {p}"][rng.integers(5)].split())

vocab = sorted({w for s in sents for w in s}); ix = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
pairs = np.array([(ix[s[i]], ix[s[j]]) for s in sents for i in range(len(s))
                  for j in range(max(0, i - 2), min(len(s), i + 3)) if j != i])      # (centre, neighbour), window 2
sig = lambda z: 1 / (1 + np.exp(-z))
d, K, lr = 8, 5, 0.05
Vc = rng.normal(0, 0.1, (V, d)); Uo = rng.normal(0, 0.1, (V, d))                       # centre and context vectors

for epoch in range(3):
    for c, o in pairs[rng.permutation(len(pairs))]:
        neg = rng.integers(0, V, K)                         # K random "fake" neighbours
        vc = Vc[c].copy()
        g_pos = sig(vc @ Uo[o]) - 1                         # gradient of -log sigma(v.u) wrt the score
        g_neg = sig(Uo[neg] @ vc)                           # gradient of -log sigma(-v.u) wrt the scores
        Vc[c] -= lr * (g_pos * Uo[o] + g_neg @ Uo[neg])
        Uo[o] -= lr * g_pos * vc
        np.add.at(Uo, neg, -lr * g_neg[:, None] * vc)

E = Vc / np.linalg.norm(Vc, axis=1, keepdims=True)         # unit length, so a dot product is cosine similarity
for w in ["cat", "bread", "paris"]:
    sim = E @ E[ix[w]]
    best = [j for j in np.argsort(-sim) if j != ix[w]][:3]  # leave out the word itself
    print(f"{w:6s} -> nearest:", [(vocab[i], round(float(sim[i]), 2)) for i in best])
print("cat vs bread:", round(float(E[ix["cat"]] @ E[ix["bread"]]), 2), "   cat vs dog:", round(float(E[ix["cat"]] @ E[ix["dog"]]), 2))`, "cat    -> nearest: [('dog', 0.99), ('horse', 0.99), ('cow', 0.98)]\nbread  -> nearest: [('pasta', 0.99), ('cake', 0.99), ('rice', 0.99)]\nparis  -> nearest: [('london', 0.99), ('madrid', 0.99), ('rome', 0.98)]\ncat vs bread: 0.02    cat vs dog: 0.99"],
    ["p", "Nobody told the model that cats are animals or that Paris is a city. It found out because those words are used in the same ways."],
    ["h", "Vector arithmetic"],
    ["p", "The famous surprise of word2vec: directions in the space carry meaning. The vector from \"man\" to \"woman\" is roughly the same as from \"king\" to \"queen\", so $\\text{king} - \\text{man} + \\text{woman}$ lands near $\\text{queen}$. The picture below uses a hand-made 2-D map (real vectors have hundreds of dimensions and the effect is less clean), but you can try other equations such as `paris - france + italy`."],
    ["viz", "dl2_embed"],
    ["h", "GloVe: counting instead of predicting"],
    ["p", "**GloVe** (Pennington et al., 2014, Stanford) reaches similar vectors from a different direction. Count how often every pair of words occurs together in the whole corpus, giving a big co-occurrence matrix $X$. Then fit vectors so that $w_i\\cdot\\tilde w_j + b_i + \\tilde b_j \\approx \\log X_{ij}$, with a weighting that stops very frequent pairs dominating. In effect it factorises the co-occurrence matrix. Even a plain factorisation shows the idea:"],
    ["code", R`
import numpy as np

sents = ["cat eats fish", "dog eats meat", "cat drinks milk", "dog drinks water", "people eat bread", "people drink water",
         "cow drinks milk", "people eat fish", "dog eats bread", "cat eats meat", "cow eats grass", "horse eats grass",
         "horse drinks water", "people drink milk"]
vocab = sorted({w for s in sents for w in s.split()}); ix = {w: i for i, w in enumerate(vocab)}
X = np.zeros((len(vocab), len(vocab)))
for s in sents:
    ws = s.split()
    for i, w in enumerate(ws):
        for j in range(max(0, i - 2), min(len(ws), i + 3)):
            if j != i:
                X[ix[w], ix[ws[j]]] += 1                      # co-occurrence count within a window of 2

P = np.log1p(X)                                              # tame the big counts
U, S, Vt = np.linalg.svd(P)
E = U[:, :3] * S[:3]                                         # keep 3 dimensions: the embeddings
En = E / np.linalg.norm(E, axis=1, keepdims=True)
def nearest(w, n=2):
    sim = En @ En[ix[w]]
    return [(vocab[i], round(float(sim[i]), 2)) for i in [j for j in np.argsort(-sim) if j != ix[w]][:n]]
print("co-occurrence matrix:", X.shape, " non-zero cells:", int((X > 0).sum()))
print("cat   ->", nearest("cat"))
print("milk  ->", nearest("milk"))`, "co-occurrence matrix: (15, 15)  non-zero cells: 68\ncat   -> [('dog', 1.0), ('horse', 1.0)]\nmilk  -> [('water', 1.0), ('people', 0.92)]"],
    ["h", "Using embeddings in PyTorch"],
    ["p", "`nn.Embedding(vocab_size, dim)` is the lookup table, trainable like any other layer. Looking up row $w$ is the same as multiplying a one-hot vector by $E$, only far faster:"],
    ["code", R`
import torch, torch.nn as nn

torch.manual_seed(0)
emb = nn.Embedding(num_embeddings=6, embedding_dim=3)
ids = torch.tensor([2, 0, 2])
one_hot = nn.functional.one_hot(ids, 6).float()
print("lookup equals one-hot @ E:", torch.allclose(emb(ids), one_hot @ emb.weight))
print("shape for a batch of 2 sentences of 4 word ids:", tuple(emb(torch.randint(0, 6, (2, 4))).shape))

# start from existing vectors (e.g. word2vec/GloVe files you loaded), and freeze or fine-tune them
pretrained = torch.randn(6, 3)
frozen = nn.Embedding.from_pretrained(pretrained, freeze=True)
print("trainable parameters, frozen:", sum(p.numel() for p in frozen.parameters() if p.requires_grad),
      "  trainable:", sum(p.numel() for p in emb.parameters() if p.requires_grad))`, "lookup equals one-hot @ E: True\nshape for a batch of 2 sentences of 4 word ids: (2, 4, 3)\ntrainable parameters, frozen: 0   trainable: 18", "PyTorch", "local"],
    ["table", ["", "word2vec", "GloVe", "Contextual (BERT, GPT)"], [
      ["Learns from", "Local windows, predicting words", "Global co-occurrence counts", "Predicting masked or next tokens with a deep network"],
      ["One word has", "One vector, always", "One vector, always", "A different vector in every sentence"],
      ["\"bank\" in 'river bank' vs 'bank loan'", "Same vector (a blend)", "Same vector (a blend)", "Two different vectors"],
      ["Cost", "Very cheap", "Cheap", "Needs a large [transformer](transformers.html#token-embeddings)"],
    ]],
    ["note", "Static embeddings like word2vec and GloVe are mostly historical now. Modern language models learn a [token embedding](transformers.html#token-embeddings) table as part of the network and then refine it through many layers, so the vector for \"bank\" depends on the sentence. But the idea carries straight over: similar meanings, nearby vectors. The same one drives [vector search](llm.html#embeddings-search)."],
    ["danger", "Trusting analogies too much. The \"king − man + woman\" demo works because the question excludes the input words from the answer; without that, the nearest word to the result is usually \"king\" itself. Embeddings also absorb the biases of the text they were trained on."],
  ],
},
{
  id: "generative", group: "Architectures", title: "Generative Models: Autoencoders, VAEs, GANs & Diffusion",
  sub: "Networks that don't just label data but learn to produce new data that looks like it.",
  tags: ["generative"],
  blocks: [
    ["p", "Most of what we have built is **discriminative**: given an input, predict a label. A **generative model** learns what the data itself looks like, well enough to produce new examples: new faces, new digits, new molecules, new music. There are four big families. They differ in *how* they learn the data distribution."],
    ["diagram", "dl2_genoverview", [], "The three classic families side by side. (Diffusion, the fourth, is shown further down.)"],
    ["h", "1. Autoencoders: compress and rebuild"],
    ["p", "An **autoencoder** has an **encoder** that squeezes the input $x$ into a small **code** $z$ (the *bottleneck*) and a **decoder** that rebuilds $\\hat x$ from the code. It is trained to make $\\hat x$ match $x$:"],
    ["math", R`L \;=\; \lVert x - \operatorname{dec}(\operatorname{enc}(x)) \rVert^2`],
    ["p", "No labels are needed (the input is its own target), so this is [unsupervised learning](ml.html#pca). Because the code is much smaller than the input, the network must keep only what matters. A linear autoencoder with a $k$-dimensional code learns exactly the same subspace as [PCA](ml.html#pca); with non-linear layers it can do better. Uses: compression, denoising, anomaly detection (a weird input rebuilds badly), pretraining."],
    ["p", "But a plain autoencoder is a poor *generator*. Pick a random point in the code space and the decoder usually gives garbage, because training only taught it to decode the codes that real data produced, and the space between them is full of gaps."],
    ["h", "2. Variational autoencoders (VAE): a smooth code space"],
    ["p", "The **VAE** fixes the gaps. The encoder outputs not one code but a small *Gaussian*: a mean $\\mu$ and a spread $\\sigma$. The decoder gets a random sample from it, $z = \\mu + \\sigma\\,\\varepsilon$ with $\\varepsilon\\sim\\mathcal N(0,1)$ (the **reparameterisation trick**, which keeps the sampling differentiable by moving the randomness into $\\varepsilon$). The loss has two parts:"],
    ["math", R`L \;=\; \underbrace{\lVert x-\hat x\rVert^2}_{\text{rebuild well}} \;+\; \underbrace{D_{KL}\big(\mathcal N(\mu,\sigma^2)\,\Vert\,\mathcal N(0,1)\big)}_{\text{keep the codes close to a standard Gaussian}}`, "the second term is the KL divergence; for diagonal Gaussians: ½ Σ (μ² + σ² − 1 − ln σ²)"],
    ["p", "The KL term squeezes all the codes into one smooth blob around zero, overlapping and without gaps. After training, you throw the encoder away, draw $z\\sim\\mathcal N(0,1)$ and decode: new data. (The loss is the negative *evidence lower bound*, ELBO, from variational inference; you can derive it from [KL divergence](math.html#entropy).) VAE samples tend to be a bit blurry, since the squared-error term averages over possibilities."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
mu, log_var = np.array([0.8, -0.5]), np.array([-1.0, 0.3])          # what an encoder might output
sigma = np.exp(0.5 * log_var)

# closed-form KL( N(mu, sigma^2) || N(0, 1) ) for a diagonal Gaussian
kl = 0.5 * np.sum(mu ** 2 + sigma ** 2 - 1 - log_var)

# check it against sampling:  KL = E[ log q(z) - log p(z) ]
eps = rng.normal(size=(500000, 2))
z = mu + sigma * eps                                                  # reparameterisation trick
log_q = -0.5 * (((z - mu) / sigma) ** 2 + log_var + np.log(2 * np.pi))
log_p = -0.5 * (z ** 2 + np.log(2 * np.pi))
print("KL closed form :", round(float(kl), 4))
print("KL by sampling :", round(float((log_q - log_p).sum(axis=1).mean()), 4))
print("a code that matches the prior has KL =", 0.5 * np.sum(0 ** 2 + 1 ** 2 - 1 - 0))`, "KL closed form : 0.6539\nKL by sampling : 0.6541\na code that matches the prior has KL = 0.0"],
    ["h", "3. GANs: a forger and a detective"],
    ["p", "A **generative adversarial network** (Goodfellow et al., 2014) skips explicit probability and sets up a game. The **generator** $G$ turns random noise into a fake sample. The **discriminator** $D$ looks at samples and tries to say which are real and which are fakes. Each one trains against the other:"],
    ["math", R`\min_G\;\max_D\;\; \mathbb E_{x\sim\text{data}}[\log D(x)] \;+\; \mathbb E_{z}[\log(1 - D(G(z)))]`, "D wants to tell real from fake; G wants D to be fooled"],
    ["p", "When it works, the generator ends up producing samples the discriminator cannot tell from real ones, and GANs produced the first strikingly sharp synthetic faces (2017-2019). But training is delicate: the two networks must stay balanced, and failures such as **mode collapse** (the generator keeps producing a few convincing samples and ignores the rest of the data) are common. A tiny version, learning a 1-D bell curve centred at 3, appears in the PyTorch example below."],
    ["h", "4. Diffusion models: learn to remove noise"],
    ["p", "Diffusion models (Sohl-Dickstein 2015, popularised by DDPM in 2020) now power most image, video and audio generators. The idea is almost the opposite of a GAN. **Forward process**: take real data and add a little Gaussian noise, again and again, until after $T\\approx1000$ steps nothing but static remains. This direction needs no learning. **Reverse process**: train a neural network to undo *one* small step of noise. To generate, start from pure noise and apply the network $T$ times."],
    ["viz", "dl2_diffusion"],
    ["diagram", "dl2_diffchain", [], "Diffusion: a fixed forward chain that destroys structure, and a learned reverse chain that restores it."],
    ["p", "The forward step is $x_t = \\sqrt{1-\\beta_t}\\,x_{t-1} + \\sqrt{\\beta_t}\\,\\varepsilon$. Because sums of Gaussians are Gaussian, we can jump straight to any step in closed form, with $\\bar\\alpha_t = \\prod_{s\\le t}(1-\\beta_s)$:"],
    ["math", R`x_t \;=\; \sqrt{\bar\alpha_t}\;x_0 \;+\; \sqrt{1-\bar\alpha_t}\;\varepsilon \qquad\qquad L \;=\; \mathbb E\,\big\lVert\, \varepsilon - \varepsilon_\theta(x_t,\,t)\,\big\rVert^2`, "the network ε_θ is trained to predict the noise that was added; this is just a regression problem"],
    ["p", "Training is therefore very stable and simple: pick a random image and a random $t$, add noise, ask the network which noise was added, minimise squared error. To make the idea concrete, we can run the whole thing with no neural network at all in a toy world where the data is only the two numbers $-1$ and $+1$. There, the best possible noise predictor has a closed form, and the real reverse sampling loop (the DDPM update rule) turns pure noise into samples at exactly $\\pm1$:"],
    ["code", R`
import numpy as np

T = 1000
betas = np.linspace(1e-4, 0.02, T)                  # the "noise schedule"
alphas = 1 - betas
abar = np.cumprod(alphas)                           # alpha-bar_t: how much signal is left after t steps
for t in [1, 100, 250, 500, 1000]:
    print(f"t={t:4d}  signal weight {np.sqrt(abar[t-1]):.3f}   noise weight {np.sqrt(1 - abar[t-1]):.3f}")

rng = np.random.default_rng(0)
x0 = rng.choice([-1.0, 1.0], size=100000)           # toy data: just -1 and +1
t = 250
eps = rng.normal(size=x0.shape)
xt = np.sqrt(abar[t-1]) * x0 + np.sqrt(1 - abar[t-1]) * eps         # jump straight to step t
print("variance of x_t (data variance is 1):", round(float(xt.var()), 3))
x0_back = (xt - np.sqrt(1 - abar[t-1]) * eps) / np.sqrt(abar[t-1])
print("knowing eps, x0 is recovered exactly:", bool(np.allclose(x0_back, x0)))

def eps_hat(x, t):                                  # the BEST possible noise predictor for this toy data
    a = abar[t]
    x0_hat = np.tanh(np.sqrt(a) / (1 - a) * x)      # E[x0 | x_t] when x0 is -1 or +1 with equal chance
    return (x - np.sqrt(a) * x0_hat) / np.sqrt(1 - a)

x = rng.normal(size=5000)                           # start from pure noise
for t in range(T - 1, -1, -1):                      # reverse process, one small denoising step at a time
    z = rng.normal(size=x.shape) if t > 0 else 0
    x = (x - betas[t] / np.sqrt(1 - abar[t]) * eps_hat(x, t)) / np.sqrt(alphas[t]) + np.sqrt(betas[t]) * z
print("generated samples: fraction above 0:", round(float((x > 0).mean()), 3), " average distance to the nearest of -1/+1:", f"{np.abs(np.abs(x) - 1).mean():.1e}")`, "t=   1  signal weight 1.000   noise weight 0.010\nt= 100  signal weight 0.947   noise weight 0.321\nt= 250  signal weight 0.724   noise weight 0.690\nt= 500  signal weight 0.280   noise weight 0.960\nt=1000  signal weight 0.006   noise weight 1.000\nvariance of x_t (data variance is 1): 0.999\nknowing eps, x0 is recovered exactly: True\ngenerated samples: fraction above 0: 0.502  average distance to the nearest of -1/+1: 1.2e-15"],
    ["p", "In real image models the closed-form predictor is replaced by a large neural network (a U-Net or a transformer) conditioned on the timestep, and often on a text prompt. Text-to-image systems also run the process in a compressed latent space (from an autoencoder) to save compute, and use \"classifier-free guidance\" to follow the prompt more strongly. The core loop is the one above."],
    ["h", "Autoencoder, VAE and GAN in PyTorch"],
    ["p", "On the 8 × 8 digits that ship with scikit-learn (64 pixels in 0-1). First, PCA with 8 components is the best *linear* 8-number code. Then an autoencoder with an 8-number code, a VAE with a 4-number code, and a 1-D GAN. For the VAE we decode random $z\\sim\\mathcal N(0,1)$ and measure how close the results are to real digits (compared with random noise images); for the GAN we check the mean and spread of the fakes against the real bell curve (mean 3, standard deviation 0.5)."],
    ["code", R`
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X = load_digits().data / 16.0
Xt = torch.tensor(X, dtype=torch.float32)

pca = PCA(8).fit(X)
print("PCA, 8 numbers : MSE", round(float(((X - pca.inverse_transform(pca.transform(X))) ** 2).mean()), 4))

torch.manual_seed(0)
ae = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8),            # encoder -> 8-number code
                   nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())   # decoder
opt = torch.optim.Adam(ae.parameters(), 3e-3)
for _ in range(300):
    loss = ((ae(Xt) - Xt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
print("autoencoder, 8 : MSE", round(loss.item(), 4))

class VAE(nn.Module):
    def __init__(self, z=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 32), nn.ReLU()); self.mu = nn.Linear(32, z); self.lv = nn.Linear(32, z)
        self.dec = nn.Sequential(nn.Linear(z, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
    def forward(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)                   # reparameterisation trick
        return self.dec(z), mu, lv

torch.manual_seed(0)
vae = VAE(); opt = torch.optim.Adam(vae.parameters(), 3e-3)
for _ in range(400):
    xr, mu, lv = vae(Xt)
    rec = ((xr - Xt) ** 2).sum(1).mean()
    kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
    loss = rec + kl; opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    fake = vae.dec(torch.randn(500, 4))                                        # sample z from the prior, then decode
print("VAE            : reconstruction", round(rec.item(), 2), " KL", round(kl.item(), 2))
print("   distance of generated digits to the nearest real digit:", round(torch.cdist(fake, Xt).min(1).values.mean().item(), 2),
      "  (random noise images:", round(torch.cdist(torch.rand(500, 64), Xt).min(1).values.mean().item(), 2), ")")

torch.manual_seed(0)
G = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
D = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
og, od = torch.optim.Adam(G.parameters(), 1e-3), torch.optim.Adam(D.parameters(), 1e-3)
bce = nn.functional.binary_cross_entropy_with_logits
for _ in range(3000):
    real = torch.randn(64, 1) * 0.5 + 3.0
    fake = G(torch.randn(64, 4))
    ld = bce(D(real), torch.ones(64, 1)) + bce(D(fake.detach()), torch.zeros(64, 1))   # D: real -> 1, fake -> 0
    od.zero_grad(); ld.backward(); od.step()
    lg = bce(D(fake), torch.ones(64, 1))                                                # G: make D say "real"
    og.zero_grad(); lg.backward(); og.step()
f = G(torch.randn(5000, 4)).detach()
print("GAN            : fake mean", round(f.mean().item(), 2), " std", round(f.std().item(), 2), "  (real: 3.0 and 0.5)")`, "PCA, 8 numbers : MSE 0.0239\nautoencoder, 8 : MSE 0.0209\nVAE            : reconstruction 4.31  KL 0.36\n   distance of generated digits to the nearest real digit: 1.43   (random noise images: 3.57 )\nGAN            : fake mean 3.02  std 0.66   (real: 3.0 and 0.5)", "PyTorch", "local"],
    ["table", ["Family", "Training signal", "Strengths", "Weaknesses"], [
      ["Autoencoder", "Rebuild the input", "Simple; compression, anomaly detection", "Not a good generator by itself"],
      ["VAE", "Rebuild + keep codes Gaussian", "Stable; smooth latent space you can interpolate in", "Blurry samples"],
      ["GAN", "Fool a discriminator", "Sharp, fast sampling (one pass)", "Unstable training, mode collapse"],
      ["Diffusion", "Predict the noise that was added", "Excellent quality and diversity; stable training", "Slow sampling (many steps); mitigated by faster samplers and distillation"],
      ["Autoregressive (GPT-style)", "Predict the next token", "The standard for text; scales very well", "Samples one token at a time; see [pretraining](llm.html#pretraining)"],
    ]],
    ["note", "As of 2026, diffusion (and closely related flow-matching) models dominate image, video and audio generation, while autoregressive [transformers](transformers.html#architectures) dominate text. GANs are still used where one-pass speed matters and VAEs (and VQ-VAEs) are widely used as the *compressors* that diffusion models and autoregressive models work inside."],
    ["warn", "A decreasing training loss on a generative model says little about sample quality. For GANs the two losses fight each other and do not go to zero; for VAEs and diffusion models, look at actual samples."],
  ],
},
{
  id: "pytorch-basics", group: "In Practice", title: "PyTorch Basics",
  sub: "Tensors, automatic gradients and modules: NumPy with a GPU and a built-in backpropagation engine.",
  tags: ["pytorch"],
  blocks: [
    ["p", "You have written [backpropagation by hand](dl.html#backprop) and built a [tiny autograd engine](dl.html#autograd). PyTorch is the industrial version of that engine. Three ideas cover 90% of it:"],
    ["list", [
      "**Tensors**: n-dimensional arrays, like NumPy's, that can also live on a GPU and remember how they were computed.",
      "**Autograd**: call `.backward()` on a number and PyTorch fills in `.grad` for every tensor that took part in computing it.",
      "**Modules** (`torch.nn`): layers that own their parameters, composed into models, plus losses and optimizers.",
    ]],
    ["note", "The code on this page runs on your own computer (PyTorch is too big to run inside the browser). Install it with `pip install torch`; everything here works on a CPU, and the outputs shown are from real runs."],
    ["h", "Tensors"],
    ["p", "A tensor has a **shape**, a **dtype** and a **device**. Almost everything you know from NumPy carries over, often with the same name."],
    ["code", R`
import numpy as np, torch

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])             # like np.array
print(a.shape, a.dtype, a.device)
print("zeros / ones / randn:", torch.zeros(2, 2).tolist(), torch.ones(1, 2).tolist(), tuple(torch.randn(3, 4).shape))
print("sum over rows (dim=0):", a.sum(dim=0).tolist(), "  keepdim shape:", tuple(a.sum(dim=1, keepdim=True).shape))
print("matmul:", (a @ a.T).tolist())
print("broadcast a + [10, 20, 30]:", (a + torch.tensor([10., 20., 30.])).tolist())
print("reshape / view:", tuple(a.reshape(3, 2).shape), tuple(a.view(-1).shape), " indexing a[:, 1]:", a[:, 1].tolist())

# integers stay integers: classification labels are int64 ("long"), activations are float32
labels = torch.tensor([0, 2, 1])
print("dtypes:", labels.dtype, a.dtype, (labels.float() * 0.5).dtype)

# NumPy bridge: from_numpy SHARES memory with the array
arr = np.arange(3.0)
t = torch.from_numpy(arr)
arr[0] = 99.0
print("tensor sees the NumPy change:", t.tolist(), "   back to NumPy:", type(t.numpy()).__name__)`, "torch.Size([2, 3]) torch.float32 cpu\nzeros / ones / randn: [[0.0, 0.0], [0.0, 0.0]] [[1.0, 1.0]] (3, 4)\nsum over rows (dim=0): [5.0, 7.0, 9.0]   keepdim shape: (2, 1)\nmatmul: [[14.0, 32.0], [32.0, 77.0]]\nbroadcast a + [10, 20, 30]: [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]\nreshape / view: (3, 2) (6,)  indexing a[:, 1]: [2.0, 5.0]\ndtypes: torch.int64 torch.float32 torch.float32\ntensor sees the NumPy change: [99.0, 1.0, 2.0]    back to NumPy: ndarray", "PyTorch", "local"],
    ["table", ["NumPy", "PyTorch", "Note"], [
      ["`np.array(...)`", "`torch.tensor(...)`", "Floats default to float32 (NumPy: float64)"],
      ["`axis=`", "`dim=`", "Also accepted: `axis` as an alias, `keepdim` instead of `keepdims`"],
      ["`x.astype(np.float32)`", "`x.float()` / `x.to(torch.float32)`", "`.long()` for class ids"],
      ["`np.concatenate`", "`torch.cat`", "`torch.stack` adds a new dimension"],
      ["`x.T`, `x.transpose`", "`x.T`, `x.permute(...)`", "`permute` reorders any dimensions, e.g. NHWC → NCHW"],
      ["`x.reshape`", "`x.reshape` / `x.view`", "`view` needs a contiguous tensor and never copies"],
      ["`float(x)`", "`x.item()`", "For a one-element tensor"],
      ["(none)", "`x.to(\"cuda\")`", "Move to a GPU; model and data must be on the same device"],
    ]],
    ["h", "Autograd"],
    ["p", "Create a tensor with `requires_grad=True` and PyTorch records every operation on it as a [computational graph](dl.html#autograd). `loss.backward()` then walks that graph backwards and stores the gradients in `.grad`. Check it against the calculus:"],
    ["code", R`
import torch

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)
x, target = 3.0, 4.0

pred = w * x + b                      # 5.0
loss = (pred - target) ** 2           # 1.0
loss.backward()                       # fills w.grad and b.grad
print("loss", loss.item(), " dL/dw", w.grad.item(), " dL/db", b.grad.item())
print("by hand: dL/dw = 2(pred-target)*x =", 2 * (5 - 4) * 3, "  dL/db = 2(pred-target) =", 2 * (5 - 4))

# gradients ACCUMULATE; that is why training loops call zero_grad()
loss2 = (w * x + b - target) ** 2
loss2.backward()
print("after a second backward, w.grad =", w.grad.item(), "(doubled)")
w.grad.zero_()

# no_grad: do not record a graph (evaluation, manual updates)
with torch.no_grad():
    w -= 0.1 * 2.0                    # a manual SGD step
print("w after manual step:", round(w.item(), 3), " requires_grad still", w.requires_grad)
y = (w * x).detach()                  # cut a tensor out of the graph
print("detached tensor tracks gradients?", y.requires_grad)`, "loss 1.0  dL/dw 6.0  dL/db 2.0\nby hand: dL/dw = 2(pred-target)*x = 6   dL/db = 2(pred-target) = 2\nafter a second backward, w.grad = 12.0 (doubled)\nw after manual step: 1.8  requires_grad still True\ndetached tensor tracks gradients? False", "PyTorch", "local"],
    ["h", "Modules, parameters and models"],
    ["p", "A layer such as `nn.Linear(in, out)` is a **Module**: it owns its weights as `nn.Parameter`s (tensors with `requires_grad=True` that the optimizer will update) and defines a `forward`. Modules nest: your model is a Module made of Modules. Calling `model(x)` runs `forward`."],
    ["code", R`
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
    def forward(self, x):                              # the forward pass: the only thing YOU write
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP(4, 8, 3)
x = torch.randn(5, 4)                                   # a batch of 5 examples
print("output shape:", tuple(model(x).shape))
for name, p in model.named_parameters():
    print(f"{name:11s} shape {tuple(p.shape)}")
print("total parameters:", sum(p.numel() for p in model.parameters()))

# the same thing written with NumPy, using the module's own weights
W1, b1 = model.fc1.weight.detach().numpy(), model.fc1.bias.detach().numpy()
W2, b2 = model.fc2.weight.detach().numpy(), model.fc2.bias.detach().numpy()
mine = np.maximum(0, x.numpy() @ W1.T + b1) @ W2.T + b2
print("NumPy forward matches PyTorch:", np.allclose(mine, model(x).detach().numpy(), atol=1e-5))

seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))     # shortcut when layers just chain
print("same architecture via nn.Sequential:", tuple(seq(x).shape))
print("training mode:", model.training, "-> after .eval():", model.eval().training)`, "output shape: (5, 3)\nfc1.weight  shape (8, 4)\nfc1.bias    shape (8,)\nfc2.weight  shape (3, 8)\nfc2.bias    shape (3,)\ntotal parameters: 67\nNumPy forward matches PyTorch: True\nsame architecture via nn.Sequential: (5, 3)\ntraining mode: True -> after .eval(): False", "PyTorch", "local"],
    ["note", "`nn.Linear` stores its weight as `(out, in)` and computes $x W^\\top + b$, so a batch `x` of shape `(N, in)` gives `(N, out)`. Our NumPy MLP in the [forward-pass section](dl.html#mlp) used the transposed layout; both are fine as long as you are consistent."],
    ["warn", "Three habits that save hours: (1) print `.shape` and `.dtype` of everything while building; (2) use `torch.no_grad()` and `model.eval()` for evaluation; (3) never convert to NumPy a tensor that still requires grad (use `.detach()` first) or one that lives on a GPU (use `.cpu()`)."],
    ["danger", "Modifying a tensor **in place** (`x += 1`, `x[0] = 5`) that autograd needs for the backward pass raises \"a variable needed for gradient computation has been modified by an inplace operation\". Write `x = x + 1` instead."],
    ["tip", "Next: wrap this into a full [training loop](dl.html#pytorch-training)."],
  ],
},
{
  id: "pytorch-training", group: "In Practice", title: "Training a Model in PyTorch",
  sub: "Datasets, data loaders, the five-line training step, validation, learning-rate schedules and saving the model.",
  tags: ["pytorch", "training"],
  blocks: [
    ["p", "The [mini-batch training loop](dl.html#training-loop) has the same shape in every PyTorch project. Learn it once and you can read or write almost any training script. This page uses the 8 × 8 handwritten digits that ship with scikit-learn (1,797 images, 10 classes), so it runs offline in seconds on a CPU."],
    ["h", "The ingredients"],
    ["table", ["Piece", "PyTorch object", "What it does"], [
      ["Data", "`Dataset`, `DataLoader`", "The `Dataset` gives one example by index; the `DataLoader` batches, shuffles and (optionally) loads in parallel"],
      ["Model", "`nn.Module`", "Holds the parameters; `model(x)` is the forward pass"],
      ["Loss", "`nn.CrossEntropyLoss`, `nn.MSELoss`, …", "Compares predictions with targets. `CrossEntropyLoss` takes raw **logits** and integer class ids (no softmax before it)"],
      ["Optimizer", "`torch.optim.AdamW(model.parameters(), lr=...)`", "Updates the parameters from their `.grad` ([optimizers](dl.html#optimizers))"],
      ["Schedule", "`torch.optim.lr_scheduler.*`", "Changes the learning rate over time ([schedules](dl.html#lr-schedules))"],
    ]],
    ["h", "The training step"],
    ["steps", [
      "`loss = loss_fn(model(xb), yb)`: forward pass and loss.",
      "`opt.zero_grad()`: clear the old gradients (they accumulate).",
      "`loss.backward()`: backpropagation fills every `.grad`.",
      "`opt.step()`: the optimizer updates the parameters.",
    ]],
    ["p", "Around it: `model.train()` before the epoch (dropout and batch-norm in training mode), `model.eval()` plus `torch.no_grad()` for validation, and a `sched.step()` once per epoch or batch. Here is the complete program, with a train / validation / test split, a cosine learning-rate schedule, dropout and saving and reloading the weights:"],
    ["code", R`
import io, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# data: 64 pixel values scaled to 0-1; labels are int64 class ids
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32); y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, random_state=0, stratify=ytr)
train = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
print("train / val / test sizes:", len(Xtr), len(Xva), len(Xte))

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

def evaluate(X, y):
    model.eval()                                            # dropout off
    with torch.no_grad():                                   # no graph needed
        logits = model(X)
        return loss_fn(logits, y).item(), (logits.argmax(1) == y).float().mean().item()

for epoch in range(1, 21):
    model.train()
    total = 0.0
    for xb, yb in train:
        loss = loss_fn(model(xb), yb)                       # 1. forward + loss
        opt.zero_grad()                                     # 2. clear old gradients
        loss.backward()                                     # 3. backprop
        opt.step()                                          # 4. update
        total += loss.item() * len(xb)
    sched.step()
    if epoch in (1, 5, 10, 20):
        val_loss, val_acc = evaluate(Xva, yva)
        print(f"epoch {epoch:2d}  train loss {total / len(Xtr):.3f}  val loss {val_loss:.3f}  val acc {val_acc:.3f}")
print("test accuracy:", round(evaluate(Xte, yte)[1], 3))

# save and reload (an in-memory buffer here; normally a file such as "model.pt")
buf = io.BytesIO(); torch.save(model.state_dict(), buf); buf.seek(0)
clone = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 10))
clone.load_state_dict(torch.load(buf)); clone.eval()
print("reloaded model test accuracy:", round((clone(Xte).argmax(1) == yte).float().mean().item(), 3))`, "train / val / test sizes: 1077 270 450\nepoch  1  train loss 2.156  val loss 1.939  val acc 0.789\nepoch  5  train loss 0.609  val loss 0.495  val acc 0.900\nepoch 10  train loss 0.328  val loss 0.276  val acc 0.933\nepoch 20  train loss 0.239  val loss 0.227  val acc 0.933\ntest accuracy: 0.949\nreloaded model test accuracy: 0.949", "PyTorch", "local"],
    ["p", "Notice the train loss is a little higher than the validation loss: dropout is active while training and off during evaluation. That is normal and a good reason to measure validation in `eval()` mode."],
    ["h", "A small CNN in the same loop"],
    ["p", "Only the model (and the input shape) changes. Images need an explicit channel dimension: `(N, 1, 8, 8)`. Two [convolution](dl.html#cnn) + [pooling](dl.html#pooling-resnet) stages shrink 8 × 8 → 4 × 4 → 2 × 2 with 16 channels, then one linear layer classifies:"],
    ["code", R`
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32).view(-1, 1, 8, 8); y = torch.tensor(y)     # add the channel dimension
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, random_state=0, stratify=ytr)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8x8 -> 4x4
                                      nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))    # 4x4 -> 2x2
        self.head = nn.Linear(16 * 2 * 2, 10)
    def forward(self, x):
        return self.head(self.features(x).flatten(1))

torch.manual_seed(0)
cnn = CNN(); opt = torch.optim.Adam(cnn.parameters(), 3e-3); loss_fn = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
for epoch in range(20):
    cnn.train()
    for xb, yb in loader:
        loss = loss_fn(cnn(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
cnn.eval()
with torch.no_grad():
    print("parameters:", sum(p.numel() for p in cnn.parameters()))
    print("validation accuracy:", round((cnn(Xva).argmax(1) == yva).float().mean().item(), 3))
    print("test accuracy:", round((cnn(Xte).argmax(1) == yte).float().mean().item(), 3))`, "parameters: 1898\nvalidation accuracy: 0.959\ntest accuracy: 0.969", "PyTorch", "local"],
    ["h", "Habits of a good training script"],
    ["list", [
      "**Set seeds** (`torch.manual_seed`, plus a seeded `Generator` for the `DataLoader`) so a run can be repeated.",
      "**Move to the GPU if there is one**: `device = \"cuda\" if torch.cuda.is_available() else \"cpu\"`, then `model.to(device)` and `xb, yb = xb.to(device), yb.to(device)` inside the loop.",
      "**Log both train and validation** metrics every epoch, and save the best checkpoint (by validation) rather than the last.",
      "**Keep the test set untouched** until the very end ([why](ml.html#splits)).",
      "**Grad clipping** for RNNs and transformers: `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` between `backward()` and `step()`.",
      "**Mixed precision and `torch.compile`** give large speed-ups on GPUs, but get the plain loop correct first.",
    ]],
    ["danger", "Putting a `softmax` before `nn.CrossEntropyLoss`. That loss applies `log_softmax` itself and expects raw logits; applying softmax twice squashes the gradients and training crawls. Likewise `BCEWithLogitsLoss` wants logits, not sigmoids. And passing one-hot labels or float labels to `CrossEntropyLoss` is a common shape/dtype error: it wants `int64` class ids."],
    ["danger", "Forgetting `model.eval()` during validation (dropout keeps randomly zeroing units, batch-norm uses the batch's statistics) or forgetting `model.train()` afterwards. Both silently give wrong numbers."],
    ["tip", "Practice for this section: [SGD with Momentum and Adam Steps](practice.html#p/sgd-momentum-adam) for the update rules the optimiser applies. When training misbehaves, see [Debugging Neural Network Training](dl.html#debugging-training)."],
  ],
},
{
  id: "transfer-learning", group: "In Practice", title: "Transfer Learning & Fine-Tuning",
  sub: "Start from a network that has already learned from a huge dataset instead of from random weights.",
  tags: ["pytorch", "pretrained"],
  blocks: [
    ["p", "Training a good image or language model from scratch takes millions of examples and days or weeks of GPU time. Most people never do that. They **start from a pretrained model**: one someone else trained on a huge dataset (ImageNet for vision, a giant text corpus for language) and **adapt** it to their own, smaller problem. This is **transfer learning**, and it is how almost all practical deep learning is done."],
    ["p", "It works because of what [deep networks learn](dl.html#what-is-dl): the early layers of a vision network detect edges, colours and textures, useful for any image; only the last layers are specific to the original labels. A language model that has read the web already knows grammar and a great deal about the world. So we keep the useful general layers and replace or retrain the last ones."],
    ["h", "Two strategies"],
    ["table", ["", "Feature extraction", "Fine-tuning"], [
      ["What you train", "Only a new final layer (the **head**); the pretrained body is **frozen**", "The head, plus some or all of the pretrained layers, with a **small** learning rate"],
      ["Cost", "Very cheap; you can even precompute the features once", "More compute and memory"],
      ["Risk", "Low; cannot damage the pretrained knowledge", "Overfitting on small data, and 'catastrophic forgetting' if the learning rate is too high"],
      ["Best when", "Very little data, or your task is close to the original", "More data, or your images/text differ from the original domain"],
    ]],
    ["p", "A common recipe: first train just the head until it is reasonable (a few epochs), then unfreeze the top blocks of the body and continue with a learning rate 10 to 100 times smaller than the head's. Early layers are more general and need less change than late ones."],
    ["h", "Freezing and replacing the head in PyTorch"],
    ["p", "torchvision's ResNet-18 is built here with `weights=None`, which creates the right architecture with random weights and downloads nothing. (With real weights the steps are identical; see the last block.) We freeze the backbone, replace `fc` with a new 3-class layer, take one training step on a synthetic batch, and check what changed. Pay attention to the last line: **frozen is not the same as unchanged**."],
    ["code", R`
import torch, torch.nn as nn, torchvision

torch.manual_seed(0)
net = torchvision.models.resnet18(weights=None)        # architecture only: no download, random weights here

for p in net.parameters():                             # 1) freeze everything
    p.requires_grad = False
net.fc = nn.Linear(net.fc.in_features, 3)              # 2) new head: 512 -> 3 classes (trainable by default)

total = sum(p.numel() for p in net.parameters())
trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"parameters: {total:,}   trainable: {trainable:,}  ({100 * trainable / total:.3f}%)")

opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)   # 3) optimise only the head
conv_before = net.layer1[0].conv1.weight.clone()
fc_before = net.fc.weight.clone()
bn = net.layer1[0].bn1
mean_before = bn.running_mean.clone()

net.train()
x, y = torch.randn(8, 3, 64, 64), torch.randint(0, 3, (8,))      # a synthetic batch
loss = nn.functional.cross_entropy(net(x), y)
opt.zero_grad(); loss.backward(); opt.step()
print("conv weight changed:", not torch.equal(conv_before, net.layer1[0].conv1.weight))
print("head weight changed:", not torch.equal(fc_before, net.fc.weight))
print("BatchNorm running mean changed even though frozen:", not torch.equal(mean_before, bn.running_mean))

# fix: keep the frozen backbone in eval mode so its BatchNorm statistics stay as pretrained
def set_backbone_eval(m):
    m.train()
    for name, child in m.named_children():
        if name != "fc":
            child.eval()
set_backbone_eval(net)
mean_before = bn.running_mean.clone()
net(x)
print("with the backbone in eval mode, running mean changed:", not torch.equal(mean_before, bn.running_mean))

# 4) later: unfreeze the last block and use a smaller learning rate for it (discriminative learning rates)
for p in net.layer4.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW([{"params": net.layer4.parameters(), "lr": 1e-4}, {"params": net.fc.parameters(), "lr": 1e-3}])
print("trainable after unfreezing layer4:", f"{sum(p.numel() for p in net.parameters() if p.requires_grad):,}")`, "parameters: 11,178,051   trainable: 1,539  (0.014%)\nconv weight changed: False\nhead weight changed: True\nBatchNorm running mean changed even though frozen: True\nwith the backbone in eval mode, running mean changed: False\ntrainable after unfreezing layer4: 8,395,267", "PyTorch", "local"],
    ["p", "That is the BatchNorm trap: with `requires_grad=False` the weights are frozen, but a layer in training mode still updates its **running statistics** every forward pass, which slowly drifts the backbone away from what it was trained with. Put frozen blocks in `eval()` mode."],
    ["h", "Does transfer actually help? A toy experiment"],
    ["p", "Real pretrained weights need a download, so here is the same idea in miniature, using only the digits that ship with scikit-learn. Step 1 (the \"pretraining\"): train a small CNN to recognise the ten digits using 1,000 images. Step 2 (the \"new task\"): learn a different question, **is the digit odd or even?**, from only **10 labelled examples**, in three ways: from scratch, with the pretrained body frozen and a new head, and by fine-tuning everything. Averages over 5 random choices of the 10 examples:"],
    ["code", R`
import copy, numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32).view(-1, 1, 8, 8); y = torch.tensor(y)
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
src, test = perm[:1000], perm[1100:]                          # pretraining pool and a test set

def make_body():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())   # 128 features
def fit(model, params, X, y, epochs, lr):
    opt = torch.optim.Adam(params, lr)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(model(X), y); opt.zero_grad(); loss.backward(); opt.step()
def acc(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

torch.manual_seed(0)
body = make_body(); pre = nn.Sequential(body, nn.Linear(128, 10))
fit(pre, pre.parameters(), X[src], y[src], 150, 3e-3)           # step 1: "pretraining" on the 10-digit task
print("pretraining task, test accuracy:", round(acc(pre, X[test], y[test]), 3))

results = {}
for name in ["from scratch", "frozen body + new head", "fine-tune everything"]:
    accs = []
    for seed in range(5):
        torch.manual_seed(seed)
        b = make_body() if name == "from scratch" else copy.deepcopy(body)
        m = nn.Sequential(b, nn.Linear(128, 2))
        if name == "frozen body + new head":
            for p in b.parameters(): p.requires_grad = False
            params, lr = m[1].parameters(), 1e-2
        elif name == "fine-tune everything":
            params, lr = [{"params": b.parameters(), "lr": 3e-4}, {"params": m[1].parameters(), "lr": 3e-3}], 3e-3
        else:
            params, lr = m.parameters(), 3e-3
        idx = src[torch.randperm(1000, generator=torch.Generator().manual_seed(seed))[:10]]       # only 10 labelled examples
        fit(m, params, X[idx], y[idx] % 2, 100, lr)                                              # new task: odd / even
        accs.append(acc(m, X[test], y[test] % 2))
    results[name] = float(np.mean(accs))
    print(f"{name:24s} accuracy with 10 examples: {results[name]:.3f}")`, "pretraining task, test accuracy: 0.984\nfrom scratch             accuracy with 10 examples: 0.760\nfrozen body + new head   accuracy with 10 examples: 0.797\nfine-tune everything     accuracy with 10 examples: 0.788", "PyTorch", "local"],
    ["p", "The pretrained body gives a small but consistent gain even on this toy: its features already separate the digit shapes. With a real network pretrained on millions of images, the difference is dramatic, often the difference between a model that works and one that does not."],
    ["h", "With real pretrained weights"],
    ["p", "The only change is asking torchvision for its pretrained weights. The first call **downloads** them (about 45 MB for ResNet-18) and caches them in `~/.cache/torch`, so this block is not run here and has no output. Two details matter: use the **same preprocessing** the weights were trained with (`weights.transforms()` gives the resize, crop and normalisation), and the 1,000-class head is replaced by your own."],
    ["code", R`
import torch.nn as nn, torchvision
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT                 # pretrained on ImageNet (downloads the weights on first use)
net = resnet18(weights=weights)
preprocess = weights.transforms()                  # resize, center-crop, normalise: use it on your own images!

for p in net.parameters():
    p.requires_grad = False
net.fc = nn.Linear(net.fc.in_features, 5)          # e.g. 5 flower species

# from here: the same training loop as before, optimising net.fc.parameters()`, "", "PyTorch", "local"],
    ["p", "For text the pattern is the same with a different library. With Hugging Face `transformers` you load a pretrained language model, add a small classification head, and fine-tune on your labelled sentences:"],
    ["code", R`
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "distilbert-base-uncased"                                  # a small pretrained encoder (downloads on first use)
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)   # new, randomly initialised head

batch = tokenizer(["great film", "dull and far too long"], padding=True, return_tensors="pt")
logits = model(**batch).logits                                    # shape (2, 2): ready for cross-entropy and the usual loop`, "", "PyTorch", "local"],
    ["table", ["Your data", "Similar to the pretraining data", "Different from the pretraining data"], [
      ["Small", "Freeze the body, train the head", "Freeze early layers, fine-tune later ones with a small learning rate; use heavy augmentation"],
      ["Large", "Fine-tune the whole model with a small learning rate", "Fine-tune everything; consider longer training"],
    ]],
    ["tip", "Practical rules: use a **learning rate 10-100x smaller** than you would from scratch for pretrained layers; **match the preprocessing** of the pretraining data exactly (image size, mean/std, tokenizer); fine-tune with a [warmup and decaying schedule](dl.html#lr-schedules); and watch the validation set closely, since a small dataset overfits fast."],
    ["note", "For very large language models, updating every weight is too expensive, so we fine-tune with far fewer trainable numbers: [LoRA and PEFT](llm.html#lora), and we usually start with [supervised fine-tuning on instructions](llm.html#sft). It is the same transfer-learning idea at a bigger scale."],
  ],
},
{
  id: "debugging-training", group: "In Practice", title: "Debugging Neural Network Training",
  sub: "A practical checklist for when the loss will not go down, goes to NaN, or the model trains but does not work.",
  tags: ["debugging", "practice"],
  blocks: [
    ["p", "Neural network bugs are unusual: the code runs, no error appears, and the model is simply... bad. A wrong sign, swapped labels or a missing `eval()` produces a slightly worse accuracy, not a crash. So debugging is a skill with its own routine. Andrej Karpathy's famous advice (\"A Recipe for Training Neural Networks\", 2019) is the backbone of what follows: **start simple, verify each piece, and only then add complexity.**"],
    ["h", "The checklist, in order"],
    ["steps", [
      "**Look at the data.** Plot or print a few raw inputs and their labels. Check shapes, value ranges, class balance, and that labels still match inputs after any shuffling or augmentation. Most bugs live here.",
      "**Check the initial loss.** For $C$ balanced classes a fresh network should give a loss near $\\ln C$ (2.303 for 10 classes). If it starts at 25, your initialisation or labels are broken.",
      "**Overfit one tiny batch.** Train on 8-32 examples until the loss is almost 0. If it cannot, you have a bug in the model, loss, optimiser or data pipeline.",
      "**Scale up one thing at a time**: the full training set, then regularisation, then a bigger model, then augmentation. Do not change five things between runs.",
      "**Read the curves.** Compare train and validation loss (table below) to decide between more capacity, more regularisation or more data.",
      "**Tune last**: learning rate first, then batch size, schedule, and everything else.",
    ]],
    ["h", "1. Sanity: initial loss and overfitting one batch"],
    ["p", "Here is the routine on a NumPy softmax classifier for the digits. The healthy run drives the loss on a batch of 16 images to nearly zero. The second run has the classic sign bug (it *adds* the gradient instead of subtracting it): the loss explodes, a clear signal that the update is wrong."],
    ["code", R`
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True); X = X / 16.0

def softmax(z):
    z = z - z.max(1, keepdims=True)                    # stability
    e = np.exp(z); return e / e.sum(1, keepdims=True)

def loss_grad(W, X, y):
    P = softmax(X @ W); n = len(y)
    L = -np.log(P[np.arange(n), y] + 1e-12).mean()
    G = P.copy(); G[np.arange(n), y] -= 1              # dL/dlogits = P - onehot
    return L, X.T @ G / n

W0 = np.zeros((64, 10))
print("initial loss:", round(loss_grad(W0, X, y)[0], 4), "   ln(10) =", round(np.log(10), 4))

rng = np.random.default_rng(0)
idx = rng.choice(len(X), 16, replace=False)
xb, yb = X[idx], y[idx]
for name, sign in [("correct update", -1), ("sign bug (+grad)", +1)]:
    W = W0.copy()
    losses = []
    for step in range(200):
        L, G = loss_grad(W, xb, yb)
        W += sign * 0.5 * G                           # sign = -1 is gradient DEscent
        losses.append(L)
    print(f"{name:17s} loss at step 0 / 50 / 199: {losses[0]:.3f} / {losses[50]:.3f} / {losses[199]:.3f}")`, "initial loss: 2.3026    ln(10) = 2.3026\ncorrect update    loss at step 0 / 50 / 199: 2.303 / 0.136 / 0.029\nsign bug (+grad)  loss at step 0 / 50 / 199: 2.303 / 27.631 / 27.631"],
    ["h", "2. The learning-rate range test"],
    ["p", "The learning rate is the first hyper-parameter to get right. The **LR finder** (Leslie Smith, popularised by fast.ai) trains for a few hundred steps while the learning rate grows exponentially, and records the loss. The loss stays flat while the rate is too small, falls fastest at a good rate, then blows up when it is too big. Pick a rate a bit **below the blow-up**, roughly where the loss was falling fastest."],
    ["code", R`
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True); X = X / 16.0
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def loss_grad(W, X, y):
    P = softmax(X @ W); n = len(y)
    G = P.copy(); G[np.arange(n), y] -= 1
    return -np.log(P[np.arange(n), y] + 1e-12).mean(), X.T @ G / n

rng = np.random.default_rng(0)
W = np.zeros((64, 10)); lr = 1e-4
lrs, losses = [], []
for step in range(150):
    idx = rng.choice(len(X), 64)
    L, G = loss_grad(W, X[idx], y[idx])
    lrs.append(lr); losses.append(L)
    if not np.isfinite(L) or L > 4 * min(losses) + 5:             # stop once it has clearly blown up
        break
    W -= lr * G
    lr *= 1.12                                                    # exponential sweep

lrs, losses = np.array(lrs), np.array(losses)
smooth = np.convolve(losses, np.ones(5) / 5, mode="same")
slope = np.gradient(smooth, np.log(lrs))                          # how fast does the loss fall per factor of lr?
steepest = int(np.argmin(slope[2:-2])) + 2
for i in range(0, len(lrs), 10):
    print(f"lr {lrs[i]:8.4f}   loss {losses[i]:.3f}")
print(f"blew up at lr about {lrs[-1]:.1f};  loss fell fastest around lr {lrs[steepest]:.2f};  a safe pick is about {lrs[steepest] / 3:.2f}")`, "lr   0.0001   loss 2.303\nlr   0.0003   loss 2.302\nlr   0.0010   loss 2.301\nlr   0.0030   loss 2.297\nlr   0.0093   loss 2.288\nlr   0.0289   loss 2.253\nlr   0.0898   loss 2.158\nlr   0.2788   loss 1.882\nlr   0.8658   loss 1.443\nlr   2.6892   loss 0.778\nblew up at lr about 6.7;  loss fell fastest around lr 1.09;  a safe pick is about 0.36"],
    ["h", "3. Gradient norms: is every layer learning?"],
    ["p", "Print the size of the gradient in every layer. A healthy network has norms of a similar magnitude in every layer. Norms that are tiny everywhere, or that shrink steadily toward the input, are [vanishing gradients](dl.html#vanishing-gradients) (bad [initialization](dl.html#initialization), saturating activations); huge values mean exploding ones. A layer with exactly zero gradient is dead or disconnected. Compare two initialisations of a 7-layer tanh network:"],
    ["code", R`
import numpy as np

def layer_grad_norms(weight_std, depth=7, n=64, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, weight_std(n), (n, n)) for _ in range(depth)]
    x = rng.normal(size=(batch, n))
    hs = [x]
    for W in Ws:                                      # forward
        hs.append(np.tanh(hs[-1] @ W))
    g = np.ones_like(hs[-1]) / batch                 # pretend dL/d(output) = 1/batch
    norms = []
    for W, h_in, h_out in zip(Ws[::-1], hs[-2::-1], hs[:0:-1]):      # backward
        gz = g * (1 - h_out ** 2)                    # through tanh
        norms.append(np.linalg.norm(h_in.T @ gz))    # gradient of the loss wrt this layer's weights
        g = gz @ W.T
    return norms[::-1]                               # layer 1 first

for name, std in [("tiny init (std 0.01)", lambda n: 0.01), ("Xavier init (std 1/sqrt(n))", lambda n: 1 / np.sqrt(n))]:
    print(f"{name:28s}", " ".join(f"{v:9.2e}" for v in layer_grad_norms(std)))
print("(columns: layer 1 ... layer 7, the output end)")
print("tiny init is about", f"{layer_grad_norms(lambda n: 1 / np.sqrt(n))[0] / layer_grad_norms(lambda n: 0.01)[0]:.1e}", "times weaker at layer 1")`, "tiny init (std 0.01)          2.41e-06  2.16e-06  2.45e-06  2.74e-06  2.73e-06  3.46e-06  3.64e-06\nXavier init (std 1/sqrt(n))   2.91e+00  2.25e+00  2.21e+00  2.72e+00  2.82e+00  3.57e+00  3.76e+00\n(columns: layer 1 ... layer 7, the output end)\ntiny init is about 1.2e+06 times weaker at layer 1"],
    ["h", "4. NaN and infinity"],
    ["p", "A loss that suddenly becomes `nan` has usually done one of four things: a learning rate that is too high (weights explode), a `log(0)` or `0/0` somewhere, an `exp` that overflowed, or bad input data (a NaN or inf in the dataset). Find the **first** place a non-finite value appears, not the last. And use the numerically stable forms:"],
    ["code", R`
import numpy as np

z = np.array([1000.0, 0.0])
with np.errstate(all="ignore"):
    naive = np.exp(z) / np.exp(z).sum()                               # exp(1000) overflows to inf -> inf/inf = nan
print("naive softmax:       ", naive)
stable = np.exp(z - z.max()) / np.exp(z - z.max()).sum()              # subtract the max first
print("stable softmax:      ", stable)

p = np.array([0.0, 1.0])
with np.errstate(divide="ignore"):
    print("log of a zero probability:", np.log(p))                    # -inf, and 0 * -inf = nan in the loss
print("with clipping:           ", np.log(np.clip(p, 1e-12, 1)))
# best of all: work with logits and log-sum-exp (what nn.CrossEntropyLoss does internally)
logits = np.array([1000.0, 0.0])
logsumexp = logits.max() + np.log(np.exp(logits - logits.max()).sum())
print("log-softmax from logits:", logits - logsumexp)

def find_nonfinite(named_arrays):
    return [name for name, a in named_arrays.items() if not np.all(np.isfinite(a))]
print("check helper flags:", find_nonfinite({"inputs": np.array([1.0, 2.0]), "labels": np.array([0.0, np.nan]), "logits": logits}))`, "naive softmax:        [nan  0.]\nstable softmax:       [1. 0.]\nlog of a zero probability: [-inf   0.]\nwith clipping:            [-27.63102112   0.        ]\nlog-softmax from logits: [    0. -1000.]\ncheck helper flags: ['labels']"],
    ["h", "In PyTorch: tools for finding the culprit"],
    ["code", R`
import torch, torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = torch.randn(64, 10), torch.randint(0, 2, (64,))

for step in range(3):
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)       # returns the norm BEFORE clipping
    per_layer = {n: round(p.grad.norm().item(), 3) for n, p in model.named_parameters()}
    print(f"step {step}  loss {loss.item():.3f}  total grad norm {norm.item():.3f}  finite: {bool(torch.isfinite(loss))}")
    opt.step()
print("per-layer gradient norms:", per_layer)

# anomaly detection turns a silent NaN into an error that names the operation which created it
x = torch.tensor([0.0], requires_grad=True)
try:
    with torch.autograd.set_detect_anomaly(True):
        (torch.sqrt(x) * 0).sum().backward()             # sqrt'(0) = inf, and inf * 0 = nan in the backward pass
except RuntimeError as e:
    print("anomaly detection says:", str(e).split(".")[0])`, "step 0  loss 0.737  total grad norm 0.354  finite: True\nstep 1  loss 0.725  total grad norm 0.316  finite: True\nstep 2  loss 0.715  total grad norm 0.288  finite: True\nper-layer gradient norms: {'0.weight': 0.16, '0.bias': 0.035, '2.weight': 0.233, '2.bias': 0.042}\nanomaly detection says: Function 'SqrtBackward0' returned nan values in its 0th output", "PyTorch", "local"],
    ["tip", "Log the **total gradient norm** every step in real projects: a spike a few steps before a NaN tells you where to look. `set_detect_anomaly` is slow, so use it only to hunt a bug."],
    ["h", "Reading the curves"],
    ["table", ["What you see", "Likely cause", "What to try"], [
      ["Loss is flat from the very start, near $\\ln C$", "Learning rate too low; or dead network; or labels random/mismatched; or a bug (e.g. no `backward`)", "Overfit one batch; raise the LR; check labels and the optimiser's parameter list"],
      ["Loss jumps up and down wildly, or rises", "Learning rate too high; a sign bug; unscaled inputs", "Lower the LR 3-10x; clip gradients; normalise inputs; try a warmup ([schedules](dl.html#lr-schedules))"],
      ["Loss becomes NaN / inf", "LR too high; `log(0)`; overflow; bad data", "See section 4; add `isfinite` checks; use stable losses"],
      ["Train loss low, validation loss high and rising", "Overfitting", "More data and augmentation; [dropout, weight decay, early stopping](dl.html#dl-regularization); smaller model"],
      ["Both train and validation loss high and similar", "Underfitting", "Bigger model; train longer; lower regularisation; better features; check the LR is not too low"],
      ["Validation better than train", "Dropout/augmentation active only in training; or validation set is easier or leaked", "Evaluate both in eval mode; double-check the split"],
      ["Loss falls but accuracy is stuck", "Class imbalance (predicting the majority); wrong metric; label noise", "Look at the confusion matrix; weighted loss; check the data"],
      ["Great validation score, poor in production", "Data leakage; train/serve preprocessing mismatch", "Re-check the split ([leakage](ml.html#splits)); use the identical transform path in both"],
      ["Training gets slower or the loss plateaus at a bad value", "Poor initialisation, vanishing gradients, learning-rate schedule too aggressive", "Print per-layer gradient norms; [normalisation](dl.html#normalization); He/Xavier init"],
    ]],
    ["h", "A short list of classic bugs"],
    ["list", [
      "Forgot `optimizer.zero_grad()` (gradients accumulate), or called it between `backward()` and `step()`.",
      "Forgot `model.eval()` (and `torch.no_grad()`) for validation, or forgot `model.train()` afterwards.",
      "Applied softmax or sigmoid before a loss that already does it ([PyTorch training](dl.html#pytorch-training)).",
      "Labels and inputs shuffled separately, so they no longer match.",
      "Normalised the training data but not the validation or test data (or computed the statistics on all of it: leakage).",
      "Wrong dimension in a reduction (`dim`), or broadcasting that silently produced a `(N, N)` matrix instead of `(N,)`.",
      "Learning rate set for one optimiser but a different default used (Adam wants about 1e-3, SGD often 1e-2 to 1e-1).",
      "Data augmentation applied to the validation set; or no augmentation where the labels need it.",
      "A very small batch with BatchNorm (use [LayerNorm or GroupNorm](dl.html#normalization)), or a batch of one in training mode.",
    ]],
    ["note", "Before launching a long run: set the seed; log the config; run 1 epoch on a subset end to end (data → loss → validation → checkpoint) to catch plumbing errors; and make sure you can resume from a checkpoint. An hour of checks saves a night of wasted GPU time."],
  ],
}
);
