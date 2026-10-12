/* Practice: CNNs & RNNs (deep learning part 2)
   Each problem: q, ex, hint, brute (plain Python), opt (NumPy), run + out (tests), dry, expl, take, mist. */
PRAC.add("CNNs & RNNs", [
{
  id: "conv2d-naive", t: "2-D Convolution with Stride and Padding", d: "Medium", p: "CNN",
  q: "A convolution layer slides a small **kernel** over an image. Implement `conv2d(x, k, stride=1, pad=0)` for one single-channel image `x` of shape `(H, W)` and one kernel `k` of shape `(kh, kw)`.\n\nFirst surround the image with `pad` rows and columns of zeros on every side. Then place the kernel at positions that are `stride` apart, multiply the covered patch by the kernel element by element and add up the products. That sum is one output pixel. Do **not** flip the kernel (this is the cross-correlation that deep learning libraries call convolution).\n\nThe output has shape `(oh, ow)` with\n\n$o = \\lfloor (n + 2p - k)/s \\rfloor + 1$\n\nfor each of height and width. Windows that would stick out past the padded border are not used.",
  ex: [["x = [[1,2,3],[4,5,6],[7,8,9]], k = [[1,0],[0,-1]]", "[[-4, -4], [-4, -4]]", "At the top-left: 1·1 + 5·(−1) = −4. The output is (3−2)/1+1 = 2 in each direction."],
    ["same x, k = ones(3,3), pad = 1", "[[12, 21, 16], [27, 45, 33], [24, 39, 28]]", "With padding 1 and a 3×3 kernel the output keeps the 3×3 size; each cell is the sum of its neighbourhood (zeros outside)."]],
  c: ["`x` and `k` are float arrays; `k` is no larger than the padded `x`", "`stride >= 1`, `pad >= 0`", "Return a float NumPy array of shape `(oh, ow)`"],
  hint: "Pad with `np.pad(x, pad)`. For output cell `(i, j)` the patch is `xp[i*stride : i*stride+kh, j*stride : j*stride+kw]`. To avoid the double loop, `np.lib.stride_tricks.sliding_window_view(xp, (kh, kw))` gives every patch at once (shape `(H', W', kh, kw)`); slice `[::stride, ::stride]` for the stride.",
  brute: {
    name: "Plain Python loops",
    idea: "Two loops over the output positions and two more over the kernel. This is the definition written straight down; it is the version to use as a reference when testing a fast one.",
    code: R`
import numpy as np

def conv2d(x, k, stride=1, pad=0):
    x, k = np.asarray(x, dtype=float), np.asarray(k, dtype=float)
    xp = np.pad(x, pad)                                   # zeros around the border
    kh, kw = k.shape
    oh = (xp.shape[0] - kh) // stride + 1
    ow = (xp.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += xp[i * stride + u, j * stride + v] * k[u, v]
            out[i, j] = total
    return out`,
    tc: "O(oh · ow · kh · kw)", tw: "Every output cell multiplies and adds kh·kw numbers: the true work of a convolution.", sc: "O(oh · ow)", sw: "The output (plus the padded copy).",
    bad: "Correct but slow: four nested Python loops. A real layer repeats this for many channels and filters, so it is hopelessly slow without vectorisation.",
  },
  opt: {
    name: "Vectorized with sliding windows",
    idea: "`sliding_window_view` presents all patches as one array of shape `(H', W', kh, kw)` without copying. Multiply by the kernel (broadcast) and sum the last two axes, after taking every `stride`-th patch.",
    steps: ["Convert to float arrays and pad: `xp = np.pad(x, pad)`.", "`win = sliding_window_view(xp, k.shape)` has shape `(H-kh+1, W-kw+1, kh, kw)`.", "Keep every `stride`-th window in both directions: `win[::stride, ::stride]`.", "`(win * k).sum(axis=(-2, -1))` multiplies each patch by the kernel and adds."],
    code: R`
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def conv2d(x, k, stride=1, pad=0):
    x, k = np.asarray(x, dtype=float), np.asarray(k, dtype=float)
    xp = np.pad(x, pad)
    win = sliding_window_view(xp, k.shape)[::stride, ::stride]      # (oh, ow, kh, kw): all patches, no copy
    return (win * k).sum(axis=(-2, -1))                              # multiply by the kernel, add up`,
    tc: "O(oh · ow · kh · kw)", tw: "Same arithmetic, but done inside NumPy in C.", sc: "O(oh · ow · kh · kw)", sw: "The product `win * k` is materialised at this size.",
    viz: "dl2_conv",
  },
  run: R`
img = np.arange(1, 10, dtype=float).reshape(3, 3)
print(conv2d(img, np.array([[1., 0.], [0., -1.]])).astype(int).tolist())
print(conv2d(img, np.ones((3, 3)), pad=1).astype(int).tolist())
print(conv2d(img, np.ones((2, 2)), stride=2).astype(int).tolist())
print(conv2d(np.ones((7, 7)), np.ones((3, 3)), stride=2, pad=1).shape)
edge = np.array([[-1., 0., 1.]] * 3)
step = np.tile([0., 0., 0., 9., 9., 9.], (6, 1))                 # dark left half, bright right half
print(conv2d(step, edge)[0].astype(int).tolist())`,
  out: R`
[[-4, -4], [-4, -4]]
[[12, 21, 16], [27, 45, 33], [24, 39, 28]]
[[12]]
(4, 4)
[0, 27, 27, 0]`,
  dryOn: "x = [[1,2,3],[4,5,6],[7,8,9]], k = [[1,0],[0,-1]], stride 1, pad 0",
  dry: { h: ["output cell", "patch", "products", "sum"], r: [["(0,0)", "[[1,2],[4,5]]", "1·1 + 2·0 + 4·0 + 5·(−1)", "**−4**"], ["(0,1)", "[[2,3],[5,6]]", "2·1 + 6·(−1)", "**−4**"], ["(1,0)", "[[4,5],[7,8]]", "4·1 + 8·(−1)", "**−4**"], ["(1,1)", "[[5,6],[8,9]]", "5·1 + 9·(−1)", "**−4**"]] },
  expl: [
    ["xp = np.pad(x, pad)", "Zeros around the border. This is what lets a 3×3 kernel produce an output as big as the input."],
    ["sliding_window_view(xp, k.shape)", "A view whose entry `[i, j]` is the `kh × kw` patch starting at `(i, j)`. No data is copied."],
    ["[::stride, ::stride]", "Keeps every `stride`-th patch position, which is exactly what stride means."],
    ["(win * k).sum(axis=(-2, -1))", "Broadcast the kernel over every patch, multiply, and add the last two axes."],
  ],
  take: "**A convolution is a dot product between the kernel and every patch.** Output size: ⌊(n + 2p − k)/s⌋ + 1.",
  mist: ["Flipping the kernel (that is the textbook convolution; deep learning does not flip).", "Padding only on one side, or forgetting that padding adds `2p` to the size.", "Mixing up `stride` with the kernel size: patches overlap when `stride < k`.", "Writing the output size as `(n - k)/s` and forgetting the `+ 1`."],
},
{
  id: "max-pool-2d", t: "Max Pooling", d: "Easy", p: "CNN",
  q: "Pooling shrinks a feature map. Implement `max_pool2d(x, size=2, stride=None)`.\n\n`x` has shape `(H, W)` or `(C, H, W)` (pool each channel separately). Slide a `size × size` window over the last two axes, moving by `stride` (default: `stride = size`, so windows do not overlap), and keep the **maximum** of each window. Windows that do not fit entirely inside the map are skipped, so the output size is $\\lfloor (n - size)/stride \\rfloor + 1$ in each direction.",
  ex: [["x = [[1,3,2,1],[4,6,5,2],[2,1,7,8],[0,2,9,4]], size = 2", "[[6, 5], [2, 9]]", "Top-left window {1,3,4,6} has maximum 6; bottom-right {7,8,9,4} has maximum 9."],
    ["a 5×5 map, size = 2", "a 2×2 output", "(5−2)//2 + 1 = 2: the last row and column do not fit a whole window and are dropped."]],
  c: ["`size >= 1`, `stride >= 1`, the map is at least `size × size`", "Return a NumPy array with the same number of leading axes as `x`"],
  hint: "`sliding_window_view(x, (size, size), axis=(-2, -1))` gives every window; keep every `stride`-th one in both directions and take the maximum over the last two axes.",
  brute: {
    name: "Plain Python loops",
    idea: "Loop over the window positions and take `max` of the patch. Handle the channel axis by pooling each 2-D slice in turn.",
    code: R`
import numpy as np

def max_pool2d(x, size=2, stride=None):
    x = np.asarray(x, dtype=float)
    if x.ndim == 3:
        return np.stack([max_pool2d(ch, size, stride) for ch in x])
    stride = stride or size
    oh = (x.shape[0] - size) // stride + 1
    ow = (x.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = x[i * stride : i * stride + size, j * stride : j * stride + size].max()
    return out`,
    tc: "O(oh · ow · size²)", tw: "Each window is scanned once.", sc: "O(oh · ow)", sw: "The output.",
    bad: "Fine for one small map, but Python loops over every window are slow on real feature maps with dozens of channels.",
  },
  opt: {
    name: "Vectorized with sliding windows",
    idea: "All windows at once as a view, thinned by the stride, then one `max` over the window axes. Because the window axes are the last two, the same line works with or without a channel axis.",
    steps: ["`stride = stride or size`.", "`win = sliding_window_view(x, (size, size), axis=(-2, -1))`: shape `(..., H-size+1, W-size+1, size, size)`.", "`win[..., ::stride, ::stride, :, :]` keeps the windows we want.", "`.max(axis=(-2, -1))` takes the maximum of each."],
    code: R`
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def max_pool2d(x, size=2, stride=None):
    x = np.asarray(x, dtype=float)
    stride = stride or size
    win = sliding_window_view(x, (size, size), axis=(-2, -1))       # (..., H', W', size, size)
    return win[..., ::stride, ::stride, :, :].max(axis=(-2, -1))`,
    tc: "O(oh · ow · size²)", tw: "One pass over every window, in C.", sc: "O(oh · ow)", sw: "A view, so no copy of the windows; the output is the only new array.",
    viz: "dl2_pool",
  },
  run: R`
x = np.array([[1, 3, 2, 1], [4, 6, 5, 2], [2, 1, 7, 8], [0, 2, 9, 4]])
print(max_pool2d(x).astype(int).tolist())
print(max_pool2d(np.arange(25).reshape(5, 5), size=2).astype(int).tolist())
print(max_pool2d(np.arange(16).reshape(4, 4), size=2, stride=1).astype(int).tolist())
cube = np.stack([x, -x])
print(max_pool2d(cube).astype(int).tolist())
print(max_pool2d(np.ones((3, 8, 8))).shape)`,
  out: R`
[[6, 5], [2, 9]]
[[6, 8], [16, 18]]
[[5, 6, 7], [9, 10, 11], [13, 14, 15]]
[[[6, 5], [2, 9]], [[-1, -1], [0, -4]]]
(3, 4, 4)`,
  dryOn: "x = [[1,3,2,1],[4,6,5,2],[2,1,7,8],[0,2,9,4]], size 2, stride 2",
  dry: { h: ["window", "values", "max"], r: [["rows 0-1, cols 0-1", "1, 3, 4, 6", "**6**"], ["rows 0-1, cols 2-3", "2, 1, 5, 2", "**5**"], ["rows 2-3, cols 0-1", "2, 1, 0, 2", "**2**"], ["rows 2-3, cols 2-3", "7, 8, 9, 4", "**9**"]] },
  expl: [
    ["stride = stride or size", "The usual default: non-overlapping windows."],
    ["sliding_window_view(x, (size, size), axis=(-2, -1))", "Windows are formed only over the last two axes, so a leading channel axis is carried along untouched."],
    ["[..., ::stride, ::stride, :, :]", "Pick every `stride`-th window position; keep both window axes whole."],
    [".max(axis=(-2, -1))", "The maximum of every window."],
  ],
  take: "**Pooling is a window reduction with no weights**: the same code with `.mean` gives average pooling.",
  mist: ["Taking the maximum over the whole channel instead of each window.", "Forgetting the default `stride = size` (a stride of 1 is *overlapping* pooling and gives a much bigger output).", "Padding or rounding up so that partial windows are included; here they are dropped."],
},
{
  id: "conv-im2col", t: "Convolution as a Matrix Multiply (im2col)", d: "Hard", p: "CNN",
  q: "Real deep learning libraries turn convolution into one large matrix multiplication. Implement `conv_forward(x, W, b, stride=1, pad=0)` for a **multi-channel** input.\n\n- `x`: input of shape `(C, H, W_in)`\n- `W`: filters of shape `(F, C, kh, kw)` (F filters, each looking at all C channels)\n- `b`: biases of shape `(F,)`\n\nReturn the output of shape `(F, oh, ow)`, where `out[f, i, j]` is the sum over channels and kernel positions of the patch times the filter, plus `b[f]`. Zero-pad `pad` pixels on each side of the height and width first.\n\nDo it the **im2col** way: copy every patch into one column of a matrix `cols` of shape `(C·kh·kw, oh·ow)`, flatten the filters to `(F, C·kh·kw)`, and compute `W_flat @ cols + b`.",
  ex: [["x of shape (2, 4, 4), W of shape (3, 2, 3, 3), stride 1, pad 0", "output shape (3, 2, 2)", "oh = ow = (4−3)/1 + 1 = 2. `cols` is (18, 4); `W_flat` is (3, 18); their product is (3, 4), reshaped to (3, 2, 2)."],
    ["x of shape (1, 5, 5), W = one 3×3 filter of ones, pad 1", "a 5×5 map of neighbourhood sums", "Padding 1 keeps the size."]],
  c: ["`x`, `W`, `b` are float arrays", "Channels of `x` and `W` agree", "Return an array of shape `(F, oh, ow)`"],
  hint: "`sliding_window_view(xp, (kh, kw), axis=(1, 2))` has shape `(C, H', W', kh, kw)`. Slice the stride, then move the axes so that `(C, kh, kw)` come first and `(oh, ow)` last, and `reshape` to `(C*kh*kw, oh*ow)`. The filters are `W.reshape(F, -1)`.",
  brute: {
    name: "Plain Python loops",
    idea: "The definition with six nested loops: for each filter, each output position, each channel and each kernel offset. Slow, but obviously correct.",
    code: R`
import numpy as np

def conv_forward(x, W, b, stride=1, pad=0):
    x, W, b = np.asarray(x, float), np.asarray(W, float), np.asarray(b, float)
    F, C, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh = (xp.shape[1] - kh) // stride + 1
    ow = (xp.shape[2] - kw) // stride + 1
    out = np.zeros((F, oh, ow))
    for f in range(F):
        for i in range(oh):
            for j in range(ow):
                total = b[f]
                for c in range(C):
                    for u in range(kh):
                        for v in range(kw):
                            total += xp[c, i * stride + u, j * stride + v] * W[f, c, u, v]
                out[f, i, j] = total
    return out`,
    tc: "O(F · C · oh · ow · kh · kw)", tw: "All the multiplications, one at a time in Python.", sc: "O(F · oh · ow)", sw: "The output.",
    bad: "Six nested Python loops. It computes the right numbers and is a good oracle, but it is thousands of times slower than a single matrix multiplication.",
  },
  opt: {
    name: "im2col + one matrix multiply",
    idea: "Gather every patch (all channels) as one column. Then every output value, for every filter, is a dot product between a filter row and a patch column, which is exactly what `W_flat @ cols` computes in one BLAS call.",
    steps: ["Pad the height and width.", "`win = sliding_window_view(xp, (kh, kw), axis=(1, 2))[:, ::stride, ::stride]` has shape `(C, oh, ow, kh, kw)`.", "`cols = win.transpose(0, 3, 4, 1, 2).reshape(C*kh*kw, oh*ow)`: channel and kernel offsets down the rows, output positions across the columns.", "`out = W.reshape(F, -1) @ cols + b[:, None]`, then reshape to `(F, oh, ow)`."],
    code: R`
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def conv_forward(x, W, b, stride=1, pad=0):
    x, W, b = np.asarray(x, float), np.asarray(W, float), np.asarray(b, float)
    F, C, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (kh, kw), axis=(1, 2))[:, ::stride, ::stride]     # (C, oh, ow, kh, kw)
    C_, oh, ow = win.shape[:3]
    cols = win.transpose(0, 3, 4, 1, 2).reshape(C * kh * kw, oh * ow)                 # im2col: one column per output position
    out = W.reshape(F, -1) @ cols + b[:, None]                                         # one matrix multiply for all filters
    return out.reshape(F, oh, ow)`,
    tc: "O(F · C · oh · ow · kh · kw)", tw: "The same multiply-adds, but inside a highly optimised matrix multiplication.", sc: "O(C · kh · kw · oh · ow)", sw: "The `cols` matrix repeats each pixel up to kh·kw times: faster, but more memory.",
    viz: "dl2_conv",
  },
  run: R`
rng = np.random.default_rng(0)
x = rng.normal(size=(2, 6, 6))
W = rng.normal(size=(3, 2, 3, 3))
b = np.array([0.5, -0.5, 0.0])
y = conv_forward(x, W, b)
print(y.shape, conv_forward(x, W, b, stride=2, pad=1).shape)
print(np.round(y[:, 0, 0], 3).tolist())
ones = conv_forward(np.arange(25.0).reshape(1, 5, 5), np.ones((1, 1, 3, 3)), np.zeros(1), pad=1)
print(ones[0].astype(int).tolist())
manual = (x[:, 1:4, 2:5] * W[1]).sum() + b[1]          # filter 1 at output position (1, 2)
print(np.isclose(y[1, 1, 2], manual))`,
  out: R`
(3, 4, 4) (3, 3, 3)
[-1.441, -3.27, -4.951]
[[12, 21, 27, 33, 24], [33, 54, 63, 72, 51], [63, 99, 108, 117, 81], [93, 144, 153, 162, 111], [72, 111, 117, 123, 84]]
True`,
  dryOn: "x (2, 4, 4), W (3, 2, 3, 3), stride 1, pad 0",
  dry: { setup: "C = 2, kh = kw = 3, F = 3; oh = ow = (4 − 3)/1 + 1 = 2", steps: [["window view", "shape (2, 2, 2, 3, 3): per channel, a 2×2 grid of 3×3 patches"], ["transpose + reshape", "cols shape (2·3·3, 2·2) = (18, 4)"], ["W.reshape(F, −1)", "shape (3, 18)"], ["W_flat @ cols + b", "shape (3, 4): 4 output positions for each of the 3 filters"], ["reshape", "(3, 2, 2)"]], result: "Output shape **(3, 2, 2)**, identical to the six-loop version." },
  expl: [
    ["sliding_window_view(xp, (kh, kw), axis=(1, 2))", "All patches, per channel, as a view of shape `(C, H', W', kh, kw)`."],
    [".transpose(0, 3, 4, 1, 2).reshape(C*kh*kw, oh*ow)", "Put (channel, row-in-patch, col-in-patch) first and (output row, output col) last; flattening gives one column per output position."],
    ["W.reshape(F, -1)", "Each filter becomes one row of length `C·kh·kw`, in the same order as the rows of `cols`."],
    ["@ cols + b[:, None]", "A row times a column is one convolution output. The bias is added per filter (one value per row)."],
  ],
  take: "**im2col turns convolution into a single matrix multiplication.** It costs extra memory (patches overlap) but makes use of the fastest operation on any CPU or GPU.",
  mist: ["Flattening the patch in a different order from the filters (the rows of `cols` and the columns of `W_flat` must match).", "Adding the bias with the wrong shape: it is `(F,)` and must broadcast along the rows, hence `b[:, None]`.", "Forgetting that the number of rows of `cols` is `C·kh·kw`, not `kh·kw`.", "Padding the channel axis as well as height and width."],
},
{
  id: "rnn-forward", t: "RNN Cell Forward Pass over a Sequence", d: "Medium", p: "RNN",
  q: "Implement the forward pass of a vanilla recurrent network. Given a sequence `xs` of shape `(T, n_in)` and the parameters `Wx` of shape `(n_hid, n_in)`, `Wh` of shape `(n_hid, n_hid)` and `b` of shape `(n_hid,)`, compute for each time step\n\n$h_t = \\tanh(W_x x_t + W_h h_{t-1} + b)$\n\nstarting from `h0` (all zeros if `h0` is `None`). Return all the hidden states as an array of shape `(T, n_hid)`: row `t` is $h_t$.",
  ex: [["n_in = 1, n_hid = 1, Wx = [[1]], Wh = [[0.5]], b = [0], xs = [[1],[1]]", "[[0.762], [0.881]]", "h₁ = tanh(1·1 + 0.5·0) = 0.762; h₂ = tanh(1 + 0.5·0.762) = tanh(1.381) = 0.881."],
    ["same weights, xs = [[0],[0]]", "[[0], [0]]", "No input and no memory: the state stays 0."]],
  c: ["`T >= 1`", "Use `tanh` as the activation", "Return a float NumPy array of shape `(T, n_hid)`"],
  hint: "The time loop cannot be removed (each step needs the previous state), but inside each step use matrix-vector products: `Wx @ x + Wh @ h + b`.",
  brute: {
    name: "Plain Python loops",
    idea: "Compute each matrix-vector product with explicit loops over rows and columns, using only `math.tanh`.",
    code: R`
import math
import numpy as np

def rnn_forward(xs, Wx, Wh, b, h0=None):
    Wx, Wh, b = np.asarray(Wx, float), np.asarray(Wh, float), np.asarray(b, float)
    n_hid = len(b)
    h = [0.0] * n_hid if h0 is None else [float(v) for v in h0]
    states = []
    for x in np.asarray(xs, float):
        new = []
        for i in range(n_hid):
            z = b[i]
            for j in range(len(x)):
                z += Wx[i, j] * x[j]
            for j in range(n_hid):
                z += Wh[i, j] * h[j]
            new.append(math.tanh(z))
        h = new
        states.append(h)
    return np.array(states)`,
    tc: "O(T · n_hid · (n_in + n_hid))", tw: "Every multiply-add in a Python loop.", sc: "O(T · n_hid)", sw: "The stored states.",
    bad: "Correct, but the inner loops are scalar Python; the matrix-vector product should be one NumPy call.",
  },
  opt: {
    name: "NumPy matrix-vector products",
    idea: "Keep the loop over time, which is inherently sequential, and replace the inner loops by `Wx @ x + Wh @ h + b`. Preallocate the output array.",
    steps: ["`h = zeros(n_hid)` unless `h0` is given.", "For each `x` in `xs`: `h = np.tanh(Wx @ x + Wh @ h + b)`.", "Store each new `h` as a row of the output."],
    code: R`
import numpy as np

def rnn_forward(xs, Wx, Wh, b, h0=None):
    xs, Wx, Wh, b = (np.asarray(a, float) for a in (xs, Wx, Wh, b))
    h = np.zeros(len(b)) if h0 is None else np.asarray(h0, float)
    out = np.empty((len(xs), len(b)))
    for t, x in enumerate(xs):
        h = np.tanh(Wx @ x + Wh @ h + b)           # new memory from the new input and the old memory
        out[t] = h
    return out`,
    tc: "O(T · n_hid · (n_in + n_hid))", tw: "T sequential steps, each a small matrix-vector product.", sc: "O(T · n_hid)", sw: "The output array.",
    viz: "dl2_rnn",
  },
  run: R`
print(np.round(rnn_forward([[1.0], [1.0]], [[1.0]], [[0.5]], [0.0]), 3).tolist())
print(rnn_forward([[0.0], [0.0]], [[1.0]], [[0.5]], [0.0]).tolist())
Wx = np.array([[0.5, -0.3], [0.2, 0.8]])
Wh = np.array([[0.1, 0.4], [-0.2, 0.3]])
b = np.array([0.1, -0.1])
xs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
H = rnn_forward(xs, Wx, Wh, b)
print(np.round(H, 3).tolist())
print(np.round(rnn_forward(xs[1:], Wx, Wh, b, h0=H[0]), 3).tolist())
long = rnn_forward(np.ones((50, 2)) * 5, Wx, Wh, b)
print(long.shape, bool(np.abs(long).max() < 1))`,
  out: R`
[[0.762], [0.881]]
[[0.0], [0.0]]
[[0.537, 0.1], [-0.106, 0.553], [0.47, 0.796]]
[[-0.106, 0.553], [0.47, 0.796]]
(50, 2) True`,
  dryOn: "Wx = [[1]], Wh = [[0.5]], b = [0], xs = [[1],[1]]",
  dry: { h: ["t", "x_t", "Wx·x + Wh·h + b", "h_t = tanh(…)"], r: [["1", "1", "1·1 + 0.5·0 + 0 = 1", "**0.762**"], ["2", "1", "1·1 + 0.5·0.762 + 0 = 1.381", "**0.881**"]] },
  expl: [
    ["h = np.zeros(len(b)) if h0 is None else ...", "The initial memory: zeros unless you pass a state in (for example to continue a sequence)."],
    ["Wx @ x + Wh @ h + b", "New input through `Wx`, old memory through `Wh`, plus a bias: both contribute to the pre-activation."],
    ["np.tanh(...)", "Squashes the new state into (−1, 1)."],
    ["out[t] = h", "Keep every state, not just the last (a tagger needs all of them, a classifier only the last)."],
  ],
  take: "**An RNN is a for-loop that reuses the same weights at every step and passes the hidden state along.**",
  mist: ["Resetting `h` to zeros inside the loop: the memory is then lost at every step.", "Using `Wh @ x` and `Wx @ h` (swapped), which only works by accident when the sizes agree.", "Returning only the last state when all of them are required.", "Forgetting the bias, or applying tanh to only part of the sum."],
},
{
  id: "lstm-step", t: "LSTM Cell Step", d: "Hard", p: "RNN",
  q: "Implement one step of an LSTM cell. `lstm_step(x, h, c, W, b)` takes the input `x` of shape `(n_in,)`, the previous hidden state `h` and cell state `c` (each `(H,)`), a weight matrix `W` of shape `(4H, H + n_in)` and a bias `b` of shape `(4H,)`.\n\nThe pre-activations are `z = W @ [h, x] + b` (the concatenation of `h` followed by `x`). Split `z` into four equal blocks in the order **forget, input, candidate, output**:\n\n- $f = \\sigma(z_f)$, $i = \\sigma(z_i)$, $\\tilde c = \\tanh(z_g)$, $o = \\sigma(z_o)$\n- $c' = f \\odot c + i \\odot \\tilde c$\n- $h' = o \\odot \\tanh(c')$\n\nReturn `(h_new, c_new)`.",
  ex: [["H = 1, n_in = 1, W = [[0.4, 0.5], [0.3, 0.8], [0.5, 1.0], [0.2, 0.6]], b = [1, 0, 0, 0], x = 1, h = 0.5, c = 1", "h' = 0.599, c' = 1.457", "f = σ(0.4·0.5+0.5·1+1) = 0.846, i = σ(0.95) = 0.721, c̃ = tanh(1.25) = 0.848, o = σ(0.7) = 0.668; c' = 0.846·1 + 0.721·0.848 = 1.457; h' = 0.668·tanh(1.457) = 0.599."],
    ["forget bias = 10, input gate closed (large negative bias)", "c' ≈ c", "A forget gate near 1 and an input gate near 0 leave the cell state untouched."]],
  c: ["`W` is ordered `[forget; input; candidate; output]`, each block `(H, H + n_in)`", "Concatenate `h` first, then `x`", "Use a numerically safe sigmoid"],
  hint: "Compute `z` with ONE matrix multiplication and slice `z[0:H]`, `z[H:2H]`, `z[2H:3H]`, `z[3H:4H]`. A safe sigmoid is `0.5 * (1 + np.tanh(z / 2))`.",
  brute: {
    name: "Plain Python loops",
    idea: "Compute each gate separately with its own loops over the hidden units, using `math.exp` and `math.tanh`.",
    code: R`
import math
import numpy as np

def lstm_step(x, h, c, W, b):
    x, h, c, W, b = (np.asarray(a, float) for a in (x, h, c, W, b))
    H = len(h)
    hx = np.concatenate([h, x])
    def pre(block, k):                                # pre-activation of unit k in gate block 'block'
        row = block * H + k
        return b[row] + sum(W[row, j] * hx[j] for j in range(len(hx)))
    sig = lambda z: 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    h_new, c_new = np.zeros(H), np.zeros(H)
    for k in range(H):
        f, i = sig(pre(0, k)), sig(pre(1, k))
        g, o = math.tanh(pre(2, k)), sig(pre(3, k))
        c_new[k] = f * c[k] + i * g
        h_new[k] = o * math.tanh(c_new[k])
    return h_new, c_new`,
    tc: "O(H · (H + n_in))", tw: "Scalar loops for the four gates.", sc: "O(H)", sw: "The two new states.",
    bad: "Correct, but it recomputes the dot products unit by unit and gate by gate in Python.",
  },
  opt: {
    name: "One stacked matrix multiply",
    idea: "All four gates read the same vector `[h, x]`, so stack their weights and compute all pre-activations with a single matrix-vector product, then slice the result.",
    steps: ["`z = W @ np.concatenate([h, x]) + b`.", "Slice four blocks of length `H`: forget, input, candidate, output.", "Apply sigmoid, sigmoid, tanh, sigmoid.", "`c_new = f*c + i*g`, then `h_new = o * tanh(c_new)`."],
    code: R`
import numpy as np

def lstm_step(x, h, c, W, b):
    x, h, c, W, b = (np.asarray(a, float) for a in (x, h, c, W, b))
    H = len(h)
    z = W @ np.concatenate([h, x]) + b                 # all four gates in one multiply
    sig = lambda v: 0.5 * (1 + np.tanh(v / 2))         # overflow-free sigmoid
    f, i = sig(z[0:H]), sig(z[H:2 * H])                # forget and input gates
    g, o = np.tanh(z[2 * H:3 * H]), sig(z[3 * H:4 * H])  # candidate and output gate
    c_new = f * c + i * g                              # keep some old memory, write some new
    h_new = o * np.tanh(c_new)                         # reveal part of the memory
    return h_new, c_new`,
    tc: "O(H · (H + n_in))", tw: "One matrix-vector product plus element-wise operations.", sc: "O(H)", sw: "The pre-activation vector and the two new states.",
    viz: "dl2_lstm",
  },
  run: R`
W = np.array([[0.4, 0.5], [0.3, 0.8], [0.5, 1.0], [0.2, 0.6]])
b = np.array([1.0, 0.0, 0.0, 0.0])
h2, c2 = lstm_step([1.0], [0.5], [1.0], W, b)
print(np.round(h2, 3).tolist(), np.round(c2, 3).tolist())
H, n_in = 3, 2
W3 = np.linspace(-1, 1, 4 * H * (H + n_in)).reshape(4 * H, H + n_in) * 0.5
b3 = np.zeros(4 * H)
h, c = np.zeros(H), np.zeros(H)
for x in [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]:
    h, c = lstm_step(x, h, c, W3, b3)
print(np.round(h, 3).tolist())
print(np.round(c, 3).tolist())
keep_b = np.zeros(4 * H); keep_b[:H] = 10.0; keep_b[H:2 * H] = -10.0
_, c_kept = lstm_step([5.0, -5.0], np.zeros(H), np.array([0.7, -0.3, 2.0]), W3, keep_b)
print(np.round(c_kept, 3).tolist())`,
  out: R`
[0.599] [1.457]
[0.046, 0.123, 0.214]
[0.069, 0.175, 0.295]
[0.7, -0.3, 2.0]`,
  dryOn: "H = 1, x = 1, h = 0.5, c = 1 (the example weights above)",
  dry: { h: ["gate", "pre-activation z", "activation", "value"], r: [["forget f", "0.4·0.5 + 0.5·1 + 1 = 1.7", "σ(1.7)", "**0.846**"], ["input i", "0.3·0.5 + 0.8·1 = 0.95", "σ(0.95)", "**0.721**"], ["candidate c̃", "0.5·0.5 + 1.0·1 = 1.25", "tanh(1.25)", "**0.848**"], ["output o", "0.2·0.5 + 0.6·1 = 0.7", "σ(0.7)", "**0.668**"], ["new cell c'", "f·c + i·c̃ = 0.846 + 0.612", "", "**1.457**"], ["new hidden h'", "o·tanh(c') = 0.668 · 0.897", "", "**0.599**"]] },
  expl: [
    ["np.concatenate([h, x])", "The gates see the previous hidden state and the new input together; the order must match the columns of `W`."],
    ["z = W @ ... + b", "One multiply gives all four gates' pre-activations stacked in a vector of length `4H`."],
    ["0.5 * (1 + np.tanh(v / 2))", "A sigmoid written with tanh, which cannot overflow."],
    ["c_new = f * c + i * g", "The cell-state update is a gated sum, which lets gradients pass along the cell with little shrinking."],
    ["h_new = o * np.tanh(c_new)", "The output gate decides how much of the squashed memory to expose."],
  ],
  take: "**An LSTM step is one stacked matrix multiply followed by four element-wise gate operations**: forget the old, write the new, reveal part of the result.",
  mist: ["Wrong order of the four blocks (frameworks differ: PyTorch uses input, forget, candidate, output).", "Concatenating `[x, h]` when `W` was built for `[h, x]`.", "Applying a sigmoid to the candidate or a tanh to the gates.", "Using `h_new = o * c_new` and forgetting the `tanh`."],
},
]);
