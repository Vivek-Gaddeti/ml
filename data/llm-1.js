/* LLMs, part 1: training (bigram LM to distillation) and inference (decoding, KV cache, quantization, serving, long context)
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/llm-1.js */
LLM.push(
{
  id: "bigram-lm", group: "Training", title: "A Bigram Language Model",
  sub: "The smallest language model there is: count which token follows which, then sample. Every bigger model does the same job with a fancier table.",
  tags: ["from scratch", "language model"],
  blocks: [
    ["p", "A **language model** answers one question: given the text so far, how likely is each possible next token? GPT, Claude and Llama are all language models. They differ from the toy in this section only in *how much context* they look at and *how* they compute the probabilities."],
    ["p", "The simplest possible version looks at **one** previous token. Is \"cat\" more likely after \"the\" or after \"sat\"? You can answer that by counting in a text. Try it below: the heatmap is the whole model."],
    ["viz", "llm1_bigram"],
    ["h", "The maths"],
    ["p", "A text is a sequence of tokens $x_1, x_2, \\dots, x_T$. The chain rule of probability splits its probability into next-token predictions:"],
    ["math", R`P(x_1,\dots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\dots,x_{t-1})`, "exact, for any language model"],
    ["p", "A **bigram** model makes one simplifying assumption: only the previous token matters, $P(x_t\\mid x_{<t})\\approx P(x_t\\mid x_{t-1})$. The best values for those probabilities (the [maximum-likelihood estimate](math.html#mle)) are just relative counts:"],
    ["math", R`P(b\mid a)=\frac{\text{count}(a,b)}{\sum_{b'}\text{count}(a,b')}`, "one row of the heatmap, divided by its sum"],
    ["p", "To **generate**, start with a token, look at its row, draw a random next token with those probabilities, and repeat. To **score** a text, multiply (or add the logs of) the probabilities of its pairs. Here it is in plain NumPy, with start and end markers so the model also learns where sentences begin and finish:"],
    ["code", R`
import numpy as np

sents = ["the cat sat on the mat", "the dog sat on the log", "the cat saw the dog",
         "a dog saw a cat", "the dog sat on the mat"]
vocab = ["<s>", "</s>"] + sorted({w for s in sents for w in s.split()})
ix = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

counts = np.zeros((V, V))
for s in sents:
    toks = ["<s>"] + s.split() + ["</s>"]
    for a, b in zip(toks, toks[1:]):
        counts[ix[a], ix[b]] += 1                 # count(a, b)

P = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)   # each row sums to 1 (</s> never has a next token)
print("vocab size:", V)
print("P(. | the):", {vocab[j]: round(float(P[ix["the"], j]), 2) for j in np.nonzero(P[ix["the"]])[0]})
print("P(. | sat):", {vocab[j]: round(float(P[ix["sat"], j]), 2) for j in np.nonzero(P[ix["sat"]])[0]})

def generate(rng, max_len=12):
    out, cur = [], ix["<s>"]
    for _ in range(max_len):
        cur = rng.choice(V, p=P[cur])             # sample the next token from the row
        if vocab[cur] == "</s>":
            break
        out.append(vocab[cur])
    return " ".join(out)

rng = np.random.default_rng(1)
for _ in range(4):
    print("sampled:", generate(rng))`, "vocab size: 11\nP(. | the): {'cat': 0.25, 'dog': 0.38, 'log': 0.12, 'mat': 0.25}\nP(. | sat): {'on': 1.0}\nsampled: the mat\nsampled: the dog sat on the dog\nsampled: the dog sat on the dog\nsampled: the cat"],
    ["p", "Some samples are copies of training sentences, some are new recombinations (\"the dog saw the cat\" never appears). That is already a tiny taste of generalisation, and also of the failure mode: the model has no memory beyond one word, so it happily produces locally plausible, globally meaningless text."],
    ["h", "Perplexity: how surprised is the model?"],
    ["p", "To measure a language model you ask how much probability it gave to real text. The average negative log-likelihood per token is the [cross-entropy](math.html#entropy) $H$ (in nats). Its exponential is **perplexity**:"],
    ["math", R`\text{PPL}=\exp\!\Big(-\frac{1}{T}\sum_{t=1}^{T}\ln P(x_t\mid x_{t-1})\Big)`, "lower is better; 1 means the model is never surprised"],
    ["p", "Read perplexity as the **effective number of equally likely choices** the model is torn between at each step. A model that guesses uniformly over a vocabulary of $V$ tokens has perplexity exactly $V$. The catch is unseen pairs: with plain counts, any pair that never occurred in training gets probability 0, so one such pair in a test text makes the perplexity infinite. The fix is **add-α smoothing**: pretend every pair was seen α extra times."],
    ["math", R`P_\alpha(b\mid a)=\frac{\text{count}(a,b)+\alpha}{\sum_{b'}\text{count}(a,b')+\alpha V}`],
    ["code", R`
import numpy as np

sents = ["the cat sat on the mat", "the dog sat on the log", "the cat saw the dog",
         "a dog saw a cat", "the dog sat on the mat"]
vocab = ["<s>", "</s>"] + sorted({w for s in sents for w in s.split()})
ix = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
counts = np.zeros((V, V))
for s in sents:
    t = ["<s>"] + s.split() + ["</s>"]
    for a, b in zip(t, t[1:]):
        counts[ix[a], ix[b]] += 1

def model(alpha):
    c = counts + alpha
    return c / np.maximum(c.sum(axis=1, keepdims=True), 1e-9)      # the </s> row has no successors: avoid 0/0

def perplexity(P, sentences):
    nll, n = 0.0, 0
    for s in sentences:
        t = ["<s>"] + s.split() + ["</s>"]
        for a, b in zip(t, t[1:]):
            if P[ix[a], ix[b]] == 0:
                return float("inf")                       # one impossible pair makes the whole text impossible
            nll -= np.log(P[ix[a], ix[b]])
            n += 1
    return float(np.exp(nll / n))

seen = ["the cat sat on the log"]           # every pair occurred in training
new = ["a cat sat on a mat"]                # 'on a' never occurred
for alpha in [0, 0.1, 1.0]:
    P = model(alpha)
    print(f"alpha={alpha}: train PPL={perplexity(P, sents):.2f}  seen-pairs PPL={perplexity(P, seen):.2f}  new-pair PPL={perplexity(P, new):.2f}")
print("uniform guess would give PPL =", V)`, "alpha=0: train PPL=2.03  seen-pairs PPL=1.98  new-pair PPL=inf\nalpha=0.1: train PPL=2.52  seen-pairs PPL=2.53  new-pair PPL=5.46\nalpha=1.0: train PPL=5.11  seen-pairs PPL=5.17  new-pair PPL=7.20\nuniform guess would give PPL = 11"],
    ["p", "With α = 0 the model is best on its own training text and falls apart on the unseen pair. Larger α trades a little training fit for robustness. Real language models face exactly this trade-off, and fight it with regularisation, more data and held-out evaluation instead of smoothing."],
    ["h", "From a count table to a neural network"],
    ["p", "Here is the bridge to everything else in this course. Instead of counting, store a table of **logits** $W\\in\\mathbb{R}^{V\\times V}$, turn row $W[a]$ into probabilities with [softmax](math.html#entropy), and *learn* $W$ by gradient descent on the cross-entropy loss. This is a neural network with one layer: an embedding lookup of the previous token, followed by softmax. Its gradient is the famous $p-\\text{onehot}$. If you train long enough it reproduces the count table."],
    ["code", R`
import numpy as np

sents = ["the cat sat on the mat", "the dog sat on the log", "the cat saw the dog",
         "a dog saw a cat", "the dog sat on the mat"]
vocab = ["<s>", "</s>"] + sorted({w for s in sents for w in s.split()})
ix = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
pairs = [(ix[a], ix[b]) for s in sents for a, b in zip(["<s>"] + s.split(), s.split() + ["</s>"])]
prev, nxt = np.array(pairs).T                  # training examples: (previous token, next token)

counts = np.zeros((V, V)); np.add.at(counts, (prev, nxt), 1)
P_count = counts / np.maximum(counts.sum(1, keepdims=True), 1)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

W = np.zeros((V, V))                           # logits table: starts uniform
for step in range(3001):
    p = softmax(W[prev])                       # (N, V) predicted distributions
    loss = -np.log(p[np.arange(len(prev)), nxt]).mean()
    if step in (0, 300, 3000):
        print(f"step {step:4d}  loss={loss:.3f}  perplexity={np.exp(loss):.2f}")
    g = p.copy(); g[np.arange(len(prev)), nxt] -= 1      # softmax + cross-entropy gradient: p - onehot
    dW = np.zeros_like(W); np.add.at(dW, prev, g / len(prev))
    W -= 2.0 * dW                              # plain gradient descent

used = counts.sum(1) > 0
print("max |learned - counted| probability:", np.abs(softmax(W) - P_count)[used].max().round(3))
print("ln(V) =", round(float(np.log(V)), 3), " (the loss at step 0 is exactly this: a uniform guess)")`, "step    0  loss=2.398  perplexity=11.00\nstep  300  loss=0.735  perplexity=2.09\nstep 3000  loss=0.709  perplexity=2.03\nmax |learned - counted| probability: 0.005\nln(V) = 2.398  (the loss at step 0 is exactly this: a uniform guess)"],
    ["p", "Two things to notice. The starting loss is $\\ln V$, the loss of a uniform guess, just like the untrained GPT in [A Tiny GPT Forward Pass](transformers.html#tiny-gpt). And the final loss equals the entropy of the *counted* model: gradient descent found the same answer that counting gave directly. The reason we bother with the neural version is that it **generalises**: if the table is replaced by a function of the whole context (a [transformer](transformers.html#tiny-gpt)), tokens that never appeared together can still get sensible probabilities."],
    ["code", R`
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
V = 10
model = nn.Embedding(V, V)                     # row i = logits for the token that follows token i
prev = torch.tensor([0, 1, 2, 1, 0, 1, 2, 3])
nxt = torch.tensor([1, 2, 3, 3, 1, 2, 3, 4])
opt = torch.optim.SGD(model.parameters(), lr=1.0)
for step in range(200):
    loss = F.cross_entropy(model(prev), nxt)   # softmax + negative log-likelihood in one call
    opt.zero_grad(); loss.backward(); opt.step()
print("loss:", round(loss.item(), 3), "  ln(V) =", round(torch.log(torch.tensor(float(V))).item(), 3))`, "loss: 0.262   ln(V) = 2.303", "PyTorch", "local"],
    ["table", ["", "Bigram counts", "Neural bigram", "Transformer LM"], [
      ["Context used", "1 token", "1 token", "thousands to millions of tokens"],
      ["Parameters", "$V^2$ counts", "$V^2$ logits", "billions, shared across positions"],
      ["Unseen combinations", "probability 0 (needs smoothing)", "same, 0 without smoothing", "generalises through embeddings and attention"],
      ["How it is fit", "counting", "gradient descent on cross-entropy", "gradient descent on cross-entropy"],
    ]],
    ["tip", "Everything in the next sections is this same recipe: a model that outputs a probability distribution over the next token, trained by minimising cross-entropy on text, then sampled from one token at a time. [Pretraining](llm.html#pretraining) scales it up; [decoding](llm.html#sampling) is the sampling loop."],
    ["danger", "Computing perplexity on the **training** text and calling it quality. It only means something on held-out text. Also, perplexities from different tokenizers cannot be compared: a model with a bigger vocabulary predicts bigger chunks per step."],
  ],
},
{
  id: "pretraining", group: "Training", title: "Pretraining: Next-Token Prediction at Scale",
  sub: "How a base model is made: one simple loss, enormous data, and a lot of engineering to keep thousands of chips busy.",
  tags: ["training", "foundations"],
  blocks: [
    ["p", "Pretraining is the first and by far the most expensive stage of building an LLM. The task is exactly the one from the [bigram model](llm.html#bigram-lm), scaled up: show the model a piece of text and make it predict each next token. No labels are needed because **the text is its own label**; this is called *self-supervised* learning. Almost all of an LLM's knowledge and fluency comes from this stage."],
    ["diagram", "llm1_pipeline", [], "The three usual stages of building a chat model. Pretraining makes the base model; the later stages shape its behaviour."],
    ["h", "The objective"],
    ["p", "For a text of tokens $x_1\\dots x_T$, the model $p_\\theta$ gives a distribution over the next token at every position. The loss is the average negative log-probability of the token that actually came next (cross-entropy, see [entropy](math.html#entropy)):"],
    ["math", R`\mathcal{L}(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\ln p_\theta\big(x_t\mid x_{<t}\big)`, "minimising this is maximum-likelihood estimation of the text distribution"],
    ["p", "Because of the causal mask ([masking](transformers.html#masking)), one forward pass over $T$ tokens yields $T$ predictions and $T$ loss terms at once. This is called **teacher forcing**: the inputs at each position are the real previous tokens, not the model's own guesses. Inputs and targets are the same sequence shifted by one:"],
    ["code", R`
import numpy as np

# toy "documents" as token ids; 0 is the end-of-text marker EOS
docs = [[5, 6, 7, 8], [9, 10], [11, 12, 13, 14, 15, 16], [17, 18, 19]]
seq_len = 6

stream = []
for d in docs:
    stream += d + [0]                           # concatenate all documents, separated by EOS
n_seq = (len(stream) - 1) // seq_len            # full sequences we can cut (each needs one extra token for the target)
stream = np.array(stream[: n_seq * seq_len + 1])
x = np.stack([stream[i * seq_len : (i + 1) * seq_len] for i in range(n_seq)])          # inputs
y = np.stack([stream[i * seq_len + 1 : (i + 1) * seq_len + 1] for i in range(n_seq)])  # targets = inputs shifted left by 1
print("inputs :\n", x)
print("targets:\n", y)
print("tokens per batch:", x.size)`, "inputs :\n [[ 5  6  7  8  0  9]\n [10  0 11 12 13 14]\n [15 16  0 17 18 19]]\ntargets:\n [[ 6  7  8  0  9 10]\n [ 0 11 12 13 14 15]\n [16  0 17 18 19  0]]\ntokens per batch: 18"],
    ["p", "This is **sequence packing**: documents are concatenated with an end-of-text token and cut into fixed-length windows, so no compute is wasted on padding. (Some pipelines also mask attention so a token cannot see the previous document in the same window.) The loss itself, from random logits so you can check the $\\ln V$ rule:"],
    ["code", R`
import torch, torch.nn.functional as F

torch.manual_seed(0)
B, T, V = 2, 6, 50
logits = torch.randn(B, T, V) * 0.01             # an untrained model: nearly uniform
tokens = torch.randint(0, V, (B, T + 1))         # a batch of packed sequences, one extra token for the last target
x, y = tokens[:, :-1], tokens[:, 1:]             # shift by one: position t predicts token t+1
# (a real model would compute logits = model(x))
loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
print("loss:", round(loss.item(), 3), "  ln(V):", round(torch.log(torch.tensor(float(V))).item(), 3))`, "loss: 3.91   ln(V): 3.912", "PyTorch", "local"],
    ["h", "Where the data comes from"],
    ["p", "A modern pretraining set is trillions of tokens (as of 2026, frontier-scale runs are commonly reported in the range of 10 to 30+ trillion). The usual pipeline:"],
    ["ol", [
      "**Collect**: web crawls, books, code repositories, scientific papers, forums, encyclopedias, plus (increasingly) synthetic text generated by other models.",
      "**Filter**: remove boilerplate, spam, adult content and low-quality pages with heuristics and trained quality classifiers. Language identification.",
      "**Deduplicate**: exact and near-duplicate removal. Repeated text is memorised and wastes compute.",
      "**Tokenize** ([BPE](transformers.html#bpe)) and **mix**: choose how much code, maths, multilingual text and so on, often changing the mix toward higher-quality data near the end of training.",
      "**Decontaminate**: remove benchmark test questions so [evaluation](llm.html#llm-evaluation) stays honest.",
    ]],
    ["p", "Data quality matters as much as quantity, which is why labs treat their data recipes as secrets. Code and maths in the mix are also widely reported to improve general reasoning, not just coding."],
    ["h", "The training run"],
    ["list", [
      "**Optimizer**: [AdamW](dl.html#optimizers) with weight decay around 0.1, betas near (0.9, 0.95), gradient clipping at norm 1.0. These are typical published values for LLM runs.",
      "**Schedule**: a short linear [warmup](dl.html#lr-schedules), then cosine (or linear) decay of the learning rate to roughly a tenth of its peak. Some recent recipes use a flat phase and a short final decay.",
      "**Batch size**: measured in *tokens*, commonly millions of tokens per optimizer step. Large batches make better use of thousands of GPUs.",
      "**Precision**: bf16 matrix multiplies with fp32 master weights and optimizer state. (Some recent runs use fp8 for parts of the computation.)",
      "**Parallelism**: one GPU cannot hold the model, so the work is split: *data parallel* (copies of the model, different batches), *tensor parallel* (split each matrix across GPUs), *pipeline parallel* (different layers on different GPUs), and sharded optimizer state (ZeRO / FSDP).",
      "**Monitoring**: loss curves, gradient norms, and regular checkpoints. Loss spikes happen; recipes include rewinding to a checkpoint and skipping the offending batches.",
    ]],
    ["h", "How big is the bill?"],
    ["p", "A very useful rule of thumb: training a dense transformer with $N$ parameters on $D$ tokens costs about"],
    ["math", R`C\approx 6\,N\,D \ \text{FLOPs}`, "2ND for the forward pass + 4ND for the backward pass (the gradient costs about twice the forward)"],
    ["p", "That makes it easy to estimate training time. An NVIDIA A100 does at most 312 trillion bf16 FLOPs per second (a published spec). Real training reaches maybe 30 to 50% of that (called *model FLOPs utilisation*, MFU), because of communication and memory stalls."],
    ["code", R`
def train_cost(n_params, n_tokens, peak_flops=312e12, mfu=0.4, n_gpus=1024):
    flops = 6 * n_params * n_tokens                      # C = 6 N D
    seconds = flops / (peak_flops * mfu * n_gpus)
    gpu_hours = flops / (peak_flops * mfu) / 3600
    return flops, gpu_hours, seconds / 86400

for name, N, D in [("1B on 100B tokens", 1e9, 100e9), ("7B on 2T tokens", 7e9, 2e12), ("70B on 15T tokens", 70e9, 15e12)]:
    flops, gh, days = train_cost(N, D)
    print(f"{name:20s} FLOPs={flops:.2e}  GPU-hours (A100, 40% MFU)={gh:,.0f}  days on 1024 GPUs={days:.1f}")

# memory for training with Adam in mixed precision: weights 2 + grads 2 + fp32 master 4 + Adam m 4 + Adam v 4 = 16 bytes/param
for N in [1e9, 7e9, 70e9]:
    print(f"{N/1e9:>4.0f}B params: training state about {16 * N / 1e9:,.0f} GB (before activations)")`, "1B on 100B tokens    FLOPs=6.00e+20  GPU-hours (A100, 40% MFU)=1,335  days on 1024 GPUs=0.1\n7B on 2T tokens      FLOPs=8.40e+22  GPU-hours (A100, 40% MFU)=186,966  days on 1024 GPUs=7.6\n70B on 15T tokens    FLOPs=6.30e+24  GPU-hours (A100, 40% MFU)=14,022,436  days on 1024 GPUs=570.6\n   1B params: training state about 16 GB (before activations)\n   7B params: training state about 112 GB (before activations)\n  70B params: training state about 1,120 GB (before activations)"],
    ["p", "The second print explains why big models must be sharded across many GPUs: just the training *state* of a 70B model is over a terabyte, far more than any single GPU holds."],
    ["h", "What you get: a base model"],
    ["p", "The result is a **base model** (or *foundation model*). It is superb at continuing text but it does not answer questions: given \"What is the capital of France?\" it may continue with more quiz questions, because that is what such text looks like on the web. Making it follow instructions is the job of [supervised fine-tuning](llm.html#sft) and [preference tuning](llm.html#rlhf). Pretraining also gives a model its **in-context learning** ability: show it a few examples in the prompt and it continues the pattern, with no weight updates (see [prompt engineering](llm.html#prompt-engineering))."],
    ["table", ["Stage", "Typical scale (as of 2026)", "What changes"], [
      ["Pretraining", "trillions of tokens, weeks to months on thousands of GPUs", "everything: knowledge, language, skills"],
      ["Mid-training / long-context extension", "hundreds of billions of tokens", "context length, domain mix (code, maths)"],
      ["SFT", "thousands to millions of examples, hours to days", "format and instruction following"],
      ["Preference tuning (RLHF, DPO, RL)", "up to millions of comparisons or prompts", "style, helpfulness, safety, reasoning"],
    ]],
    ["warn", "Pretraining loss going down does not automatically mean the model is better at *your* task. Loss is an average over everything; downstream ability is measured separately ([evaluating LLMs](llm.html#llm-evaluation)). Loss values also depend on the tokenizer and data, so they cannot be compared across different runs."],
    ["tip", "Intuition for why one simple loss produces broad skills: to predict the next word of a physics paper, a chess game, a Python file and a legal contract, the model is pushed to represent facts, syntax and logic. Compression of the training data is, in effect, learning."],
  ],
},
{
  id: "scaling-laws", group: "Training", title: "Scaling Laws & Chinchilla",
  sub: "Loss falls smoothly and predictably as you add parameters, data and compute, and that lets you plan expensive runs on paper.",
  tags: ["training", "planning"],
  blocks: [
    ["p", "Before spending millions of dollars on a training run, you want to know what you will get. The surprising discovery of the early 2020s is that you can know quite well: language-model loss follows **smooth power laws** in model size, dataset size and compute. Plot loss against any of them on log-log axes and you get a nearly straight line, over many orders of magnitude."],
    ["h", "Kaplan et al. (2020) and Chinchilla (2022)"],
    ["p", "Kaplan et al. at OpenAI (2020) measured these power laws and concluded that, for a given compute budget, you should spend most of the extra compute on a **bigger model** and train it on relatively little data. Models like GPT-3 (175B parameters, about 300B training tokens) followed that advice."],
    ["p", "Hoffmann et al. at DeepMind (2022, \"Training Compute-Optimal Large Language Models\", the *Chinchilla* paper) trained over 400 models of different sizes and re-did the analysis. Their conclusion: **model size and training tokens should grow together, in equal proportion**. Their model Chinchilla (70B parameters, 1.4T tokens) beat Gopher (280B parameters, 300B tokens), which used the same compute but was far too large for its data. The rule of thumb that came out of it: about **20 training tokens per parameter**."],
    ["p", "The paper models the loss as a sum of three terms:"],
    ["math", R`L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}`, "E = irreducible loss of text itself; the other terms shrink as the model (N) and the data (D) grow"],
    ["p", "A too-small model is limited by the $A/N^\\alpha$ term; too little data by $B/D^\\beta$. With the budget fixed at $C\\approx 6ND$, the best split balances the two. Play with it (the constants are round made-up numbers of the same shape as published fits, so the picture is **illustrative**, not a prediction for any real model):"],
    ["viz", "llm1_scaling"],
    ["h", "Using it: compute-optimal sizing"],
    ["p", "If $D=20N$ and $C=6ND$, then $C=120N^2$. So for any compute budget:"],
    ["math", R`N^{*}\approx\sqrt{C/120},\qquad D^{*}\approx 20\,N^{*}`],
    ["code", R`
import numpy as np

def compute_optimal(C, tokens_per_param=20):
    N = np.sqrt(C / (6 * tokens_per_param))      # from C = 6 N D and D = 20 N
    return N, tokens_per_param * N

# Chinchilla itself: 70B parameters, 1.4T tokens -> compute
C_chin = 6 * 70e9 * 1.4e12
print(f"Chinchilla compute: {C_chin:.2e} FLOPs;  tokens/param = {1.4e12 / 70e9:.0f}")
N, D = compute_optimal(C_chin)
print(f"compute-optimal for that budget: N = {N/1e9:.0f}B, D = {D/1e12:.2f}T")
print(f"Gopher: 280B params, 300B tokens -> tokens/param = {300e9 / 280e9:.1f}   (far below 20: under-trained)")

for C in [1e21, 1e22, 1e23, 1e24]:
    N, D = compute_optimal(C)
    print(f"C={C:.0e}: N={N/1e9:7.2f}B  D={D/1e9:9.1f}B tokens")`, "Chinchilla compute: 5.88e+23 FLOPs;  tokens/param = 20\ncompute-optimal for that budget: N = 70B, D = 1.40T\nGopher: 280B params, 300B tokens -> tokens/param = 1.1   (far below 20: under-trained)\nC=1e+21: N=   2.89B  D=     57.7B tokens\nC=1e+22: N=   9.13B  D=    182.6B tokens\nC=1e+23: N=  28.87B  D=    577.4B tokens\nC=1e+24: N=  91.29B  D=   1825.7B tokens"],
    ["h", "Fitting a power law yourself"],
    ["p", "A power law $L = a\\,x^{-b}$ is a straight line after taking logs: $\\ln L=\\ln a - b\\ln x$. So fitting one is just linear regression in log-log space. The data below are **synthetic** (a power law plus noise), made only to show the method:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
N = np.logspace(6, 10, 9)                               # model sizes from 1M to 10B
true_a, true_b = 12.0, 0.08
loss = true_a * N ** (-true_b) * np.exp(rng.normal(0, 0.003, N.size))   # synthetic data, 0.3% noise

slope, intercept = np.polyfit(np.log(N), np.log(loss), 1)   # straight line in log-log space
print("fitted exponent b =", round(-slope, 3), " (true 0.08)")
print("fitted a =", round(float(np.exp(intercept)), 2), " (true 12.0)")
pred = np.exp(intercept) * (1e11) ** slope
print("extrapolated loss at N=1e11:", round(float(pred), 3))`, "fitted exponent b = 0.08  (true 0.08)\nfitted a = 11.99  (true 12.0)\nextrapolated loss at N=1e11: 1.584"],
    ["p", "This is how labs plan: train a family of small models (cheap), fit the curve, and extrapolate to the big run. The extrapolation is the whole point and also the risk: it is only trustworthy if nothing changes regime between the sizes you tried and the size you want."],
    ["h", "Beyond Chinchilla: training for inference"],
    ["p", "Chinchilla-optimal minimises *training* cost for a given loss. But a deployed model is used for billions of requests, and every token it generates costs about $2N$ FLOPs. A **smaller model trained on more data** reaches the same loss and is cheaper to serve forever. So modern open models deliberately **over-train**: Llama 3 8B was trained on roughly 15T tokens, close to 2000 tokens per parameter, 100 times the Chinchilla ratio."],
    ["code", R`
def lifetime_flops(N, D_train, tokens_served):
    return 6 * N * D_train + 2 * N * tokens_served        # training + inference

served = 1e13                                             # suppose the model serves 10 trillion tokens in its life
big   = lifetime_flops(70e9, 1.4e12, served)              # roughly Chinchilla-optimal
small = lifetime_flops(8e9, 15e12, served)                # smaller but heavily over-trained (assume similar quality)
print(f"70B @1.4T tokens: training {6*70e9*1.4e12:.1e}, inference {2*70e9*served:.1e}, total {big:.2e}")
print(f"8B @15T tokens  : training {6*8e9*15e12:.1e}, inference {2*8e9*served:.1e}, total {small:.2e}")
print("cheaper overall:", "8B" if small < big else "70B")`, "70B @1.4T tokens: training 5.9e+23, inference 1.4e+24, total 1.99e+24\n8B @15T tokens  : training 7.2e+23, inference 1.6e+23, total 8.80e+23\ncheaper overall: 8B"],
    ["p", "(The quality of the 8B model is *assumed* equal here; in reality over-training a small model reduces but does not close the gap to a larger one. The point is the trade-off: the more you will serve, the more you should over-train a smaller model.)"],
    ["h", "What scaling laws do not tell you"],
    ["list", [
      "They predict **loss**, not specific abilities. Some skills appear to switch on suddenly as models grow, but how sharp that looks depends heavily on the metric used.",
      "They assume enough **fresh, high-quality data**. Repeating the same data for several epochs works for a few passes, then gives diminishing returns; the supply of high-quality text is finite, which is why synthetic data and data quality are active areas (as of 2026).",
      "The constants depend on architecture, tokenizer and data mix. A mixture-of-experts model ([MoE](transformers.html#moe)) has different effective curves.",
      "Scaling now also happens at **inference**: letting a model think longer improves answers on hard problems (see [reasoning models](llm.html#reasoning-models)). That is a second axis to scale.",
    ]],
    ["tip", "Memorise three numbers and one formula: $C\\approx 6ND$, about 20 tokens per parameter for compute-optimal training, about 16 bytes per parameter of training state with Adam, and about 2 bytes per parameter to *store* a model in 16-bit precision ([quantization](llm.html#quantization) shrinks that)."],
    ["danger", "Treating the 20:1 ratio as a law of nature. It is the optimum for minimising training compute under one fitted model. If inference cost, data availability or a target size matters (and they usually do), the right ratio is different."],
  ],
},
{
  id: "sft", group: "Training", title: "Supervised Fine-Tuning & Instruction Tuning",
  sub: "Teach a base model to behave like an assistant by showing it examples of good answers to prompts.",
  tags: ["post-training", "alignment"],
  blocks: [
    ["p", "A base model continues text; an assistant answers. **Supervised fine-tuning (SFT)** bridges the gap with a surprisingly small, simple recipe: collect examples of the form *(prompt, ideal answer)* and keep training the model with the **same next-token loss** as in [pretraining](llm.html#pretraining), but only on the answer tokens. This is also called **instruction tuning** when the prompts are varied instructions (\"summarise this\", \"write a SQL query\", \"translate to French\")."],
    ["p", "The key insight (shown clearly by the 2022 InstructGPT work and by LIMA, which used only about 1000 curated examples) is that SFT mostly teaches the model *which of the behaviours it already has* to show, and in what **format**. It adds style and instruction following far more than new knowledge."],
    ["h", "The data"],
    ["list", [
      "**Human demonstrations**: contractors write ideal answers. Highest quality, expensive.",
      "**Synthetic data**: a stronger model writes answers (or whole conversations) that are then filtered. Cheap and now very common. The student inherits the teacher's strengths and mistakes (see [distillation](llm.html#distillation)).",
      "**Task collections** converted to instructions (FLAN-style): existing datasets rewritten as prompt/answer pairs.",
      "**Domain data**: customer-support tickets, code with tests, medical dialogues, for specialised models.",
    ]],
    ["p", "Quality dominates quantity: a few thousand clean, diverse, correct examples usually beat a million noisy ones. Bad answers teach bad habits, and a model trained on confident-sounding wrong answers learns to sound confident."],
    ["h", "Chat templates"],
    ["p", "A conversation must be turned into one token sequence. Each model family uses its own **chat template**: special tokens that mark who is speaking. A typical shape (the exact tokens differ per model):"],
    ["code", R`
<|system|>You are a helpful assistant.<|end|>
<|user|>What is the capital of France?<|end|>
<|assistant|>The capital of France is Paris.<|end|>`, "", "Chat template (illustrative, token names vary)"],
    ["p", "Only the assistant part is the training target. The system and user parts are given as context. In training, that means a **loss mask**: positions whose next token belongs to the prompt contribute nothing to the loss. In NumPy, with a tiny made-up model output:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
V = 20
tokens = np.array([1, 4, 5, 6, 2,  7, 8, 9, 3])         # [prompt tokens ... | answer tokens ...]
is_answer = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])      # 1 where the token is part of the answer

logits = rng.normal(size=(len(tokens) - 1, V))          # fake model output: position t predicts token t+1
p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
nll = -np.log(p[np.arange(len(tokens) - 1), tokens[1:]])          # loss for each next-token prediction
mask = is_answer[1:]                                     # a target counts only if the target token is in the answer

print("per-token loss:", nll.round(2))
print("mask          :", mask)
print("loss on all tokens   :", nll.mean().round(3))
print("loss on answer only  :", ((nll * mask).sum() / mask.sum()).round(3))`, "per-token loss: [3.67 3.22 2.35 2.05 2.08 5.66 4.47 2.57]\nmask          : [0 0 0 0 1 1 1 1]\nloss on all tokens   : 3.257\nloss on answer only  : 3.693"],
    ["p", "In PyTorch this is done by setting the label of every prompt token to `-100`, which `F.cross_entropy` ignores by default. Hugging Face's TRL library handles templates and masks for you:"],
    ["code", R`
# Hugging Face TRL (needs model weights, so it is not run here)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

ds = load_dataset("trl-lib/Capybara", split="train")      # conversations: [{"role": "user", ...}, {"role": "assistant", ...}]
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",                             # any causal LM id or a loaded model
    train_dataset=ds,
    args=SFTConfig(output_dir="sft-out", learning_rate=2e-5, num_train_epochs=1,
                   per_device_train_batch_size=4, assistant_only_loss=True),   # loss only on assistant tokens
)
trainer.train()`, "", "TRL", "local"],
    ["h", "Hyperparameters and practice"],
    ["table", ["", "Pretraining", "SFT"], [
      ["Data", "trillions of raw tokens", "10^3 to 10^6 curated examples"],
      ["Learning rate", "around 1e-4 to 6e-4 for small models, lower for big ones", "much lower: typically 1e-5 to 2e-5 for full fine-tuning"],
      ["Epochs", "usually about 1 over the data", "1 to 3; more tends to overfit"],
      ["Cost", "millions of GPU-hours", "hours to days"],
    ]],
    ["list", [
      "**Catastrophic forgetting**: training hard on a narrow set can erase general abilities. Use a low learning rate, few epochs, and mix in general data. [LoRA](llm.html#lora) limits forgetting because the base weights stay frozen.",
      "**Match the template at inference time.** The model learned one specific token format; if you prompt it with a different one, quality drops noticeably.",
      "**Add an end-of-turn token** to the training target so the model learns to stop. Otherwise it keeps writing a fake next user turn.",
      "SFT cannot teach the model what it cannot do. If the answers need knowledge the base model lacks, it learns to *sound* right, which feeds [hallucinations](llm.html#hallucinations).",
    ]],
    ["p", "SFT is imitation: it makes the model produce answers like the examples, but it never tells the model that one answer is *better* than another. For that you need preferences: [RLHF](llm.html#rlhf) and [DPO](llm.html#dpo)."],
    ["danger", "Computing the loss on the prompt tokens too. The model then partly learns to generate user messages, and for short answers the prompt can dominate the average loss. Always mask the prompt (unless you deliberately want to train on it)."],
  ],
},
{
  id: "lora", group: "Training", title: "LoRA, QLoRA & PEFT",
  sub: "Fine-tune a huge model by training two thin matrices per layer instead of the whole weight matrix.",
  tags: ["fine-tuning", "efficiency"],
  blocks: [
    ["p", "Fully fine-tuning a 7B-parameter model needs to hold gradients and Adam state for every weight: about 16 bytes per parameter, over 100 GB. And every task would produce a full 14 GB copy of the model. **Parameter-efficient fine-tuning (PEFT)** freezes the original weights and trains a tiny number of new ones. The most popular method by far is **LoRA** (Low-Rank Adaptation, Hu et al. 2021)."],
    ["h", "The idea"],
    ["p", "When we fine-tune a weight matrix $W\\in\\mathbb{R}^{d\\times d}$, we learn an update $\\Delta W$. LoRA bets that this update has **low rank** (see [SVD](math.html#svd)): it can be well approximated by a product of two thin matrices. So instead of learning the $d^2$ numbers of $\\Delta W$, learn $B\\in\\mathbb{R}^{d\\times r}$ and $A\\in\\mathbb{R}^{r\\times d}$ with a small rank $r$ (4, 8, 16, 64):"],
    ["math", R`h = Wx + \frac{\alpha}{r}\,B\,(A x)`, "W is frozen. Only A and B are trained."],
    ["list", [
      "$A$ is initialised with small random values, $B$ with **zeros**. So at the start $BA=0$ and the model is exactly the base model.",
      "$\\alpha$ is a scaling constant (often $\\alpha=r$ or $2r$); dividing by $r$ keeps the update size comparable when you change $r$.",
      "Trainable parameters per matrix: $2dr$ instead of $d^2$, a fraction $2r/d$. For $d=4096$, $r=8$ that is 0.39%.",
    ]],
    ["viz", "llm1_lora"],
    ["h", "LoRA from scratch"],
    ["p", "The forward pass, the parameter count, and the fact that after training you can **merge** the update into the weights, so inference costs nothing extra:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
d, r, alpha = 64, 4, 8
W = rng.normal(0, 0.1, size=(d, d))                  # frozen pretrained weight
A = rng.normal(0, 0.01, size=(r, d))                 # random init
B = np.zeros((d, r))                                 # zero init: BA = 0 at the start
scale = alpha / r

def lora_forward(x):
    return x @ W.T + scale * (x @ A.T) @ B.T         # Wx + (alpha/r) B A x, for a batch of row vectors

x = rng.normal(size=(5, d))
print("at init, output equals the base model:", np.allclose(lora_forward(x), x @ W.T))

B = rng.normal(0, 0.1, size=(d, r))                  # pretend training has changed B
W_merged = W + scale * B @ A                         # merge: one ordinary weight matrix again
print("merged weights give the same output:", np.allclose(lora_forward(x), x @ W_merged.T))

print("full matrix:", W.size, " LoRA:", A.size + B.size, " fraction:", round((A.size + B.size) / W.size, 4))
print("rank of the update B@A:", np.linalg.matrix_rank(B @ A))`, "at init, output equals the base model: True\nmerged weights give the same output: True\nfull matrix: 4096  LoRA: 512  fraction: 0.125\nrank of the update B@A: 4"],
    ["h", "Does low rank really work? A toy"],
    ["p", "Suppose the ideal change to a layer really is rank 2. Train LoRA adapters of rank 1, 2 and 4 with plain gradient descent to match it. Rank 1 cannot represent the target, while ranks 2 and 4 can:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
d, n = 16, 400
W0 = rng.normal(0, 0.3, size=(d, d))
delta = rng.normal(size=(d, 2)) @ rng.normal(size=(2, d)) * 0.3     # the "true" fine-tuning update: rank 2
X = rng.normal(size=(n, d))
Y = X @ (W0 + delta).T                                               # targets from the fine-tuned layer

def fit_lora(r, steps=4000, lr=0.2):
    A = rng.normal(0, 0.3, size=(r, d)); B = np.zeros((d, r))
    for _ in range(steps):
        err = X @ (W0 + B @ A).T - Y                  # (n, d)
        G = 2 / (n * d) * err.T @ X                   # dLoss / d(BA), shape (d, d)
        dB, dA = G @ A.T, B.T @ G                     # chain rule through the product
        B -= lr * dB; A -= lr * dA
    return float(((X @ (W0 + B @ A).T - Y) ** 2).mean())

print("loss with no update:", round(float(((X @ W0.T - Y) ** 2).mean()), 4))
for r in [1, 2, 4]:
    print(f"rank {r}: final loss = {fit_lora(r):.4f}   trainable = {2 * d * r} of {d * d}")`, "loss with no update: 3.6854\nrank 1: final loss = 0.4522   trainable = 32 of 256\nrank 2: final loss = 0.0000   trainable = 64 of 256\nrank 4: final loss = 0.0000   trainable = 128 of 256"],
    ["p", "That is the whole bet of LoRA: if the useful change really has low intrinsic rank, a very thin adapter captures it. Empirically this holds well for adapting to a task or style, and less well for teaching a model lots of genuinely new knowledge."],
    ["h", "In PyTorch, and with PEFT"],
    ["p", "A hand-written LoRA layer is only a few lines. It wraps a frozen `nn.Linear`:"],
    ["code", R`
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                       # freeze the pretrained weight
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

torch.manual_seed(0)
layer = LoRALinear(nn.Linear(64, 64), r=4)
x = torch.randn(3, 64)
print("same as base at init:", torch.allclose(layer(x), layer.base(x)))
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print("trainable:", trainable, "of", total)`, "same as base at init: True\ntrainable: 512 of 4672", "PyTorch", "local"],
    ["p", "In practice you use the Hugging Face **PEFT** library, which injects adapters into a loaded model and can save only the adapter (a few megabytes):"],
    ["code", R`
# Hugging Face PEFT (needs model weights and the peft package, so it is not run here)
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],   # which linear layers get adapters
                 task_type="CAUSAL_LM")
model = get_peft_model(base, cfg)
model.print_trainable_parameters()          # prints trainable params and the (tiny) percentage
model.save_pretrained("my-adapter")          # saves only the adapter weights
merged = model.merge_and_unload()            # fold the update into W for adapter-free inference`, "", "PEFT", "local"],
    ["h", "QLoRA: LoRA on a quantized base"],
    ["p", "**QLoRA** (Dettmers et al., 2023) makes fine-tuning fit on a single consumer or workstation GPU by combining LoRA with [quantization](llm.html#quantization): the frozen base model is stored in **4 bits** (a format called NF4, designed for normally distributed weights), while the small LoRA matrices stay in 16-bit and are the only things trained. Gradients flow through the dequantized base weights. Extra tricks: *double quantization* (quantize the scale factors too) and *paged optimizers* (spill optimizer state to CPU memory on spikes)."],
    ["code", R`
GB = 1e9
N = 7e9                                              # a 7B-parameter model
adapter = 0.01 * N                                   # LoRA trains about 1% of the parameters (depends on rank and layers)

full_ft = 16 * N / GB                                # fp16 weights + grads + fp32 master copy + Adam m and v
lora    = (2 * N + 16 * adapter) / GB                # frozen fp16 base + full training state for the adapter only
qlora   = (0.5 * N + 16 * adapter) / GB              # frozen 4-bit base (0.5 byte per parameter) + adapter state
print(f"full fine-tune    : ~{full_ft:5.0f} GB")
print(f"LoRA (fp16 base)  : ~{lora:5.1f} GB")
print(f"QLoRA (4-bit base): ~{qlora:5.1f} GB")
print("activations and the KV cache come on top of all of these")`, "full fine-tune    : ~  112 GB\nLoRA (fp16 base)  : ~ 15.1 GB\nQLoRA (4-bit base): ~  4.6 GB\nactivations and the KV cache come on top of all of these"],
    ["table", ["Method", "What is trained", "Memory", "Notes"], [
      ["Full fine-tuning", "all weights", "highest (about 16 bytes per parameter)", "best quality ceiling, one full copy per task"],
      ["LoRA", "low-rank adapters on chosen layers", "base model + small state", "adapters are megabytes, swappable, mergeable"],
      ["QLoRA", "LoRA on a 4-bit base", "lowest", "slightly lower quality than 16-bit LoRA in some cases, makes big models fine-tunable on one GPU"],
      ["Prompt / prefix tuning", "a few learned virtual tokens", "tiny", "less expressive, mostly superseded by LoRA"],
    ]],
    ["p", "Practical defaults (as of 2026, treat as starting points): rank 8 to 64, $\\alpha$ about $r$ to $2r$, learning rate around $10^{-4}$ to $2\\times10^{-4}$ (higher than full fine-tuning because few parameters move), adapters on **all** linear layers (attention and MLP), which usually works better than only $W_q, W_v$."],
    ["warn", "A LoRA adapter is tied to the exact base model it was trained on. It cannot be moved to a different model or a different quantization without quality loss. Also, training only a few parameters is not a guarantee against forgetting or against learning bad behaviour: a LoRA on harmful data still changes the model's behaviour."],
    ["tip", "Serving many fine-tunes? Keep one base model in memory and swap or batch adapters per request. This is how multi-tenant systems offer hundreds of custom models cheaply."],
  ],
},
{
  id: "rlhf", group: "Training", title: "RLHF: Learning from Human Feedback",
  sub: "After SFT, use human preferences between pairs of answers to push the model toward what people actually want.",
  tags: ["alignment", "reinforcement learning"],
  blocks: [
    ["p", "SFT shows the model good answers. But for most prompts there is no single right answer, and it is far easier for a person to say *which of two answers is better* than to write a perfect one. **Reinforcement Learning from Human Feedback (RLHF)**, the method behind InstructGPT and the first ChatGPT, turns those comparisons into a training signal in three steps."],
    ["steps", [
      "**Start from the SFT model**, call it the reference policy $\\pi_{\\text{ref}}$.",
      "**Train a reward model** $r_\\phi(x,y)$ on human comparisons: for a prompt $x$, annotators pick the better of two answers $y_w$ (winner) and $y_l$ (loser). The model learns to give winners a higher score.",
      "**Optimise the policy** $\\pi_\\theta$ with reinforcement learning to produce answers that the reward model scores highly, while staying close to $\\pi_{\\text{ref}}$.",
    ]],
    ["diagram", "llm1_pipeline", [], "Where RLHF sits: step 3 of the usual post-training pipeline."],
    ["h", "Step 2: the reward model"],
    ["p", "The standard assumption is the **Bradley-Terry** model: the probability that answer $y_w$ beats $y_l$ is a sigmoid of the difference of their scores. Training minimises the negative log-likelihood of the human choices:"],
    ["math", R`P(y_w\succ y_l\mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),\qquad \mathcal{L}_{RM}=-\mathbb{E}\,\ln\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)`, "only the difference of scores matters, so the absolute scale is arbitrary"],
    ["p", "In practice the reward model is the same kind of LLM with a single-number head on its last token. A toy version with a linear scorer, trained on synthetic comparisons made from a hidden \"true\" preference with some noise:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
d = 5
w_true = rng.normal(size=d)                           # the hidden human preference (a score is w . features)

# 400 comparisons: two answers, each described by a feature vector; the human picks the higher true score (with noise)
A, B = rng.normal(size=(400, d)), rng.normal(size=(400, d))
noisy = (A - B) @ w_true + rng.normal(0, 1.0, size=400)
chosen   = np.where((noisy > 0)[:, None], A, B)       # y_w
rejected = np.where((noisy > 0)[:, None], B, A)       # y_l

sigmoid = lambda z: 1 / (1 + np.exp(-z))
w = np.zeros(d)                                       # reward model parameters
for step in range(501):
    margin = (chosen - rejected) @ w                  # r(y_w) - r(y_l)
    loss = -np.log(sigmoid(margin)).mean()            # Bradley-Terry loss
    grad = -((1 - sigmoid(margin))[:, None] * (chosen - rejected)).mean(axis=0)
    w -= 1.0 * grad
    if step in (0, 20, 100, 500):
        print(f"step {step:3d}  loss={loss:.3f}  accuracy on training pairs={(margin > 1e-9).mean():.3f}" if step else f"step {step:3d}  loss={loss:.3f}  (all scores equal, so every pair is a coin flip)")

# fresh test pairs, labelled by the TRUE preference without noise
At, Bt = rng.normal(size=(2000, d)), rng.normal(size=(2000, d))
agree = ((At - Bt) @ w > 0) == ((At - Bt) @ w_true > 0)
print("agreement with the true preference on new pairs:", agree.mean().round(3))
print("cosine(learned, true):", round(float(w @ w_true / np.linalg.norm(w) / np.linalg.norm(w_true)), 3))`, "step   0  loss=0.693  (all scores equal, so every pair is a coin flip)\nstep  20  loss=0.414  accuracy on training pairs=0.820\nstep 100  loss=0.413  accuracy on training pairs=0.820\nstep 500  loss=0.413  accuracy on training pairs=0.820\nagreement with the true preference on new pairs: 0.946\ncosine(learned, true): 0.989"],
    ["p", "Note the loss starts at $\\ln 2\\approx0.693$: with equal scores the model thinks each answer is equally likely to win. It cannot get to zero because the labels are noisy (people disagree, and sometimes err), which is realistic: real reward models reach agreement with held-out human labels of well below 100%."],
    ["h", "Step 3: optimise against the reward, with a leash"],
    ["p", "Now we want a policy that maximises the reward. If we only maximised $r_\\phi$, the policy would find odd outputs that fool the reward model (it is an imperfect proxy) and drift away from fluent language. So the objective has a **KL penalty** that charges the policy for straying from the reference:"],
    ["math", R`\max_{\theta}\ \ \mathbb{E}_{x,\;y\sim\pi_\theta(\cdot\mid x)}\Big[r_\phi(x,y)\Big]\;-\;\beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)`, "beta controls the leash: large beta stays close to the SFT model"],
    ["p", "Classic RLHF optimises this with **PPO** (Proximal Policy Optimization), which takes policy-gradient steps but *clips* how much the probability ratio of any action may change per update, for stability. The system needs four models in memory: the policy, the frozen reference, the reward model and a value network (a \"critic\") that reduces gradient variance. That is heavy, which motivated simpler methods ([DPO](llm.html#dpo), and the critic-free GRPO used for [reasoning models](llm.html#reasoning-models))."],
    ["p", "To see the objective at work without any neural network, take a \"policy\" that picks one of four canned answers. We can compute the expected gradient exactly, and we know the optimum has a closed form, $\\pi^*(y)\\propto\\pi_{\\text{ref}}(y)\\,e^{r(y)/\\beta}$ (an exponential tilt of the reference by the reward):"],
    ["code", R`
import numpy as np

softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
answers = ["rude", "terse", "helpful", "helpful + verbose"]
r = np.array([-1.0, 0.0, 1.0, 1.2])                  # reward model scores
ref = np.array([0.30, 0.40, 0.25, 0.05])             # SFT model: mostly terse

for beta in [1.0, 0.1]:
    theta = np.log(ref).copy()                       # start at the reference policy
    for _ in range(20000):
        pi = softmax(theta)
        g = r - beta * np.log(pi / ref)              # reward minus KL "cost" per answer
        theta += 0.1 * pi * (g - (pi * g).sum())     # exact gradient of E[r] - beta*KL through the softmax
    pi = softmax(theta)
    star = ref * np.exp(r / beta); star /= star.sum()    # closed-form optimum
    kl = (pi * np.log(pi / ref)).sum()
    print(f"beta={beta}: policy={pi.round(3)}  closed form={star.round(3)}  E[r]={(pi * r).sum():.3f}  KL={kl:.3f}")
print("reference           :", ref, " E[r] =", round(float((ref * r).sum()), 3))`, "beta=1.0: policy=[0.081 0.295 0.501 0.122]  closed form=[0.081 0.295 0.501 0.122]  E[r]=0.567  KL=0.262\nbeta=0.1: policy=[0.    0.001 0.403 0.596]  closed form=[0.    0.    0.404 0.596]  E[r]=1.118  KL=1.665\nreference           : [0.3  0.4  0.25 0.05]  E[r] = 0.01"],
    ["p", "With a large $\\beta$ the policy barely moves from the reference; with a small $\\beta$ it piles nearly all its mass on the highest-reward answer. In the real system the same trade-off is controlled by $\\beta$, and the optimisation is done on sampled text with a neural policy instead of exactly on four options."],
    ["h", "What goes wrong"],
    ["table", ["Problem", "What happens", "Mitigation"], [
      ["Reward hacking", "the policy finds outputs that score high but are not actually good (the reward model is only a proxy)", "KL penalty, early stopping, better and larger reward models, periodic refresh of preference data"],
      ["Length bias", "reward models often prefer longer, more elaborate answers, so the policy becomes wordy", "length-normalised or length-controlled rewards, balanced data"],
      ["Sycophancy", "labellers like agreeable answers, so the model learns to flatter and agree", "train on data where truthfulness is rewarded over agreement"],
      ["Mode collapse", "less diverse outputs after tuning", "keep KL strong, mix in SFT loss, evaluate diversity"],
      ["Label noise and bias", "human preferences are inconsistent and reflect the annotators' demographics and guidelines", "clear guidelines, several annotators per pair, audits"],
    ]],
    ["note", "Many labs now also use **verifiable rewards** instead of a learned reward model for tasks with checkable answers (a maths result, passing unit tests). There is nothing to hack in a correct unit test, which is a big reason for the progress in [reasoning models](llm.html#reasoning-models). Human-preference RLHF is still the main tool for subjective qualities such as helpfulness and tone."],
    ["danger", "Treating the reward model's score as ground truth. A policy optimised hard against any fixed proxy will eventually exploit its blind spots; the reward goes up while real quality goes down. Always evaluate the final model with independent human or held-out checks ([evaluating LLMs](llm.html#llm-evaluation))."],
  ],
},
{
  id: "dpo", group: "Training", title: "DPO, RLAIF & Constitutional AI",
  sub: "Skip the reward model and the RL loop: train directly on preference pairs, and let AI feedback replace some of the human labelling.",
  tags: ["alignment", "preference tuning"],
  blocks: [
    ["p", "[RLHF](llm.html#rlhf) works but is complicated: a reward model, four networks in memory, an unstable RL loop. **Direct Preference Optimization (DPO)**, introduced by Rafailov et al. in 2023, showed that you can get the same effect with a loss you compute on the preference pairs directly, as simply as a classifier."],
    ["h", "The derivation in three lines"],
    ["ol", [
      "The RLHF objective has a known optimal policy: $\\pi^*(y\\mid x)=\\dfrac{1}{Z(x)}\\pi_{\\text{ref}}(y\\mid x)\\exp\\!\\big(r(x,y)/\\beta\\big)$.",
      "Solve that for the reward: $r(x,y)=\\beta\\ln\\dfrac{\\pi^*(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)}+\\beta\\ln Z(x)$. The reward is a log-ratio between the policy and the reference (plus a term that depends only on $x$).",
      "Put this into the Bradley-Terry preference model. $Z(x)$ appears in both answers' rewards and **cancels in the difference**, leaving a loss that needs only the policy and the reference model:",
    ]],
    ["math", R`\mathcal{L}_{DPO}=-\,\mathbb{E}\;\ln\sigma\!\Big(\beta\Big[\ln\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\ln\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big]\Big)`, "log-probabilities are sums of token log-probabilities over the whole answer"],
    ["p", "In words: increase the probability of the chosen answer *relative to the reference* and decrease the rejected one, with the sigmoid stopping you from pushing already-correct pairs further. The quantity $\\beta\\ln(\\pi_\\theta/\\pi_{\\text{ref}})$ is called the **implicit reward**: the model itself is the reward model."],
    ["code", R`
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))

def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta=0.1):
    # each argument: total log-probability of the chosen / rejected answer under the policy / reference model
    margin = beta * ((logp_w - ref_w) - (logp_l - ref_l))      # difference of implicit rewards
    return -np.log(sigmoid(margin)), margin

# at the start the policy IS the reference, so the margin is 0 and the loss is ln 2
loss, m = dpo_loss(-42.0, -40.0, -42.0, -40.0)
print("at init  : margin =", m, " loss =", round(float(loss), 4), " ln2 =", round(float(np.log(2)), 4))

# the policy raises P(chosen) by 3 nats and lowers P(rejected) by 2 nats
loss, m = dpo_loss(-39.0, -42.0, -42.0, -40.0)
print("improved : margin =", round(float(m), 2), " loss =", round(float(loss), 4))

# the policy makes the wrong answer more likely
loss, m = dpo_loss(-45.0, -38.0, -42.0, -40.0)
print("worse    : margin =", round(float(m), 2), " loss =", round(float(loss), 4))`, "at init  : margin = 0.0  loss = 0.6931  ln2 = 0.6931\nimproved : margin = 0.5  loss = 0.4741\nworse    : margin = -0.5  loss = 0.9741"],
    ["p", "Now a complete toy DPO training loop. The policy chooses among four canned answers (as in the RLHF toy), and the gradient of the DPO loss with respect to the logits is wonderfully simple: with $z=\\beta[\\ldots]$, it is $-\\sigma(-z)\\,\\beta\\,(e_w-e_l)$, i.e. push the winner's logit up and the loser's down, by less and less as the pair is already satisfied."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()

answers = ["rude", "terse", "helpful", "helpful + verbose"]
quality = np.array([0, 1, 2, 3])                     # hidden ground-truth ranking used by the "labeller"
ref = np.array([0.30, 0.40, 0.25, 0.05])
theta = np.log(ref).copy()                           # policy logits start at the reference
beta = 0.5

# preference data: random pairs, labelled by the ranking
pairs = []
while len(pairs) < 200:
    a, b = rng.choice(4, 2, replace=False)
    pairs.append((a, b) if quality[a] > quality[b] else (b, a))     # (winner, loser)

for epoch in range(300):
    grad = np.zeros(4)
    for w, l in pairs:
        pi = softmax(theta)
        z = beta * ((np.log(pi[w]) - np.log(ref[w])) - (np.log(pi[l]) - np.log(ref[l])))
        grad[w] += -sigmoid(-z) * beta           # d loss / d theta
        grad[l] -= -sigmoid(-z) * beta
    theta -= 0.5 * grad / len(pairs)

print("reference:", ref)
print("after DPO:", softmax(theta).round(3))
print("most likely answer:", answers[int(np.argmax(softmax(theta)))])`, "reference: [0.3  0.4  0.25 0.05]\nafter DPO: [0.    0.005 0.063 0.932]\nmost likely answer: helpful + verbose"],
    ["p", "DPO has no sampling loop, no reward model and no critic: it is a supervised-style loss, so it is cheap, stable and easy to implement. With Hugging Face TRL:"],
    ["code", R`
# Hugging Face TRL (needs model weights, so it is not run here)
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")   # columns: prompt, chosen, rejected
trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",              # the SFT model; the reference copy is created internally
    args=DPOConfig(output_dir="dpo-out", beta=0.1, learning_rate=5e-7),   # tiny learning rate: DPO is sensitive
    train_dataset=ds,
)
trainer.train()`, "", "TRL", "local"],
    ["table", ["", "RLHF (PPO)", "DPO"], [
      ["Needs a reward model", "yes", "no (implicit)"],
      ["Samples from the model during training", "yes (on-policy)", "no, uses a fixed dataset (off-policy)"],
      ["Models in memory", "policy, reference, reward, value", "policy, reference"],
      ["Stability and simplicity", "tricky, many hyperparameters", "simple and stable"],
      ["Weak spot", "cost, reward hacking", "limited by the quality and coverage of the fixed pairs; can lower the likelihood of both answers; length drift"],
    ]],
    ["p", "Many variants exist (IPO, KTO, ORPO, SimPO and others) that change the loss to fix specific weaknesses, for example KTO works with single thumbs-up/down labels instead of pairs. As of 2026, simple DPO-style methods are widely used for preference tuning of open models, while the largest labs combine preference tuning with on-policy RL."],
    ["h", "RLAIF: let a model do the labelling"],
    ["p", "Human preference labels are slow and costly. **RLAIF** (Reinforcement Learning from AI Feedback) asks a strong model to judge which of two answers is better, following written criteria, and uses those AI-made labels in place of (or alongside) human ones. The training step afterwards is unchanged: a reward model plus RL, or DPO directly. Studies report that AI labels can come close to human labels on tasks such as summarisation and helpfulness, but the judge inherits its own biases (a preference for long, confident, or its-own-style answers)."],
    ["code", R`
# A rule-based "AI judge" turns principles into preference labels. Real judges are LLMs given the same principles in a prompt.
principles = [
    ("is polite",       lambda a: 1 if any(w in a.lower() for w in ["please", "happy to", "sure"]) else 0),
    ("is not rude",     lambda a: -2 if any(w in a.lower() for w in ["stupid", "shut up"]) else 0),
    ("is on topic",     lambda a: 1 if "refund" in a.lower() else 0),
    ("is concise",      lambda a: 1 if len(a.split()) <= 12 else 0),
]

def judge(a, b):
    sa, sb = sum(f(a) for _, f in principles), sum(f(b) for _, f in principles)
    return (a, b) if sa >= sb else (b, a)                 # (chosen, rejected)

pairs = [("Shut up, that is a stupid question.", "Sure, I am happy to help with your refund."),
         ("Your refund will arrive in 5 days.", "Please wait, we are happy to look into the weather forecast for you today and tomorrow and next week.")]
for a, b in pairs:
    chosen, rejected = judge(a, b)
    print("chosen  :", chosen, "\nrejected:", rejected, "\n")`, "chosen  : Sure, I am happy to help with your refund. \nrejected: Shut up, that is a stupid question. \n\nchosen  : Your refund will arrive in 5 days. \nrejected: Please wait, we are happy to look into the weather forecast for you today and tomorrow and next week."],
    ["h", "Constitutional AI"],
    ["p", "**Constitutional AI** (Bai et al., Anthropic, 2022) applies this idea to safety. Instead of humans labelling thousands of harmful outputs, the model is given a short written list of principles (a *constitution*) and trains itself against it, in two phases:"],
    ["list", [
      "**Supervised phase (critique and revise).** The model answers a prompt, then is asked to critique its own answer against a principle, then to rewrite it. The revised answers are used for SFT.",
      "**RL phase (RLAIF).** The model compares pairs of answers according to the principles; those AI preferences train a preference model, which then drives RL like in RLHF.",
    ]],
    ["p", "The benefits are scale and transparency: the rules are written down and can be inspected and changed, rather than being implicit in thousands of individual human labels. Humans still write the constitution and still check the results."],
    ["warn", "A preference-tuned model only learns what the preference data rewards. If chosen answers are systematically longer, more flattering, or more cautious than rejected ones, the model learns exactly that. Inspect your pairs, control for length, and keep a held-out check of what you actually care about."],
    ["tip", "Practical order of operations (as of 2026): pretrain, then SFT, then DPO-style preference tuning, then optional on-policy RL (with verifiable rewards where possible). Each stage is cheaper than the one before it."],
  ],
},
{
  id: "reasoning-models", group: "Training", title: "Reasoning Models & Test-Time Compute",
  sub: "Models that think before they answer: more computation at answer time, learned through reinforcement learning on checkable problems.",
  tags: ["reasoning", "reinforcement learning"],
  blocks: [
    ["p", "A normal LLM spends the same effort on \"2+2\" and on a hard maths proof: one forward pass per token, whatever the question. A **reasoning model** is trained to produce a long internal chain of thought first (trying approaches, checking itself, backtracking) and only then the final answer. Spending more tokens on thinking is a form of **test-time compute**: computation done when answering rather than during training."],
    ["h", "Why thinking longer helps"],
    ["p", "A transformer does a fixed amount of computation per generated token. Hard problems need more serial steps than one pass can do. Writing intermediate steps into the context gives the model scratch space: each new token can read all earlier steps (this is **chain of thought**, CoT). Simple prompting (\"let's think step by step\") already helps a lot on arithmetic and logic. Reasoning models take this much further by *training* the behaviour."],
    ["h", "Test-time scaling without retraining"],
    ["p", "Even a normal model can trade compute for accuracy by sampling several answers:"],
    ["list", [
      "**Self-consistency / majority vote**: sample $n$ reasoning chains at a temperature above 0, extract each final answer, return the most common one.",
      "**Best-of-N with a verifier**: sample $n$ answers and let a reward model or a checker (unit tests, a calculator) pick one.",
      "**Search**: expand and prune partial reasoning steps (tree search), scoring the steps with a process reward model.",
    ]],
    ["p", "A simulation shows why voting works: wrong answers are scattered over many different values, while right answers all agree. Suppose a single sample is right 40% of the time and the 60% wrong samples are spread uniformly over 9 wrong values:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
p_correct, n_wrong_values, trials = 0.4, 9, 20000

def vote_accuracy(n):
    correct = 0
    for _ in range(trials):
        # answer 0 is correct; each sample is correct with prob p, otherwise one of the wrong values 1..9
        s = np.where(rng.random(n) < p_correct, 0, rng.integers(1, n_wrong_values + 1, size=n))
        counts = np.bincount(s, minlength=n_wrong_values + 1)
        correct += counts.argmax() == 0                       # ties go to the smallest value; fine for a toy
    return correct / trials

for n in [1, 3, 5, 9, 17, 33]:
    print(f"n={n:2d} samples: majority-vote accuracy = {vote_accuracy(n):.3f}")`, "n= 1 samples: majority-vote accuracy = 0.395\nn= 3 samples: majority-vote accuracy = 0.735\nn= 5 samples: majority-vote accuracy = 0.778\nn= 9 samples: majority-vote accuracy = 0.880\nn=17 samples: majority-vote accuracy = 0.963\nn=33 samples: majority-vote accuracy = 0.997"],
    ["p", "Accuracy climbs from 0.4 towards 1 as $n$ grows, but with diminishing returns and a cost that grows linearly with $n$. The trick stops working when wrong answers are *correlated* (the model makes the same mistake every time), because then the wrong answer wins the vote."],
    ["h", "Training reasoning: RL on verifiable rewards"],
    ["p", "The 2024 to 2025 breakthrough (OpenAI's o1, then DeepSeek-R1 and many others) was to train long chains of thought with **reinforcement learning against automatic checkers**: for maths, compare the final number with the known answer; for code, run the unit tests. This is called RL with verifiable rewards (RLVR). There is no learned reward model to fool, and the reward is cheap and exact. The model discovers by itself behaviours like re-checking its work and trying a different approach, because those lead to correct answers more often."],
    ["p", "DeepSeek's **GRPO** (Group Relative Policy Optimization) is a popular, simple algorithm for this. For each prompt, sample a group of $G$ answers, score each with the checker, and use the **group-normalised reward as the advantage**, which removes the need for a separate value network:"],
    ["math", R`A_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}`, "answers better than their siblings are reinforced, worse ones are discouraged"],
    ["p", "The policy is then updated with a clipped, PPO-style objective weighted by $A_i$ (plus a small KL term to a reference model). A toy of the principle: a policy chooses one of three \"strategies\" with hidden success rates 0.2, 0.5 and 0.8, and learns from group-relative advantages only:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
success = np.array([0.2, 0.5, 0.8])         # hidden: how often each strategy solves the problem
theta = np.zeros(3)
G, lr = 16, 0.5

for step in range(201):
    pi = softmax(theta)
    acts = rng.choice(3, size=G, p=pi)                       # a group of G attempts
    rewards = (rng.random(G) < success[acts]).astype(float)  # verifiable reward: 1 if correct, else 0
    if rewards.std() > 0:
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)   # group-relative advantage
    else:
        adv = np.zeros(G)                                    # all right or all wrong: no signal
    grad = np.zeros(3)
    for a, A in zip(acts, adv):
        grad += A * (np.eye(3)[a] - pi)                      # grad of log pi(a) times advantage
    theta += lr * grad / G
    if step in (0, 50, 200):
        print(f"step {step:3d}  policy = {softmax(theta).round(3)}  expected success = {(softmax(theta) * success).sum():.3f}")`, "step   0  policy = [0.285 0.306 0.409]  expected success = 0.537\nstep  50  policy = [0.009 0.019 0.972]  expected success = 0.789\nstep 200  policy = [0.003 0.003 0.994]  expected success = 0.797"],
    ["p", "Real GRPO does exactly this at a vastly larger scale: the \"action\" is an entire chain of thought of thousands of tokens, the policy is a billion-parameter LLM, and the probability of each token is nudged up or down by the same advantage."],
    ["h", "Using reasoning models"],
    ["table", ["Aspect", "Standard chat model", "Reasoning model"], [
      ["Output", "answer directly", "hidden or summarised thinking tokens, then the answer"],
      ["Latency and cost", "low", "higher: you pay for the thinking tokens (often many times the answer length)"],
      ["Best at", "chat, writing, extraction, simple Q&A", "multi-step maths, code, planning, puzzles, careful analysis"],
      ["Prompting", "show examples, ask for steps", "state the problem and constraints clearly; avoid micromanaging the steps"],
      ["Controls (as of 2026)", "temperature, max tokens", "often a thinking budget or effort level"],
    ]],
    ["list", [
      "**Overthinking**: on easy questions, a reasoning model can waste tokens. Effort or budget controls exist for this reason.",
      "**Faithfulness is not guaranteed**: the written reasoning is not necessarily an accurate account of how the model reached its answer, so do not treat it as a proof or an audit log.",
      "**Verifiable domains improve fastest**: where answers can be checked automatically, RL works well. Open-ended tasks without a cheap checker improve more slowly.",
      "Reasoning is a **second scaling axis** next to [training scale](llm.html#scaling-laws): for the same pretrained model, more thinking gives better answers on hard problems, up to a point.",
    ]],
    ["tip", "A cheap way to get some of this benefit from any model: ask for the reasoning first and the final answer last, put the answer in a fixed format (see [structured output](llm.html#structured-output)), and, for important cases, sample several times and vote or verify."],
    ["danger", "Asking a model to \"explain its reasoning\" **after** it already gave an answer. The explanation is then a plausible-sounding justification, not the computation. Reasoning only helps if it comes *before* the answer."],
  ],
},
{
  id: "distillation", group: "Training", title: "Distillation & Small Models",
  sub: "Train a small, cheap student to imitate a large teacher and keep most of the quality at a fraction of the serving cost.",
  tags: ["efficiency", "training"],
  blocks: [
    ["p", "Big models are expensive to serve ([about $2N$ FLOPs per generated token](llm.html#scaling-laws)). **Knowledge distillation** (Hinton, Vinyals and Dean, 2015) trains a small **student** to reproduce the behaviour of a large **teacher**. The student usually ends up much better than the same small model trained from scratch on the same data, because the teacher's outputs carry more information than the raw labels."],
    ["h", "Soft targets carry \"dark knowledge\""],
    ["p", "A hard label says only \"this image is a cat\". The teacher's probabilities say \"90% cat, 9% dog, 0.001% car\": cat is far more similar to dog than to car. That ranking of the wrong answers is extra training signal. A **temperature** $T>1$ in the softmax flattens the teacher's output so those small probabilities become visible:"],
    ["math", R`p_i^{(T)}=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}`],
    ["math", R`\mathcal{L}=(1-\lambda)\,\mathrm{CE}\big(y,\;p^{\,\text{student}}\big)\;+\;\lambda\,T^{2}\,\mathrm{KL}\big(p^{(T)}_{\text{teacher}}\;\|\;p^{(T)}_{\text{student}}\big)`, "the T squared keeps the size of the soft-target gradients comparable as T changes"],
    ["code", R`
import numpy as np

def softmax(z, T=1.0):
    e = np.exp((z - z.max()) / T)
    return e / e.sum()

teacher_logits = np.array([6.0, 3.5, 0.5, -2.0])        # classes: cat, dog, fox, car
for T in [1, 3, 8]:
    print(f"T={T}: teacher probabilities = {softmax(teacher_logits, T).round(4)}")

def distill_loss(student_logits, teacher_logits, y, T=3.0, lam=0.5):
    ce = -np.log(softmax(student_logits)[y])                              # ordinary loss on the hard label
    pt, ps = softmax(teacher_logits, T), softmax(student_logits, T)
    kl = (pt * np.log(pt / ps)).sum()                                      # match the teacher's softened distribution
    return (1 - lam) * ce + lam * T * T * kl

student = np.array([2.0, 1.0, 0.5, 0.0])
print("loss for a student that agrees weakly:", round(float(distill_loss(student, teacher_logits, 0)), 3))
print("loss for a student equal to teacher  :", round(float(distill_loss(teacher_logits, teacher_logits, 0)), 3))`, "T=1: teacher probabilities = [9.204e-01 7.550e-02 3.800e-03 3.000e-04]\nT=3: teacher probabilities = [0.601  0.2612 0.0961 0.0418]\nT=8: teacher probabilities = [0.3843 0.2811 0.1932 0.1414]\nloss for a student that agrees weakly: 1.133\nloss for a student equal to teacher  : 0.041"],
    ["p", "A small experiment: a teacher network is trained on plenty of labelled data. A student (a plain linear classifier) either learns from only **20 labelled points**, or from **600 unlabelled points that the teacher labels with its probabilities**. Distillation lets the student use unlabelled data, which is cheap:"],
    ["code", R`
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

X, y = make_blobs(n_samples=3000, centers=[[0, 0], [2.2, 0], [1.1, 1.9]], cluster_std=1.1, random_state=0)
Xtr, ytr, Xte, yte = X[:2000], y[:2000], X[2000:], y[2000:]
teacher = MLPClassifier((32,), max_iter=400, random_state=0).fit(Xtr, ytr)
print("teacher test accuracy:", round(teacher.score(Xte, yte), 3))

def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def train_student(Xs, targets, steps=2000, lr=0.5):       # softmax regression; targets may be hard or soft
    W, b = np.zeros((2, 3)), np.zeros(3)
    for _ in range(steps):
        g = (softmax(Xs @ W + b) - targets) / len(Xs)
        W -= lr * Xs.T @ g; b -= lr * g.sum(0)
    return W, b

accuracy = lambda W, b: (np.argmax(Xte @ W + b, 1) == yte).mean()
hard_acc, soft_acc = [], []
for seed in range(5):
    idx = np.random.default_rng(seed).choice(2000, 20, replace=False)         # only 20 labelled points
    W, b = train_student(Xtr[idx], np.eye(3)[ytr[idx]]); hard_acc.append(accuracy(W, b))
    W, b = train_student(Xtr[:600], teacher.predict_proba(Xtr[:600])); soft_acc.append(accuracy(W, b))
print("student, 20 hard labels      : mean accuracy", round(float(np.mean(hard_acc)), 3))
print("student, 600 teacher-soft-labelled points: mean accuracy", round(float(np.mean(soft_acc)), 3))`, "teacher test accuracy: 0.743\nstudent, 20 hard labels      : mean accuracy 0.698\nstudent, 600 teacher-soft-labelled points: mean accuracy 0.743"],
    ["h", "Distillation for LLMs"],
    ["table", ["Style", "What the student learns from", "Notes"], [
      ["Logit (white-box) distillation", "the teacher's full next-token probability distribution at every position", "needs the same tokenizer and access to the teacher's logits; richest signal. Used when training small models inside one family"],
      ["Sequence-level (black-box) distillation", "text the teacher **generated**, used as ordinary SFT data", "works with any teacher, even behind an API; this is how most open \"distilled\" models are made"],
      ["Reasoning-trace distillation", "long chains of thought written by a reasoning teacher", "e.g. the smaller DeepSeek-R1-Distill models were produced by fine-tuning open base models on R1's outputs"],
      ["Preference / judge distillation", "a strong model's preferences or scores", "this is [RLAIF](llm.html#dpo)"],
    ]],
    ["p", "Distillation is one of several routes to small-but-capable models. Others: **pruning** (remove weights or layers, then retrain briefly), [quantization](llm.html#quantization) (fewer bits per weight), and simply **training a small model on far more data** than Chinchilla recommends ([scaling laws](llm.html#scaling-laws)). Several small model families released in recent years combine these: for example, Meta described its Llama 3.2 1B and 3B models as produced with pruning and distillation from larger Llama models."],
    ["warn", "Distilling from a commercial model's outputs may violate its terms of service, and the student inherits the teacher's mistakes, biases and hallucinations, usually with less ability to recover from them. Check the licence of every teacher and dataset."],
    ["danger", "Evaluating a student only on the teacher-style prompts it was trained on. Students are good at imitating the teacher's *style* before they match its *ability*, so test on held-out and harder tasks."],
    ["tip", "Rule of thumb: use a big model to build the data and the evaluation, then distil or fine-tune a small model for the one task you serve at scale. Small + specialised often beats large + general on cost and latency."],
  ],
},
{
  id: "greedy-beam", group: "Inference", title: "Greedy & Beam Search",
  sub: "Deterministic ways to turn next-token probabilities into a sentence: take the best token each time, or keep several candidates alive.",
  tags: ["decoding"],
  blocks: [
    ["p", "A language model only outputs a probability for the next token. **Decoding** is the loop that turns those probabilities into text: compute the distribution, choose a token, append it, feed the longer text back in, repeat until an end-of-text token or a length limit. All decoding methods differ only in *how they choose*. Two of them are deterministic."],
    ["h", "Greedy decoding"],
    ["p", "Always take the single most probable token: $x_t=\\arg\\max_v P(v\\mid x_{<t})$. It is fast and needs no randomness, but it is **short-sighted**: the best token now may lead to a poor continuation, and a locally best choice cannot be undone. It also tends to loop (\"the the the\", repeated sentences) in open-ended text."],
    ["h", "Beam search"],
    ["p", "What we would like is the sequence with the highest *total* probability $P(x_1\\dots x_T)=\\prod_t P(x_t\\mid x_{<t})$. There are $V^T$ sequences, so we cannot check them all. **Beam search** keeps the best $B$ partial sequences (the *beam*) at each step: extend each by every possible token, score the resulting $B\\times V$ candidates by cumulative **log**-probability (sums of logs, so no underflow), keep the top $B$, repeat. With $B=1$ it is greedy decoding. Step through the toy below: the amber outline is where greedy ends up."],
    ["viz", "llm1_beam"],
    ["p", "From scratch, on the same toy table (the next-token distribution depends only on the last token), with an exhaustive search as ground truth:"],
    ["code", R`
import numpy as np
from itertools import product

# P(next token | last token): rows a, b, c; the first row is the start
table = {"": [0.5, 0.4, 0.1], "a": [0.4, 0.3, 0.3], "b": [0.1, 0.8, 0.1], "c": [0.3, 0.3, 0.4]}
toks = ["a", "b", "c"]
steps = 3

def beam_search(width):
    beams = [("", 0.0)]                                    # (sequence, cumulative log-probability)
    for _ in range(steps):
        cand = []
        for seq, lp in beams:
            for t, p in zip(toks, table[seq[-1:]]):
                cand.append((seq + t, lp + np.log(p)))
        cand.sort(key=lambda c: -c[1])
        beams = cand[:width]                               # prune to the best 'width'
    return beams[0]

def exhaustive():
    best = max(product(toks, repeat=steps),
               key=lambda s: sum(np.log(table[("".join(s[:i]))[-1:]][toks.index(s[i])]) for i in range(steps)))
    return "".join(best)

for w in [1, 2, 3]:
    seq, lp = beam_search(w)
    print(f"beam width {w}: {seq}  P={np.exp(lp):.3f}" + ("   (this is greedy)" if w == 1 else ""))
print("exhaustive search best:", exhaustive())`, "beam width 1: aaa  P=0.080   (this is greedy)\nbeam width 2: bbb  P=0.256\nbeam width 3: bbb  P=0.256\nexhaustive search best: bbb"],
    ["h", "Length bias"],
    ["p", "Every token multiplies the probability by something below 1, so *shorter* sequences always score higher in raw probability. Beam search therefore tends to end too early. The usual fix is **length normalisation**: rank finished candidates by the average log-probability per token, $\\frac{1}{|y|^{\\alpha}}\\sum_t\\ln P(y_t\\mid\\cdot)$, with $\\alpha$ between 0 (no normalisation) and 1."],
    ["code", R`
import math

short = ("The answer is 4.", -2.0, 5)                                   # (text, total log-prob, length in tokens)
long_ = ("The answer is 4, because 2 plus 2 equals 4.", -3.5, 12)
for name, alpha in [("raw total log-prob", 0.0), ("normalised by length", 1.0)]:
    best = max([short, long_], key=lambda c: c[1] / c[2] ** alpha)
    print(f"{name:22s} -> picks: {best[0]}")`, "raw total log-prob     -> picks: The answer is 4.\nnormalised by length   -> picks: The answer is 4, because 2 plus 2 equals 4."],
    ["h", "When to use which"],
    ["table", ["Method", "Strengths", "Weaknesses", "Typical use"], [
      ["Greedy", "fastest, deterministic, simple", "repetitive, can miss better sequences", "extraction, classification, code with temperature 0"],
      ["Beam search", "finds higher-probability sequences; stable", "$B$ times the compute and memory; bland, repetitive text for open-ended generation", "machine translation, speech recognition, summarisation"],
      ["[Sampling](llm.html#sampling)", "diverse, natural, creative", "can wander; not repeatable without a seed", "chat, creative writing, brainstorming"],
    ]],
    ["p", "Why not always use the highest-probability sequence? For open-ended text the most probable continuation is often dull and repetitive; human text is not the most likely string, it is a typical sample. That is why chat models sample, and why a famous finding (Holtzman et al., 2019) is that beam search output reads *less* human than sampled output even though its probability is higher."],
    ["code", R`
# Hugging Face generate: the same strategies as arguments (needs model weights, so it is not run here)
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
inputs = tok("The capital of France is", return_tensors="pt")

greedy = model.generate(**inputs, max_new_tokens=20, do_sample=False)
beam   = model.generate(**inputs, max_new_tokens=20, num_beams=4, length_penalty=1.0, early_stopping=True)
print(tok.decode(greedy[0]), "\n", tok.decode(beam[0]))`, "", "Hugging Face", "local"],
    ["warn", "Greedy decoding is not guaranteed to be bit-for-bit repeatable in production: floating-point rounding differs across GPUs, batch sizes and kernels, and a near-tie between two tokens can flip. Treat \"temperature 0\" as *nearly* deterministic."],
    ["danger", "Using a large beam width for chat. Quality does not keep improving with $B$; past a small width, output often gets shorter, blander and more repetitive. For open-ended text use [sampling](llm.html#sampling)."],
  ],
},
{
  id: "sampling", group: "Inference", title: "Sampling: Temperature, Top-k & Top-p",
  sub: "Pick the next token at random in proportion to its probability, then reshape the distribution to control creativity versus reliability.",
  tags: ["decoding"],
  blocks: [
    ["p", "**Sampling** draws the next token at random from the model's distribution: a token with probability 0.6 is chosen 60% of the time. This gives varied, natural text. The danger is the long tail: a vocabulary of 100,000 tokens has thousands of nonsensical options, each with tiny probability, and together they can add up to a few percent per step. Over hundreds of steps, one of them eventually gets picked and the text goes off the rails. The controls below reshape the distribution to tame that tail."],
    ["h", "Temperature"],
    ["p", "Divide the logits by a temperature $T$ before the softmax:"],
    ["math", R`p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}`],
    ["list", [
      "$T<1$ **sharpens**: the top tokens get even more of the mass. As $T\\to0$ it becomes greedy decoding.",
      "$T=1$ is the model's own distribution.",
      "$T>1$ **flattens**: unlikely tokens become more likely. As $T\\to\\infty$ it becomes a uniform random choice.",
    ]],
    ["viz", "llm1_temperature"],
    ["h", "Top-k and top-p"],
    ["p", "Temperature keeps every token possible. **Top-k** and **top-p** instead cut the tail off entirely:"],
    ["list", [
      "**Top-k**: keep only the $k$ most likely tokens (say $k=40$), set the rest to zero, renormalise. Simple, but $k$ is the same whether the model is sure or confused: if one token has 99% probability you still keep 40 candidates, and if 200 tokens are all reasonable, 40 is too few.",
      "**Top-p (nucleus sampling)**: sort by probability and keep the smallest set whose probabilities add up to at least $p$ (say 0.9), then renormalise. The set size **adapts**: one token when the model is confident, many when it is not. Introduced by Holtzman et al. (2019).",
      "**Min-p** (a newer, popular variant): drop tokens whose probability is below a fraction of the top token's probability.",
    ]],
    ["viz", "llm1_topkp"],
    ["h", "All together, from scratch"],
    ["code", R`
import numpy as np

def sample(logits, T=1.0, top_k=None, top_p=None, rng=None):
    z = np.asarray(logits, dtype=float) / max(T, 1e-8)          # 1. temperature
    p = np.exp(z - z.max()); p /= p.sum()                       #    softmax (shifted for stability)
    order = np.argsort(-p)                                      # token ids from most to least likely
    if top_k is not None:                                       # 2. top-k: zero everything beyond the k best
        p[order[top_k:]] = 0.0
        p /= p.sum()
    if top_p is not None:                                       # 3. top-p: smallest prefix with cumulative mass >= p
        csum = np.cumsum(p[order])
        cut = int(np.searchsorted(csum, top_p - 1e-12)) + 1     #    number of tokens to keep
        p[order[cut:]] = 0.0
        p /= p.sum()
    return int(rng.choice(len(p), p=p)), p                      # 4. draw one token

rng = np.random.default_rng(0)
logits = np.array([3.2, 2.8, 2.5, 2.2, 1.6, 1.1, 0.5, 0.0, -1.0, -2.5])
words = ["the", "a", "cat", "dog", "sat", "mat", "ran", "on", "zebra", "qux"]

_, p_full = sample(logits, 1.0, rng=rng)
_, p_cut = sample(logits, 1.0, top_k=5, top_p=0.9, rng=rng)
print("probs T=1       :", p_full.round(3))
print("probs k=5,p=0.9 :", p_cut.round(3), " kept:", int((p_cut > 0).sum()), "tokens")

for T in [0.2, 1.0, 2.0]:
    draws = [sample(logits, T, rng=rng)[0] for _ in range(4000)]
    counts = np.bincount(draws, minlength=len(words))
    print(f"T={T}: most common = {words[counts.argmax()]!r} ({counts.max() / 4000:.2f}),  distinct tokens drawn = {int((counts > 0).sum())}")`, "probs T=1       : [0.335 0.225 0.166 0.123 0.068 0.041 0.023 0.014 0.005 0.001]\nprobs k=5,p=0.9 : [0.395 0.264 0.196 0.145 0.    0.    0.    0.    0.    0.   ]  kept: 4 tokens\nT=0.2: most common = 'the' (0.86),  distinct tokens drawn = 4\nT=1.0: most common = 'the' (0.34),  distinct tokens drawn = 10\nT=2.0: most common = 'the' (0.21),  distinct tokens drawn = 10"],
    ["h", "Order matters, and other knobs"],
    ["p", "Libraries usually apply: repetition/frequency penalties (adjust logits of tokens already used) → temperature → top-k → top-p (or min-p) → sample. Setting top-k and top-p together is allowed; both filters apply. **Repetition, presence and frequency penalties** subtract from the logit of tokens that already appeared, which reduces loops but can also stop a model from legitimately repeating a name or code identifier. A **seed** makes sampling reproducible."],
    ["table", ["Task", "Typical setting (a starting point, not a rule)"], [
      ["Code, extraction, classification, factual Q&A", "temperature 0 to 0.3 (near-greedy)"],
      ["General chat and assistants", "temperature around 0.6 to 0.8, top-p around 0.9 to 0.95"],
      ["Creative writing, brainstorming", "temperature around 1.0 or a bit above, top-p 0.95, perhaps min-p"],
      ["Sampling many answers to vote or search ([reasoning](llm.html#reasoning-models))", "temperature around 0.6 to 1.0, so the samples differ"],
    ]],
    ["code", R`
# Hugging Face generate (needs model weights, so it is not run here)
out = model.generate(**inputs, max_new_tokens=50, do_sample=True,
                     temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1)`, "", "Hugging Face", "local"],
    ["tip", "Change **one** knob at a time, and usually only temperature or top-p, not both: they overlap in what they do. Hosted APIs expose different subsets of these parameters, and the best defaults differ per model, so check the provider's documentation (see [using LLM APIs](llm.html#llm-apis))."],
    ["warn", "A higher temperature does not make a model \"more creative\" in a deep sense; it makes it more random. Random is often wrong: factual errors and [hallucinations](llm.html#hallucinations) increase with temperature. Also, temperature does not change what the model *knows*; it only changes how often it picks lower-ranked options."],
    ["danger", "Applying temperature **after** top-k/top-p, or forgetting to renormalise after cutting. Always: scale logits, softmax, filter, then renormalise so the probabilities sum to 1 before drawing."],
  ],
},
{
  id: "kv-cache", group: "Inference", title: "The KV Cache",
  sub: "Remember the keys and values of every earlier token, so each new token costs one step instead of a re-read of the whole text.",
  tags: ["inference", "memory"],
  blocks: [
    ["p", "Generating text is a loop: run the model, get one token, append it, run again. Done naively, step $t$ pushes all $t$ tokens through the model again, although the first $t-1$ have not changed. Look at [attention](transformers.html#scaled-dot-product): the new token's **query** must be compared with the **keys** of all earlier tokens and then mixes their **values**. And thanks to the causal mask, the keys and values of earlier tokens never change when a new token is added. So we can compute them once and store them: the **KV cache**."],
    ["viz", "llm1_kvcache"],
    ["h", "Two phases of generation"],
    ["list", [
      "**Prefill**: the whole prompt goes through the model in one parallel pass. All its keys and values are computed and stored. This is what happens before the first token appears. It is compute-heavy and fast per token.",
      "**Decode**: now one token at a time. For each new token: compute its $q,k,v$, append $k,v$ to the cache, attend over the whole cache, produce the next token. Each step does little arithmetic but must **read the entire cache and all the model weights** from memory, so it is limited by memory bandwidth, not by compute.",
    ]],
    ["p", "Here is the whole idea in NumPy for one attention head: the cached version gives exactly the same output as recomputing everything, with far fewer K/V projections."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
d, T = 8, 6
Wq, Wk, Wv = (rng.normal(0, 0.5, size=(d, d)) for _ in range(3))
X = rng.normal(size=(T, d))                                # embeddings of T tokens

def softmax(s):
    e = np.exp(s - s.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)

# --- full attention over the whole sequence (what training does) ---
Q, K, V = X @ Wq, X @ Wk, X @ Wv
S = Q @ K.T / np.sqrt(d)
S = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, S)       # causal mask
full = softmax(S) @ V

# --- incremental attention with a KV cache (what generation does) ---
k_cache, v_cache, outs, projections = [], [], [], 0
for t in range(T):
    x = X[t]
    q, k, v = x @ Wq, x @ Wk, x @ Wv                       # only the NEW token is projected
    projections += 1
    k_cache.append(k); v_cache.append(v)                   # append to the cache
    Kc, Vc = np.array(k_cache), np.array(v_cache)          # (t+1, d): everything so far
    w = softmax(Kc @ q / np.sqrt(d))                       # attention of the new query over all cached keys
    outs.append(w @ Vc)
cached = np.array(outs)

print("cached output equals full attention:", np.allclose(cached, full))
print("K/V projections with cache:", projections, "  without cache (recompute every step):", sum(range(1, T + 1)))`, "cached output equals full attention: True\nK/V projections with cache: 6   without cache (recompute every step): 21"],
    ["h", "The price: memory"],
    ["p", "The cache holds a key and a value vector for every layer, for every token, for every sequence being served:"],
    ["math", R`\text{KV bytes}=2\times n_{\text{layers}}\times n_{\text{kv heads}}\times d_{\text{head}}\times T\times \text{bytes per number}`, "the 2 is one K and one V"],
    ["code", R`
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * bytes_per

# (name, layers, kv heads, head dim). 16-bit numbers = 2 bytes
models = [("7B with full multi-head attention (Llama-2-7B shape)", 32, 32, 128),
          ("8B with grouped-query attention, 8 KV heads (Llama-3-8B shape)", 32, 8, 128),
          ("same, with multi-query attention (1 KV head)", 32, 1, 128)]
for name, L, h, dh in models:
    per_tok = kv_bytes_per_token(L, h, dh)
    print(f"{name}\n   {per_tok / 1024:6.0f} KiB per token;  4k tokens: {per_tok * 4096 / 2**30:5.2f} GiB;  128k tokens: {per_tok * 131072 / 2**30:6.1f} GiB")

per_tok = kv_bytes_per_token(32, 8, 128)
print("\nbatch of 32 users, 8k tokens each, GQA model:", round(32 * 8192 * per_tok / 2**30, 1), "GiB of cache, versus about 15 GiB for the 8B weights in 16-bit")`, "7B with full multi-head attention (Llama-2-7B shape)\n      512 KiB per token;  4k tokens:  2.00 GiB;  128k tokens:   64.0 GiB\n8B with grouped-query attention, 8 KV heads (Llama-3-8B shape)\n      128 KiB per token;  4k tokens:  0.50 GiB;  128k tokens:   16.0 GiB\nsame, with multi-query attention (1 KV head)\n       16 KiB per token;  4k tokens:  0.06 GiB;  128k tokens:    2.0 GiB\n\nbatch of 32 users, 8k tokens each, GQA model: 32.0 GiB of cache, versus about 15 GiB for the 8B weights in 16-bit"],
    ["p", "The cache grows linearly with context length **and** with the number of simultaneous users, and soon rivals the model weights. This is the reason for most of the efficiency work in [efficient attention](transformers.html#efficient-attention):"],
    ["list", [
      "**MQA / GQA**: share keys and values among groups of query heads, cutting the cache by 4 to 32 times.",
      "**MLA** (multi-head latent attention, used by DeepSeek models): cache a compressed latent vector instead of full K and V.",
      "**PagedAttention** (vLLM): allocate the cache in small blocks like virtual-memory pages, so no space is wasted on reserved-but-unused length. See [serving](llm.html#serving).",
      "**Cache quantization**: store K and V in 8 or 4 bits ([quantization](llm.html#quantization)).",
      "**Sliding window / eviction**: keep only recent tokens (and a few initial \"sink\" tokens), see [long context](llm.html#long-context).",
      "**Prefix caching / prompt caching**: if many requests share the same beginning (a long system prompt, a document), reuse its cache instead of recomputing. Many hosted APIs offer this as discounted \"cached input tokens\".",
    ]],
    ["code", R`
# Hugging Face keeps the cache for you; here it is made visible (needs model weights, so it is not run here)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

ids = tok("The KV cache stores", return_tensors="pt").input_ids
out = model(ids, use_cache=True)                         # prefill: processes the prompt, returns past_key_values
past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
for _ in range(10):                                      # decode: feed ONE token plus the cache
    out = model(nxt, past_key_values=past, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    ids = torch.cat([ids, nxt], dim=1)
print(tok.decode(ids[0]))`, "", "Hugging Face", "local"],
    ["tip", "Attention scores are still computed against all $t$ cached keys, so decoding step $t$ costs $O(t)$ instead of $O(t^2)$ for attention, and the feed-forward layers cost the same for every step. The cache removes *recomputation*, not *reading*, which is why long contexts still make each token slower."],
    ["danger", "Forgetting that the cache depends on the exact prefix. If you change any earlier token (editing a system prompt, inserting a timestamp at the top), every later cached entry is invalid. Put stable content first and varying content last so prompt caching can hit."],
  ],
},
{
  id: "quantization", group: "Inference", title: "Quantization",
  sub: "Store the weights with fewer bits (8, 4, even less) so a model takes a fraction of the memory and runs faster, at a small cost in accuracy.",
  tags: ["efficiency", "inference"],
  blocks: [
    ["p", "A 7-billion-parameter model in 16-bit numbers takes 14 GB. In 8 bits it takes 7 GB, in 4 bits about 3.5 GB. That is the difference between needing a data-centre GPU and fitting on a laptop. And because token generation is limited by how fast the weights can be **read** from memory ([KV cache](llm.html#kv-cache) section), smaller weights also mean faster generation. **Quantization** maps the high-precision numbers to a small set of integer codes, plus a scale to convert back."],
    ["viz", "llm1_quant"],
    ["h", "The recipe: absmax (symmetric) quantization"],
    ["p", "For $b$ bits and a block of weights $w$:"],
    ["math", R`s=\frac{\max|w|}{2^{\,b-1}-1},\qquad q=\operatorname{round}\!\big(w/s\big),\qquad \hat w = q\cdot s`, "for int8 the integer range is -127 to 127; for int4 it is -7 to 7"],
    ["p", "Each weight is rounded to the nearest multiple of $s$, so the error is at most $s/2$. Fewer bits means a bigger $s$ and bigger errors. A variant called **asymmetric** quantization adds a zero-point so the range need not be centred on 0. In NumPy:"],
    ["code", R`
import numpy as np

def quantize(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(w).max() / qmax                            # one scale for the whole tensor
    q = np.clip(np.round(w / s), -qmax, qmax).astype(np.int8)
    return q, s

def dequantize(q, s):
    return q.astype(np.float32) * s

rng = np.random.default_rng(0)
w = rng.normal(0, 0.5, size=1000).astype(np.float32)
for bits in [8, 6, 4, 3, 2]:
    q, s = quantize(w, bits)
    err = dequantize(q, s) - w
    print(f"{bits} bits: scale={s:.4f}  max|error|={np.abs(err).max():.4f}  (<= s/2 = {s/2:.4f})  RMS error={np.sqrt((err**2).mean()):.4f}")`, "8 bits: scale=0.0154  max|error|=0.0077  (<= s/2 = 0.0077)  RMS error=0.0045\n6 bits: scale=0.0629  max|error|=0.0314  (<= s/2 = 0.0314)  RMS error=0.0180\n4 bits: scale=0.2785  max|error|=0.1391  (<= s/2 = 0.1393)  RMS error=0.0817\n3 bits: scale=0.6499  max|error|=0.3243  (<= s/2 = 0.3250)  RMS error=0.1864\n2 bits: scale=1.9497  max|error|=0.9741  (<= s/2 = 0.9749)  RMS error=0.4466"],
    ["h", "Outliers and why we quantize in groups"],
    ["p", "One huge weight forces a large scale, which squeezes all the ordinary weights into a few buckets (try the **outlier size** slider above). Large language models are known for such outliers, especially in activations (the LLM.int8() paper described a few feature dimensions with huge values appearing at larger model sizes). The cure is to give each small **group** of weights its own scale: per output channel, or, for 4-bit, blocks of typically 32 to 128 values. The extra cost is one 16-bit scale per group (about 0.1 to 0.5 extra bits per weight)."],
    ["code", R`
import numpy as np

def quant_error(w, bits, group):
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, group)                              # rows of 'group' weights, each with its own scale
    s = np.abs(g).max(axis=1, keepdims=True) / qmax
    deq = np.clip(np.round(g / s), -qmax, qmax) * s
    return np.sqrt(((deq - g) ** 2).mean())

rng = np.random.default_rng(0)
w = rng.normal(0, 0.5, size=4096)
w[100] = 12.0; w[2000] = -9.0                             # two outliers
for group in [4096, 512, 128, 32]:
    print(f"4-bit, group size {group:4d}: RMS error = {quant_error(w, 4, group):.4f}")
print("overhead of one fp16 scale per 128 weights:", 16 / 128, "extra bits per weight")`, "4-bit, group size 4096: RMS error = 0.4372\n4-bit, group size  512: RMS error = 0.2068\n4-bit, group size  128: RMS error = 0.1153\n4-bit, group size   32: RMS error = 0.0683\noverhead of one fp16 scale per 128 weights: 0.125 extra bits per weight"],
    ["p", "Two 4-bit codes fit into one byte, and that packing is what actually saves the memory:"],
    ["code", R`
import numpy as np

q = np.array([-7, 3, 0, 5, -2, 7, 1, -1], dtype=np.int8)        # eight signed 4-bit codes
u = (q + 8).astype(np.uint8)                                     # shift to 0..15 so each fits in 4 bits
packed = (u[0::2] << 4) | u[1::2]                                # two codes per byte
unpacked = np.empty_like(u); unpacked[0::2] = packed >> 4; unpacked[1::2] = packed & 0x0F
print("packed bytes:", packed.tolist(), "(", packed.nbytes, "bytes for", q.size, "weights )")
print("round trip ok:", np.array_equal(unpacked.astype(np.int8) - 8, q))`, "packed bytes: [27, 141, 111, 151] ( 4 bytes for 8 weights )\nround trip ok: True"],
    ["h", "What gets quantized, and how"],
    ["table", ["Choice", "Options", "Notes"], [
      ["What", "weights only (W4A16, W8A16); weights and activations (W8A8, FP8); the KV cache", "weight-only is the most common for serving: it shrinks memory and speeds up memory-bound decoding. Quantizing activations too lets the matrix multiply itself run in low precision"],
      ["When", "**Post-training (PTQ)** on an already trained model; **quantization-aware training (QAT)**, simulating rounding during training", "PTQ is cheap and dominant for LLMs; QAT recovers more quality at low bit-widths"],
      ["PTQ methods", "round-to-nearest; **GPTQ** (round one group at a time, correcting the remaining weights using second-order information); **AWQ** (protect the weights that matter most for the activations by scaling them)", "all work with a small calibration set of sample text"],
      ["Formats", "int8, int4, FP8, FP4 variants, NF4 (used by [QLoRA](llm.html#lora)), GGUF 'k-quants' in llama.cpp", "names change quickly; the principle (low-bit codes + group scales) does not"],
    ]],
    ["p", "How much do you lose? As a rule of thumb (varies by model and method; always measure on your own task): 8-bit is nearly lossless, 4-bit group-wise quantization of a large model costs a small amount of quality, and 3 bits or fewer degrades noticeably unless special methods are used. **Larger models tolerate quantization better than small ones**, so a 4-bit 70B model usually beats a 16-bit 13B model at a similar memory budget."],
    ["code", R`
def model_gb(n_params, bits, group=128):
    return n_params * (bits + (16 / group if bits < 16 else 0)) / 8 / 1e9

for name, n in [("7B", 7e9), ("13B", 13e9), ("70B", 70e9)]:
    row = "  ".join(f"{b:>2}-bit: {model_gb(n, b):6.1f} GB" for b in [16, 8, 4])
    print(f"{name:>4}  {row}")
print("(weights only; add the KV cache, activations and runtime overhead)")`, "  7B  16-bit:   14.0 GB   8-bit:    7.1 GB   4-bit:    3.6 GB\n 13B  16-bit:   26.0 GB   8-bit:   13.2 GB   4-bit:    6.7 GB\n 70B  16-bit:  140.0 GB   8-bit:   71.1 GB   4-bit:   36.1 GB\n(weights only; add the KV cache, activations and runtime overhead)"],
    ["p", "A framework does the heavy lifting for you. PyTorch even has a dynamic int8 mode for CPU linear layers that runs without downloads:"],
    ["code", R`
import io, warnings, torch, torch.nn as nn
warnings.filterwarnings("ignore")

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)   # weights stored as int8

def size_bytes(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return buf.getbuffer().nbytes

x = torch.randn(4, 512)
print("float32 model:", size_bytes(model), "bytes")
print("int8 model   :", size_bytes(qmodel), "bytes")
print("max output difference:", round((model(x) - qmodel(x)).abs().max().item(), 4), " typical output size:", round(model(x).abs().mean().item(), 4))`, "float32 model: 2103517 bytes\nint8 model   : 531981 bytes\nmax output difference: 0.0156  typical output size: 0.1882", "PyTorch", "local"],
    ["code", R`
# Loading an LLM in 4 bits with bitsandbytes (needs a GPU and model weights, so it is not run here)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", quantization_config=cfg)`, "", "bitsandbytes", "local"],
    ["warn", "Quantization error is not uniform across tasks. Perplexity can barely move while a specific capability (long reasoning chains, rare languages, exact arithmetic, code) drops more. Evaluate the quantized model on the tasks you care about ([evaluating LLMs](llm.html#llm-evaluation)), not only on perplexity."],
    ["danger", "Quantizing a tensor with a single scale when it contains outliers, or quantizing the layers that are most sensitive (embeddings, final output layer, normalisation parameters). Practical schemes keep those in higher precision."],
  ],
},
{
  id: "serving", group: "Inference", title: "Serving LLMs: Batching & Speculative Decoding",
  sub: "How to answer many users at once on expensive GPUs: keep them busy with continuous batching, and cheat the one-token-at-a-time rule with a draft model.",
  tags: ["inference", "systems"],
  blocks: [
    ["p", "Training a model is a one-off cost; **serving** it is a bill that never stops. The goals pull against each other: low latency for each user, high throughput for the GPU (that is, low cost per token), and enough memory for the weights and everyone's [KV cache](llm.html#kv-cache)."],
    ["h", "Why decoding is slow: memory bandwidth"],
    ["p", "In the decode phase, producing one token needs the GPU to read **all the weights** from memory (every parameter is used once per token) but do only about 2 FLOPs per weight. GPUs can compute far faster than they can fetch data, so decode is *memory-bound*. A back-of-the-envelope upper limit for one stream:"],
    ["math", R`\text{tokens/s}\ \lesssim\ \frac{\text{memory bandwidth}}{\text{bytes of weights}}`, "plus the KV cache bytes read per token, so real numbers are lower"],
    ["code", R`
def max_tokens_per_s(n_params, bytes_per_param, bandwidth_gb_s):
    return bandwidth_gb_s * 1e9 / (n_params * bytes_per_param)

# bandwidth is a published spec; 2000 GB/s is a round number typical of recent data-centre GPUs, 100 GB/s of a laptop
for dev, bw in [("data-centre GPU (about 2000 GB/s)", 2000), ("laptop (about 100 GB/s)", 100)]:
    for label, bpp in [("7B fp16", 2.0), ("7B 4-bit", 0.5)]:
        print(f"{dev:36s} {label:9s}: at most {max_tokens_per_s(7e9, bpp, bw):7.0f} tokens/s for ONE stream")`, "data-centre GPU (about 2000 GB/s)    7B fp16  : at most     143 tokens/s for ONE stream\ndata-centre GPU (about 2000 GB/s)    7B 4-bit : at most     571 tokens/s for ONE stream\nlaptop (about 100 GB/s)              7B fp16  : at most       7 tokens/s for ONE stream\nlaptop (about 100 GB/s)              7B 4-bit : at most      29 tokens/s for ONE stream"],
    ["p", "This explains two things at once. [Quantization](llm.html#quantization) speeds up generation (fewer bytes to read). And **batching** is nearly free: if 32 users share a decode step, the weights are read once and used for all 32, so throughput rises almost 32 times while per-user latency barely changes, until compute or the KV cache becomes the limit."],
    ["h", "Batching, and why continuous batching wins"],
    ["p", "**Static batching** groups requests and runs them together until the *last* one finishes. Requests have very different output lengths, so short ones sit idle holding a GPU slot while the longest one finishes. **Continuous batching** (also called in-flight or iteration-level batching; introduced by the Orca system and used by vLLM, TGI, TensorRT-LLM and SGLang) re-forms the batch at **every decode step**: as soon as a sequence finishes, a waiting request takes its slot."],
    ["code", R`
import heapq

lengths = [5, 30, 8, 12, 25, 6, 40, 10, 7, 22, 9, 15]       # tokens each request will generate
slots = 4                                                    # how many sequences fit in the batch at once

def static_steps(lengths, slots):
    steps = 0
    for i in range(0, len(lengths), slots):
        steps += max(lengths[i:i + slots])                   # a group runs until its longest request is done
    return steps

def continuous_steps(lengths, slots):
    running, queue, steps = [], list(lengths), 0             # running holds remaining tokens of active sequences
    while queue or running:
        while queue and len(running) < slots:
            running.append(queue.pop(0))                     # a free slot is refilled immediately
        steps += 1                                           # one decode step produces a token for every active sequence
        running = [r - 1 for r in running if r - 1 > 0]
    return steps

total = sum(lengths)
for name, fn in [("static", static_steps), ("continuous", continuous_steps)]:
    s = fn(lengths, slots)
    print(f"{name:10s}: {s:3d} decode steps for {total} tokens -> slot utilisation {total / (s * slots):.0%}")`, "static    :  92 decode steps for 189 tokens -> slot utilisation 51%\ncontinuous:  52 decode steps for 189 tokens -> slot utilisation 91%"],
    ["h", "Other building blocks"],
    ["table", ["Technique", "What it does"], [
      ["Continuous batching", "refills the batch every step, so GPUs are never waiting on the longest request"],
      ["PagedAttention (vLLM)", "stores the KV cache in fixed-size blocks that need not be contiguous, so memory is not wasted on over-reservation and identical prefixes can share blocks"],
      ["Prefix / prompt caching", "reuse the cache of a shared prompt beginning across requests"],
      ["Chunked prefill", "split long prompts into pieces and mix them with decode steps so a long prompt does not stall everyone else"],
      ["Prefill/decode disaggregation", "run the compute-bound prefill and the memory-bound decode on different machines, each tuned for its job"],
      ["Tensor / pipeline parallelism", "split one model over several GPUs when it does not fit on one"],
      ["Quantization, FlashAttention, GQA", "less memory traffic per token ([efficient attention](transformers.html#efficient-attention))"],
    ]],
    ["p", "Metrics worth knowing: **TTFT** (time to first token, dominated by queueing and prefill), **TPOT** or inter-token latency (time per generated token after the first), **throughput** (tokens per second across all users) and cost per million tokens. Chat UIs care about TTFT and TPOT; batch jobs only about throughput."],
    ["h", "Speculative decoding"],
    ["p", "Decoding is sequential: each token needs the previous one. But the big model can **check** many tokens as cheaply as it generates one, because verifying k positions is one parallel pass, just like prefill. **Speculative decoding** (Leviathan et al. and Chen et al., 2023) exploits that: a small, fast **draft model** guesses the next $k$ tokens; the large **target model** scores all of them in a single pass; we keep the correct prefix and gain several tokens for the price of one target pass."],
    ["diagram", "llm1_specdec", [], "A draft model proposes four tokens, the target model verifies them in one pass: three are accepted and the fourth is replaced by the target's own choice."],
    ["p", "The rule that makes it *exactly* correct (not an approximation): let $q$ be the draft model's distribution and $p$ the target's at some position, and let the draft propose token $x\\sim q$. **Accept** it with probability $\\min(1,\\,p(x)/q(x))$. If it is rejected, **resample** from the corrected distribution $\\propto\\max(0,\\,p-q)$ and stop. The tokens you end up with are distributed exactly as if sampled from the target alone. A quick check by simulation:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
p = np.array([0.50, 0.30, 0.15, 0.05])      # target model's distribution at one position
q = np.array([0.25, 0.25, 0.25, 0.25])      # a poor draft model

def speculative_step():
    x = rng.choice(4, p=q)                                   # the draft proposes a token
    if rng.random() < min(1.0, p[x] / q[x]):                 # accept with probability min(1, p/q)
        return x, True
    resid = np.maximum(p - q, 0); resid /= resid.sum()       # otherwise resample from max(0, p - q), normalised
    return rng.choice(4, p=resid), False

draws = [speculative_step() for _ in range(100000)]
freq = np.bincount([d[0] for d in draws], minlength=4) / len(draws)
print("target p          :", p)
print("speculative output:", freq.round(3), " (matches the target)")
print("acceptance rate   :", round(float(np.mean([d[1] for d in draws])), 3), "  theory sum(min(p,q)) =", round(float(np.minimum(p, q).sum()), 3))`, "target p          : [0.5  0.3  0.15 0.05]\nspeculative output: [0.501 0.301 0.149 0.049]  (matches the target)\nacceptance rate   : 0.702   theory sum(min(p,q)) = 0.7"],
    ["p", "How much speed-up? If each drafted token is accepted with probability $\\alpha$ (assumed independent) and the draft proposes $k$ tokens per round, the expected number of tokens per target pass is $\\dfrac{1-\\alpha^{k+1}}{1-\\alpha}$. The draft also costs time, so the real gain is less."],
    ["code", R`
def tokens_per_pass(alpha, k):
    return (1 - alpha ** (k + 1)) / (1 - alpha)

for alpha in [0.5, 0.7, 0.9]:
    row = "  ".join(f"k={k}: {tokens_per_pass(alpha, k):4.2f}" for k in [1, 2, 4, 8])
    print(f"acceptance {alpha}: expected tokens per target pass ->  {row}")`, "acceptance 0.5: expected tokens per target pass ->  k=1: 1.50  k=2: 1.75  k=4: 1.94  k=8: 2.00\nacceptance 0.7: expected tokens per target pass ->  k=1: 1.70  k=2: 2.19  k=4: 2.77  k=8: 3.20\nacceptance 0.9: expected tokens per target pass ->  k=1: 1.90  k=2: 2.71  k=4: 4.10  k=8: 6.13"],
    ["p", "Variants: drafts that are small separate models, extra \"heads\" on the big model itself (Medusa, EAGLE), n-gram lookup from the prompt (very effective for editing code or summarising, where output copies the input), and multi-token prediction heads. Speculative decoding helps most for **low batch sizes** (latency-sensitive use), where the GPU is under-used anyway; at high batch sizes the spare compute is already consumed."],
    ["code", R`
# vLLM (needs a GPU and model weights, so it is not run here)
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")        # continuous batching and PagedAttention are on by default
outs = llm.generate(["Explain the KV cache in one sentence.", "What is TTFT?"], SamplingParams(temperature=0.7, max_tokens=64))
for o in outs:
    print(o.outputs[0].text)`, "", "vLLM", "local"],
    ["tip", "For local use, llama.cpp / Ollama run quantized GGUF models on laptops, while vLLM and SGLang are common server-side engines (as of 2026). Pick the engine first, then the model format it likes."],
    ["danger", "Benchmarking a server with a single request at a time. Throughput, latency and cost per token all depend on the batch; measure with realistic concurrency, prompt lengths and output lengths."],
  ],
},
{
  id: "long-context", group: "Inference", title: "Long Context",
  sub: "Letting a model read whole books or codebases at once: what makes it hard, how models stretch their window, and why a big window is not the same as using it well.",
  tags: ["context", "inference"],
  blocks: [
    ["p", "The **context window** is how many tokens a model can look at in one go: your prompt, any documents, the conversation so far, and the answer it is writing. Early GPT models had a few thousand tokens. As of 2026, many models offer hundreds of thousands, and some a million or more. A longer window lets you paste a whole contract, a codebase or hours of transcript. But it comes with three difficulties, one for compute, one for memory and one for quality."],
    ["h", "Difficulty 1: attention cost"],
    ["p", "Self-attention compares every token with every earlier token, so compute grows with the **square** of the length: doubling the context quadruples the attention work (see [scaled dot-product attention](transformers.html#scaled-dot-product)). Naive attention also stores the $T\\times T$ score matrix per head. [FlashAttention](transformers.html#efficient-attention) avoids storing it (same result, far less memory), but the compute is still quadratic."],
    ["h", "Difficulty 2: the KV cache"],
    ["p", "The [KV cache](llm.html#kv-cache) grows linearly with length. At 128K tokens even a GQA 8B model needs about 16 GiB of cache for a *single* user (computed in that section), as much as the weights themselves."],
    ["code", R`
for T in [4_096, 32_768, 131_072, 1_048_576]:
    heads = 32
    score_matrix_gb = heads * T * T * 2 / 1e9              # naive attention, fp16 scores, ONE layer (flash attention never stores this)
    kv_gib = 2 * 32 * 8 * 128 * 2 * T / 2**30              # GQA 8B shape, fp16 cache, all layers
    print(f"T={T:>9,}: naive score matrix per layer = {score_matrix_gb:>12,.1f} GB   KV cache (GQA 8B) = {kv_gib:7.1f} GiB")
print("attention work relative to 4k tokens at 128k:", (131072 / 4096) ** 2, "x")`, "T=    4,096: naive score matrix per layer =          1.1 GB   KV cache (GQA 8B) =     0.5 GiB\nT=   32,768: naive score matrix per layer =         68.7 GB   KV cache (GQA 8B) =     4.0 GiB\nT=  131,072: naive score matrix per layer =      1,099.5 GB   KV cache (GQA 8B) =    16.0 GiB\nT=1,048,576: naive score matrix per layer =     70,368.7 GB   KV cache (GQA 8B) =   128.0 GiB\nattention work relative to 4k tokens at 128k: 1024.0 x"],
    ["h", "Difficulty 3: positions the model never saw"],
    ["p", "A model trained on sequences of at most 4,096 tokens has only ever seen position information up to that length. With [RoPE](transformers.html#positional) each token's query and key are rotated by an angle proportional to its position. At position 16,000 the angles fall outside everything seen in training, and quality collapses. Two standard remedies:"],
    ["list", [
      "**Position interpolation**: multiply positions by $L_{\\text{train}}/L_{\\text{new}}$ so every new position maps back inside the trained range, then fine-tune briefly on long text.",
      "**NTK-aware scaling, YaRN** and similar: rescale the rotation frequencies unevenly, interpolating the slow dimensions (which encode long-range position) while leaving the fast ones (which encode precise local order) mostly alone.",
    ]],
    ["code", R`
import numpy as np

d, L_train, L_new = 64, 4096, 16384
theta = 10000.0 ** (-np.arange(0, d, 2) / d)              # RoPE: one rotation speed per pair of dimensions

slow = theta[-1]                                           # the slowest rotating pair (encodes long-range position)
print("slowest pair, largest angle seen in training :", round(L_train * slow, 4), "rad")
print("slowest pair at position 16384 (extrapolated):", round(L_new * slow, 4), "rad  -> beyond anything trained")
scale = L_train / L_new
print("slowest pair at 16384 with interpolation     :", round(L_new * scale * slow, 4), "rad  -> back inside the trained range")

fast = theta[0]
print("fastest pair: neighbouring tokens differ by", round(fast, 3), "rad;  with interpolation they would differ by only", round(fast * scale, 3), "rad (local detail gets blurred, which NTK/YaRN avoid)")`, "slowest pair, largest angle seen in training : 0.5462 rad\nslowest pair at position 16384 (extrapolated): 2.1848 rad  -> beyond anything trained\nslowest pair at 16384 with interpolation     : 0.5462 rad  -> back inside the trained range\nfastest pair: neighbouring tokens differ by 1.0 rad;  with interpolation they would differ by only 0.25 rad (local detail gets blurred, which NTK/YaRN avoid)"],
    ["h", "Cheaper attention patterns"],
    ["p", "Some models restrict who may look at whom. **Sliding-window attention** lets each token attend only to the previous $w$ tokens (Mistral 7B used $w=4096$); stacked layers still pass information further, since layer 2 sees what layer 1 saw. Some architectures alternate full-attention and windowed layers. A banded mask is all it takes:"],
    ["code", R`
import numpy as np

T, w = 8, 3
i, j = np.indices((T, T))
mask = (j <= i) & (i - j < w)                             # causal AND within the last w tokens
print(mask.astype(int))
print("allowed pairs:", int(mask.sum()), "of", T * (T + 1) // 2, "causal pairs")`, "[[1 0 0 0 0 0 0 0]\n [1 1 0 0 0 0 0 0]\n [1 1 1 0 0 0 0 0]\n [0 1 1 1 0 0 0 0]\n [0 0 1 1 1 0 0 0]\n [0 0 0 1 1 1 0 0]\n [0 0 0 0 1 1 1 0]\n [0 0 0 0 0 1 1 1]]\nallowed pairs: 21 of 36 causal pairs"],
    ["h", "Having room is not the same as using it"],
    ["list", [
      "**Lost in the middle**: Liu et al. (2023) found that models retrieve facts placed at the beginning or end of a long prompt better than facts buried in the middle. Newer models are better, but position effects remain.",
      "**Needle-in-a-haystack** tests (hide one fact in a long text, ask for it) are easy, so passing them says little. Harder tests need *reasoning over many* pieces of the context. An advertised window is the maximum accepted, not the length at which quality stays high.",
      "**Distractors hurt**: irrelevant text in the window can lower accuracy. More context is not automatically better.",
      "**Cost**: you pay (and wait) for every input token. Prefill time grows with length, and the attention cost grows faster than linearly. Prompt caching helps when the same long prefix is reused.",
    ]],
    ["table", ["Approach", "Good when", "Weak when"], [
      ["Put everything in the long context", "one-off analysis of a document set that fits; tasks needing the whole text in view (summaries, cross-references)", "repeated queries over a huge corpus (expensive); facts get diluted"],
      ["Retrieve only the relevant pieces ([RAG](llm.html#rag))", "large or changing knowledge bases; cost matters; you need citations", "the answer needs to compare passages across the whole text; retrieval misses"],
      ["Summarise / compress history ([agent memory](agents.html#memory))", "long conversations and long-running agents", "details that were summarised away are lost"],
    ]],
    ["tip", "Order your prompt sensibly: instructions and question at the beginning *and* the end for long inputs, documents in between, clear separators and titles for each document. Test with your own data at your real context length."],
    ["danger", "Assuming that a window of N tokens means the model performs equally well across all N. Always measure quality at the length you actually use, with the facts placed where they would naturally be (including the middle)."],
  ],
}
);
