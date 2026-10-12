/* Transformers: tokens, embeddings, attention and the transformer block, built from scratch in NumPy
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/tf.js */
TF.push(
/* =========================================================================================== Tokens */
{
  id: "why-tokens", group: "Tokens", title: "Why Tokens?",
  sub: "A model can only do maths on numbers, so text is first cut into pieces called tokens, and each piece is replaced by an ID.",
  tags: ["tokenization"],
  blocks: [
    ["p", "A neural network multiplies matrices. It cannot multiply the word *cat*. So before a transformer sees any text, a **tokenizer** cuts the text into small pieces called **tokens**, taken from a fixed list called the **vocabulary**, and replaces each piece by its position in that list: an integer **token ID**. The model only ever sees those IDs. At the other end it outputs one probability per vocabulary entry, and the tokenizer turns the chosen IDs back into text."],
    ["p", "The simplest tokenizer uses single characters. The vocabulary is every distinct character, encoding is a dictionary lookup, and decoding is the reverse lookup:"],
    ["code", R`
text = "the cat sat on the mat"
chars = sorted(set(text))                      # the vocabulary: every distinct character
stoi = {ch: i for i, ch in enumerate(chars)}   # string -> id
itos = {i: ch for ch, i in stoi.items()}       # id -> string

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

ids = encode("the cat")
print("vocab:", chars)
print("vocab size:", len(chars))
print("ids:", ids)
print("back:", decode(ids))`, "vocab: [' ', 'a', 'c', 'e', 'h', 'm', 'n', 'o', 's', 't']\nvocab size: 10\nids: [9, 4, 3, 0, 2, 1, 9]\nback: the cat"],
    ["p", "That works, but every character becomes its own step for the model, so sequences get long. The opposite extreme is one token per word. Sequences are short, but any word the tokenizer never saw becomes an unknown token:"],
    ["code", R`
corpus = "the cat sat on the mat . the dog sat on the log ."
vocab = ["<unk>"] + sorted(set(corpus.split()))
stoi = {w: i for i, w in enumerate(vocab)}

def encode(s):
    return [stoi.get(w, 0) for w in s.split()]     # unknown words -> id 0 (<unk>)

print(vocab)
print(encode("the dog sat on the mat"))
print(encode("the cats sat on the sofa"))          # 'cats' and 'sofa' were never seen`, "['<unk>', '.', 'cat', 'dog', 'log', 'mat', 'on', 'sat', 'the']\n[8, 3, 7, 6, 8, 5]\n[8, 0, 7, 6, 8, 0]"],
    ["p", "The model now knows *cat* but has no idea what *cats* is: it sees the same `<unk>` it would see for *sofa*. A real language has millions of word forms (names, typos, plurals, code, other languages), so a word vocabulary is either enormous or full of holes."],
    ["h", "Three ways to cut text"],
    ["table", ["Unit", "Vocabulary size", "Sequence length", "Unknown words?", "\"unhappiness\" becomes"], [
      ["Characters (or bytes)", "Tiny: a few hundred characters, or exactly 256 bytes", "Very long", "Never", "u n h a p p i n e s s (11 tokens)"],
      ["Words", "Huge: hundreds of thousands or more", "Short", "Often (`<unk>`)", "unhappiness (1 token, if it's in the list)"],
      ["**Subwords**", "Medium: roughly 30,000 to 200,000+ (as of 2026)", "Medium", "Rarely or never", "un · happi · ness (3 tokens)"],
    ]],
    ["p", "Modern models use **subwords**. Frequent words (*the*, *model*) become a single token, while rare words are split into reusable parts (*un* + *happi* + *ness*). The vocabulary stays a manageable size, sequences stay reasonably short, and any word can still be written. The next two sections show how such a vocabulary is learned: [Byte-Pair Encoding](transformers.html#bpe) and its relatives [WordPiece, SentencePiece and byte-level BPE](transformers.html#tokenizers)."],
    ["h", "What a tokenizer does, step by step"],
    ["steps", [
      "**Normalize** (optional): clean up Unicode, sometimes lowercase.",
      "**Pre-tokenize**: split the text into rough chunks, usually at spaces and punctuation, so pieces never cross word boundaries.",
      "**Model**: split each chunk into subword tokens with the learned rules (BPE merges, a WordPiece vocabulary, or a Unigram model).",
      "**Post-process**: add special tokens such as a beginning-of-text marker.",
      "**Map to IDs**: look up each token in the vocabulary.",
    ]],
    ["h", "Special tokens"],
    ["table", ["Token (names vary by model)", "Meaning"], [
      ["`<bos>` / `<s>`", "Beginning of a sequence."],
      ["`<eos>` / `</s>` / `<|endoftext|>`", "End of a sequence. When a model generates it, generation stops."],
      ["`<pad>`", "Filler so that sequences in a batch have the same length. Must be ignored by attention ([padding masks](transformers.html#masking))."],
      ["`<unk>`", "Unknown piece. Byte-level tokenizers never need it."],
      ["`[CLS]`, `[SEP]`, `[MASK]`", "BERT-style markers: a summary slot, a separator between two texts, and a hidden word to predict."],
      ["Chat markers", "Chat models wrap each message in role tokens (system / user / assistant). The exact format is the model's **chat template**."],
    ]],
    ["h", "Why tokens matter in practice"],
    ["list", [
      "**Limits and prices are counted in tokens.** The context window and API costs are measured in tokens, not words. For English with GPT-style tokenizers a rough rule is one token ≈ 4 characters ≈ ¾ of a word; other languages and source code often need more tokens for the same content. See [long context](llm.html#long-context).",
      "**Compute grows with the number of tokens.** Attention compares every token with every other one, so twice the tokens costs about four times the attention work ([scaled dot-product attention](transformers.html#scaled-dot-product)).",
      "**Odd failure modes.** The model sees *straw* + *berry*, not letters, which is why counting letters in a word is surprisingly hard for it. Numbers can be split into irregular chunks (*1234* → *123* + *4*), which hurts arithmetic; some tokenizers split every digit separately for this reason.",
      "**Spaces are part of tokens.** In GPT-style tokenizers *\" cat\"* (with a leading space) and *\"cat\"* are different tokens with different IDs.",
    ]],
    ["h", "With a library"],
    ["p", "In practice you load the exact tokenizer that a model was trained with. This needs to download the tokenizer files once, so it is shown without output:"],
    ["code", R`
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")        # downloads the tokenizer files the first time
ids = tok("Tokenization is surprisingly important!")["input_ids"]
print(ids)
print(tok.convert_ids_to_tokens(ids))              # 'Ġ' marks a leading space in GPT-2's vocabulary
print(tok.decode(ids))`, "", "Hugging Face", "local"],
    ["danger", "Assuming one token = one word. Budget prompts and context with the model's **own** tokenizer: different models split the same text into different numbers of tokens, and a model only understands IDs from its own vocabulary."],
  ],
},
{
  id: "bpe", group: "Tokens", title: "Byte-Pair Encoding (BPE)",
  sub: "Learn a subword vocabulary by starting from characters and repeatedly gluing together the pair of symbols that appears most often.",
  tags: ["tokenization", "algorithm"],
  blocks: [
    ["p", "Byte-Pair Encoding started life as a simple compression trick and was adapted for machine translation in 2016 (Sennrich et al.). A byte-level version of it is the tokenizer behind GPT-2 and many later models. The idea fits in one sentence: **find the most frequent pair of neighbouring symbols, merge it into a new symbol, and repeat.**"],
    ["h", "The algorithm"],
    ["steps", [
      "Split every word of the training text into characters, and add an end-of-word marker (here `_`) so that *est* at the end of a word can differ from *est* inside one.",
      "Count every pair of neighbouring symbols, weighted by how often each word occurs.",
      "Take the most frequent pair, say (`e`, `s`), and replace it everywhere by the new symbol `es`. Add `es` to the vocabulary and **record the merge**.",
      "Repeat steps 2–3 until you have done the number of merges you want. Vocabulary size = number of starting characters + number of merges.",
    ]],
    ["p", "Try it below on the classic toy corpus: *low* ×5, *lower* ×2, *newest* ×6, *widest* ×3. Type your own words (`word:count` or just repeated words) and press Run."],
    ["viz", "tf_bpe"],
    ["h", "BPE from scratch"],
    ["p", "Words are stored once with their counts, as tuples of symbols. Each merge step counts pairs and rewrites every word:"],
    ["code", R`
from collections import Counter

corpus = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3
word_freq = Counter(corpus.split())
words = {tuple(w) + ("_",): c for w, c in word_freq.items()}   # "_" marks the end of a word

def pair_counts(words):
    counts = Counter()
    for symbols, c in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += c          # a pair counts as often as its word appears
    return counts

def merge(words, pair):
    a, b = pair
    out = {}
    for symbols, c in words.items():
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b); i += 2       # glue the pair into one symbol
            else:
                new.append(symbols[i]); i += 1
        out[tuple(new)] = c
    return out

merges = []
for step in range(10):
    counts = pair_counts(words)
    best = max(counts, key=counts.get)       # most frequent pair (ties: the first one seen)
    words = merge(words, best)
    merges.append(best)
    print(f"merge {step + 1:2d}: {best[0]} + {best[1]} -> {best[0] + best[1]}  (seen {counts[best]} times)")
print()
for symbols, c in words.items():
    print(c, "x", " ".join(symbols))`, "merge  1: e + s -> es  (seen 9 times)\nmerge  2: es + t -> est  (seen 9 times)\nmerge  3: est + _ -> est_  (seen 9 times)\nmerge  4: l + o -> lo  (seen 7 times)\nmerge  5: lo + w -> low  (seen 7 times)\nmerge  6: n + e -> ne  (seen 6 times)\nmerge  7: ne + w -> new  (seen 6 times)\nmerge  8: new + est_ -> newest_  (seen 6 times)\nmerge  9: low + _ -> low_  (seen 5 times)\nmerge 10: w + i -> wi  (seen 3 times)\n\n5 x low_\n2 x low e r _\n6 x newest_\n3 x wi d est_"],
    ["p", "Ten merges turned *newest* and *low* into single tokens, and `est_` (\"est at the end of a word\") into a reusable suffix. The ordered list of merges **is** the tokenizer: to encode a new word, split it into characters and replay the merges in the order they were learned."],
    ["code", R`
merges = [("e", "s"), ("es", "t"), ("est", "_"), ("l", "o"), ("lo", "w"),
          ("n", "e"), ("ne", "w"), ("new", "est_"), ("low", "_"), ("w", "i")]   # learned above

def encode_word(word, merges):
    symbols = list(word) + ["_"]
    for a, b in merges:                       # replay the merges in the order they were learned
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b); i += 2
            else:
                new.append(symbols[i]); i += 1
        symbols = new
    return symbols

for w in ["lowest", "newer", "wider", "slow"]:
    print(f"{w:7s} -> {encode_word(w, merges)}")`, "lowest  -> ['low', 'est_']\nnewer   -> ['new', 'e', 'r', '_']\nwider   -> ['wi', 'd', 'e', 'r', '_']\nslow    -> ['s', 'low_']"],
    ["p", "None of these four words was in the training text, yet each one is encoded with known pieces: *lowest* = `low` + `est_`. The worst case is falling back to single characters, which is why BPE (almost) never needs an unknown token. Real implementations such as GPT-2's store a **rank** for every merge and repeatedly merge the neighbouring pair with the lowest rank; that gives the same result as replaying the list, only faster."],
    ["h", "With a library"],
    ["p", "The Hugging Face `tokenizers` library (written in Rust) trains the same kind of tokenizer in milliseconds on gigabytes of text. Training on a small list in memory runs fully offline:"],
    ["code", R`
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tok = Tokenizer(models.BPE(unk_token="[UNK]", end_of_word_suffix="_"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()            # split at spaces first
trainer = trainers.BpeTrainer(vocab_size=30, special_tokens=["[UNK]"], end_of_word_suffix="_", show_progress=False)
corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
tok.train_from_iterator(corpus, trainer)

print("vocab size:", tok.get_vocab_size())
print(tok.encode("lowest newer wider").tokens)`, "vocab size: 27\n['lo', 'w', 'est_', 'new', 'er_', 'w', 'i', 'd', 'er_']", "Hugging Face tokenizers", "local"],
    ["h", "Choosing the vocabulary size"],
    ["table", ["Vocabulary", "Effect"], [
      ["Smaller (more characters per word split)", "Longer sequences (more compute per sentence), but a smaller embedding table and every token is seen often in training."],
      ["Larger (more merges)", "Shorter sequences and better coverage of many languages, but a bigger embedding and output layer, and rare tokens get little training."],
    ]],
    ["note", "As of 2026, vocabularies of widely used models range from about 32,000 tokens to well over 100,000; larger vocabularies have become more common, mainly to cover many languages and code efficiently."],
    ["warn", "The tokenizer is trained **before** the model, often on different data. Tokens that were common in the tokenizer's data but rare in the model's training data end up with barely trained embeddings, and can make the model behave strangely when they appear (the so-called \"glitch tokens\")."],
    ["danger", "Merging across word boundaries. Without a pre-tokenizer (split at spaces first) BPE happily learns tokens like `e_t` that glue the end of one word to the start of the next, which wastes vocabulary and generalizes badly."],
    ["p", "Practise it: [One BPE merge step](practice.html#p/bpe-merge-step) and [Train a tiny BPE tokenizer](practice.html#p/bpe-tokenizer)."],
  ],
},
{
  id: "tokenizers", group: "Tokens", title: "WordPiece, SentencePiece & Byte-Level BPE",
  sub: "The three tokenizer families you will meet in real models: how each one builds its vocabulary, splits words, and marks spaces.",
  tags: ["tokenization"],
  blocks: [
    ["p", "Every subword tokenizer answers the same two questions: **how do we choose the vocabulary**, and **how do we split a new word with it**? BPE answers \"merge the most frequent pair\" and \"replay the merges\". The other popular designs change one of the two answers."],
    ["h", "WordPiece (BERT)"],
    ["p", "WordPiece also merges pairs bottom-up, but it does not simply pick the most frequent pair. It picks the pair whose merge raises the likelihood of the training data the most, which works out to this score:"],
    ["math", R`\text{score}(a, b) = \frac{\text{count}(ab)}{\text{count}(a)\cdot\text{count}(b)}`, "high when a and b appear together much more often than you'd expect from how common each is alone"],
    ["p", "Pieces that continue a word carry a `##` prefix, so *playing* becomes `play` + `##ing`. Here both rules choose a first merge on the same tiny corpus:"],
    ["code", R`
from collections import Counter

word_freq = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
splits = {w: [w[0]] + ["##" + ch for ch in w[1:]] for w in word_freq}   # later letters get ##

sym, pair = Counter(), Counter()
for w, f in word_freq.items():
    s = splits[w]
    for t in s:
        sym[t] += f
    for a, b in zip(s, s[1:]):
        pair[(a, b)] += f

bpe_pick = max(pair, key=pair.get)                                       # most frequent
scores = {p: c / (sym[p[0]] * sym[p[1]]) for p, c in pair.items()}
wp_pick = max(scores, key=scores.get)                                    # best likelihood score
print("BPE picks      :", bpe_pick, "count", pair[bpe_pick])
print("WordPiece picks:", wp_pick, "score", round(scores[wp_pick], 4))`, "BPE picks      : ('##u', '##g') count 20\nWordPiece picks: ('##g', '##s') score 0.05"],
    ["p", "BPE merges the very common `##u ##g`, but `##u` is everywhere, so WordPiece prefers `##g ##s`: the rarer pair whose parts almost always appear together."],
    ["p", "To **encode**, WordPiece doesn't replay merges. It walks through the word and always takes the **longest** piece in the vocabulary that matches (greedy longest-match-first). If some part can't be matched at all, the whole word becomes `[UNK]`:"],
    ["code", R`
vocab = {"[UNK]", "un", "##aff", "##able", "play", "##ing", "##ed", "h", "##u", "##g", "##s"}

def wordpiece(word, vocab):
    tokens, start = [], 0
    while start < len(word):
        end, piece = len(word), None
        while start < end:                       # try the longest piece first
            sub = word[start:end]
            if start > 0:
                sub = "##" + sub                 # pieces inside a word carry ##
            if sub in vocab:
                piece = sub
                break
            end -= 1
        if piece is None:
            return ["[UNK]"]                     # nothing fits: the whole word is unknown
        tokens.append(piece)
        start = end
    return tokens

for w in ["unaffable", "playing", "played", "hugs", "xyz"]:
    print(f"{w:10s} -> {wordpiece(w, vocab)}")`, "unaffable  -> ['un', '##aff', '##able']\nplaying    -> ['play', '##ing']\nplayed     -> ['play', '##ed']\nhugs       -> ['h', '##u', '##g', '##s']\nxyz        -> ['[UNK]']"],
    ["h", "Unigram language model and SentencePiece"],
    ["p", "The **Unigram** method works top-down. It starts from a big list of candidate pieces, gives each piece a probability, and repeatedly throws away the pieces whose removal hurts the likelihood of the training text the least. To split a word it picks the segmentation with the **highest total probability**, found with dynamic programming (the Viterbi algorithm). Here with made-up log-probabilities:"],
    ["code", R`
import math

logp = {"h": -4.0, "e": -3.5, "l": -3.0, "o": -3.2, "he": -3.0, "ll": -3.4,
        "lo": -3.6, "llo": -4.8, "hell": -5.0, "hello": -8.0}       # a toy unigram model

def viterbi(word):
    n = len(word)
    best = [0.0] + [-math.inf] * n        # best[i] = best log-probability of word[:i]
    back = [0] * (n + 1)                  # where the last piece of that best split starts
    for end in range(1, n + 1):
        for start in range(end):
            piece = word[start:end]
            if piece in logp and best[start] + logp[piece] > best[end]:
                best[end] = best[start] + logp[piece]
                back[end] = start
    pieces, i = [], n
    while i > 0:                          # walk back to recover the pieces
        pieces.append(word[back[i]:i])
        i = back[i]
    return pieces[::-1], round(best[n], 2)

print(viterbi("hello"))     # he + llo (-7.8) beats hello (-8.0) and hell + o (-8.2)
print(viterbi("hell"))`, "(['he', 'llo'], -7.8)\n(['hell'], -5.0)"],
    ["p", "**SentencePiece** is a library (from Google) that implements both BPE and Unigram with one important twist: it treats the input as a raw stream of characters **including spaces**. A space is replaced by the visible symbol `▁` and becomes part of the next piece, so *Hello world* becomes `▁Hello` `▁world`. No language-specific word splitter is needed (handy for Chinese or Japanese, which don't use spaces), and decoding is exact: join the pieces and turn `▁` back into spaces."],
    ["h", "Byte-level BPE (GPT-2 and later)"],
    ["p", "Character-level BPE still needs `<unk>` for characters it never saw (a rare symbol, an unusual script). **Byte-level BPE** runs BPE on the UTF-8 **bytes** of the text instead. There are only 256 possible bytes, so the starting vocabulary is exactly 256 symbols and *every* possible string can be encoded. GPT-2's vocabulary of 50,257 is 256 bytes + 50,000 merges + 1 end-of-text token."],
    ["p", "To keep the merge tables readable, GPT-2 maps every byte to a printable character. Printable bytes keep their own character and the others are shifted up, which is why the space byte shows up as `Ġ` in GPT-2 token lists:"],
    ["code", R`
def bytes_to_unicode():
    # printable bytes keep their own character; the others are shifted to 256+ so every byte is visible
    keep = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1)))
    table, n = {}, 0
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + n)
            n += 1
    return table

B2U = bytes_to_unicode()
for text in ["hi there", "café", "日本"]:
    raw = list(text.encode("utf-8"))           # 'é' is 2 bytes, each of these characters is 3
    print(f"{text!r:11} {str(raw):38} {''.join(B2U[b] for b in raw)}")`, "'hi there'  [104, 105, 32, 116, 104, 101, 114, 101] hiĠthere\n'café'      [99, 97, 102, 195, 169]                cafÃ©\n'日本'        [230, 151, 165, 230, 156, 172]         æĹ¥æľ¬"],
    ["p", "Before merging, byte-level tokenizers split the text with a regular expression so that merges never cross from a word into punctuation or the next word. A simplified (ASCII-only) version of GPT-2's pattern:"],
    ["code", R`
import re

pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
print(pat.findall("Hello world, it's 2026!  Let's go."))`, "['Hello', ' world', ',', ' it', \"'s\", ' 2026', '!', ' ', ' Let', \"'s\", ' go', '.']"],
    ["p", "Notice that each word keeps its **leading** space (`' world'`). That's why *\" world\"* and *\"world\"* are different tokens."],
    ["h", "Side by side"],
    ["table", ["", "BPE", "WordPiece", "Unigram (SentencePiece)", "Byte-level BPE"], [
      ["Builds the vocabulary by", "Merging the most frequent pair", "Merging the pair with the best likelihood score", "Pruning a large candidate list", "BPE merges, starting from 256 bytes"],
      ["Splits a new word by", "Replaying the merges", "Greedy longest match", "Most probable split (Viterbi)", "Replaying the merges (by rank)"],
      ["Unknown symbols", "`<unk>` unless there's a fallback", "`[UNK]`", "`<unk>` or byte fallback", "Never"],
      ["Marks word boundaries with", "End-of-word marker (`</w>` in the original paper)", "`##` on pieces inside a word", "`▁` in front of a word", "`Ġ` = a leading space"],
      ["Used by (as of 2026)", "The original NMT work", "BERT, DistilBERT", "T5, ALBERT, XLNet; Llama 2 used SentencePiece in BPE mode", "GPT-2 onward, RoBERTa, Llama 3"],
    ]],
    ["p", "Vocabulary sizes of some well-known models: BERT-base 30,522, GPT-2 50,257, T5 32,000, Llama 2 32,000, Llama 3 about 128,000."],
    ["code", R`
from transformers import AutoTokenizer

for name in ["bert-base-uncased", "t5-small", "gpt2"]:
    tok = AutoTokenizer.from_pretrained(name)          # downloads the tokenizer files the first time
    print(f"{name:18s}", tok.tokenize("Tokenizers split unbelievably long words."))`, "", "Hugging Face", "local"],
    ["tip", "Always pair a model with **its own** tokenizer (`AutoTokenizer.from_pretrained(<same model id>)`). The IDs are only meaningful to the embedding table they were trained with; a different tokenizer produces valid-looking IDs that mean nonsense to the model."],
    ["danger", "Stripping or adding spaces before tokenizing. With byte-level BPE `\"Paris\"` and `\" Paris\"` are different tokens, so a prompt that ends with a stray space can change what the model predicts next."],
  ],
},
/* =========================================================================================== Embeddings */
{
  id: "token-embeddings", group: "Embeddings", title: "Token Embeddings",
  sub: "Each token ID gets its own learned vector: one row of a big table called the embedding matrix.",
  tags: ["embeddings"],
  blocks: [
    ["p", "A token ID is just a name tag. Token 523 isn't \"bigger\" or \"more\" than token 17 in any meaningful way, so feeding the raw number into a network would make no sense. Instead every ID gets its own vector of `d` learned numbers. All those vectors are stacked in an **embedding matrix** $E$ with one row per token: shape $(V, d)$ for a vocabulary of $V$ tokens."],
    ["diagram", "tf_embed", [], "Embedding lookup: each ID picks one row of the table. Repeated IDs pick the same row."],
    ["math", R`x_t = E[\,\text{id}_t\,] = \text{onehot}(\text{id}_t)\, E`, "picking row id from E is the same as multiplying a one-hot vector by E"],
    ["p", "In code, the lookup is plain indexing. Multiplying by a one-hot vector gives the same result, which is why an embedding layer is just a linear layer that is evaluated cleverly:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
V, d = 10, 4                                   # vocabulary size, embedding size
E = rng.normal(0, 1, size=(V, d)).round(2)     # the embedding matrix: one row per token id

ids = np.array([3, 7, 3])                      # a sequence of 3 token ids (3 appears twice)
X = E[ids]                                     # lookup = pick rows
print(X.shape)
print(X)

one_hot = np.eye(V)[ids]                       # (3, 10): a single 1 in each row
print(np.allclose(one_hot @ E, X))             # the same as a matrix product`, "(3, 4)\n[[-2.33 -0.22 -1.25 -0.73]\n [-0.46  0.22 -1.01 -0.21]\n [-2.33 -0.22 -1.25 -0.73]]\nTrue"],
    ["p", "With batches the IDs have shape `(batch, seq)` and the result has shape `(batch, seq, d)`: one vector per token per sequence. That 3-D tensor is what flows through every transformer layer."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
E = rng.normal(size=(50, 8))                   # V = 50 tokens, d = 8
ids = np.array([[4, 9, 9, 1],
                [7, 2, 0, 0]])                 # (batch=2, seq=4)
X = E[ids]
print(ids.shape, "->", X.shape)                # (2, 4) -> (2, 4, 8)`, "(2, 4) -> (2, 4, 8)"],
    ["h", "How embeddings learn"],
    ["p", "The embedding matrix is trained by gradient descent like every other weight. In one training step only the rows of tokens that actually appeared get a gradient, and a token that appears twice collects **both** gradients. Use `np.add.at` for that scatter-add; ordinary fancy indexing silently keeps only one of the repeats:"],
    ["code", R`
import numpy as np

V, d = 6, 3
ids = np.array([2, 5, 2])                      # token 2 appears twice
dX = np.ones((3, d))                           # gradient arriving at each looked-up vector

dE = np.zeros((V, d))
np.add.at(dE, ids, dX)                         # scatter-add: repeated ids accumulate
print(dE)

dE_wrong = np.zeros((V, d))
dE_wrong[ids] += dX                            # fancy-index += does NOT accumulate repeats
print("row 2, right:", dE[2], " wrong:", dE_wrong[2])`, "[[0. 0. 0.]\n [0. 0. 0.]\n [2. 2. 2.]\n [0. 0. 0.]\n [0. 0. 0.]\n [1. 1. 1.]]\nrow 2, right: [2. 2. 2.]  wrong: [1. 1. 1.]"],
    ["p", "After training, tokens used in similar ways end up with similar vectors, so the [cosine similarity](math.html#dot-product) of their embeddings is high. This is the same idea as [word2vec and GloVe](dl.html#word-embeddings), except that the embedding is learned as part of the whole model."],
    ["note", "These input embeddings are **static**: *bank* gets the same vector in *river bank* and *bank account*. The attention layers then mix in information from the surrounding tokens, so the vectors deeper in the model are **contextual**."],
    ["h", "Size, sharing and scaling"],
    ["list", [
      "**The table is big.** GPT-2 small has $V = 50{,}257$ and $d = 768$, so its embedding matrix holds about 38.6 million numbers, roughly a third of its 124 million parameters.",
      "**Weight tying.** The output layer that turns the final vector into one score per token needs a $(d, V)$ matrix. Many models (GPT-2 among them) simply reuse $E^\\top$ for it, which saves $V \\cdot d$ parameters and tends to work as well or better.",
      "**Scaling.** The original Transformer multiplies the embeddings by $\\sqrt{d}$ before adding the [positional encodings](transformers.html#positional), so that the two have a similar size.",
    ]],
    ["h", "In PyTorch"],
    ["code", R`
import torch

torch.manual_seed(0)
emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=4)
ids = torch.tensor([[3, 7, 3], [1, 0, 9]])     # (batch=2, seq=3)
x = emb(ids)
print(x.shape)
print(torch.equal(x[0, 0], emb.weight[3]))     # the lookup is a row of the weight matrix

x.sum().backward()
print(emb.weight.grad[3])                      # token 3 was used twice -> gradient 2
print(emb.weight.grad[5])                      # token 5 was never used -> gradient 0`, "torch.Size([2, 3, 4])\nTrue\ntensor([2., 2., 2., 2.])\ntensor([0., 0., 0., 0.])", "PyTorch", "local"],
    ["danger", "Feeding token IDs to a network as numbers (for example `x = ids / V`). The IDs are arbitrary labels; their numeric order means nothing. Always go through an embedding lookup."],
  ],
},
{
  id: "positional", group: "Embeddings", title: "Positional Encodings: Sinusoidal, Learned & RoPE",
  sub: "Attention on its own can't tell word order apart, so we inject each token's position: by adding a position vector, or by rotating queries and keys.",
  tags: ["embeddings", "attention"],
  blocks: [
    ["p", "*Dog bites man* and *man bites dog* use the same three tokens. Attention compares every token with every other token, but nothing in that comparison says **where** each token is. Shuffle the input and the outputs are shuffled in exactly the same way; nothing else changes. Attention treats its input as a **set**:"],
    ["code", R`
import numpy as np

def attention(X):
    S = X @ X.T / np.sqrt(X.shape[1])
    A = np.exp(S - S.max(1, keepdims=True))
    A /= A.sum(1, keepdims=True)
    return A @ X

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 8))                    # 4 token vectors: "dog bites man ."
perm = [2, 1, 0, 3]                            # reorder to "man bites dog ."
out, out_perm = attention(X), attention(X[perm])
print(np.allclose(out[perm], out_perm))        # the same vectors, just shuffled: order is invisible`, "True"],
    ["p", "So we must add position information ourselves. There are three main families."],
    ["h", "1. Sinusoidal encodings (the original Transformer)"],
    ["p", "Give every position a fixed vector made of sine and cosine waves, and **add** it to the token embedding. Each pair of dimensions $(2i, 2i+1)$ is like the hand of a clock that turns at its own speed: the first pair spins quickly (changes a lot from one position to the next), later pairs turn more and more slowly. Together the hands give every position a unique pattern, a bit like the digits of a counter."],
    ["math", R`PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right) \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)`, "pos: the token's position · i: which pair of dimensions · d: the model width"],
    ["viz", "tf_sinusoid"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_pe(T, d):
    pos = np.arange(T)[:, None]                    # (T, 1)
    i = np.arange(0, d, 2)[None, :]                # (1, d/2): the even dimensions 0, 2, 4, ...
    angle = pos / 10000 ** (i / d)                 # (T, d/2)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angle)                    # even dimensions: sine
    pe[:, 1::2] = np.cos(angle)                    # odd dimensions: cosine
    return pe

pe = sinusoidal_pe(50, 64)
print(pe.shape)
print(pe[1, :6].round(3))

plt.figure(figsize=(6, 3.5))
plt.imshow(pe, aspect="auto", cmap="RdBu")
plt.xlabel("dimension")
plt.ylabel("position")
plt.title("Sinusoidal positional encoding (d = 64)")
plt.colorbar()
plt.show()`, "(50, 64)\n[0.841 0.54  0.682 0.732 0.533 0.846]"],
    ["p", "A nice property: because $\\sin a \\sin b + \\cos a \\cos b = \\cos(a - b)$, the dot product of two position vectors depends only on **how far apart** the positions are, not on where they are:"],
    ["code", R`
import numpy as np

def sinusoidal_pe(T, d):
    angle = np.arange(T)[:, None] / 10000 ** (np.arange(0, d, 2)[None, :] / d)
    pe = np.zeros((T, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe

pe = sinusoidal_pe(100, 64)
print(round(pe[10] @ pe[13], 4), round(pe[50] @ pe[53], 4))   # both 3 apart: identical
print(round(pe[10] @ pe[30], 4))                               # 20 apart: a different value`, "25.587 25.587\n18.8791"],
    ["p", "The model input is then `X = E[ids] + PE[:T]`: token meaning plus position, in the same vector."],
    ["h", "2. Learned absolute positions (BERT, GPT-2)"],
    ["p", "Simpler still: make a second embedding table with one learned vector per position, shape `(max_len, d)`, and add `P[:T]` to the token embeddings. It works well, but the model has no vector at all for positions beyond `max_len` (512 for BERT, 1024 for GPT-2), so it can't handle longer inputs."],
    ["h", "3. Rotary position embeddings, RoPE (most modern LLMs)"],
    ["p", "RoPE (Su et al., 2021) doesn't add anything to the input. Instead, inside every attention layer, it **rotates** each query and key vector by an angle proportional to its position. Dimensions are again taken in pairs, and pair $i$ is rotated by $pos \\cdot \\theta_i$ with $\\theta_i = 10000^{-2i/d}$: fast rotation for the first pairs, slow for the last."],
    ["math", R`\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}`, "the pair (x₂ᵢ, x₂ᵢ₊₁) of a query or key at position m is turned by the angle m·θᵢ"],
    ["p", "Why rotate? The attention score is a dot product, and the dot product of two rotated vectors depends only on the **difference** of their angles. So the score between a query at position $m$ and a key at position $n$ depends on the content and on $m - n$ only: attention becomes naturally **relative**. Drag the sliders and watch the score stay fixed when both positions move together:"],
    ["viz", "tf_rope"],
    ["code", R`
import numpy as np

def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)       # one rotation speed per pair of dimensions
    cos, sin = np.cos(pos * theta), np.sin(pos * theta)
    x1, x2 = x[0::2], x[1::2]                       # pairs (0,1), (2,3), ...
    out = np.empty_like(x)
    out[0::2] = x1 * cos - x2 * sin                 # a 2-D rotation of every pair
    out[1::2] = x1 * sin + x2 * cos
    return out

rng = np.random.default_rng(0)
q, k = rng.normal(size=8), rng.normal(size=8)
print(round(rope(q, 5) @ rope(k, 2), 4))           # positions 5 and 2: 3 apart
print(round(rope(q, 105) @ rope(k, 102), 4))       # positions 105 and 102: 3 apart again, same score
print(round(rope(q, 5) @ rope(k, 4), 4))           # 1 apart: a different score
print(np.isclose(np.linalg.norm(rope(q, 7)), np.linalg.norm(q)))   # rotations keep the length`, "-1.5865\n-1.5865\n-1.6954\nTrue"],
    ["p", "A fourth idea, **ALiBi** (Press et al., 2022), uses no position vectors at all: it subtracts a penalty proportional to the distance $|i - j|$ from each attention score, with a different slope per head."],
    ["table", ["Method", "Where it acts", "Relative?", "Longer inputs than in training", "Used by (as of 2026)"], [
      ["Sinusoidal", "Added to the input embeddings", "Partly (dot products depend on the offset)", "Defined for any position, but quality drops", "The original Transformer"],
      ["Learned absolute", "Added to the input embeddings", "No", "Impossible beyond `max_len`", "BERT, GPT-2"],
      ["**RoPE**", "Rotates q and k in every attention layer", "Yes", "Degrades, but can be stretched (position interpolation, NTK-aware scaling, YaRN)", "Llama, Mistral, Qwen and most open LLMs"],
      ["ALiBi", "A distance penalty on attention scores", "Yes", "Extrapolates fairly well", "BLOOM, MPT"],
    ]],
    ["p", "Stretching RoPE to longer contexts is a big topic of its own: see [long context](llm.html#long-context)."],
    ["warn", "There are two RoPE layouts. The paper rotates neighbouring pairs (0,1), (2,3)…, while many codebases (including Hugging Face's Llama code) rotate the first half against the second half, pairs (0, d/2), (1, d/2+1)…. Both are fine on their own, but porting weights from one layout to the other without permuting them silently breaks the model."],
    ["qa", "Going deeper: why the number 10000?", [
      ["p", "The wavelengths of the waves form a geometric series from $2\\pi$ (the fastest pair, one full turn about every 6 positions) up to $10000 \\cdot 2\\pi$ (the slowest). A large base makes the slowest waves much longer than any sequence the model will see, so far-apart positions never get confused with each other. Long-context models often raise the RoPE base (to 500,000 or more) for the same reason."],
    ]],
    ["danger", "Adding the positional encoding to the queries and keys *and* also using RoPE, or forgetting positions entirely. Without any position signal a transformer is a bag-of-words model: it literally cannot tell *dog bites man* from *man bites dog*."],
  ],
},
/* =========================================================================================== Attention */
{
  id: "attention-intuition", group: "Attention", title: "Attention: Queries, Keys & Values",
  sub: "Attention is a soft lookup: each token asks a question (query), compares it with every token's label (key), and collects a blend of their contents (values).",
  tags: ["attention"],
  blocks: [
    ["p", "Think of a dictionary. You give it a key (`\"cat\"`), it finds the one entry whose key matches exactly and returns its value. That is a **hard lookup**: all or nothing. Attention is a **soft lookup**. You give it a *query*; it compares the query with **every** key, scores how well each one matches, and returns a **blend** of all the values, weighted by those scores. Good matches contribute a lot, poor matches almost nothing."],
    ["diagram", "tf_qkv", [], "One query is compared with every key. The match scores become weights; the output is the weighted sum of the values."],
    ["code", R`
import numpy as np

# A dict is a HARD lookup: the key must match exactly and you get exactly one value
book = {"cat": "animal", "paris": "city", "red": "colour"}
print(book["cat"])

# Attention is a SOFT lookup: compare the query with every key, then blend all the values
keys = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])       # 3 keys (one per row)
values = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])   # 3 values
query = np.array([0.9, 0.1])                                # looks most like key 0

scores = keys @ query                                       # dot product = similarity with each key
weights = np.exp(scores) / np.exp(scores).sum()             # softmax: positive numbers that sum to 1
print("scores :", scores.round(2))
print("weights:", weights.round(3))
print("result :", (weights @ values).round(2))              # a blend, leaning towards value 0`, "animal\nscores : [0.9 0.1 0.7]\nweights: [0.441 0.198 0.361]\nresult : [6.21 3.79]"],
    ["p", "That is the whole idea. Everything else in a transformer's attention layer is about **where the queries, keys and values come from** and doing it for all tokens at once."],
    ["h", "Where Q, K and V come from"],
    ["p", "In **self-attention** every token plays all three roles. The token vectors $X$ (one row per token) are multiplied by three learned matrices:"],
    ["math", R`Q = X W_Q \qquad K = X W_K \qquad V = X W_V`, "three different linear views of the same tokens"],
    ["table", ["Role", "Question it answers", "Analogy"], [
      ["**Query** $q_i$", "What am I looking for?", "The search terms you type"],
      ["**Key** $k_j$", "What do I contain / what am I about?", "The title or tags of each document"],
      ["**Value** $v_j$", "What do I pass on if I'm chosen?", "The document's actual content"],
    ]],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
T, d, d_k = 4, 8, 4                         # 4 tokens, width 8, query/key width 4
X = rng.normal(size=(T, d))                 # one row per token
Wq, Wk, Wv = (rng.normal(size=(d, d_k)) * 0.3 for _ in range(3))   # three learned matrices

Q, K, V = X @ Wq, X @ Wk, X @ Wv
print(Q.shape, K.shape, V.shape)            # one query, key and value per token
print((Q @ K.T).shape)                      # (T, T): query i against key j for every pair`, "(4, 4) (4, 4) (4, 4)\n(4, 4)"],
    ["p", "Why three separate matrices instead of comparing the raw token vectors? Because *finding* and *giving* are different jobs. The word *it* wants to find a noun, but what it should *receive* from that noun is its meaning, not its \"I am a noun\" label. Separate $W_Q$, $W_K$ and $W_V$ let the model learn the matching rule and the content independently. The queries and keys are only used to compute weights; the values carry the information that actually flows."],
    ["p", "Step through it on a real sentence. Every number below is computed (with small random weights, so the pattern is arbitrary: training is what makes real attention patterns meaningful). Use the slider to choose which token to follow:"],
    ["viz", "tf_attention"],
    ["h", "What attention gives you"],
    ["list", [
      "**Direct connections.** Token 50 can read token 3 in one step, no matter the distance. An [RNN](dl.html#rnn) has to pass information through every token in between.",
      "**Parallelism.** All the queries are compared with all the keys in one matrix multiplication, so the GPU processes the whole sequence at once.",
      "**Content-based reading.** What each token reads depends on what it contains. The same word gets different context in different sentences.",
    ]],
    ["p", "Attention was first used as an add-on to RNN translators ([Seq2Seq and the birth of attention](dl.html#seq2seq)); the 2017 paper *Attention Is All You Need* showed you can drop the RNN and build the whole model from attention plus feed-forward layers. The next section writes the exact formula: [scaled dot-product attention](transformers.html#scaled-dot-product)."],
    ["warn", "Attention weights look like an explanation (\"the model looked at *cat*\"), but they are only one intermediate step. Later layers, the value vectors and the MLPs all matter too. Treat attention maps as a useful hint, not proof of why the model said something."],
    ["danger", "Thinking the query, key and value are the token itself. They are three *projections* of it, produced by three different learned matrices."],
  ],
},
{
  id: "scaled-dot-product", group: "Attention", title: "Scaled Dot-Product Attention",
  sub: "The one formula at the heart of every transformer: softmax(QKᵀ / √d) V, in a handful of lines.",
  tags: ["attention", "maths"],
  blocks: [
    ["p", "Given matrices of queries $Q$, keys $K$ and values $V$ (one row per token), attention is:"],
    ["math", R`\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V`, "d_k: the width of a query / key vector"],
    ["p", "Read it from the inside out, and track the shapes for $T$ tokens:"],
    ["steps", [
      "$QK^\\top$ has shape $(T, T)$. Entry $(i, j)$ is the dot product of query $i$ with key $j$: how much token $i$ wants to read from token $j$.",
      "Divide by $\\sqrt{d_k}$ to keep the scores in a sensible range (explained below).",
      "**Softmax along each row** turns the scores of one query into weights that are positive and sum to 1. This is the attention matrix $A$, shape $(T, T)$.",
      "Multiply by $V$ (shape $(T, d_v)$): each output row is a weighted average of the value rows. Result shape $(T, d_v)$.",
    ]],
    ["p", "A tiny example with hand-sized numbers. One query, two keys, $d_k = 2$:"],
    ["code", R`
import numpy as np

q = np.array([1.0, 0.0])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[1.0, 2.0], [3.0, 4.0]])

scores = K @ q / np.sqrt(2)                 # [1, 0] / 1.414
w = np.exp(scores) / np.exp(scores).sum()   # softmax
print("scores:", scores.round(3))
print("weights:", w.round(3))
print("output :", (w @ V).round(3))         # 0.670 * [1, 2] + 0.330 * [3, 4]`, "scores: [0.707 0.   ]\nweights: [0.67 0.33]\noutput : [1.66 2.66]"],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # shift so exp() cannot overflow
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)   # (..., T_q, T_k)
    weights = softmax(scores)                        # each row sums to 1
    return weights @ V, weights                      # (..., T_q, d_v)

rng = np.random.default_rng(0)
Q, K, V = (rng.normal(size=(5, 4)) for _ in range(3))
out, w = attention(Q, K, V)
print(out.shape, w.shape)
print("rows sum to 1:", w.sum(axis=-1).round(3))
print("first row of weights:", w[0].round(3))`, "(5, 4) (5, 5)\nrows sum to 1: [1. 1. 1. 1. 1.]\nfirst row of weights: [0.162 0.171 0.149 0.226 0.292]"],
    ["p", "Because of `swapaxes(-1, -2)` and the `...` in the shapes, the same function works unchanged on batches `(B, T, d)` and on multiple heads `(B, h, T, d)`. That is exactly how it is used in [multi-head attention](transformers.html#multi-head)."],
    ["h", "Why divide by √d"],
    ["p", "If the entries of a query and a key are independent with variance 1, their dot product is a sum of $d_k$ such products, so its variance is $d_k$ and its standard deviation $\\sqrt{d_k}$. Wider heads mean bigger scores. Large scores push the softmax to put almost all weight on one token (**saturation**), where the gradient is nearly zero and learning stalls. Dividing by $\\sqrt{d_k}$ brings the scores back to variance 1."],
    ["code", R`
import numpy as np

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

rng = np.random.default_rng(0)
for dk in [4, 64, 1024]:
    q, k = rng.normal(size=(2000, dk)), rng.normal(size=(2000, dk))
    s = (q * k).sum(axis=1)                          # 2000 dot products
    print(f"d_k={dk:5d}  std of q.k = {s.std():6.2f}   after / sqrt(d_k) = {(s / np.sqrt(dk)).std():.2f}")

q, K = rng.normal(size=1024), rng.normal(size=(8, 1024))   # one query, 8 keys, d_k = 1024
raw = K @ q
print("biggest weight, unscaled:", softmax(raw).max().round(3))
print("biggest weight, scaled  :", softmax(raw / np.sqrt(1024)).max().round(3))`, "d_k=    4  std of q.k =   2.04   after / sqrt(d_k) = 1.02\nd_k=   64  std of q.k =   7.85   after / sqrt(d_k) = 0.98\nd_k= 1024  std of q.k =  32.72   after / sqrt(d_k) = 1.02\nbiggest weight, unscaled: 0.96\nbiggest weight, scaled  : 0.25"],
    ["p", "Without the scaling, one token grabs essentially all the weight for a wide head. With it, the weights stay spread out and trainable."],
    ["h", "The cost: attention is quadratic"],
    ["p", "The score matrix has $T \\times T$ entries per head per layer. Double the length and you quadruple it:"],
    ["code", R`
for T in [1_000, 10_000, 100_000]:
    entries = T * T
    gb = entries * 2 / 1e9                           # 2 bytes per entry in float16
    print(f"T = {T:>7,}: {entries:>14,} scores  =  {gb:8.3f} GB for ONE head in ONE layer")`, "T =   1,000:      1,000,000 scores  =     0.002 GB for ONE head in ONE layer\nT =  10,000:    100,000,000 scores  =     0.200 GB for ONE head in ONE layer\nT = 100,000: 10,000,000,000 scores  =    20.000 GB for ONE head in ONE layer"],
    ["p", "This is why long context is expensive and why [efficient attention](transformers.html#efficient-attention) (FlashAttention and friends) exists: they never store the full matrix."],
    ["h", "In PyTorch"],
    ["p", "PyTorch ships the whole formula as one fused function. It matches our NumPy version:"],
    ["code", R`
import numpy as np
import torch
import torch.nn.functional as F

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

rng = np.random.default_rng(0)
Q, K, V = (rng.normal(size=(5, 4)) for _ in range(3))
mine = softmax(Q @ K.T / np.sqrt(4)) @ V

theirs = F.scaled_dot_product_attention(*(torch.tensor(a)[None] for a in (Q, K, V)))   # adds a batch dim
print(theirs.shape)
print(np.allclose(mine, theirs[0].numpy()))`, "torch.Size([1, 5, 4])\nTrue", "PyTorch", "local"],
    ["tip", "Use `F.scaled_dot_product_attention` in real PyTorch code. It picks a fast fused kernel (FlashAttention-style) when it can and still gives the same numbers."],
    ["danger", "Taking the softmax over the wrong axis. It must be over the **keys** (the last axis of the score matrix), so that each query's weights sum to 1. Softmax over the queries axis produces garbage that still runs without errors."],
    ["p", "Practise it: [Scaled dot-product attention](practice.html#p/scaled-dot-product-attention)."],
  ],
},
{
  id: "masking", group: "Attention", title: "Causal & Padding Masks",
  sub: "Masks set some attention scores to −∞ before the softmax, so a token cannot look at the future or at padding.",
  tags: ["attention"],
  blocks: [
    ["p", "Attention by default lets every token see every other token. Two situations need to forbid some pairs:"],
    ["list", [
      "**Causal (look-ahead) mask.** A model that predicts the *next* token must not see it. Token $i$ may only attend to tokens $j \\le i$. Without the mask, training would let the model copy the answer from the input.",
      "**Padding mask.** Sentences in a batch have different lengths, so shorter ones are filled with `<pad>`. Those filler tokens carry no information and must never be attended to.",
    ]],
    ["p", "The trick: add $-\\infty$ to the forbidden scores **before** the softmax. Since $e^{-\\infty} = 0$, those positions get weight exactly 0, and the remaining weights still sum to 1."],
    ["viz", "tf_mask"],
    ["h", "Causal mask in NumPy"],
    ["code", R`
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

T = 5
rng = np.random.default_rng(0)
scores = rng.normal(size=(T, T)).round(2)

mask = np.triu(np.ones((T, T), dtype=bool), k=1)     # True above the diagonal = the future
masked = np.where(mask, -np.inf, scores)             # forbidden pairs get -inf
A = softmax(masked)
print(A.round(2))
print("row sums:", A.sum(axis=1))`, "[[1.   0.   0.   0.   0.  ]\n [0.28 0.72 0.   0.   0.  ]\n [0.32 0.62 0.06 0.   0.  ]\n [0.15 0.18 0.22 0.46 0.  ]\n [0.1  0.43 0.06 0.15 0.27]]\nrow sums: [1. 1. 1. 1. 1.]"],
    ["p", "Row 0 is `[1, 0, 0, 0, 0]`: the first token can only see itself. Row 4 sees all five tokens. The upper triangle is exactly zero."],
    ["note", "Why this is so useful for training: one forward pass over a sentence of $T$ tokens gives $T$ next-token predictions at once. Position $i$ predicts token $i+1$ and, thanks to the mask, only used tokens up to $i$. That is $T$ training examples for the price of one pass. At generation time the mask is not needed: the future does not exist yet (see [the KV cache](llm.html#kv-cache))."],
    ["h", "Padding mask"],
    ["p", "A padding mask has one entry per key position and is the same for all queries of that sentence. It is built from the lengths, and combined with the causal mask:"],
    ["code", R`
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

lengths = np.array([3, 5])                           # two sentences in a batch, padded to T = 5
T = 5
is_pad = np.arange(T)[None, :] >= lengths[:, None]   # (B, T): True where the token is <pad>
print(is_pad.astype(int))

rng = np.random.default_rng(1)
scores = rng.normal(size=(2, T, T))                  # (B, T_q, T_k)
causal = np.triu(np.ones((T, T), dtype=bool), k=1)
forbidden = causal[None, :, :] | is_pad[:, None, :]  # broadcast: (B, T, T)
A = softmax(np.where(forbidden, -np.inf, scores))
print(A[0].round(2))                                 # sentence 0: columns 3, 4 are padding -> all 0`, "[[0 0 0 1 1]\n [0 0 0 0 0]]\n[[1.   0.   0.   0.   0.  ]\n [0.73 0.27 0.   0.   0.  ]\n [0.32 0.53 0.15 0.   0.  ]\n [0.5  0.29 0.21 0.   0.  ]\n [0.19 0.14 0.67 0.   0.  ]]"],
    ["h", "Pitfalls"],
    ["code", R`
import numpy as np

with np.errstate(all="ignore"):
    row = np.array([-np.inf, -np.inf, -np.inf])      # a query that is not allowed to see anything
    e = np.exp(row - row.max())                      # -inf - (-inf) = nan
    print(e / e.sum())

# fix: always let a token see at least itself, or use a large finite negative number
safe = np.array([-1e9, -1e9, -1e9])
e = np.exp(safe - safe.max())
print((e / e.sum()).round(3))`, "[nan nan nan]\n[0.333 0.333 0.333]"],
    ["p", "A fully masked row gives NaN (0/0), and one NaN poisons the whole network. This can happen for padded query rows with a padding-only mask; real implementations either keep the diagonal visible or zero such rows out. Many libraries use a very large negative number such as -1e9 (or the dtype's minimum) instead of a true $-\\infty$ for the same reason."],
    ["h", "In PyTorch"],
    ["code", R`
import torch
import torch.nn.functional as F

torch.manual_seed(0)
q = k = v = torch.randn(1, 1, 4, 8)                  # (batch, heads, T, d)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))   # True = allowed
out2 = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
print(torch.allclose(out, out2))
print(torch.allclose(out[0, 0, 0], v[0, 0, 0]))      # token 0 sees only itself, so its output is v[0]`, "True\nTrue", "PyTorch", "local"],
    ["table", ["Mask", "Shape", "Used in"], [
      ["Causal", "(T, T), same for all", "Decoder-only models (GPT, Llama), the decoder in encoder–decoder models"],
      ["Padding", "(B, 1, T) per sentence", "Any batched input with different lengths"],
      ["None (bidirectional)", "-", "Encoders such as BERT"],
      ["Prefix / custom", "(T, T)", "Prefix-LM: bidirectional over the prompt, causal over the answer"],
    ]],
    ["warn", "Mask conventions differ between libraries: in some `True` means \"allowed\", in others \"forbidden\" (`nn.MultiheadAttention`'s boolean `attn_mask` marks `True` as **not allowed**, while `F.scaled_dot_product_attention` treats `True` as **take part**). Check the docs and test with a tiny example."],
    ["danger", "Applying the mask **after** the softmax (zeroing the weights). The remaining weights no longer sum to 1, so the output is scaled wrongly. Always mask the scores first, then softmax."],
    ["p", "Practise it: [Causal masked softmax](practice.html#p/causal-mask-softmax)."],
  ],
},
{
  id: "multi-head", group: "Attention", title: "Multi-Head Attention",
  sub: "Run several smaller attentions in parallel on different slices of the vector, then join the results, so the model can look for different things at once.",
  tags: ["attention"],
  blocks: [
    ["p", "A single attention pattern is a limitation. In one sentence a token might want to find its subject, the previous word, and a matching bracket, all at the same time. **Multi-head attention** gives the model $h$ separate attentions (heads). Each head has its own view of the tokens, makes its own attention matrix, and reads its own information. The heads' outputs are then concatenated."],
    ["p", "The trick that keeps it cheap: the model width $d$ is **divided** among the heads, $d_k = d / h$. So eight heads on width 512 each work with 64 dimensions, and the total cost is about the same as one big head."],
    ["math", R`\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O \qquad \text{head}_a = \text{Attention}(X W_Q^{(a)},\; X W_K^{(a)},\; X W_V^{(a)})`, "each head a has its own slice of the projection matrices; W_O mixes the concatenated result"],
    ["viz", "tf_multihead"],
    ["h", "In practice: one big matmul, then reshape"],
    ["p", "Nobody loops over heads. We project once with full-size $W_Q, W_K, W_V$ (shape $d \\times d$), then **reshape** the result so the heads become a batch dimension. This is the shape juggling that trips people up:"],
    ["code", R`
import numpy as np

T, d, h = 4, 8, 2
d_k = d // h
X = np.arange(T * d, dtype=float).reshape(T, d)         # pretend this is Q
print("Q:", X.shape)

heads = X.reshape(T, h, d_k).transpose(1, 0, 2)         # (T, d) -> (T, h, d_k) -> (h, T, d_k)
print("split:", heads.shape)
print("head 0, token 0:", heads[0, 0], " head 1, token 0:", heads[1, 0])   # head 1 owns columns 4..7

merged = heads.transpose(1, 0, 2).reshape(T, d)         # undo: (h, T, d_k) -> (T, h, d_k) -> (T, d)
print("merge round-trips:", np.array_equal(merged, X))`, "Q: (4, 8)\nsplit: (2, 4, 4)\nhead 0, token 0: [0. 1. 2. 3.]  head 1, token 0: [4. 5. 6. 7.]\nmerge round-trips: True"],
    ["p", "Now the full layer, from scratch. Attention runs on tensors of shape $(h, T, d_k)$, so the single-head function from before works for all heads at the same time:"],
    ["code", R`
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def multi_head_attention(X, Wq, Wk, Wv, Wo, h, causal=False):
    T, d = X.shape
    d_k = d // h
    split = lambda M: M.reshape(T, h, d_k).transpose(1, 0, 2)          # (h, T, d_k)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)                   # (h, T, T)
    if causal:
        scores = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, scores)
    A = softmax(scores)                                                # one attention matrix per head
    out = (A @ V).transpose(1, 0, 2).reshape(T, d)                     # concat heads -> (T, d)
    return out @ Wo, A

rng = np.random.default_rng(0)
T, d = 6, 16
X = rng.normal(size=(T, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.25 for _ in range(4))
for h in [1, 2, 4]:
    y, A = multi_head_attention(X, Wq, Wk, Wv, Wo, h, causal=True)
    print(f"h={h}: output {y.shape}, attention maps {A.shape}")
print("parameters in the layer:", 4 * d * d, "(the same for any number of heads)")`, "h=1: output (6, 16), attention maps (1, 6, 6)\nh=2: output (6, 16), attention maps (2, 6, 6)\nh=4: output (6, 16), attention maps (4, 6, 6)\nparameters in the layer: 1024 (the same for any number of heads)"],
    ["p", "The number of heads does not change the parameter count (always $4d^2$ for $W_Q, W_K, W_V, W_O$) or the output shape. It only changes how the same width is partitioned. We can check our function against PyTorch's `nn.MultiheadAttention` by copying the weights over. This runs fully offline:"],
    ["code", R`
import numpy as np
import torch

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def mha(X, Wq, Wk, Wv, Wo, h):
    T, d = X.shape
    d_k = d // h
    split = lambda M: M.reshape(T, h, d_k).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_k))
    return (A @ V).transpose(1, 0, 2).reshape(T, d) @ Wo

torch.manual_seed(0)
d, h, T = 8, 2, 5
layer = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True)
x = torch.randn(1, T, d)
y, w = layer(x, x, x, need_weights=True, average_attn_weights=False)
print(y.shape, w.shape)                                  # weights: (batch, heads, T, T)

Win = layer.in_proj_weight.detach().numpy()              # rows are [W_Q; W_K; W_V], each used as x @ W.T
Wq, Wk, Wv = Win[:d].T, Win[d:2 * d].T, Win[2 * d:].T
Wo = layer.out_proj.weight.detach().numpy().T
mine = mha(x[0].numpy(), Wq, Wk, Wv, Wo, h)
print("same as PyTorch:", bool(np.abs(mine - y[0].detach().numpy()).max() < 1e-5))`, "torch.Size([1, 5, 8]) torch.Size([1, 2, 5, 5])\nsame as PyTorch: True", "PyTorch", "local"],
    ["h", "What do heads learn?"],
    ["p", "Nobody designs the heads' jobs; they emerge in training. Studies of trained models find heads that attend to the previous token, to the first token, to matching earlier occurrences (so-called *induction heads*, important for in-context learning), to syntactic partners, and many that are hard to describe. Some heads can be removed with little loss, others are critical."],
    ["table", ["Choice", "Effect"], [
      ["More heads, smaller $d_k$", "More different patterns per layer, but each head has fewer dimensions to compute a match."],
      ["Fewer heads, larger $d_k$", "Richer matching per head, fewer patterns."],
      ["Typical values (as of 2026)", "$d_k$ is usually 64 or 128; the number of heads follows from the width: 12 heads of 64 in GPT-2 small, 32 heads of 128 in Llama-class 7B models."],
    ]],
    ["p", "To save memory at inference time, newer models share the key and value heads between several query heads: see [MQA and GQA](transformers.html#efficient-attention)."],
    ["danger", "Reshaping `(T, d)` straight to `(h, T, d_k)`. That scrambles tokens and heads. The correct order is reshape to `(T, h, d_k)` first, **then** transpose to `(h, T, d_k)`."],
    ["p", "Practise it: [Multi-head split and merge](practice.html#p/multi-head-split-merge)."],
  ],
},
{
  id: "self-vs-cross", group: "Attention", title: "Self-Attention vs Cross-Attention",
  sub: "Same formula, different sources: in self-attention Q, K and V come from one sequence; in cross-attention the queries come from one sequence and the keys and values from another.",
  tags: ["attention"],
  blocks: [
    ["p", "The attention formula never says where $Q$, $K$ and $V$ come from. That choice gives two flavours:"],
    ["table", ["", "Self-attention", "Cross-attention"], [
      ["Queries come from", "the sequence itself", "sequence A (e.g. the text being generated)"],
      ["Keys and values come from", "the same sequence", "sequence B (e.g. the encoded source sentence, or an image)"],
      ["Attention matrix shape", "$(T, T)$", "$(T_A, T_B)$: rows are A's tokens, columns are B's"],
      ["Output shape", "$(T, d)$", "$(T_A, d)$: always as many rows as there are **queries**"],
      ["Question it answers", "How do the tokens of this sequence relate to each other?", "Which parts of B matter for each token of A?"],
      ["Causal mask?", "Yes in decoders, no in encoders", "No (B is fully known)"],
    ]],
    ["p", "In code the only difference is the argument you pass for the keys and values:"],
    ["code", R`
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attend(Xq, Xkv, Wq, Wk, Wv):
    Q, K, V = Xq @ Wq, Xkv @ Wk, Xkv @ Wv          # queries from one place, keys/values from another
    A = softmax(Q @ K.T / np.sqrt(Q.shape[1]))
    return A @ V, A

rng = np.random.default_rng(0)
d = 8
Wq, Wk, Wv = (rng.normal(size=(d, d)) * 0.4 for _ in range(3))
dec = rng.normal(size=(3, d))                      # 3 decoder tokens (what we are writing)
enc = rng.normal(size=(7, d))                      # 7 encoder tokens (the source sentence)

out_self, A_self = attend(dec, dec, Wq, Wk, Wv)    # self-attention
out_cross, A_cross = attend(dec, enc, Wq, Wk, Wv)  # cross-attention
print("self :", A_self.shape, "->", out_self.shape)
print("cross:", A_cross.shape, "->", out_cross.shape)
print("each decoder token spreads weight over the 7 encoder tokens:", A_cross.sum(axis=1).round(2))`, "self : (3, 3) -> (3, 8)\ncross: (3, 7) -> (3, 8)\neach decoder token spreads weight over the 7 encoder tokens: [1. 1. 1.]"],
    ["h", "Where cross-attention is used"],
    ["list", [
      "**Encoder–decoder transformers** (the original Transformer, T5, BART, Whisper): each decoder layer has self-attention over the text generated so far, then cross-attention that reads the encoder's output (the source sentence, or the audio).",
      "**Image generation with diffusion models:** the image latents are the queries and the text prompt embeddings are the keys and values, which is how a prompt steers the picture.",
      "**Multimodal models:** some architectures let text tokens attend to image features through cross-attention layers; others skip cross-attention and simply put image tokens into the same sequence (see [Vision Transformers and multimodal models](transformers.html#vit-multimodal)).",
      "**Retrieval-style models** that attend to retrieved passages.",
    ]],
    ["p", "A decoder-only LLM (GPT, Llama) has **no** cross-attention at all. The prompt and the answer live in one sequence and are handled by causal self-attention. See [architectures](transformers.html#architectures)."],
    ["p", "PyTorch's `nn.MultiheadAttention` takes separate `query`, `key` and `value` arguments, so cross-attention is just passing different tensors. Runs offline:"],
    ["code", R`
import torch

torch.manual_seed(0)
mha = torch.nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
decoder_states = torch.randn(2, 5, 16)           # (batch, 5 target tokens)
encoder_states = torch.randn(2, 9, 16)           # (batch, 9 source tokens)

out, weights = mha(query=decoder_states, key=encoder_states, value=encoder_states)
print(out.shape)                                 # as many rows as queries
print(weights.shape)                             # (batch, queries, keys), averaged over heads
print(weights.sum(dim=-1)[0].detach().numpy().round(3))                    # each query's weights over the 9 source tokens sum to 1`, "torch.Size([2, 5, 16])\ntorch.Size([2, 5, 9])\n[1. 1. 1. 1. 1.]", "PyTorch", "local"],
    ["tip", "Rule of thumb for shapes: **queries decide the output length; keys and values decide what can be read**. The keys and values must have the same length as each other, but not the same as the queries."],
    ["danger", "Applying the causal mask in cross-attention. The source sequence is fully available, so each target token should see all of it; masking it would just hide information."],
  ],
},
/* =========================================================================================== Architecture */
{
  id: "transformer-block", group: "Architecture", title: "The Transformer Block",
  sub: "Attention to mix information between tokens, an MLP to process each token, both wrapped in LayerNorm and residual connections, stacked N times.",
  tags: ["architecture"],
  blocks: [
    ["p", "A transformer is one **block** repeated many times (12 in GPT-2 small, dozens in large models). Every block has the same two sublayers:"],
    ["list", [
      "**Multi-head self-attention** ([the previous section](transformers.html#multi-head)): the only place where tokens exchange information.",
      "**A feed-forward network (MLP)**: two linear layers with a nonlinearity between them, applied to each token separately. It expands the vector (usually to $4d$), applies GELU, and shrinks it back to $d$. Much of the model's stored knowledge is thought to live here.",
    ]],
    ["p", "Each sublayer is wrapped with two helpers that make deep stacks trainable: a **residual connection** (add the sublayer's input to its output) and **LayerNorm**."],
    ["diagram", "tf_block", [], "A pre-LN transformer block. The blue path is the residual stream: the sublayers read a normalised copy and only add to it."],
    ["math", R`x \leftarrow x + \text{MHA}(\text{LN}(x)) \qquad x \leftarrow x + \text{MLP}(\text{LN}(x))`, "the pre-LN block used by GPT-2 and most modern LLMs"],
    ["h", "LayerNorm and GELU from scratch"],
    ["p", "LayerNorm normalises **each token's vector on its own**: subtract its mean, divide by its standard deviation. (BatchNorm, in contrast, normalises across the batch; see [BatchNorm & LayerNorm](dl.html#normalization).) Real LayerNorm then multiplies by a learned scale $\\gamma$ and adds a learned shift $\\beta$; we leave them out here."],
    ["code", R`
import numpy as np

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)           # per token, over the feature axis
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

rng = np.random.default_rng(0)
x = rng.normal(5, 3, size=(2, 6))                   # 2 tokens with a large mean and spread
y = layer_norm(x)
print("mean per token:", y.mean(axis=-1).round(4))
print("std  per token:", y.std(axis=-1).round(3))
print("gelu:", gelu(np.array([-2.0, -0.5, 0.0, 0.5, 2.0])).round(3))`, "mean per token: [-0. -0.]\nstd  per token: [1. 1.]\ngelu: [-0.045 -0.154  0.     0.346  1.955]"],
    ["h", "The whole block in NumPy"],
    ["p", "Put the pieces together. Notice that the output has the same shape as the input, so blocks can be stacked, and that the causal mask makes the output of token $i$ independent of the tokens after it:"],
    ["code", R`
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def block(x, p, h):
    T, d = x.shape
    d_k = d // h
    a = layer_norm(x)                                                  # pre-LN: normalise a COPY
    split = lambda M: M.reshape(T, h, d_k).transpose(1, 0, 2)
    Q, K, V = split(a @ p["Wq"]), split(a @ p["Wk"]), split(a @ p["Wv"])
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)       # causal mask
    att = (softmax(s) @ V).transpose(1, 0, 2).reshape(T, d) @ p["Wo"]
    x = x + att                                                        # residual 1: add, don't replace
    mlp = gelu(layer_norm(x) @ p["W1"]) @ p["W2"]                      # expand to 4d, GELU, shrink to d
    return x + mlp                                                     # residual 2

rng = np.random.default_rng(0)
d, h, T = 16, 4, 6
p = {k: rng.normal(0, 0.2, size=s) for k, s in
     [("Wq", (d, d)), ("Wk", (d, d)), ("Wv", (d, d)), ("Wo", (d, d)), ("W1", (d, 4 * d)), ("W2", (4 * d, d))]}
x = rng.normal(size=(T, d))

y = block(x, p, h)
print("shape in/out:", x.shape, y.shape)

x2 = x.copy()
x2[-1] += 5.0                                                          # change only the LAST token
y2 = block(x2, p, h)
print("earlier tokens unchanged:", np.allclose(y[:-1], y2[:-1]), "| last token changed:", not np.allclose(y[-1], y2[-1]))

z = x
for _ in range(4):                                                     # stack 4 blocks (same weights, just to see the shape)
    z = block(z, p, h)
print("after 4 blocks:", z.shape)`, "shape in/out: (6, 16) (6, 16)\nearlier tokens unchanged: True | last token changed: True\nafter 4 blocks: (6, 16)"],
    ["viz", "tf_residual"],
    ["h", "Why residual connections and LayerNorm"],
    ["list", [
      "**Residual connections** give gradients a direct highway from the loss back to the first layer (the same idea as in [ResNets](dl.html#pooling-resnet)). They also change how to think about the model: the vector $x$ is a **residual stream** that every layer reads from and writes a small update into, rather than a value that each layer replaces.",
      "**LayerNorm** keeps the scale of activations under control, which stabilises training. It works per token, so it does not depend on the batch size or on padding.",
    ]],
    ["h", "Pre-LN vs post-LN"],
    ["table", ["", "Post-LN (original 2017 paper)", "Pre-LN (GPT-2 and later)"], [
      ["Formula", "$x \\leftarrow \\text{LN}(x + \\text{Sublayer}(x))$", "$x \\leftarrow x + \\text{Sublayer}(\\text{LN}(x))$"],
      ["Residual path", "Passes through a LayerNorm in every layer", "Clean: nothing in the way"],
      ["Training of deep stacks", "Needs careful learning-rate warm-up; can diverge", "Much more stable"],
      ["Final layer", "-", "Needs one extra LayerNorm after the last block"],
    ]],
    ["p", "As of 2026 almost all large language models use pre-LN, usually with two refinements: **RMSNorm** instead of LayerNorm (only divides by the root-mean-square, no mean subtraction, slightly cheaper) and a **gated MLP such as SwiGLU** instead of the plain GELU MLP. The structure of the block stays the same."],
    ["h", "Where the parameters are"],
    ["code", R`
def block_params(d, ratio=4):
    attn = 4 * d * d                     # W_Q, W_K, W_V, W_O
    mlp = 2 * ratio * d * d              # d -> 4d and 4d -> d
    return attn + mlp                    # = 12 d^2 (ignoring biases and LayerNorm)

d, layers, vocab, ctx = 768, 12, 50257, 1024     # GPT-2 small
blocks = layers * block_params(d)
tok_emb, pos_emb = vocab * d, ctx * d
print(f"per block   : {block_params(d):,}")
print(f"12 blocks   : {blocks:,}")
print(f"embeddings  : {tok_emb + pos_emb:,}")
print(f"total (approx): {blocks + tok_emb + pos_emb:,}")`, "per block   : 7,077,888\n12 blocks   : 84,934,656\nembeddings  : 39,383,808\ntotal (approx): 124,318,464"],
    ["p", "That total matches the roughly 124M usually quoted for GPT-2 small (the exact count differs slightly because of biases and LayerNorm). Two thirds of each block's parameters are in the MLP, one third in attention. (In mixture-of-experts models the MLP is what gets multiplied: see [Mixture of Experts](transformers.html#moe).)"],
    ["h", "In PyTorch"],
    ["code", R`
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)    # True = may NOT attend
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))

torch.manual_seed(0)
blk = Block(d=64, h=4)
x = torch.randn(2, 10, 64)                                 # (batch, tokens, width)
print(blk(x).shape)
print("parameters:", sum(p.numel() for p in blk.parameters()))`, "torch.Size([2, 10, 64])\nparameters: 49984", "PyTorch", "local"],
    ["warn", "Forgetting the extra LayerNorm after the last block in a pre-LN model. The residual stream is never normalised inside the blocks, so its scale grows with depth; the final LayerNorm is needed before the output layer."],
    ["danger", "Normalising the residual stream itself (`x = layer_norm(x)`) and then adding. In pre-LN you normalise a **copy** that goes into the sublayer, and add the sublayer output to the original `x`."],
    ["p", "Practise it: [LayerNorm forward](practice.html#p/layernorm-forward) and [A pre-LN transformer block](practice.html#p/pre-ln-block)."],
  ],
},
{
  id: "architectures", group: "Architecture", title: "Encoder-Only, Decoder-Only & Encoder–Decoder",
  sub: "The same blocks arranged three ways: read everything at once (BERT), generate left to right (GPT), or read then write (T5).",
  tags: ["architecture"],
  blocks: [
    ["p", "The idea is simple: you take the transformer block, change **which tokens may look at which** (the mask), choose **what the model is trained to predict**, and you get three families."],
    ["diagram", "tf_archs", [], "The three families. Blue cells = allowed attention. Decoder-only models use a causal (lower-triangular) mask."],
    ["table", ["", "Encoder-only", "Decoder-only", "Encoder–decoder"], [
      ["Attention", "Bidirectional", "Causal", "Encoder: bidirectional. Decoder: causal + cross-attention to the encoder"],
      ["Training task", "Masked language modelling: hide ~15% of tokens, predict them", "Next-token prediction", "Corrupt or transform the input, generate the output (e.g. fill masked spans, translate)"],
      ["Output", "A contextual vector per token", "A probability over the next token", "A new sequence"],
      ["Typical uses", "Classification, search embeddings, rerankers, tagging", "Chat, writing, code, reasoning", "Translation, summarisation, speech recognition"],
      ["Examples", "BERT, RoBERTa, DeBERTa", "GPT family, Llama, Claude-style assistants, Mistral, Qwen", "Original Transformer, T5, BART, Whisper"],
    ]],
    ["p", "The masks, side by side. A *prefix-LM* is a hybrid: the prompt is bidirectional, the answer is causal."],
    ["code", R`
import numpy as np

T, P = 6, 3                                         # 6 tokens; the first 3 are the prompt
full = np.ones((T, T), dtype=int)                   # encoder: everyone sees everyone
causal = np.tril(np.ones((T, T), dtype=int))        # decoder: only the past (and itself)
prefix = causal.copy()
prefix[:P, :P] = 1                                  # prefix-LM: prompt tokens also see each other

for name, m in [("bidirectional", full), ("causal", causal), ("prefix-LM", prefix)]:
    print(name)
    print(m)`, "bidirectional\n[[1 1 1 1 1 1]\n [1 1 1 1 1 1]\n [1 1 1 1 1 1]\n [1 1 1 1 1 1]\n [1 1 1 1 1 1]\n [1 1 1 1 1 1]]\ncausal\n[[1 0 0 0 0 0]\n [1 1 0 0 0 0]\n [1 1 1 0 0 0]\n [1 1 1 1 0 0]\n [1 1 1 1 1 0]\n [1 1 1 1 1 1]]\nprefix-LM\n[[1 1 1 0 0 0]\n [1 1 1 0 0 0]\n [1 1 1 0 0 0]\n [1 1 1 1 0 0]\n [1 1 1 1 1 0]\n [1 1 1 1 1 1]]"],
    ["h", "Encoder-only: understand"],
    ["p", "BERT (2018) reads the whole text with bidirectional attention, so every token's vector knows both its left and right context. It is trained by hiding words and predicting them (**masked language modelling**). A special `[CLS]` token at the start collects a summary of the sentence for classification. Encoders cannot generate text token by token, but they are small, fast and excellent at producing embeddings, which is why they still power much of search and reranking (see [embeddings and vector search](llm.html#embeddings-search))."],
    ["h", "Decoder-only: generate"],
    ["p", "GPT-style models read left to right with a causal mask and are trained to predict the next token ([pretraining](llm.html#pretraining)). To use them you just continue the text. Every task becomes \"continue this prompt\": translation, question answering, classification, code. As of 2026 nearly all large general-purpose LLMs are decoder-only."],
    ["h", "Encoder–decoder: transform"],
    ["p", "The encoder reads the input once with bidirectional attention. The decoder generates the output token by token with causal self-attention **and** [cross-attention](transformers.html#self-vs-cross) into the encoder's final vectors. It is a natural fit when there is a clear input and a clear output, such as translation. T5 casts everything as text-to-text, including classification."],
    ["h", "Building them without downloading anything"],
    ["p", "Hugging Face can create a model from just a configuration (random weights, no download). That makes the difference visible in the parameter names:"],
    ["code", R`
from transformers import (BertConfig, BertModel, GPT2Config, GPT2LMHeadModel,
                          T5Config, T5ForConditionalGeneration)

bert = BertModel(BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=2, intermediate_size=64))
gpt = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_embd=32, n_layer=2, n_head=2, n_positions=64))
t5 = T5ForConditionalGeneration(T5Config(vocab_size=100, d_model=32, d_ff=64, num_layers=2,
                                         num_decoder_layers=2, num_heads=2, d_kv=16))

for name, m in [("BERT   (encoder-only)", bert), ("GPT-2  (decoder-only)", gpt), ("T5     (enc-dec)", t5)]:
    n_cross = sum("EncDecAttention" in k for k, _ in m.named_parameters())
    print(f"{name:22s} parameters: {sum(p.numel() for p in m.parameters()):>6,}   cross-attention tensors: {n_cross}")`, "BERT   (encoder-only)  parameters: 37,856   cross-attention tensors: 0\nGPT-2  (decoder-only)  parameters: 30,720   cross-attention tensors: 0\nT5     (enc-dec)       parameters: 44,672   cross-attention tensors: 8", "Hugging Face", "local"],
    ["h", "Why decoder-only won"],
    ["list", [
      "**One simple objective** (predict the next token) that needs no labels and scales to trillions of tokens.",
      "**One model for every task**: just describe the task in the prompt (see [prompt engineering](llm.html#prompt-engineering)).",
      "**Efficient generation**: the [KV cache](llm.html#kv-cache) makes each new token cheap, and the same architecture serves training and inference.",
      "**Strong scaling behaviour** with data and parameters ([scaling laws](llm.html#scaling-laws)).",
    ]],
    ["note", "Encoders and encoder–decoders are not obsolete. Small, fast encoders are the standard choice for embeddings, classification and reranking, and encoder–decoders remain common for speech recognition and translation."],
    ["danger", "Using a decoder-only model's hidden states as if they saw the whole sentence. Because of the causal mask, the vector of an early token knows nothing about later tokens; only the **last** token has seen everything."],
  ],
},
{
  id: "tiny-gpt", group: "Architecture", title: "A Tiny GPT Forward Pass in NumPy",
  sub: "Assemble embeddings, positions, blocks and the output layer into a complete language model, and read off next-token probabilities.",
  tags: ["from scratch", "architecture"],
  blocks: [
    ["p", "We now have every ingredient. A GPT is just this pipeline:"],
    ["diagram", "tf_gpt", [], "The forward pass of a GPT. Shapes are for T tokens, width d and vocabulary size V."],
    ["steps", [
      "**Tokens → IDs** with the tokenizer (here: one ID per character).",
      "**Embed + position**: $x = E[\\text{ids}] + P[:T]$, shape $(T, d)$.",
      "**N transformer blocks**, each with causal self-attention and an MLP.",
      "**Final LayerNorm**, then multiply by $E^\\top$ (weight tying) to get **logits**, one score per vocabulary entry, for every position: shape $(T, V)$.",
      "**Softmax of the last row** gives the probabilities of the next token.",
    ]],
    ["p", "Below is the complete model in one runnable block with **random, untrained** weights. The numbers are meaningless as language, but every shape, mask and operation is exactly what a real GPT does. Training would adjust the weights so the probabilities become good ([next-token pretraining](llm.html#pretraining))."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)

# ---- tokenizer: one id per character -------------------------------------------------
text = "hello world"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
ids = np.array([stoi[c] for c in text])

# ---- parameters (random) -------------------------------------------------------------
d, h, L, T_max = 16, 2, 2, 32
w = lambda *shape: rng.normal(0, 0.1, size=shape)
E, P = w(V, d), w(T_max, d)                                   # token and position embeddings
blocks = [dict(Wq=w(d, d), Wk=w(d, d), Wv=w(d, d), Wo=w(d, d), W1=w(d, 4 * d), W2=w(4 * d, d)) for _ in range(L)]

# ---- building blocks -----------------------------------------------------------------
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def block(x, p):
    T = len(x)
    a = layer_norm(x)
    split = lambda M: M.reshape(T, h, d // h).transpose(1, 0, 2)
    Q, K, Vv = split(a @ p["Wq"]), split(a @ p["Wk"]), split(a @ p["Wv"])
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(d // h)
    s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)
    x = x + (softmax(s) @ Vv).transpose(1, 0, 2).reshape(T, d) @ p["Wo"]
    return x + gelu(layer_norm(x) @ p["W1"]) @ p["W2"]

def gpt(ids):
    x = E[ids] + P[:len(ids)]                                 # (T, d)
    for p in blocks:
        x = block(x, p)
    return layer_norm(x) @ E.T                                # (T, V): weight tying, logits for every position

# ---- run it --------------------------------------------------------------------------
logits = gpt(ids)
print("logits shape:", logits.shape)
probs = softmax(logits[-1])                                   # next-token distribution after "hello world"
print("vocab:", chars)
print("P(next):", probs.round(3), " sum =", probs.sum().round(3))

# the loss of an untrained model is about ln(V): it is guessing
p_all = softmax(logits[:-1])                                  # predict ids[1:] from positions 0..T-2
loss = -np.log(p_all[np.arange(len(ids) - 1), ids[1:]]).mean()
print("loss:", loss.round(3), " ln(V):", np.log(V).round(3))

# causal check: changing the last token cannot change earlier logits
ids2 = ids.copy(); ids2[-1] = stoi["h"]
print("earlier positions unchanged:", np.allclose(logits[:-1], gpt(ids2)[:-1]))

n_params = E.size + P.size + sum(a.size for p in blocks for a in p.values())
print("parameters:", n_params)`, "logits shape: (11, 8)\nvocab: [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']\nP(next): [0.12  0.179 0.063 0.213 0.059 0.1   0.133 0.134]  sum = 1.0\nloss: 2.187  ln(V): 2.079\nearlier positions unchanged: True\nparameters: 6784"],
    ["p", "What to notice:"],
    ["list", [
      "The loss of the untrained model is close to $\\ln V$, the loss of picking uniformly at random. A freshly initialised language model should start there; if it starts far above, something is wrong. See [loss functions for neural networks](dl.html#dl-losses).",
      "`logits[i]` uses only tokens $0..i$, thanks to the causal mask. During training **all** $T$ positions are scored at once (each predicts the following token), which is where the $T$ training examples per sentence come from.",
      "There is no recurrence and no loop over time: the whole sequence flows through as matrices.",
    ]],
    ["h", "Generating text"],
    ["p", "To generate, repeat: run the model on the tokens so far, take the distribution at the last position, choose a token, append it. Greedy decoding picks the largest probability; the other strategies are in [greedy and beam search](llm.html#greedy-beam) and [sampling](llm.html#sampling). Because the weights are random, the output is gibberish, but the loop is the real one:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
V, d = 8, 16
E = rng.normal(0, 1, size=(V, d))
W = rng.normal(0, 0.5, size=(d, d))

def fake_gpt(ids):                          # stand-in for a model: returns logits (T, V)
    x = E[ids].cumsum(axis=0) / np.arange(1, len(ids) + 1)[:, None]   # each position averages its past
    return np.tanh(x @ W) @ E.T

ids = [3]                                   # a one-token prompt
for _ in range(8):
    logits = fake_gpt(np.array(ids))
    ids.append(int(np.argmax(logits[-1])))  # greedy: take the most likely next token
print(ids)`, "[3, 2, 7, 3, 3, 3, 3, 3, 5]"],
    ["p", "A real model recomputes nothing it has already computed: the keys and values of past tokens are stored in the [KV cache](llm.html#kv-cache), so each new token costs one step instead of a full pass."],
    ["tip", "Writing a model like this once is the best way to understand the shape bookkeeping. When you later read any LLM codebase (nanoGPT, Hugging Face's `modeling_*.py`), you will recognise every line."],
    ["danger", "Forgetting to use only the **last** row of the logits when generating. Row $i$ is the prediction for position $i+1$; only the last row predicts the token you do not have yet."],
  ],
},
{
  id: "efficient-attention", group: "Architecture", title: "Efficient Attention: FlashAttention, MQA & GQA",
  sub: "Standard attention is slow and memory-hungry for long inputs. Three fixes: never write the big score matrix, and share keys and values between heads.",
  tags: ["efficiency", "attention"],
  blocks: [
    ["p", "Attention has two costs that hurt at long lengths $T$: the **score matrix** ($T \\times T$ entries, see the [quadratic cost](transformers.html#scaled-dot-product)) and, during generation, the **KV cache**: the keys and values of every past token, kept in memory for every layer. Modern models attack both."],
    ["h", "FlashAttention: same result, less memory traffic"],
    ["p", "Moving numbers in and out of the GPU's main memory (HBM) is often slower than the arithmetic itself. A naive implementation writes the full $T \\times T$ matrix to HBM, reads it back for the softmax, writes it again, and so on. **FlashAttention** (Dao et al., 2022) computes attention in small **tiles** that fit in fast on-chip memory (SRAM) and writes only the final output. It is **exact**, not an approximation; it is just a better schedule."],
    ["diagram", "tf_flash", [], "FlashAttention keeps the intermediate scores in fast on-chip memory and never stores the full matrix."],
    ["p", "The trick that makes tiling possible is the **online softmax**: you can compute a softmax over a long row piece by piece, if you keep a running maximum $m$ and a running sum $l$ and rescale what you have accumulated whenever the maximum changes:"],
    ["code", R`
import numpy as np

def attention_one_query(q, K, V):                  # the normal way: needs ALL scores at once
    s = K @ q / np.sqrt(len(q))
    e = np.exp(s - s.max())
    return (e / e.sum()) @ V

def online_attention(q, K, V, block=4):            # the flash way: one block of keys at a time
    m, l, o = -np.inf, 0.0, np.zeros(V.shape[1])   # running max, running sum, running output
    for s0 in range(0, len(K), block):
        s = K[s0:s0 + block] @ q / np.sqrt(len(q))
        m_new = max(m, s.max())
        scale = np.exp(m - m_new)                  # shrink what we accumulated under the old max
        p = np.exp(s - m_new)
        l = l * scale + p.sum()
        o = o * scale + p @ V[s0:s0 + block]
        m = m_new
    return o / l

rng = np.random.default_rng(0)
q, K, V = rng.normal(size=8), rng.normal(size=(20, 8)), rng.normal(size=(20, 8))
a, b = attention_one_query(q, K, V), online_attention(q, K, V)
print(np.allclose(a, b))
print(a.round(3))`, "True\n[ 0.129  0.135 -0.627  0.236 -0.117 -0.494 -0.565 -0.059]"],
    ["p", "The result is identical, yet the blockwise version never needs more than `block` scores in memory. Real FlashAttention does this for tiles of queries and keys inside one GPU kernel, so the memory needed grows linearly with $T$ instead of quadratically. In PyTorch you get it automatically through `F.scaled_dot_product_attention` on a supported GPU. Newer versions of the algorithm and of PyTorch keep improving speed on new hardware, but the idea is unchanged."],
    ["h", "The KV cache problem"],
    ["p", "When generating, each new token needs the keys and values of all earlier tokens. Recomputing them would be wasteful, so they are cached ([the KV cache](llm.html#kv-cache)). Its size is:"],
    ["math", R`\text{bytes} = 2 \times L \times H_{kv} \times d_{head} \times T \times \text{bytes per number} \times \text{batch}`, "2 = keys and values · L layers · H_kv key/value heads · T tokens"],
    ["code", R`
def kv_cache_gb(layers, kv_heads, d_head, T, bytes_per=2, batch=1):
    return 2 * layers * kv_heads * d_head * T * bytes_per * batch / 1e9

# an example 7B-class shape: 32 layers, 32 query heads of width 128, float16 cache
for name, kv in [("MHA  (32 kv heads)", 32), ("GQA  ( 8 kv heads)", 8), ("MQA  ( 1 kv head )", 1)]:
    print(f"{name}: {kv_cache_gb(32, kv, 128, T=8192):7.3f} GB for one 8192-token sequence")`, "MHA  (32 kv heads):   4.295 GB for one 8192-token sequence\nGQA  ( 8 kv heads):   1.074 GB for one 8192-token sequence\nMQA  ( 1 kv head ):   0.134 GB for one 8192-token sequence"],
    ["h", "MQA and GQA: share keys and values"],
    ["diagram", "tf_gqa", [], "Many query heads, fewer key/value heads. Fewer K/V heads means a smaller KV cache."],
    ["list", [
      "**Multi-head attention (MHA):** every query head has its own key and value head.",
      "**Multi-query attention (MQA):** all query heads share **one** key head and one value head. Tiny cache, but quality can drop a little.",
      "**Grouped-query attention (GQA):** query heads are split into groups, and each group shares one key/value head. This keeps most of MHA's quality at close to MQA's cost, and is the common choice in open LLMs as of 2026.",
    ]],
    ["p", "Implementing GQA means only one change to multi-head attention: there are fewer K and V heads, and each one is repeated to serve its group of query heads:"],
    ["code", R`
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def gqa(X, Wq, Wk, Wv, h, g):
    T, d = X.shape
    d_k = d // h
    Q = (X @ Wq).reshape(T, h, d_k).transpose(1, 0, 2)          # (h, T, d_k)   h query heads
    K = (X @ Wk).reshape(T, g, d_k).transpose(1, 0, 2)          # (g, T, d_k)   only g key heads
    V = (X @ Wv).reshape(T, g, d_k).transpose(1, 0, 2)
    K, V = np.repeat(K, h // g, axis=0), np.repeat(V, h // g, axis=0)   # each K/V head serves h/g query heads
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_k))
    return (A @ V).transpose(1, 0, 2).reshape(T, d)

rng = np.random.default_rng(0)
T, d, h = 6, 16, 8
X = rng.normal(size=(T, d))
for g in [8, 2, 1]:                                             # MHA, GQA, MQA
    Wq = rng.normal(size=(d, d)) * 0.3
    Wk, Wv = (rng.normal(size=(d, g * (d // h))) * 0.3 for _ in range(2))
    y = gqa(X, Wq, Wk, Wv, h, g)
    print(f"kv heads g={g}: output {y.shape}, K+V cache entries per token = {2 * g * (d // h)}")`, "kv heads g=8: output (6, 16), K+V cache entries per token = 32\nkv heads g=2: output (6, 16), K+V cache entries per token = 8\nkv heads g=1: output (6, 16), K+V cache entries per token = 4"],
    ["h", "Other ways to make attention cheaper"],
    ["table", ["Idea", "What it does", "Trade-off"], [
      ["**Sliding-window attention**", "Each token attends only to the last $w$ tokens (often mixed with a few full-attention layers).", "Cost $O(Tw)$, but distant information must hop through layers."],
      ["**Sparse / strided patterns**", "A fixed pattern of allowed pairs (local + a few global tokens).", "Cheaper, more engineering, may miss relations."],
      ["**Linear attention / state-space models**", "Replace the softmax with something that can be computed incrementally, with a constant-size state (Mamba-style models).", "Cost linear in $T$, but exact retrieval of old details is harder. Hybrids that mix a few attention layers with such layers are an active area as of 2026."],
      ["**Latent KV compression**", "Store a small learned compression of K and V per token (used in DeepSeek models as multi-head latent attention).", "Much smaller cache; more complex."],
      ["**KV-cache quantization**", "Store the cache in 8 or 4 bits.", "Memory saving with a small accuracy cost; see [quantization](llm.html#quantization)."],
    ]],
    ["p", "These are the engineering behind [long context](llm.html#long-context) and fast [serving](llm.html#serving)."],
    ["tip", "If you are tuning inference for memory, the KV cache usually dominates at long lengths and large batch sizes, not the weights. GQA and cache quantization attack exactly that."],
    ["danger", "Thinking FlashAttention approximates attention or changes the model. It computes exactly the same numbers (up to floating-point rounding); only the memory access pattern differs."],
  ],
},
{
  id: "moe", group: "Architecture", title: "Mixture of Experts",
  sub: "Replace the single MLP with many expert MLPs and a router that sends each token to only a few of them: far more parameters for the same compute per token.",
  tags: ["architecture", "scaling"],
  blocks: [
    ["p", "Most of a block's parameters are in the MLP ([the transformer block](transformers.html#transformer-block)). A **mixture-of-experts (MoE)** layer keeps the attention as it is, but replaces the one MLP with $E$ separate MLPs called **experts**, plus a small **router**. For each token the router scores the experts, and only the top $k$ (commonly 1 or 2, up to 8 in some recent designs) actually run. Their outputs are combined with the router's weights."],
    ["diagram", "tf_moe", [], "A token goes through a router, which picks the top-2 of 4 experts. The output is the weighted sum of just those two."],
    ["math", R`y = \sum_{e \in \text{top-}k} g_e(x)\, \text{Expert}_e(x) \qquad g = \text{softmax}(x W_r)\ \text{(renormalised over the chosen } k)`, "g: router weights · W_r: the router's matrix, shape (d, number of experts)"],
    ["p", "The point: **total parameters** (what the model can store) grows with the number of experts, but **active parameters** (what each token pays for in compute) stays about the same as a dense model with $k$ experts. A mixture-of-experts model with eight experts and top-2 routing has nearly eight times the MLP parameters at roughly twice the MLP compute of one expert."],
    ["h", "A router and experts in NumPy"],
    ["code", R`
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(0)
T, d, n_exp, k = 8, 8, 4, 2
X = rng.normal(size=(T, d))                                      # 8 token vectors
Wr = rng.normal(size=(d, n_exp)) * 0.5                           # the router
experts = [(rng.normal(size=(d, 16)) * 0.3, rng.normal(size=(16, d)) * 0.3) for _ in range(n_exp)]

probs = softmax(X @ Wr)                                          # (T, n_exp): router probabilities
top = np.argsort(-probs, axis=1)[:, :k]                          # the k best experts per token
print("experts chosen per token:")
print(top.T)

out = np.zeros_like(X)
for t in range(T):
    chosen = top[t]
    gate = probs[t, chosen] / probs[t, chosen].sum()             # renormalise over the chosen experts
    for e, g in zip(chosen, gate):
        W1, W2 = experts[e]
        out[t] += g * (np.maximum(X[t] @ W1, 0) @ W2)            # only k experts run for this token

load = np.bincount(top.ravel(), minlength=n_exp)
print("tokens per expert:", load, "(each token uses", k, "experts)")
dense = 2 * d * 16
print("parameters: all experts =", n_exp * dense, " active per token =", k * dense)`, "experts chosen per token:\n[[3 1 3 0 3 3 2 3]\n [1 0 1 2 2 2 0 1]]\ntokens per expert: [3 4 4 5] (each token uses 2 experts)\nparameters: all experts = 1024  active per token = 512"],
    ["h", "The load-balancing problem"],
    ["p", "Nothing forces the router to spread tokens evenly. Left alone, training tends to collapse onto a few favourite experts while the others get no data and never improve. The standard remedy is an **auxiliary loss** that is smallest when tokens are spread evenly:"],
    ["math", R`\mathcal{L}_{\text{balance}} = E \sum_{e=1}^{E} f_e \, P_e`, "f_e: fraction of tokens whose top choice is expert e · P_e: average router probability of expert e. Equals 1 when perfectly balanced."],
    ["code", R`
import numpy as np

def balance_loss(probs):
    n_exp = probs.shape[1]
    f = np.bincount(probs.argmax(axis=1), minlength=n_exp) / len(probs)   # share of tokens per expert
    P = probs.mean(axis=0)                                                # mean router probability
    return n_exp * (f * P).sum()

even = np.full((100, 4), 0.25)                                            # perfectly uniform router
skew = np.tile([0.85, 0.05, 0.05, 0.05], (100, 1))                        # everything goes to expert 0
print("balanced :", balance_loss(even).round(3))
print("collapsed:", balance_loss(skew).round(3))`, "balanced : 1.0\ncollapsed: 3.4"],
    ["p", "Some recent models use bias-based or other balancing schemes instead of a loss term, but the goal is the same. In practice there are also **capacity limits** (an expert can only take so many tokens per batch) and some designs keep a few always-on **shared experts** next to the routed ones."],
    ["h", "Trade-offs"],
    ["table", ["", "Dense model", "Mixture of experts"], [
      ["Compute per token", "All parameters", "Only the active ones (a fraction)"],
      ["Total parameters", "= active parameters", "Much larger than active"],
      ["Memory needed to serve", "Proportional to parameters", "All experts must be in memory, even those a given token skips"],
      ["Training", "Straightforward", "Router instability, load balancing, communication between GPUs"],
      ["Quality at a given compute budget", "Baseline", "Usually better"],
    ]],
    ["p", "Some well-known examples (as of 2026): Mixtral 8x7B uses 8 experts with top-2 routing, about 47B total parameters and about 13B active per token; DeepSeek-V3 has 671B total and about 37B active parameters. Several frontier models are widely believed to be MoE, but vendors often do not publish details."],
    ["warn", "\"7B active\" does not mean it runs on a 7B model's hardware. The memory footprint is that of the **total** parameter count, because any token might need any expert."],
    ["danger", "Judging an MoE model by its total parameter count when comparing with a dense model. Compare active parameters for speed and cost, total parameters for memory."],
  ],
},
{
  id: "vit-multimodal", group: "Architecture", title: "Vision Transformers & Multimodal Models",
  sub: "Cut an image into patches, treat each patch as a token, and the same transformer works on pictures; mix image and text tokens in one sequence and you get a multimodal model.",
  tags: ["vision", "multimodal"],
  blocks: [
    ["p", "Nothing in a transformer block is specific to words. It takes a sequence of vectors. If you can turn an image into a sequence of vectors, the same architecture applies. That is the idea of the **Vision Transformer (ViT)**, introduced in the 2020 paper *An Image Is Worth 16x16 Words*."],
    ["diagram", "tf_vit", [], "ViT cuts the image into patches, projects each one into a vector, adds positions and a [CLS] token, and runs a standard transformer. Multimodal LLMs put image tokens and text tokens into one sequence."],
    ["steps", [
      "**Cut** the image into non-overlapping square patches (16×16 pixels is typical). A 224×224 image gives 14×14 = 196 patches.",
      "**Flatten** each patch (16·16·3 = 768 numbers) and multiply by a learned matrix to get a $d$-dimensional token: the *patch embedding*.",
      "Prepend a learned **[CLS]** token and add **position embeddings** (otherwise the model would not know which patch is where).",
      "Run a standard **transformer encoder** (bidirectional attention). The [CLS] output goes to a classifier, or the patch outputs are used for dense tasks.",
    ]],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
img = rng.normal(size=(32, 32, 3))                       # a tiny 32x32 RGB "image"
P, d = 8, 64                                             # 8x8 patches, token width 64

# cut into patches: (32,32,3) -> (4,8,4,8,3) -> (4,4,8,8,3) -> (16, 192)
patches = img.reshape(32 // P, P, 32 // P, P, 3).transpose(0, 2, 1, 3, 4).reshape(-1, P * P * 3)
print("patches:", patches.shape)
print("patch 1 is the block right of patch 0:", np.array_equal(patches[1], img[0:8, 8:16].reshape(-1)))

W = rng.normal(size=(P * P * 3, d)) * 0.05               # the patch-embedding matrix
tokens = patches @ W                                     # (16, 64): 16 image tokens
cls = rng.normal(size=(1, d))
pos = rng.normal(size=(17, d)) * 0.02
x = np.vstack([cls, tokens]) + pos                       # (17, 64): ready for the transformer
print("sequence fed to the transformer:", x.shape)

print("ViT-B/16 on 224x224:", (224 // 16) ** 2, "patches + 1 [CLS] =", (224 // 16) ** 2 + 1, "tokens")`, "patches: (16, 192)\npatch 1 is the block right of patch 0: True\nsequence fed to the transformer: (17, 64)\nViT-B/16 on 224x224: 196 patches + 1 [CLS] = 197 tokens"],
    ["p", "Frameworks implement the cut-and-project step as one strided convolution, whose kernel size and stride equal the patch size. It is exactly equal to the reshape-and-matmul above:"],
    ["code", R`
import torch
import torch.nn.functional as F

torch.manual_seed(0)
conv = torch.nn.Conv2d(3, 64, kernel_size=8, stride=8)          # 3 colour channels -> 64-dim token per 8x8 patch
img = torch.randn(1, 3, 32, 32)

a = conv(img).flatten(2).transpose(1, 2)                         # (1, 16 patches, 64)
patches = F.unfold(img, kernel_size=8, stride=8).transpose(1, 2) # (1, 16, 192): the flattened patches
b = patches @ conv.weight.flatten(1).T + conv.bias               # the same as a linear layer on patches
print(a.shape, torch.allclose(a, b, atol=1e-5))`, "torch.Size([1, 16, 64]) True", "PyTorch", "local"],
    ["note", "ViT has less built-in knowledge about images than a CNN (no assumption of locality or translation invariance), so on small datasets CNNs can win, and ViTs shine with lots of data or pretraining. Most modern vision backbones are ViT-style or hybrids as of 2026. Compare with [CNNs](dl.html#cnn)."],
    ["h", "Connecting text and images: CLIP-style models"],
    ["p", "CLIP (2021) trains an image encoder and a text encoder together on hundreds of millions of image–caption pairs. Both output a vector, and the loss pushes the vector of an image **close to its own caption and far from every other caption in the batch**. This is a symmetric cross-entropy over a matrix of cosine similarities (the [contrastive](llm.html#embeddings-search) idea):"],
    ["code", R`
import numpy as np

def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

rng = np.random.default_rng(0)
B, d = 4, 16
img = rng.normal(size=(B, d))
txt = img + 1.0 * rng.normal(size=(B, d))                # captions that roughly match their images
img, txt = (a / np.linalg.norm(a, axis=1, keepdims=True) for a in (img, txt))

sim = img @ txt.T / 0.07                                 # cosine similarities / temperature
print((img @ txt.T).round(2))                            # the diagonal (true pairs) should be largest
labels = np.arange(B)
loss_i = -log_softmax(sim)[labels, labels].mean()        # image -> which caption?
loss_t = -log_softmax(sim.T)[labels, labels].mean()      # caption -> which image?
print("contrastive loss:", ((loss_i + loss_t) / 2).round(4))`, "[[ 0.6   0.26  0.17  0.02]\n [ 0.14  0.47 -0.29  0.2 ]\n [ 0.06  0.28  0.51 -0.04]\n [-0.29  0.1   0.02  0.79]]\ncontrastive loss: 0.0246"],
    ["p", "After training, a text query and an image can be compared directly, which allows zero-shot classification (\"a photo of a dog\" vs \"a photo of a cat\") and image search."],
    ["h", "Multimodal LLMs: images as tokens"],
    ["p", "Chat models that accept images typically follow a simple recipe, used with many variations: a vision encoder (a ViT) turns the image into patch vectors, a small **projector** maps them into the language model's embedding space, and they are placed in the token sequence next to the text tokens. The language model then attends to image tokens and text tokens alike, with its usual causal self-attention. See [multimodal LLMs](llm.html#multimodal-llms) for the full story."],
    ["code", R`
side, patch = 336, 14                                    # a 336-pixel image with 14-pixel patches
n_img = (side // patch) ** 2
text = 40                                                # tokens in the user's question
print("image tokens:", n_img, "| text tokens:", text, "| total sequence:", n_img + text)
print("a 2x larger image side needs", ((2 * side) // patch) ** 2, "tokens: cost grows with the SQUARE of the side")`, "image tokens: 576 | text tokens: 40 | total sequence: 616\na 2x larger image side needs 2304 tokens: cost grows with the SQUARE of the side"],
    ["tip", "Image tokens count against the context window and the bill just like text tokens, and attention cost grows quadratically with their number. That is why many systems resize large images, tile them, or merge neighbouring patch tokens before the language model."],
    ["list", [
      "**Audio:** speech models such as Whisper cut a spectrogram into frames and run an encoder–decoder transformer.",
      "**Video:** frames are patched like images (often with time as one more dimension), which makes sequences very long.",
      "**Diffusion transformers:** newer image-generation models use transformer blocks on latent patches in place of the older U-Net, with text conditioning by cross-attention or by joint attention (see [generative models](dl.html#generative))."
    ]],
    ["danger", "Forgetting the position embeddings in a ViT. Without them the patches are an unordered bag, and the model cannot tell where in the image anything is."],
  ],
},
/* @@more@@ */
);
