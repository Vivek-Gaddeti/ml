/* Statistics for ML: descriptive statistics, sampling, hypothesis tests, A/B testing, correlation, regression inference
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/stats.js */
STATS.push(
{
  id: "descriptive", group: "Basics", title: "Descriptive Statistics",
  sub: "A few numbers (centre, spread, shape) that summarise a whole column of data before you model it.",
  tags: ["basics"],
  blocks: [
    ["p", "Why does a machine-learning course need statistics? Because a model is only useful if you can tell **when a difference is real**. Is model B really better than model A, or did it just get lucky on this test set? Did the new button raise sign-ups, or is that normal day-to-day wobble? How accurate is your 92% accuracy, given you only measured it on 500 examples? Statistics gives honest answers to these questions, and stops you shipping noise."],
    ["p", "We start with the simplest tools: **descriptive statistics**, a handful of numbers that summarise a column of data. Imagine the salaries of 12 people in a small team. You cannot read 12 numbers and 'see' the picture, so you ask three questions: where is the **centre**, how **spread out** is it, and what is its **shape**?"],
    ["table", ["Question", "Statistic", "Formula / idea", "Robust to outliers?"], [
      ["Centre", "**Mean** $\\bar x$", "add everything up, divide by $n$", "No: one huge value drags it"],
      ["Centre", "**Median**", "the middle value after sorting", "Yes"],
      ["Spread", "**Variance** $s^2$, **std dev** $s$", "average squared distance from the mean", "No"],
      ["Spread", "**IQR**", "75th percentile minus 25th percentile", "Yes"],
      ["Shape", "**Skewness**", "positive = long tail to the right", "No"],
    ]],
    ["viz", "stats_descriptive"],
    ["h", "The formulas"],
    ["math", R`\bar x=\frac{1}{n}\sum_{i=1}^{n}x_i \qquad s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar x)^2 \qquad s=\sqrt{s^2}`],
    ["list", [
      "$\\bar x$ is the sample mean and $n$ is the number of data points.",
      "$s^2$ is the **sample variance**: the average squared distance from the mean. Squaring keeps negatives from cancelling positives and punishes far-away points.",
      "We divide by $n-1$, not $n$ (**Bessel's correction**). The mean was estimated from the same data, so the points sit slightly closer to it than to the true mean; dividing by $n-1$ corrects that bias.",
      "$s$ is the **standard deviation**, back in the original units. A percentile such as the median (50th) or the 25th and 75th are read off the sorted data.",
    ]],
    ["p", "**Worked example.** Data: 2, 4, 4, 4, 5, 5, 7, 9. The sum is 40 so $\\bar x=5$. The squared distances from 5 are 9, 1, 1, 1, 0, 0, 4, 16, which add to 32. Dividing by $n=8$ gives 4.0 (the population variance); dividing by $n-1=7$ gives $32/7\\approx4.571$ (the sample variance), so $s\\approx2.138$. The median is the average of the two middle values, $(4+5)/2=4.5$."],
    ["code", R`
import numpy as np

x = np.array([2, 4, 4, 4, 5, 5, 7, 9])
n = len(x)
mean = x.sum() / n
dev = x - mean                                  # distance of each point from the mean
var_pop = (dev ** 2).sum() / n                  # divide by n
var_samp = (dev ** 2).sum() / (n - 1)           # divide by n-1 (Bessel's correction)

print("mean", mean, " median", np.median(x))
print("population variance", var_pop, " sample variance", round(var_samp, 3), " sample std", round(var_samp ** 0.5, 3))
print("NumPy: var() uses ddof=0 ->", x.var(), "  var(ddof=1) ->", round(x.var(ddof=1), 3))

q1, q3 = np.percentile(x, [25, 75])
print("Q1", q1, " Q3", q3, " IQR", q3 - q1)`, "mean 5.0  median 4.5\npopulation variance 4.0  sample variance 4.571  sample std 2.138\nNumPy: var() uses ddof=0 -> 4.0   var(ddof=1) -> 4.571\nQ1 4.0  Q3 5.5  IQR 1.5"],
    ["warn", "NumPy's `np.var` and `np.std` divide by $n$ by default (`ddof=0`), while pandas' `.var()` / `.std()` divide by $n-1$. For a sample, almost always pass `ddof=1` to NumPy. This mismatch silently changes results on small data."],
    ["h", "Skewed data: use the median"],
    ["p", "Real data is rarely symmetric: incomes, response times, file sizes and prices have a **long right tail**. Then the mean is pulled toward the tail and the median is the better 'typical value'. The gap between mean and median is itself a quick skew detector."],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = rng.lognormal(mean=3.0, sigma=0.8, size=500)      # a skewed 'salary-like' column (k$)

print("mean", x.mean().round(2), " median", np.median(x).round(2), " std", x.std(ddof=1).round(2))
print("IQR", (np.percentile(x, 75) - np.percentile(x, 25)).round(2))
print("skewness", round(stats.skew(x), 2), "  (0 = symmetric, > 0 = right tail)")

x2 = np.append(x, 5000)                                # add one absurd outlier
print("after one outlier: mean", x2.mean().round(2), " median", np.median(x2).round(2))

import pandas as pd
print(pd.Series(x).describe().round(2).to_dict())`, "mean 27.05  median 19.29  std 24.72\nIQR 23.34\nskewness 2.95   (0 = symmetric, > 0 = right tail)\nafter one outlier: mean 36.98  median 19.3\n{'count': 500.0, 'mean': 27.05, 'std': 24.72, 'min': 0.89, '25%': 11.44, '50%': 19.29, '75%': 34.78, 'max': 233.42}"],
    ["tip", "Always **plot a histogram** before you trust a single summary number. Two datasets can share the same mean and standard deviation yet look completely different (see the Anscombe example in [Correlation](stats.html#correlation)). Link back to the maths of the mean as an expectation in [Expectation & Variance](math.html#expectation)."],
    ["note", "In ML, descriptive statistics are the first step of every project: check each feature's range, missing values and skew, then decide on [scaling and transforms](ml.html#feature-engineering). The same mean and standard deviation drive standardisation ($z=(x-\\bar x)/s$)."],
  ],
},
{
  id: "clt", group: "Basics", title: "Sampling Distributions & the Central Limit Theorem",
  sub: "Averages wobble from sample to sample in a predictable, bell-shaped way, and that is what lets us put error bars on things.",
  tags: ["sampling"],
  blocks: [
    ["p", "You never see the whole population. You see a **sample**: 500 test images, 2,000 website visitors, 40 patients. Everything you compute from it (an accuracy, an average, a conversion rate) is a **statistic**, and a different sample would give a slightly different value. The distribution of those values over many imaginary repeated samples is the **sampling distribution**. Its width tells you how much to trust the number you got."],
    ["p", "The remarkable fact is the **central limit theorem (CLT)**: if you average $n$ independent values, the sampling distribution of the average becomes close to a **normal (bell) curve**, almost whatever the shape of the original data. Watch it happen. The population below is very skewed (an exponential), yet the histogram of sample means turns into a bell as $n$ grows. Step through the samples and slide $n$:"],
    ["viz", "stats_clt"],
    ["h", "The key formula: the standard error"],
    ["math", R`\bar X \;\approx\; \mathcal N\!\Big(\mu,\ \frac{\sigma^2}{n}\Big) \qquad \text{SE}(\bar X)=\frac{\sigma}{\sqrt n}`],
    ["list", [
      "$\\mu$ and $\\sigma$ are the mean and standard deviation of the **population** (one value). $\\bar X$ is the mean of a random sample of size $n$.",
      "**Standard error** (SE) is the standard deviation of the sample mean. It shrinks like $1/\\sqrt n$: to halve your uncertainty you need **four times** as much data.",
      "In practice $\\sigma$ is unknown, so we plug in the sample standard deviation $s$ and use $\\text{SE}=s/\\sqrt n$.",
    ]],
    ["p", "**Worked example.** IQ scores have $\\sigma=15$. The mean of $n=36$ people has $\\text{SE}=15/\\sqrt{36}=2.5$. With $n=144$ the SE is $15/12=1.25$: four times the people, half the SE. By the 68-95-99.7 rule from [Common Distributions](math.html#distributions), about 95% of such sample means land within $\\pm2\\times2.5=5$ points of the true mean."],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
pop_mean, pop_sd = 1.0, 1.0                       # an exponential(1) population: skewed
for n in [1, 5, 30, 200]:
    means = rng.exponential(1.0, size=(10000, n)).mean(axis=1)       # 10,000 samples of size n
    print(f"n={n:3d}  mean of means {means.mean():.3f}   sd of means {means.std():.3f}   "
          f"theory {pop_sd / np.sqrt(n):.3f}   skewness {stats.skew(means):.2f}")

n = 30
means = rng.exponential(1.0, size=(10000, n)).mean(axis=1)
inside = np.mean(np.abs(means - pop_mean) < 1.96 * pop_sd / np.sqrt(n))
print("share within 1.96 SE of the truth:", round(inside, 3), "(normal would give 0.95)")`, "n=  1  mean of means 0.993   sd of means 0.998   theory 1.000   skewness 1.98\nn=  5  mean of means 1.002   sd of means 0.450   theory 0.447   skewness 0.91\nn= 30  mean of means 1.001   sd of means 0.185   theory 0.183   skewness 0.38\nn=200  mean of means 0.999   sd of means 0.071   theory 0.071   skewness 0.14\nshare within 1.96 SE of the truth: 0.951 (normal would give 0.95)"],
    ["p", "Notice the skewness of the sample means falls toward 0 as $n$ grows (it shrinks like $1/\\sqrt n$), and the spread matches $\\sigma/\\sqrt n$ closely. The CLT is an approximation that improves with $n$; for a very skewed population, $n=30$ is already decent, $n=5$ is not."],
    ["h", "The same idea for model accuracy"],
    ["p", "Each test example is correct or not: a Bernoulli variable with success probability $p$ (the true accuracy). The measured accuracy is a mean of $n$ such values, so its standard error is $\\sqrt{p(1-p)/n}$. With $p=0.9$ and $n=1000$ that is $\\sqrt{0.09/1000}\\approx0.0095$, about one percentage point. A model with 90.0% and another with 90.8% on 1,000 test examples are **not** clearly different; the gap is smaller than one standard error."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(1)
p_true, n = 0.90, 1000
acc = rng.binomial(n, p_true, size=20000) / n            # accuracy measured on 20,000 different test sets
print("measured accuracy ranges from", acc.min().round(3), "to", acc.max().round(3))
print("sd of measured accuracy", acc.std().round(4), "  formula", round((p_true * (1 - p_true) / n) ** 0.5, 4))`, "measured accuracy ranges from 0.863 to 0.936\nsd of measured accuracy 0.0095   formula 0.0095"],
    ["warn", "The CLT needs **independent** values with **finite variance**. With heavy tails (Cauchy-like data, rare gigantic purchases) the sample mean converges painfully slowly or not at all, and with strongly dependent data (time series, repeated measurements of the same user) the effective $n$ is much smaller than the row count. Check that rows are really independent before quoting $\\sigma/\\sqrt n$."],
    ["tip", "The CLT is why the normal distribution, and the tests built on it, appear everywhere. It does **not** say your data is normal; it says your **averages** are. Next: use the SE to build [confidence intervals](stats.html#confidence-intervals). Background on the bell curve: [Common Distributions](math.html#distributions)."],
  ],
},
{
  id: "confidence-intervals", group: "Basics", title: "Confidence Intervals",
  sub: "A range around your estimate that shows how precisely you know it.",
  tags: ["estimation"],
  blocks: [
    ["p", "A single number like 'accuracy is 91.2%' hides how sure you are. A **confidence interval** (CI) adds the honesty: 'accuracy is 91.2%, 95% CI from 89.4% to 93.0%'. The width is the uncertainty, and it comes straight from the standard error you met in [the CLT](stats.html#clt)."],
    ["p", "Here is the right way to read it. Imagine repeating the whole study 50 times, each with fresh data, and computing an interval each time. A 95% procedure gives intervals of which about 95% contain the true value. Press Play or Next to draw more intervals:"],
    ["viz", "stats_ci"],
    ["math", R`\bar x\ \pm\ t_{\,n-1,\,1-\alpha/2}\cdot\frac{s}{\sqrt n}`, "CI for a mean"],
    ["list", [
      "$\\bar x$ is the sample mean, $s$ the sample standard deviation and $s/\\sqrt n$ the standard error.",
      "$t_{n-1,\\,1-\\alpha/2}$ is the critical value of the **t distribution** with $n-1$ degrees of freedom (2.06 for $n=25$ and 95% confidence). It is slightly wider than the normal's 1.96 because $s$ is itself uncertain; for $n$ above 100 they are nearly the same.",
      "The part after $\\pm$ is the **margin of error**. More confidence (99%) means a bigger critical value and a wider interval; more data means a smaller SE and a narrower one.",
    ]],
    ["p", "**Worked example.** A new feature changes a model's latency. Over $n=25$ runs: $\\bar x=52$ ms, $s=10$ ms. Then $\\text{SE}=10/\\sqrt{25}=2$, $t_{24,0.975}=2.064$, margin $=2.064\\times2=4.13$, so the 95% CI is $52\\pm4.13=(47.87,\\,56.13)$ ms."],
    ["code", R`
import numpy as np
from scipy import stats

x = np.array([52.1, 45.3, 60.2, 49.8, 55.0, 38.7, 63.1, 51.4, 47.9, 58.6])   # latency in ms
n = len(x)
mean, s = x.mean(), x.std(ddof=1)
se = s / np.sqrt(n)
tcrit = stats.t.ppf(0.975, df=n - 1)                    # critical value for 95%
print("mean", mean.round(2), " SE", se.round(2), " t*", tcrit.round(3))
print("95% CI:", (mean - tcrit * se).round(2), "to", (mean + tcrit * se).round(2))

lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)    # the library one-liner
print("scipy :", round(lo, 2), "to", round(hi, 2))`, "mean 52.21  SE 2.32  t* 2.262\n95% CI: 46.95 to 57.47\nscipy : 46.95 to 57.47"],
    ["h", "Checking the coverage claim by simulation"],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
true_mean, n, trials = 10.0, 15, 5000
tcrit = stats.t.ppf(0.975, df=n - 1)
hits = 0
for _ in range(trials):
    x = rng.normal(true_mean, 3.0, n)
    half = tcrit * x.std(ddof=1) / np.sqrt(n)
    hits += (x.mean() - half <= true_mean <= x.mean() + half)
print("coverage of the 95% interval:", round(hits / trials, 3))`, "coverage of the 95% interval: 0.949"],
    ["h", "A CI for a proportion (accuracy, click rate)"],
    ["p", "For a share $\\hat p$ out of $n$ trials the SE is $\\sqrt{\\hat p(1-\\hat p)/n}$. For small samples or $\\hat p$ near 0 or 1 the simple normal formula misbehaves; the **Wilson interval** is the safer default."],
    ["code", R`
import numpy as np
from scipy import stats

correct, n = 91, 100                      # 91 of 100 test examples right
p = correct / n
z = stats.norm.ppf(0.975)

se = np.sqrt(p * (1 - p) / n)
print("normal  (Wald) CI:", round(p - z * se, 3), "to", round(p + z * se, 3))

centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)           # Wilson interval
half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
print("Wilson        CI:", round(centre - half, 3), "to", round(centre + half, 3))`, "normal  (Wald) CI: 0.854 to 0.966\nWilson        CI: 0.838 to 0.952"],
    ["danger", "A 95% CI does **not** mean 'there is a 95% chance the true value is in this interval'. The true value is fixed; the interval is the random thing. The 95% describes the long-run hit rate of the **method**. (A Bayesian [credible interval](stats.html#bayes-vs-freq) does make the direct probability statement.)"],
    ["tip", "Overlapping CIs for two models do **not** automatically mean 'no significant difference' (a test on the difference is more precise, especially for paired data). Conversely, if two 95% CIs do not overlap, the difference is significant. When there is no formula for your metric (AUC, F1, a median), use the [bootstrap](stats.html#permutation-bootstrap)."],
  ],
},
{
  id: "hypothesis-testing", group: "Hypothesis Testing", title: "Hypothesis Testing: p-values & Errors",
  sub: "A recipe for asking 'could this result be just chance?' and the two ways the answer can be wrong.",
  tags: ["testing"],
  blocks: [
    ["p", "A drug trial gives 50 patients a pill and 50 a placebo, and the pill group does better by 4 points. Is the pill working, or did the lucky patients end up in that group? A **hypothesis test** answers by setting up a skeptic's position and checking how embarrassing the data is for it."],
    ["steps", [
      "State the **null hypothesis** $H_0$: nothing is going on (no difference, no effect). The **alternative** $H_1$ is what you suspect.",
      "Choose a **significance level** $\\alpha$ (usually 0.05) before looking at the data. It is the false-alarm rate you will tolerate.",
      "Compute a **test statistic** that measures how far the data is from what $H_0$ predicts (a z, t, chi-square or F value).",
      "Find the **p-value**: the probability, assuming $H_0$ is true, of a result at least as extreme as yours.",
      "If $p<\\alpha$, reject $H_0$ ('statistically significant'). Otherwise you **fail to reject** it (which is not the same as proving it true).",
    ]],
    ["viz", "stats_pvalue"],
    ["h", "A worked z-test"],
    ["p", "A model's latency is known to average $\\mu_0=100$ ms with $\\sigma=15$. After a code change, $n=36$ runs average $\\bar x=105$. Is that a real slowdown? The test statistic is how many standard errors the mean is from the null value:"],
    ["math", R`z=\frac{\bar x-\mu_0}{\sigma/\sqrt n}=\frac{105-100}{15/\sqrt{36}}=\frac{5}{2.5}=2.0`],
    ["p", "For a two-sided test the p-value is $2\\times P(Z\\ge2)=2\\times0.0228=0.0455$. Since $0.0455<0.05$ we reject $H_0$: there is evidence the latency changed. (With $z=1.9$ instead, $p=0.057$ and we would not; 0.05 is a convention, not a cliff.)"],
    ["code", R`
import numpy as np
from scipy import stats

mu0, sigma, n, xbar = 100, 15, 36, 105
z = (xbar - mu0) / (sigma / np.sqrt(n))
p_two = 2 * stats.norm.sf(abs(z))              # sf = 1 - cdf: the upper tail area
p_one = stats.norm.sf(z)
print("z", round(z, 2), " two-sided p", round(p_two, 4), " one-sided p", round(p_one, 4))
print("critical value for alpha=0.05 (two-sided):", round(stats.norm.ppf(0.975), 3))`, "z 2.0  two-sided p 0.0455  one-sided p 0.0228\ncritical value for alpha=0.05 (two-sided): 1.96"],
    ["h", "Two ways to be wrong"],
    ["viz", "stats_power"],
    ["table", ["", "$H_0$ actually true (no effect)", "$H_0$ actually false (real effect)"], [
      ["**Test says: significant**", "**Type I error** (false positive), rate $\\alpha$", "Correct: **power** $=1-\\beta$"],
      ["**Test says: not significant**", "Correct (rate $1-\\alpha$)", "**Type II error** (false negative), rate $\\beta$"],
    ]],
    ["p", "**Power** is the chance of catching a real effect. It rises with a larger effect, a larger sample, lower noise and a larger $\\alpha$. A study with low power usually misses real effects, and the 'significant' ones it does find are exaggerated. Aim for about 80% power, which you plan **before** collecting data ([A/B Testing](stats.html#ab-testing))."],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
n, trials, alpha = 30, 4000, 0.05

def rejection_rate(true_diff):
    a = rng.normal(0, 1, (trials, n))
    b = rng.normal(true_diff, 1, (trials, n))
    return np.mean(stats.ttest_ind(a, b, axis=1).pvalue < alpha)

print("no real effect  -> rejects", round(rejection_rate(0.0), 3), "(this is the Type I error rate ~ alpha)")
print("effect d=0.5    -> rejects", round(rejection_rate(0.5), 3), "(this is the power)")
print("effect d=0.8    -> rejects", round(rejection_rate(0.8), 3))`, "no real effect  -> rejects 0.054 (this is the Type I error rate ~ alpha)\neffect d=0.5    -> rejects 0.485 (this is the power)\neffect d=0.8    -> rejects 0.858"],
    ["h", "What a p-value is not"],
    ["list", [
      "**Not** the probability that $H_0$ is true. It assumes $H_0$ is true and asks how surprising the data is. (To get $P(H_0\\mid\\text{data})$ you need [Bayes' theorem](math.html#bayes) and a prior; see [Bayesian vs Frequentist](stats.html#bayes-vs-freq).)",
      "**Not** the probability that the result will replicate, and not the size or importance of the effect.",
      "$p=0.06$ and $p=0.04$ are nearly identical evidence; treating 0.05 as a wall is arbitrary.",
      "A large $p$ is not proof of 'no effect', only that the data cannot rule it out (maybe the study had low power).",
    ]],
    ["h", "Statistical vs practical significance"],
    ["p", "With enough data, a tiny difference becomes 'highly significant'. Test two models on 5 million examples and a 0.02-point accuracy gain can give $p<0.0001$ while being worthless in production. Always report the **effect size** and a confidence interval next to the p-value, and ask whether the difference is big enough to matter."],
    ["h", "Peeking and p-hacking"],
    ["p", "The 5% false-alarm guarantee holds only if you decide the sample size in advance and test **once**. If you check an A/B test every day and stop the first time $p<0.05$ (**peeking** or optional stopping), you get many chances to be fooled. Likewise, trying many metrics, subgroups or model variants and reporting the one that worked (**p-hacking**) turns noise into 'discoveries'."],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
trials, per_look, looks = 3000, 20, 10            # A and B are IDENTICAL: any 'win' is a false alarm
a = rng.normal(0, 1, (trials, per_look * looks))
b = rng.normal(0, 1, (trials, per_look * looks))

fixed = stats.ttest_ind(a, b, axis=1).pvalue < 0.05          # one test at the end
peek = np.zeros(trials, dtype=bool)
for k in range(1, looks + 1):                                # test after every 20 new users, stop if significant
    m = k * per_look
    peek |= stats.ttest_ind(a[:, :m], b[:, :m], axis=1).pvalue < 0.05
print("false positive rate, test once :", fixed.mean().round(3))
print("false positive rate, peek 10x  :", peek.mean().round(3))`, "false positive rate, test once : 0.056\nfalse positive rate, peek 10x  : 0.195"],
    ["warn", "Peeking more than doubles the real false-positive rate. Fix the sample size beforehand, or use a method designed for continuous monitoring (sequential tests, alpha-spending, Bayesian decision rules). When you run many tests, correct for it: [Multiple Testing](stats.html#multiple-testing)."],
  ],
},
{
  id: "t-tests", group: "Hypothesis Testing", title: "t-Tests: One-Sample, Two-Sample & Paired",
  sub: "The workhorse tests for comparing means, from 'is this different from a target?' to 'is model A better than model B?'.",
  tags: ["testing", "means"],
  blocks: [
    ["p", "A z-test needs the true $\\sigma$, which you almost never know. The **t-test** replaces it with the sample standard deviation $s$ and uses the **t distribution**, which has fatter tails for small samples to account for that extra uncertainty. There are three versions; the right choice depends on how the data was collected."],
    ["table", ["Test", "Question", "Statistic", "Use when"], [
      ["**One-sample**", "Is the mean different from a target $\\mu_0$?", "$t=\\dfrac{\\bar x-\\mu_0}{s/\\sqrt n}$", "One group, a known benchmark"],
      ["**Two-sample (Welch)**", "Do two **independent** groups have different means?", "$t=\\dfrac{\\bar x_1-\\bar x_2}{\\sqrt{s_1^2/n_1+s_2^2/n_2}}$", "Different users/patients in each group"],
      ["**Paired**", "Is the mean of the **differences** zero?", "one-sample test on $d_i=a_i-b_i$", "Same unit measured twice (before/after, model A vs B on the same folds)"],
    ]],
    ["list", [
      "**Welch's** version does not assume the two groups have equal variance and is a safe default; its degrees of freedom come from the Welch-Satterthwaite formula. The classic 'pooled' Student t-test assumes equal variances.",
      "The **paired** test is more powerful whenever the pairs are correlated, because differencing cancels the variation between units.",
      "Degrees of freedom: $n-1$ for one-sample and paired; roughly $n_1+n_2-2$ (Student) or the Welch formula for two independent groups.",
    ]],
    ["p", "**Worked example (Welch).** Group A: $n_1=10$, $\\bar x_1=12.1$, $s_1=2.0$. Group B: $n_2=12$, $\\bar x_2=10.5$, $s_2=2.5$. The SE of the difference is $\\sqrt{4/10+6.25/12}=\\sqrt{0.921}=0.960$, so $t=1.6/0.960=1.667$ with about 19.98 degrees of freedom, giving a two-sided $p\\approx0.111$: not significant at 5%, even though A is 1.6 higher on average."],
    ["code", R`
import numpy as np
from scipy import stats

def welch_t(a, b):
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2          # variance of each mean
    t = (a.mean() - b.mean()) / np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))     # Welch-Satterthwaite
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p

rng = np.random.default_rng(0)
a = rng.normal(12, 2.0, 10)
b = rng.normal(10.5, 2.5, 12)

t, df, p = welch_t(a, b)
print("from scratch:", round(t, 3), round(df, 2), round(p, 4))
r = stats.ttest_ind(a, b, equal_var=False)                       # Welch (equal_var=False)
print("scipy Welch :", round(r.statistic, 3), round(r.df, 2), round(r.pvalue, 4))
print("scipy Student (pooled):", round(stats.ttest_ind(a, b).pvalue, 4))
print("one-sample vs mu0=10:", round(stats.ttest_1samp(a, 10).pvalue, 4))`, "from scratch: 2.724 18.86 0.0135\nscipy Welch : 2.724 18.86 0.0135\nscipy Student (pooled): 0.0165\none-sample vs mu0=10: 0.0017"],
    ["h", "Paired t-test: before and after"],
    ["p", "Ten users rate a page before and after a redesign. Each user is their own control, so work with the **differences**. The unpaired test would waste the pairing and lose power."],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
user_level = rng.normal(6, 1.5, 10)                       # some users are generous raters, some harsh
before = user_level + rng.normal(0, 0.4, 10)
after = user_level + 0.5 + rng.normal(0, 0.4, 10)         # the redesign adds about 0.5

d = after - before
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print("paired by hand :", round(t, 3), " p", round(2 * stats.t.sf(abs(t), len(d) - 1), 4))
print("ttest_rel      :", [round(float(v), 4) for v in stats.ttest_rel(after, before)])
print("WRONG unpaired :", [round(float(v), 4) for v in stats.ttest_ind(after, before)])`, "paired by hand : 2.488  p 0.0345\nttest_rel      : [2.4882, 0.0345]\nWRONG unpaired : [1.0772, 0.2956]"],
    ["h", "Effect size: Cohen's d"],
    ["math", R`d=\frac{\bar x_1-\bar x_2}{s_p},\qquad s_p=\sqrt{\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}}`],
    ["p", "$d$ says how many standard deviations apart the groups are, independent of sample size. A rough guide: 0.2 small, 0.5 medium, 0.8 large. Report it next to the p-value."],
    ["h", "Comparing two ML models properly"],
    ["p", "You cross-validate model A and model B on the same 10 folds and get 10 pairs of scores. They are **paired** (same folds), so a paired t-test on the 10 differences is the natural tool. But beware: the folds share training data, so the scores are **not independent** and the plain paired t-test is too optimistic. The corrected resampled t-test of Nadeau and Bengio inflates the variance by a factor $(1/k+n_{test}/n_{train})$."],
    ["code", R`
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)       # same 10 splits for both models
a = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y, cv=cv)
b = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=0), X, y, cv=cv)
print("logistic :", a.mean().round(4), "  forest:", b.mean().round(4))

d = a - b
k = len(d)
naive = stats.ttest_rel(a, b).pvalue
n_test, n_train = len(y) / 5, len(y) * 4 / 5
t_corr = d.mean() / np.sqrt(d.var(ddof=1) * (1 / k + n_test / n_train))      # Nadeau-Bengio correction
print("naive paired p :", round(naive, 4))
print("corrected p    :", round(2 * stats.t.sf(abs(t_corr), k - 1), 4))`, "logistic : 0.9754   forest: 0.9622\nnaive paired p : 0.0342\ncorrected p    : 0.2152"],
    ["table", ["Assumption", "How to check", "If it fails"], [
      ["Independent observations", "think about how the data was collected", "Pair the data, or use a mixed model; do not just ignore it"],
      ["Roughly normal **means** (not necessarily the data)", "histogram / Q-Q plot; fine for $n\\gtrsim30$ by the [CLT](stats.html#clt)", "Small $n$ and skewed: [Mann-Whitney / Wilcoxon](stats.html#nonparametric) or a [permutation test](stats.html#permutation-bootstrap)"],
      ["Equal variances (Student only)", "compare $s_1$ and $s_2$", "Use Welch (`equal_var=False`)"],
      ["No extreme outliers", "box plot", "Investigate them; use a rank-based or bootstrap approach"],
    ]],
    ["tip", "Related: [Confidence Intervals](stats.html#confidence-intervals) (the CI of a difference carries the same information as the test), and for **more than two groups** use [ANOVA](stats.html#anova) instead of many t-tests. For comparing two classifiers on one test set, see McNemar's test in [Chi-Square Tests](stats.html#chi-square). Fold structure: [Cross-Validation](ml.html#cross-validation)."],
  ],
},
);
STATS.push(
{
  id: "chi-square", group: "Hypothesis Testing", title: "Chi-Square Tests",
  sub: "Tests for counts: is the pattern in a table of categories more than chance, and do two classifiers really disagree?",
  tags: ["testing", "counts"],
  blocks: [
    ["p", "t-tests compare averages of numbers. When your data is **counts in categories** (clicked / did not click, spam / not spam, which of three layouts a user chose) you compare the counts you saw with the counts you would **expect if nothing was going on**. That is the idea of the **chi-square ($\\chi^2$) test**."],
    ["math", R`\chi^2=\sum_{\text{cells}}\frac{(O-E)^2}{E}`],
    ["list", [
      "$O$ is the observed count in a cell and $E$ is the count expected under $H_0$. A cell far from its expected value adds a lot; dividing by $E$ makes a miss of 15 matter more when you only expected 20 than when you expected 2000.",
      "For a **test of independence** on a table, $E_{ij}=\\dfrac{(\\text{row total})(\\text{column total})}{\\text{grand total}}$. The degrees of freedom are $(r-1)(c-1)$.",
      "For a **goodness-of-fit** test (do die rolls look fair?), $E$ comes from the claimed probabilities and $\\text{df}=k-1$.",
    ]],
    ["p", "**Worked example (A/B test on a button).** Variant A: 120 clicks out of 1,000 visitors. Variant B: 150 out of 1,000. Overall 270 of 2,000 clicked (13.5%). If button colour did nothing, each variant would expect $1000\\times0.135=135$ clicks and 865 non-clicks. Then $\\chi^2=\\dfrac{(120-135)^2}{135}+\\dfrac{(150-135)^2}{135}+\\dfrac{(880-865)^2}{865}+\\dfrac{(850-865)^2}{865}=1.667+1.667+0.260+0.260=3.85$, with 1 degree of freedom, giving $p\\approx0.0496$."],
    ["code", R`
import numpy as np
from scipy import stats

observed = np.array([[120, 880],      # variant A: clicked, did not click
                     [150, 850]])     # variant B
row = observed.sum(axis=1, keepdims=True)
col = observed.sum(axis=0, keepdims=True)
expected = row * col / observed.sum()                   # what we'd see if A/B made no difference
chi2 = ((observed - expected) ** 2 / expected).sum()
df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
print("expected:", expected.tolist())
print("chi2", round(chi2, 3), " df", df, " p", round(stats.chi2.sf(chi2, df), 4))

res = stats.chi2_contingency(observed, correction=False)          # same thing in one call
print("scipy      :", round(res.statistic, 3), round(res.pvalue, 4), " df", res.dof)
res = stats.chi2_contingency(observed)                            # default: Yates continuity correction for 2x2
print("with Yates :", round(res.statistic, 3), round(res.pvalue, 4))`, "expected: [[135.0, 865.0], [135.0, 865.0]]\nchi2 3.854  df 1  p 0.0496\nscipy      : 3.854 0.0496  df 1\nwith Yates : 3.601 0.0577"],
    ["h", "Goodness of fit: is the die fair?"],
    ["code", R`
import numpy as np
from scipy import stats

rolls = np.array([5, 8, 9, 8, 10, 20])           # counts of faces 1..6 in 60 rolls
expected = np.full(6, rolls.sum() / 6)           # a fair die: 10 of each
chi2 = ((rolls - expected) ** 2 / expected).sum()
print("by hand:", round(chi2, 2), " p", round(stats.chi2.sf(chi2, 5), 4))
r = stats.chisquare(rolls, expected)
print("scipy  :", round(r.statistic, 2), round(r.pvalue, 4))`, "by hand: 13.4  p 0.0199\nscipy  : 13.4 0.0199"],
    ["h", "Sample ratio mismatch: a free health check for A/B tests"],
    ["p", "You intended a 50/50 split but got 5,300 users in A and 4,700 in B. Is that just chance? A goodness-of-fit test tells you. If $p$ is tiny (a common alarm level is 0.001) your assignment is broken (bots, a bug, a redirect that drops users), and the experiment's results cannot be trusted."],
    ["code", R`
from scipy import stats

print("5300 vs 4700:", stats.chisquare([5300, 4700]).pvalue, "(expected 50/50)  -> broken assignment")
print("5040 vs 4960:", round(stats.chisquare([5040, 4960]).pvalue, 4), "-> fine, plain chance")`, "5300 vs 4700: 1.9731752900753933e-09 (expected 50/50)  -> broken assignment\n5040 vs 4960: 0.4237 -> fine, plain chance"],
    ["h", "McNemar's test: comparing two classifiers on the same test set"],
    ["p", "Two models are scored on the same 1,000 test examples. Most of the time they agree, and those cases say nothing about which is better. Only the **disagreements** matter: $b$ = A right and B wrong, $c$ = A wrong and B right. Under $H_0$ (equal accuracy) a disagreement is equally likely to go either way, so $b$ should be about $(b+c)/2$:"],
    ["math", R`\chi^2=\frac{(|b-c|-1)^2}{b+c}\quad(\text{df}=1)`, "McNemar's statistic, with continuity correction"],
    ["code", R`
from scipy import stats

b, c = 60, 40            # A right & B wrong  vs  A wrong & B right (the other 900 examples: both agree)
chi2 = (abs(b - c) - 1) ** 2 / (b + c)
print("McNemar chi2", round(chi2, 2), " p", round(stats.chi2.sf(chi2, 1), 4))
print("exact binomial version p:", round(stats.binomtest(b, b + c, 0.5).pvalue, 4))
print("accuracy gap A - B:", (b - c) / 1000)`, "McNemar chi2 3.61  p 0.0574\nexact binomial version p: 0.0569\naccuracy gap A - B: 0.02"],
    ["table", ["Test", "Data", "Question"], [
      ["$\\chi^2$ independence", "r x c table of counts", "Are the two categorical variables related?"],
      ["$\\chi^2$ goodness of fit", "counts across k categories", "Do they match a claimed distribution?"],
      ["Fisher's exact (`stats.fisher_exact`)", "2 x 2 table with **small counts**", "Same as independence, exact p-value"],
      ["McNemar", "paired yes/no (same examples, two models)", "Is one model right more often?"],
    ]],
    ["warn", "The chi-square approximation needs **expected** counts of at least about 5 in every cell; with fewer, use Fisher's exact test. Observations must be independent (one row per user, not per click). And a significant chi-square only says 'not independent', not how strongly: report the rates themselves (13.5% vs 15.0%) and a CI for the difference. Cross-links: [Confusion Matrix, Precision, Recall & F1](ml.html#classification-metrics), [A/B Testing](stats.html#ab-testing)."],
  ],
},
{
  id: "anova", group: "Hypothesis Testing", title: "ANOVA",
  sub: "One test for 'are three or more group means all equal?', instead of a pile of t-tests.",
  tags: ["testing", "groups"],
  blocks: [
    ["p", "You compare three recommender models, or four page layouts, or five learning-rate settings. Running a t-test on every pair multiplies the chance of a false alarm: with 3 groups there are 3 pairs, and the chance that at least one is 'significant' by luck is already about $1-0.95^3\\approx14\\%$. **ANOVA** (analysis of variance) asks one question first: **could all the group means be equal?**"],
    ["p", "The idea: compare the variation **between** group means with the variation **within** groups. If the groups are all really the same, the between-group variation is just noise and is similar in size to the within-group variation. If a group truly differs, the between-group part becomes much larger."],
    ["math", R`F=\frac{MS_{\text{between}}}{MS_{\text{within}}}=\frac{SS_B/(k-1)}{SS_W/(N-k)}`],
    ["list", [
      "$k$ groups, $N$ observations in total, $\\bar x_j$ the mean of group $j$ (size $n_j$) and $\\bar x$ the grand mean.",
      "$SS_B=\\sum_j n_j(\\bar x_j-\\bar x)^2$: how far the group means are from the grand mean.",
      "$SS_W=\\sum_j\\sum_i (x_{ij}-\\bar x_j)^2$: the scatter of points inside their own group.",
      "Dividing by degrees of freedom ($k-1$ and $N-k$) turns the sums into averages ('mean squares'). Under $H_0$, $F$ follows the **F distribution**. A big $F$ means the means differ more than noise explains.",
    ]],
    ["p", "**Worked example.** Three models, four validation scores each: A = 4, 5, 6, 5 (mean 5); B = 7, 8, 6, 7 (mean 7); C = 9, 8, 10, 9 (mean 9). The grand mean is 7. $SS_B=4(4+0+4)=32$. Within each group the squared deviations add to 2, so $SS_W=6$. Then $MS_B=32/2=16$, $MS_W=6/9=0.667$ and $F=24$ with (2, 9) degrees of freedom: $p\\approx0.00025$."],
    ["code", R`
import numpy as np
from scipy import stats

groups = [np.array([4, 5, 6, 5]), np.array([7, 8, 6, 7]), np.array([9, 8, 10, 9])]
allx = np.concatenate(groups)
N, k = len(allx), len(groups)
grand = allx.mean()

ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
F = (ss_between / (k - 1)) / (ss_within / (N - k))
print("SS_between", ss_between, " SS_within", ss_within, " F", F)
print("p-value from the F distribution:", round(stats.f.sf(F, k - 1, N - k), 5))

res = stats.f_oneway(*groups)
print("scipy f_oneway:", float(res.statistic), round(float(res.pvalue), 5))`, "SS_between 32.0  SS_within 6.0  F 24.0\np-value from the F distribution: 0.00025\nscipy f_oneway: 24.0 0.00025"],
    ["h", "ANOVA says 'something differs', not what"],
    ["p", "A significant $F$ only tells you the means are not all equal. To find **which** pairs differ, run a **post-hoc** test that controls the overall error, such as Tukey's HSD (or pairwise t-tests with a [Bonferroni or Holm correction](stats.html#multiple-testing))."],
    ["code", R`
import numpy as np
from scipy import stats

a, b, c = np.array([4, 5, 6, 5]), np.array([7, 8, 6, 7]), np.array([9, 8, 10, 9])
res = stats.tukey_hsd(a, b, c)
print("pairwise p-values (rows/cols = A, B, C):")
print(np.round(res.pvalue, 4).tolist())`, "pairwise p-values (rows/cols = A, B, C):\n[[1.0, 0.0176, 0.0002], [0.0176, 1.0, 0.0176], [0.0002, 0.0176, 1.0]]"],
    ["h", "Why ANOVA is a regression in disguise"],
    ["p", "A one-way ANOVA is the same as a [linear regression](ml.html#linear-regression) on group indicator (dummy) variables, and its $F$ test is the 'is the whole model better than just the grand mean?' test from [Regression Inference](stats.html#regression-inference). That is also why $R^2=SS_B/(SS_B+SS_W)$, the share of variance explained by the grouping, is a natural effect size ($\\eta^2$). Here it is $32/38=0.84$."],
    ["table", ["Assumption", "Check", "If it fails"], [
      ["Independent observations", "design of the study", "Repeated measures ANOVA / mixed model (e.g. several scores per user)"],
      ["Roughly normal residuals", "Q-Q plot; fine for moderate $n$", "[Kruskal-Wallis](stats.html#nonparametric)"],
      ["Similar variance in each group", "compare group SDs; Levene's test (`stats.levene`)", "Welch's ANOVA, or Kruskal-Wallis"],
    ]],
    ["note", "Comparing several ML models across the same CV folds? The folds are shared, so the scores are paired and a plain one-way ANOVA is not right: use a repeated-measures design or the Friedman test (`stats.friedmanchisquare`), then a corrected post-hoc test. See [Cross-Validation](ml.html#cross-validation)."],
  ],
},
{
  id: "nonparametric", group: "Hypothesis Testing", title: "Non-Parametric Tests",
  sub: "Tests that work on ranks instead of raw values, for skewed data, small samples and outliers.",
  tags: ["testing", "ranks"],
  blocks: [
    ["p", "t-tests and ANOVA assume the sample **means** are roughly normal. With a small sample of skewed data (response times, revenue per user), or with ordinal ratings (1 to 5 stars), that can fail. **Non-parametric** tests make fewer assumptions about the shape of the data. Most of them replace each value by its **rank** (1 for the smallest, 2 for the next, and so on), so a huge outlier is just 'the biggest' and cannot dominate."],
    ["p", "**Mann-Whitney U** (a.k.a. Wilcoxon rank-sum) compares two independent groups. Pool all values, rank them, and add up the ranks of group 1: if group 1 tends to be larger, its rank sum is big. $U_1=R_1-\\dfrac{n_1(n_1+1)}{2}$ counts, over all pairs (one from each group), how often a group-1 value beats a group-2 value. Under $H_0$, $U$ has mean $n_1n_2/2$ and variance $n_1n_2(n_1+n_2+1)/12$, which gives a z-score for large samples."],
    ["code", R`
import numpy as np
from scipy import stats

a = np.array([1.2, 3.4, 2.2, 8.9, 2.8, 3.1, 4.4])        # e.g. page load times (seconds)
b = np.array([5.1, 6.3, 3.9, 12.5, 7.4, 5.9, 6.8, 9.9])
n1, n2 = len(a), len(b)

ranks = stats.rankdata(np.concatenate([a, b]))             # rank all 15 values together
R1 = ranks[:n1].sum()
U1 = R1 - n1 * (n1 + 1) / 2
U = max(U1, n1 * n2 - U1)
mu, sd = n1 * n2 / 2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
z = (U - mu - 0.5) / sd                                     # 0.5 = continuity correction
print("rank sum of A", R1, " U1", U1, " z", round(z, 3), " p", round(2 * stats.norm.sf(z), 4))

r = stats.mannwhitneyu(a, b, method="asymptotic")
print("scipy (asymptotic):", r.statistic, round(r.pvalue, 4))
print("scipy exact p     :", round(stats.mannwhitneyu(a, b).pvalue, 4))
print("Welch t-test p    :", round(stats.ttest_ind(a, b, equal_var=False).pvalue, 4))`, "rank sum of A 35.0  U1 7.0  z 2.372  p 0.0177\nscipy (asymptotic): 7.0 0.0177\nscipy exact p     : 0.014\nWelch t-test p    : 0.0226"],
    ["h", "The family"],
    ["table", ["Situation", "Parametric test", "Non-parametric alternative", "SciPy"], [
      ["Two independent groups", "Welch t-test", "**Mann-Whitney U**", "`mannwhitneyu`"],
      ["Two paired measurements", "paired t-test", "**Wilcoxon signed-rank** (ranks of the differences)", "`wilcoxon`"],
      ["3+ independent groups", "one-way ANOVA", "**Kruskal-Wallis** H", "`kruskal`"],
      ["3+ paired groups (same folds)", "repeated-measures ANOVA", "**Friedman** test", "`friedmanchisquare`"],
      ["Relationship of two variables", "Pearson correlation", "**Spearman** rank correlation", "`spearmanr`"],
      ["Proportion of paired successes", "-", "**Sign test**, McNemar", "`binomtest`, see [Chi-Square](stats.html#chi-square)"],
    ]],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
before = rng.lognormal(1.0, 0.7, 12)                         # skewed 'seconds to finish a task'
after = before * rng.lognormal(-0.25, 0.3, 12)               # most users get faster

print("Wilcoxon signed-rank p :", round(stats.wilcoxon(before, after).pvalue, 4))
print("paired t-test p        :", round(stats.ttest_rel(before, after).pvalue, 4))

g1, g2, g3 = rng.lognormal(1.0, 0.6, 10), rng.lognormal(1.3, 0.6, 10), rng.lognormal(1.8, 0.6, 10)
print("Kruskal-Wallis p       :", round(stats.kruskal(g1, g2, g3).pvalue, 4))
print("one-way ANOVA p        :", round(stats.f_oneway(g1, g2, g3).pvalue, 4))`, "Wilcoxon signed-rank p : 0.0024\npaired t-test p        : 0.0033\nKruskal-Wallis p       : 0.0004\none-way ANOVA p        : 0.0003"],
    ["h", "Which test when?"],
    ["table", ["Your data", "First choice", "If assumptions break"], [
      ["One mean vs a target", "one-sample t-test", "Wilcoxon signed-rank, bootstrap CI"],
      ["Two groups, numeric outcome", "Welch t-test", "Mann-Whitney, [permutation test](stats.html#permutation-bootstrap)"],
      ["Same units measured twice", "paired t-test", "Wilcoxon signed-rank"],
      ["3+ groups, numeric outcome", "ANOVA + Tukey", "Kruskal-Wallis + Dunn / Holm-corrected Mann-Whitney"],
      ["Counts in categories / two proportions", "chi-square / z-test for proportions", "Fisher's exact (small counts)"],
      ["Two models, same test set (right/wrong)", "McNemar", "exact binomial on the disagreements"],
      ["Two numeric variables related?", "Pearson r", "Spearman rho"],
      ["Something with no formula (AUC, median, F1)", "bootstrap CI / permutation test", "-"],
    ]],
    ["warn", "Non-parametric does not mean 'assumption-free'. Mann-Whitney still needs independent observations, and strictly it compares the whole distributions (it is sensitive to differences in spread too), not just medians. It also has slightly less power than the t-test when the data really is normal, so do not use it by reflex: it shines with skew, outliers, ordinal data and tiny samples."],
    ["tip", "With big samples ($n$ in the hundreds or more) the [CLT](stats.html#clt) makes the t-test robust even for skewed data, so the choice matters little. With tiny samples, ranks and permutation tests are your friends."],
  ],
},
{
  id: "permutation-bootstrap", group: "Hypothesis Testing", title: "Permutation Tests & the Bootstrap",
  sub: "Let the computer simulate chance: shuffle labels for a p-value, resample your data for an error bar.",
  tags: ["resampling"],
  blocks: [
    ["p", "Classic tests rely on formulas that need assumptions (normality, equal variance). **Resampling** methods replace the formulas with brute force and work for almost any statistic: a median, a ratio, an AUC, the difference in F1 between two models. There are two key tools, with two different jobs."],
    ["table", ["", "Permutation test", "Bootstrap"], [
      ["Answers", "'Could this difference be chance?' (a **p-value**)", "'How uncertain is my estimate?' (a **standard error / CI**)"],
      ["Trick", "**Shuffle** the group labels (no real difference means labels are interchangeable)", "**Resample with replacement** from your data (pretend the sample is the population)"],
      ["Builds", "the null distribution", "the sampling distribution"],
    ]],
    ["h", "Permutation test"],
    ["p", "Take two groups of 8 values. If group membership did nothing, every way of splitting the 16 numbers into 8 + 8 is equally likely. So shuffle the labels many times, recompute the difference in means each time, and see how often chance alone gives a difference as big as the real one. That fraction is the p-value."],
    ["viz", "stats_permutation"],
    ["code", R`
import numpy as np
from scipy import stats

A = np.array([14.1, 15.3, 12.8, 16.2, 13.9, 15.8, 14.7, 16.5])
B = np.array([12.9, 13.4, 11.8, 14.0, 12.2, 13.1, 12.7, 14.4])
obs = A.mean() - B.mean()

rng = np.random.default_rng(0)
pooled = np.concatenate([A, B])
n_perm = 10000
shuffled = rng.permuted(np.tile(pooled, (n_perm, 1)), axis=1)         # 10,000 random re-orderings of the 16 numbers
diffs = shuffled[:, :8].mean(axis=1) - shuffled[:, 8:].mean(axis=1)    # first 8 = 'group A', rest = 'group B'

p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)       # +1: the real labelling counts as one shuffle
print("observed difference", round(obs, 3), "  permutation p", round(p, 4))
print("Welch t-test p      ", round(stats.ttest_ind(A, B, equal_var=False).pvalue, 4))

res = stats.permutation_test((A, B), lambda x, y: x.mean() - y.mean(), n_resamples=9999, random_state=0)
print("scipy permutation_test p", round(res.pvalue, 4))`, "observed difference 1.85   permutation p 0.0055\nWelch t-test p       0.0051\nscipy permutation_test p 0.0072"],
    ["h", "Bootstrap"],
    ["p", "You have one sample and want the uncertainty of its median (no textbook formula). **Resample with replacement**: draw $n$ values from your own data (some repeat, some are left out), compute the statistic, and repeat thousands of times. The spread of those values estimates the standard error; their 2.5th and 97.5th percentiles give a 95% **percentile confidence interval**."],
    ["viz", "stats_bootstrap"],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = rng.lognormal(1.0, 0.8, 40)                          # a skewed sample of 40 (e.g. session length, minutes)

B = 10000
idx = rng.integers(0, len(x), size=(B, len(x)))          # B resamples of 40 indices, with replacement
boot_median = np.median(x[idx], axis=1)

print("sample median", round(np.median(x), 3))
print("bootstrap SE ", round(boot_median.std(ddof=1), 3))
print("95% percentile CI:", np.percentile(boot_median, [2.5, 97.5]).round(3).tolist())

res = stats.bootstrap((x,), np.median, confidence_level=0.95, n_resamples=9999, method="percentile", random_state=0)
print("scipy bootstrap  :", np.round([res.confidence_interval.low, res.confidence_interval.high], 3).tolist())`, "sample median 2.452\nbootstrap SE  0.314\n95% percentile CI: [1.998, 3.235]\nscipy bootstrap  : [1.998, 3.235]"],
    ["h", "Bootstrapping a model's test metric"],
    ["p", "A test AUC measured on 300 examples is a single noisy number. Resample the **test examples** (keeping each true label with its prediction) and recompute the metric to get an interval."],
    ["code", R`
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=800, n_features=10, n_informative=4, flip_y=0.1, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=300, random_state=0, stratify=y)
score = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
print("test AUC", round(roc_auc_score(yte, score), 3))

rng = np.random.default_rng(0)
aucs = []
for _ in range(1000):
    i = rng.integers(0, len(yte), len(yte))                  # resample test examples with replacement
    if yte[i].min() == yte[i].max():
        continue                                             # need both classes to compute AUC
    aucs.append(roc_auc_score(yte[i], score[i]))
print("bootstrap 95% CI:", np.percentile(aucs, [2.5, 97.5]).round(3).tolist())`, "test AUC 0.736\nbootstrap 95% CI: [0.682, 0.795]"],
    ["warn", "Resampling is not magic. The bootstrap needs **independent** rows (if one user has 50 rows, resample users, not rows), works poorly for extremes such as the maximum, and can be unreliable for very small samples. It cannot invent information that is not in your sample. For a permutation test, the labels must be exchangeable under $H_0$ (shuffle within blocks if you have paired or grouped data)."],
    ["tip", "Rule of thumb: 1,000 resamples for a quick look, 10,000 for a reported result. Compare with the formula-based intervals in [Confidence Intervals](stats.html#confidence-intervals); related idea: bagging in [Random Forests](ml.html#random-forests) is a bootstrap of the training set."],
  ],
},
{
  id: "multiple-testing", group: "Hypothesis Testing", title: "Multiple Testing: Bonferroni & FDR",
  sub: "Test enough things and something will look significant by luck; these corrections keep you honest.",
  tags: ["testing"],
  blocks: [
    ["p", "A 5% false-alarm rate per test sounds safe, until you run 20 tests. If every null hypothesis is true, the chance that **at least one** looks significant is $1-0.95^{20}\\approx64\\%$. Test 20 metrics on an A/B test, 100 customer segments, or 10,000 genes, and 'discoveries' appear from pure noise. This is the **multiple comparisons problem**, and it is the engine behind p-hacking and many irreproducible results."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
m, trials = 20, 5000
p = rng.uniform(size=(trials, m))                           # p-values when ALL 20 nulls are true are uniform on [0,1]
print("theory: 1 - 0.95^20 =", round(1 - 0.95 ** 20, 3))
print("simulated chance of at least one p < 0.05:", round((p < 0.05).any(axis=1).mean(), 3))
print("average number of false 'discoveries':", (p < 0.05).sum(axis=1).mean().round(2))`, "theory: 1 - 0.95^20 = 0.642\nsimulated chance of at least one p < 0.05: 0.645\naverage number of false 'discoveries': 1.0"],
    ["h", "Two goals, two kinds of correction"],
    ["list", [
      "**Family-wise error rate (FWER)**: the probability of **any** false positive. Strict. Use when one false claim is costly (a clinical trial, a final launch decision).",
      "**False discovery rate (FDR)**: the expected **share** of your discoveries that are false. Looser, more powerful. Use when screening many candidates and a few false leads are acceptable (feature selection, gene screens, finding promising segments).",
    ]],
    ["table", ["Method", "Controls", "Rule for $m$ tests", "Power"], [
      ["**Bonferroni**", "FWER", "reject if $p_i\\le\\alpha/m$", "lowest, simplest"],
      ["**Holm**", "FWER", "sort $p_{(1)}\\le\\dots$; reject while $p_{(i)}\\le\\alpha/(m-i+1)$, stop at the first failure", "always at least Bonferroni's"],
      ["**Benjamini-Hochberg**", "FDR", "find the largest $i$ with $p_{(i)}\\le\\dfrac{i}{m}\\,q$; reject the $i$ smallest", "highest"],
    ]],
    ["p", "**Worked example.** Six p-values: 0.001, 0.009, 0.020, 0.041, 0.045, 0.60, with $\\alpha=q=0.05$ and $m=6$. **Bonferroni** cut-off $=0.05/6=0.0083$: only 0.001 passes (1 discovery). **Holm**: thresholds 0.0083, 0.01, 0.0125, ...: 0.001 and 0.009 pass, but 0.020 > 0.0125 stops it (2 discoveries). **BH**: thresholds $\\frac{i}{6}\\times0.05$ are 0.0083, 0.0167, 0.025, 0.033, 0.042, 0.05. The p-values 0.001, 0.009 and 0.020 are under their thresholds, 0.041 > 0.033 and 0.045 > 0.042 are not, so the largest passing $i$ is 3 (3 discoveries). Looser correction, more findings, at the price of accepting that about 5% of them may be false."],
    ["code", R`
import numpy as np
from scipy import stats

p = np.array([0.001, 0.009, 0.020, 0.041, 0.045, 0.60])
m, alpha = len(p), 0.05

bonf = p <= alpha / m

order = np.argsort(p)                                    # Holm: step-down
holm = np.zeros(m, dtype=bool)
for rank, i in enumerate(order):
    if p[i] <= alpha / (m - rank):
        holm[i] = True
    else:
        break

ps = p[order]                                            # Benjamini-Hochberg: step-up
passed = np.nonzero(ps <= (np.arange(1, m + 1) / m) * alpha)[0]
k = passed.max() + 1 if len(passed) else 0
bh = np.zeros(m, dtype=bool)
bh[order[:k]] = True

print("Bonferroni rejects:", bonf.sum(), bonf.tolist())
print("Holm rejects      :", holm.sum(), holm.tolist())
print("BH rejects        :", bh.sum(), bh.tolist())
print("BH adjusted p (q-values):", stats.false_discovery_control(p).round(4).tolist())`, "Bonferroni rejects: 1 [True, False, False, False, False, False]\nHolm rejects      : 2 [True, True, False, False, False, False]\nBH rejects        : 3 [True, True, True, False, False, False]\nBH adjusted p (q-values): [0.006, 0.027, 0.04, 0.054, 0.054, 0.6]"],
    ["h", "Simulation: what each correction buys you"],
    ["p", "1,000 hypotheses, 900 truly null and 100 with a real effect. Compare no correction, Bonferroni and BH on how many true effects they find and how many false alarms they let through."],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
m, m_true = 1000, 100
z = np.concatenate([rng.normal(0, 1, m - m_true), rng.normal(3, 1, m_true)])      # last 100 are real effects (z ~ 3)
p = 2 * stats.norm.sf(np.abs(z))
real = np.arange(m) >= m - m_true

def report(name, rej):
    print(f"{name:11s} discoveries {rej.sum():4d}   true {np.sum(rej & real):3d}   false {np.sum(rej & ~real):3d}   "
          f"false share {np.sum(rej & ~real) / max(rej.sum(), 1):.2f}")

report("none", p < 0.05)
report("Bonferroni", p < 0.05 / m)
q = stats.false_discovery_control(p)                    # BH-adjusted p-values
report("BH (q=0.05)", q < 0.05)`, "none        discoveries  115   true  80   false  35   false share 0.30\nBonferroni  discoveries    8   true   8   false   0   false share 0.00\nBH (q=0.05) discoveries   34   true  32   false   2   false share 0.06"],
    ["code", R`
import numpy as np
from statsmodels.stats.multitest import multipletests

p = np.array([0.001, 0.009, 0.020, 0.041, 0.045, 0.60])
for method in ["bonferroni", "holm", "fdr_bh"]:
    reject, adj, _, _ = multipletests(p, alpha=0.05, method=method)
    print(f"{method:10s}", reject.tolist(), adj.round(4).tolist())`, "bonferroni [True, False, False, False, False, False] [0.006, 0.054, 0.12, 0.246, 0.27, 1.0]\nholm       [True, True, False, False, False, False] [0.006, 0.045, 0.08, 0.123, 0.123, 0.6]\nfdr_bh     [True, True, True, False, False, False] [0.006, 0.027, 0.04, 0.054, 0.054, 0.6]", "statsmodels", "local"],
    ["danger", "Corrections must cover **all** the tests you ran, including the ones you did not report. If you tried 40 feature sets and show the best, the honest m is 40. Deciding on the comparisons **before** looking at the data (one primary metric, a few guardrail metrics) is the cleanest way to avoid the problem. The same applies to ML: picking the best of 200 hyperparameter settings by validation score makes that score optimistic, so keep an untouched [test set](ml.html#splits)."],
  ],
},
);
STATS.push(
{
  id: "ab-testing", group: "Applied", title: "A/B Testing: Power, Sample Size & Effect Size",
  sub: "How to run an experiment so that its result means something: decide the effect, the error rates and the sample size in advance.",
  tags: ["experiments"],
  blocks: [
    ["p", "A product team changes a button from blue to green. 10% of visitors click the old button. Is the new one better? You split visitors at random into **A** (old) and **B** (new), count clicks, and run a test. The hard part is not the test. It is deciding **beforehand** how many visitors you need, because too few means you will miss a real improvement and too many wastes time and exposes users to a possibly worse variant."],
    ["h", "The four linked quantities"],
    ["table", ["Quantity", "Symbol", "Typical value", "Meaning"], [
      ["Significance level", "$\\alpha$", "0.05", "Accepted false-positive rate (see [Hypothesis Testing](stats.html#hypothesis-testing))"],
      ["Power", "$1-\\beta$", "0.80", "Chance of detecting the effect if it is real"],
      ["Minimum detectable effect (MDE)", "$\\delta$", "e.g. 10% to 12%", "The smallest improvement worth caring about"],
      ["Sample size", "$n$ per group", "computed", "Fix any three and the fourth follows"],
    ]],
    ["viz", "stats_power"],
    ["p", "Play with the sliders: a bigger effect, a bigger $n$ or a looser $\\alpha$ all raise power. The most important lesson is the cost of small effects: **halving the effect you want to detect needs about four times the users**, because $n\\propto1/\\delta^2$."],
    ["h", "Sample size for comparing two proportions"],
    ["math", R`n=\frac{\Big(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}+z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\Big)^2}{(p_2-p_1)^2},\qquad \bar p=\frac{p_1+p_2}{2}`],
    ["list", [
      "$p_1$ is the baseline rate (10%) and $p_2$ the rate you hope to detect (12%). $n$ is the number needed **in each group**.",
      "$z_{1-\\alpha/2}=1.96$ for a two-sided test at $\\alpha=0.05$; $z_{1-\\beta}=0.84$ for 80% power.",
      "**Worked example.** $\\bar p=0.11$, so $\\sqrt{2\\cdot0.11\\cdot0.89}=0.4425$ and $\\sqrt{0.09+0.1056}=0.4423$. Then $n=\\dfrac{(1.96\\cdot0.4425+0.84\\cdot0.4423)^2}{0.02^2}=\\dfrac{(1.2386)^2}{0.0004}\\approx3{,}841$ per group, about 7,700 visitors in total.",
    ]],
    ["code", R`
import numpy as np
from scipy import stats

def n_per_group(p1, p2, alpha=0.05, power=0.80):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (p2 - p1) ** 2

print("10% -> 12%:", int(np.ceil(n_per_group(0.10, 0.12))), "per group")
for p2 in [0.11, 0.12, 0.15, 0.20]:
    print(f"10% -> {p2:.0%}: {int(np.ceil(n_per_group(0.10, p2))):6d} per group")`, "10% -> 12%: 3841 per group\n10% -> 11%:  14751 per group\n10% -> 12%:   3841 per group\n10% -> 15%:    686 per group\n10% -> 20%:    199 per group"],
    ["h", "Check the formula by simulation"],
    ["p", "Run the experiment 2,000 times with a true 10% vs 12% rate and the computed $n$. About 80% of runs should come out significant. And when there is **no** true difference, about 5% should."],
    ["code", R`
import numpy as np
from scipy import stats

def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)                               # pooled rate under H0
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return z, 2 * stats.norm.sf(abs(z))

print("one example (400/4000 vs 480/4000):", [round(float(v), 4) for v in two_prop_ztest(400, 4000, 480, 4000)])

rng = np.random.default_rng(0)
n, runs = 3841, 2000
for p2 in [0.12, 0.10]:
    a = rng.binomial(n, 0.10, runs)
    b = rng.binomial(n, p2, runs)
    z, p = two_prop_ztest(a, n, b, n)
    print(f"true rates 10% vs {p2:.0%}: significant in {np.mean(p < 0.05):.3f} of runs")`, "one example (400/4000 vs 480/4000): [2.8586, 0.0043]\ntrue rates 10% vs 12%: significant in 0.792 of runs\ntrue rates 10% vs 10%: significant in 0.052 of runs"],
    ["code", R`
import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

h = proportion_effectsize(0.12, 0.10)                       # Cohen's h for 12% vs 10%
n = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided")
print("effect size h", round(h, 4), " n per group", int(np.ceil(n)))

pw = NormalIndPower().solve_power(effect_size=h, nobs1=2000, alpha=0.05, ratio=1.0)
print("power with 2000 per group:", round(pw, 3))

d_n = TTestIndPower().solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print("t-test, d=0.5, 80% power -> n per group", int(np.ceil(d_n)))`, "effect size h 0.064  n per group 3835\npower with 2000 per group: 0.525\nt-test, d=0.5, 80% power -> n per group 64", "statsmodels power analysis", "local"],
    ["note", "statsmodels gives 3,835 instead of 3,841: it uses Cohen's h (an arcsine transform of each rate) rather than the pooled-variance formula above. Both are approximations of the same thing; planning tools differ by a few users, which never matters. The last line shows the power of a t-test: a medium effect ($d=0.5$) needs 64 per group for 80% power."],
    ["h", "Running the experiment well"],
    ["list", [
      "**Decide first**: the primary metric, the MDE, $\\alpha$, power and the sample size (or run length) are written down before launch. Run whole weeks to cover weekday effects.",
      "**Do not peek.** Looking daily and stopping at the first $p<0.05$ inflates false positives (see the simulation in [Hypothesis Testing](stats.html#hypothesis-testing)). If you must monitor, use a sequential method designed for it.",
      "**Check the split** with a sample-ratio-mismatch test ([Chi-Square](stats.html#chi-square)). A broken split invalidates everything.",
      "**One primary metric**, a few guardrail metrics (latency, errors, revenue). Testing 20 metrics needs a [multiple-testing](stats.html#multiple-testing) correction.",
      "**Randomise at the right unit** (user, not page view), or the observations are not independent and your p-values are too small.",
      "**Statistical is not practical significance**: report the effect size with a [confidence interval](stats.html#confidence-intervals), e.g. 'lift of +1.9 points, 95% CI 0.2 to 3.6', and decide if even the low end is worth shipping.",
    ]],
    ["warn", "Underpowered tests are worse than useless: they mostly miss real effects, and the few that reach significance exaggerate the effect size (the 'winner's curse'). Also, 'no significant difference' after a small test does not mean 'no difference'. Compute power or the CI before concluding anything."],
    ["tip", "The same logic applies to ML evaluation: comparing a new model with the old one on a held-out set is an A/B test of sorts, and a 0.3-point gain on a small test set is within noise (see [the CLT](stats.html#clt) and [t-tests](stats.html#t-tests)). To lower the variance, test on the same examples (paired) or use a metric closer to the user's goal. Offline model metrics: [Confusion Matrix, Precision, Recall & F1](ml.html#classification-metrics)."],
  ],
},
{
  id: "correlation", group: "Applied", title: "Correlation & Covariance",
  sub: "One number from -1 to 1 for how closely two variables move together in a straight line, and why you must still look at the plot.",
  tags: ["relationships"],
  blocks: [
    ["p", "Do taller people weigh more? Does more training data go with higher accuracy? **Covariance** and **correlation** measure how two numeric variables move together. If $x$ is above its average when $y$ is above its average (and below when below), they are positively related."],
    ["math", R`\text{cov}(x,y)=\frac{1}{n-1}\sum_{i}(x_i-\bar x)(y_i-\bar y),\qquad r=\frac{\text{cov}(x,y)}{s_x\,s_y}=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2}\sqrt{\sum(y_i-\bar y)^2}}`],
    ["list", [
      "**Covariance** has the units of $x$ times the units of $y$ (metres times kilograms), so its size is hard to interpret. Positive means they rise together, negative means one rises as the other falls. See [Expectation & Variance](math.html#expectation).",
      "**Pearson's $r$** divides by both standard deviations, giving a unit-free number from $-1$ to $+1$: $+1$ a perfect upward line, $-1$ a perfect downward line, $0$ no **linear** relationship. It is the cosine of the angle between the two centred vectors ([Dot Product & Cosine Similarity](math.html#dot-product)).",
      "$r^2$ is the share of the variance of $y$ explained by a straight line in $x$ (the $R^2$ of [simple linear regression](ml.html#linear-regression)).",
    ]],
    ["p", "**Worked example.** $x=1,2,3,4,5$ and $y=2,4,5,4,5$. Means: $\\bar x=3$, $\\bar y=4$. Products of deviations: $(-2)(-2)+(-1)(0)+0\\cdot1+1\\cdot0+2\\cdot1=6$. Sums of squares: $\\sum(x-\\bar x)^2=10$, $\\sum(y-\\bar y)^2=6$. So $r=6/\\sqrt{10\\cdot6}=0.775$."],
    ["viz", "stats_corr"],
    ["code", R`
import numpy as np
from scipy import stats

x = np.array([1, 2, 3, 4, 5.0])
y = np.array([2, 4, 5, 4, 5.0])
dx, dy = x - x.mean(), y - y.mean()
cov = (dx * dy).sum() / (len(x) - 1)
r = (dx * dy).sum() / np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
print("covariance", cov, "  r", round(r, 4), "  r^2", round(r ** 2, 3))
print("np.cov     :", np.cov(x, y)[0, 1], "  np.corrcoef:", np.corrcoef(x, y)[0, 1].round(4))

res = stats.pearsonr(x, y)
print("pearsonr   : r", round(res.statistic, 4), " p", round(res.pvalue, 4))

t = r * np.sqrt(len(x) - 2) / np.sqrt(1 - r ** 2)             # the test that rho = 0
print("by hand    : t", round(t, 3), " p", round(2 * stats.t.sf(abs(t), len(x) - 2), 4))`, "covariance 1.5   r 0.7746   r^2 0.6\nnp.cov     : 1.5   np.corrcoef: 0.7746\npearsonr   : r 0.7746  p 0.124\nby hand    : t 2.121  p 0.124"],
    ["h", "Pearson vs Spearman, and the Anscombe warning"],
    ["p", "**Spearman's** $\\rho$ is Pearson's $r$ computed on the **ranks**. It captures any steadily increasing or decreasing (monotonic) relationship and is hardly bothered by outliers. Below, $y=x^2$ is perfectly predictable from $x$ but not linear; and a single outlier wrecks Pearson. Then Anscombe's famous four datasets: identical means, variances and $r=0.816$, yet totally different pictures."],
    ["code", R`
import numpy as np
from scipy import stats

x = np.linspace(-3, 3, 61)
print("y = x^2     : pearson", round(stats.pearsonr(x, x ** 2).statistic, 3), "  spearman", round(stats.spearmanr(x, x ** 2).statistic, 3), " (a perfect U-shape has r = 0)")
print("y = exp(x)  : pearson", round(stats.pearsonr(x, np.exp(x)).statistic, 3), "  spearman", round(stats.spearmanr(x, np.exp(x)).statistic, 3), " (monotonic but curved)")

rng = np.random.default_rng(0)
a = rng.normal(size=30); b = a + rng.normal(scale=0.5, size=30)
print("clean data  : pearson", round(stats.pearsonr(a, b).statistic, 3), "  spearman", round(stats.spearmanr(a, b).statistic, 3))
a2, b2 = np.append(a, 8.0), np.append(b, -8.0)               # one wild point
print("one outlier : pearson", round(stats.pearsonr(a2, b2).statistic, 3), "  spearman", round(stats.spearmanr(a2, b2).statistic, 3))

xs = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
sets = {"I": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        "II": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        "III": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]}
for name, ys in sets.items():
    print("Anscombe", name, ": r =", round(stats.pearsonr(xs, ys).statistic, 3), " mean y", round(np.mean(ys), 2), " var y", round(np.var(ys, ddof=1), 2))`, "y = x^2     : pearson 0.0   spearman 0.016  (a perfect U-shape has r = 0)\ny = exp(x)  : pearson 0.816   spearman 1.0  (monotonic but curved)\nclean data  : pearson 0.829   spearman 0.84\none outlier : pearson -0.574   spearman 0.668\nAnscombe I : r = 0.816  mean y 7.5  var y 4.13\nAnscombe II : r = 0.816  mean y 7.5  var y 4.13\nAnscombe III : r = 0.816  mean y 7.5  var y 4.12"],
    ["h", "A correlation matrix"],
    ["code", R`
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

df = load_diabetes(as_frame=True).frame[["age", "bmi", "bp", "s1", "s2", "target"]]
print(df.corr().round(2))`, "         age   bmi    bp    s1    s2  target\nage     1.00  0.19  0.34  0.26  0.22    0.19\nbmi     0.19  1.00  0.40  0.25  0.26    0.59\nbp      0.34  0.40  1.00  0.24  0.19    0.44\ns1      0.26  0.25  0.24  1.00  0.90    0.21\ns2      0.22  0.26  0.19  0.90  1.00    0.17\ntarget  0.19  0.59  0.44  0.21  0.17    1.00"],
    ["warn", "A significant correlation is not a strong one: with $n=10{,}000$ even $r=0.03$ gives $p<0.01$. And a high $r$ is not a causal link ([next section](stats.html#causation)). Three more traps: **non-linear** patterns (r near 0 for a perfect U), **outliers** and **restricted range** (look only at tall basketball players and height seems unrelated to ability), and pooling **subgroups** with different trends."],
    ["tip", "In ML, check the correlation matrix to spot features that duplicate each other (multicollinearity makes [regression coefficients](stats.html#regression-inference) unstable) and features that correlate suspiciously well with the target (possible [data leakage](ml.html#splits)). Always scatter-plot the pairs that matter."],
  ],
},
{
  id: "causation", group: "Applied", title: "Correlation vs Causation & Confounders",
  sub: "Why 'A goes with B' does not mean 'A causes B', and the third variables that fool us.",
  tags: ["causality"],
  blocks: [
    ["p", "Ice-cream sales and drownings rise and fall together all year. Does ice cream cause drowning? No: hot weather makes people buy ice cream **and** go swimming. The weather is a **confounder**: a third variable that influences both the 'cause' and the 'effect', creating a link that disappears if you account for it."],
    ["diagram", "stats_confounder", [], "A confounder Z drives both X and Y, so X and Y are correlated even if X has no effect on Y."],
    ["p", "Three explanations are always on the table when two variables are associated: $X$ causes $Y$, $Y$ causes $X$ (reverse causation), or something else causes both (confounding). A fourth, pure coincidence, matters when you look at many pairs ([Multiple Testing](stats.html#multiple-testing))."],
    ["h", "Simpson's paradox"],
    ["p", "Confounding can even **reverse** a trend. In the plot below a higher dose helps in each patient group, but severe patients receive higher doses and recover worse for other reasons, so across everyone the trend looks negative. Turn the colouring off to see what a naive analysis would conclude."],
    ["viz", "stats_simpson"],
    ["p", "A real, famous example (kidney-stone treatments, Charig et al. 1986): treatment A has the better success rate for small stones and for large stones, yet treatment B looks better overall, because doctors gave the harder (large-stone) cases more often to A."],
    ["code", R`
import numpy as np

# (successes, patients)
A = {"small stones": (81, 87), "large stones": (192, 263)}
B = {"small stones": (234, 270), "large stones": (55, 80)}

for g in A:
    print(f"{g:13s} A: {A[g][0] / A[g][1]:.1%}   B: {B[g][0] / B[g][1]:.1%}")
sa, na = np.sum(list(A.values()), axis=0)
sb, nb = np.sum(list(B.values()), axis=0)
print(f"{'overall':13s} A: {sa / na:.1%}   B: {sb / nb:.1%}   <- the order flips")
print("share of A's patients with large stones:", round(263 / 350, 2), "  B's:", round(80 / 350, 2))`, "small stones  A: 93.1%   B: 86.7%\nlarge stones  A: 73.0%   B: 68.8%\noverall       A: 78.0%   B: 82.6%   <- the order flips\nshare of A's patients with large stones: 0.75   B's: 0.23"],
    ["h", "A confounder in simulated data"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
n = 5000
z = rng.normal(size=n)                          # confounder, e.g. temperature
x = 0.8 * z + rng.normal(size=n)                # ice-cream sales: driven by z, NO effect on y
y = 0.8 * z + rng.normal(size=n)                # drownings: driven by z

print("corr(x, y):", np.corrcoef(x, y)[0, 1].round(3), "  <- looks like a relationship")

def ols(features, target):
    A = np.column_stack([np.ones(len(target))] + features)
    return np.linalg.lstsq(A, target, rcond=None)[0]

print("regress y on x       : slope of x =", ols([x], y)[1].round(3))
print("regress y on x and z : slope of x =", ols([x, z], y)[1].round(3), " (adjusting for z removes it)")`, "corr(x, y): 0.392   <- looks like a relationship\nregress y on x       : slope of x = 0.385\nregress y on x and z : slope of x = 0.013  (adjusting for z removes it)"],
    ["h", "How to get at causation"],
    ["list", [
      "**Randomised experiment (A/B test)**: assigning treatment by coin flip breaks every link between the treatment and any confounder, known or unknown. It is the gold standard ([A/B Testing](stats.html#ab-testing)).",
      "**Adjustment** (regression, stratification, matching): compare like with like. It only removes confounders you **measured**; unmeasured ones remain.",
      "**Quasi-experiments**: natural experiments, difference-in-differences, instrumental variables, regression discontinuity.",
      "**Draw the causal diagram first** (as above) and decide what to adjust for. Careful: adjusting for a **collider** (a variable caused by both $X$ and $Y$) *creates* a fake association, so 'control for everything' is not safe.",
    ]],
    ["warn", "Observational data plus a good predictive model still does not answer 'what if we change X?'. A model can predict that hospital patients with oxygen masks die more, without masks being harmful. Prediction needs correlation; decisions need causation."],
    ["tip", "ML models happily latch onto confounders: a classifier that identifies 'pneumonia' from which hospital's X-ray scanner made the image, or 'husky vs wolf' from snow in the background (a **shortcut**). These patterns fail when the environment changes. Related: [data leakage](ml.html#splits), and [Correlation](stats.html#correlation) for the measures themselves."],
  ],
},
{
  id: "regression-inference", group: "Applied", title: "Regression Inference",
  sub: "Beyond the fitted line: standard errors, t-tests and confidence intervals for the coefficients, and what R-squared does and does not tell you.",
  tags: ["regression"],
  blocks: [
    ["p", "In [Linear Regression](ml.html#linear-regression) you fit weights to make good predictions. **Inference** asks different questions: how precisely is each coefficient estimated, and is a feature's effect distinguishable from zero? 'Each extra year of age lowers the price by 1.47, give or take 0.24' is a statement about the **population**, and it needs standard errors."],
    ["p", "The model is $y=X\\beta+\\varepsilon$ with noise $\\varepsilon\\sim\\mathcal N(0,\\sigma^2)$. The least-squares solution and its uncertainty come from the same matrix:"],
    ["math", R`\hat\beta=(X^\top X)^{-1}X^\top y,\qquad \hat\sigma^2=\frac{\sum_i(y_i-\hat y_i)^2}{n-p},\qquad \text{SE}(\hat\beta_j)=\sqrt{\hat\sigma^2\,\big[(X^\top X)^{-1}\big]_{jj}}`],
    ["list", [
      "$n$ rows, $p$ columns of $X$ (including the intercept column of ones). $\\hat\\sigma^2$ is the residual variance, dividing by $n-p$ because $p$ parameters were estimated.",
      "$\\text{SE}(\\hat\\beta_j)$ says how much coefficient $j$ would vary from sample to sample. It is smaller with more data, less noise, and **more spread in feature $j$**; and larger if features are correlated with each other.",
      "To test $H_0:\\beta_j=0$ use $t_j=\\hat\\beta_j/\\text{SE}(\\hat\\beta_j)$ with $n-p$ degrees of freedom. The 95% CI is $\\hat\\beta_j\\pm t_{n-p,0.975}\\,\\text{SE}(\\hat\\beta_j)$.",
      "The overall **F-test** asks whether the model beats predicting the mean: $F=\\dfrac{R^2/(p-1)}{(1-R^2)/(n-p)}$. $R^2=1-\\dfrac{SS_{res}}{SS_{tot}}$.",
    ]],
    ["code", R`
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
n = 100
size = rng.uniform(50, 150, n)                    # house size (m2)
age = rng.uniform(0, 40, n)                       # age (years)
price = 20 + 2.0 * size - 1.5 * age + rng.normal(0, 25, n)

X = np.column_stack([np.ones(n), size, age])
p = X.shape[1]
XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ price                      # least squares
resid = price - X @ beta
sigma2 = resid @ resid / (n - p)
se = np.sqrt(sigma2 * np.diag(XtX_inv))
t = beta / se
pval = 2 * stats.t.sf(np.abs(t), n - p)
tcrit = stats.t.ppf(0.975, n - p)

for name, b, s, tt, pp in zip(["const", "size", "age"], beta, se, t, pval):
    print(f"{name:6s} coef {b:8.3f}  se {s:6.3f}  t {tt:7.2f}  p {pp:.4f}  95% CI [{b - tcrit * s:.3f}, {b + tcrit * s:.3f}]")

r2 = 1 - resid @ resid / ((price - price.mean()) ** 2).sum()
F = (r2 / (p - 1)) / ((1 - r2) / (n - p))
print("R^2", round(r2, 3), " F", round(F, 1), " residual std", round(np.sqrt(sigma2), 2))`, "const  coef   21.397  se 10.466  t    2.04  p 0.0436  95% CI [0.625, 42.169]\nsize   coef    1.947  se  0.096  t   20.37  p 0.0000  95% CI [1.758, 2.137]\nage    coef   -1.472  se  0.242  t   -6.08  p 0.0000  95% CI [-1.952, -0.992]\nR^2 0.811  F 208.1  residual std 28.09"],
    ["p", "The professional tool prints all of this in one table. statsmodels' OLS (a local example; it is not available in the browser) gives the identical numbers, plus diagnostics:"],
    ["code", R`
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 100
size = rng.uniform(50, 150, n)
age = rng.uniform(0, 40, n)
price = 20 + 2.0 * size - 1.5 * age + rng.normal(0, 25, n)

X = sm.add_constant(np.column_stack([size, age]))
res = sm.OLS(price, X).fit()
print("R2", round(res.rsquared, 3), " adj R2", round(res.rsquared_adj, 3), " F", round(res.fvalue, 1), " p(F)", res.f_pvalue)
print(res.summary(xname=["const", "size", "age"]).tables[1])`, "R2 0.811  adj R2 0.807  F 208.1  p(F) 8.113641146212058e-36\n==============================================================================\n                 coef    std err          t      P>|t|      [0.025      0.975]\n------------------------------------------------------------------------------\nconst         21.3972     10.466      2.044      0.044       0.625      42.169\nsize           1.9473      0.096     20.370      0.000       1.758       2.137\nage           -1.4720      0.242     -6.083      0.000      -1.952      -0.992\n==============================================================================", "statsmodels OLS", "local"],
    ["h", "Assumptions and how to check them"],
    ["table", ["Assumption", "If violated", "Check / fix"], [
      ["The relationship is linear in the features", "biased coefficients", "residual vs fitted plot should show no curve; add [polynomial features](ml.html#polynomial) or transform"],
      ["Independent errors", "SEs too small; false 'significance'", "time series / clustered rows: robust or clustered SEs, mixed models"],
      ["Constant error variance (homoscedasticity)", "SEs wrong, estimates still unbiased", "residuals should fan out evenly; use robust (HC) standard errors or transform $y$"],
      ["Roughly normal errors", "mostly matters for small $n$", "Q-Q plot of residuals; CLT helps at larger $n$"],
      ["Features not highly correlated (no multicollinearity)", "huge SEs, unstable signs", "[correlation matrix](stats.html#correlation), variance inflation factor (VIF above 5 to 10)"],
    ]],
    ["danger", "Do not read coefficients as causal effects (see [Causation](stats.html#causation)), and do not compare raw coefficient sizes across features measured in different units. A tiny $p$ for a coefficient does not mean a useful model, and a high $R^2$ does not mean a correct one: $R^2$ never decreases when you add features, even useless ones. Use adjusted $R^2$, held-out error, or [cross-validation](ml.html#cross-validation) for predictive quality."],
    ["tip", "Inference and prediction are different goals. For prediction, regularisation ([Regularization: L1, L2 & ElasticNet](ml.html#regularization)) helps, but it biases coefficients, so ordinary p-values no longer apply. For a Bayesian take where the coefficients get a full posterior distribution, see [Bayesian Linear Regression](classical.html#bayes-linreg)."],
  ],
},
{
  id: "bayes-vs-freq", group: "Applied", title: "Bayesian vs Frequentist Inference",
  sub: "Two ways to reason about uncertainty: long-run error rates of a procedure, or a probability for the unknown given the data.",
  tags: ["bayesian"],
  blocks: [
    ["p", "So far we used the **frequentist** view: the true value (the real click rate) is a fixed unknown number, and probability describes how often our **procedure** would be right over many repeated experiments. That is where p-values and 95% confidence intervals come from. The **Bayesian** view treats the unknown itself as uncertain: you start with a **prior** belief, update it with the data using [Bayes' theorem](math.html#bayes), and end with a **posterior** distribution that you can read directly as probabilities."],
    ["math", R`p(\theta\mid\text{data})\;\propto\;p(\text{data}\mid\theta)\;p(\theta)\qquad\text{posterior}\propto\text{likelihood}\times\text{prior}`],
    ["h", "A/B test with Beta-Binomial"],
    ["p", "For a click rate $\\theta$, start with a $\\text{Beta}(\\alpha,\\beta)$ prior. After seeing $k$ clicks in $n$ trials the posterior is again a Beta, with no integration required: $\\text{Beta}(\\alpha+k,\\ \\beta+n-k)$ (a **conjugate** prior). $\\text{Beta}(1,1)$ is flat: every rate equally plausible. **Worked example.** A: 120 clicks of 1,000 gives $\\text{Beta}(121,881)$; B: 150 of 1,000 gives $\\text{Beta}(151,851)$. Sample many values from each posterior and count how often B's rate is higher."],
    ["code", R`
import numpy as np
from scipy import stats

clicks_a, n_a = 120, 1000
clicks_b, n_b = 150, 1000
post_a = stats.beta(1 + clicks_a, 1 + n_a - clicks_a)          # Beta(1,1) prior + data
post_b = stats.beta(1 + clicks_b, 1 + n_b - clicks_b)

print("posterior mean A", round(post_a.mean(), 4), "  B", round(post_b.mean(), 4))
print("95% credible interval A:", post_a.interval(0.95)[0].round(4), "to", post_a.interval(0.95)[1].round(4))
print("95% credible interval B:", post_b.interval(0.95)[0].round(4), "to", post_b.interval(0.95)[1].round(4))

rng = np.random.default_rng(0)
a = post_a.rvs(100000, random_state=rng)
b = post_b.rvs(100000, random_state=rng)
print("P(B better than A)      :", (b > a).mean().round(3))
print("expected lift           :", (b - a).mean().round(4))
print("P(lift > 1 point)       :", ((b - a) > 0.01).mean().round(3))
print("frequentist two-sided p (for comparison):", round(stats.chi2_contingency([[120, 880], [150, 850]], correction=False)[1], 4))`, "posterior mean A 0.1208   B 0.1507\n95% credible interval A: 0.1013 to 0.1416\n95% credible interval B: 0.1292 to 0.1735\nP(B better than A)      : 0.975\nexpected lift           : 0.0299\nP(lift > 1 point)       : 0.903\nfrequentist two-sided p (for comparison): 0.0496"],
    ["p", "Notice what the Bayesian output offers: 'there is a 97.5% probability that B is better' and 'a 90% chance the lift exceeds one point'. These are the statements people usually **want**, and a p-value cannot give them. The price is that the answer depends on the prior."],
    ["h", "How much does the prior matter?"],
    ["code", R`
from scipy import stats

# 3 clicks out of 10 visitors: very little data
for name, (a0, b0) in {"flat Beta(1,1)": (1, 1), "skeptical Beta(10,90) (rates near 10%)": (10, 90)}.items():
    post = stats.beta(a0 + 3, b0 + 7)
    print(f"{name:40s} posterior mean {post.mean():.3f}")

# 300 clicks out of 1000: lots of data, the prior is overwhelmed
for name, (a0, b0) in {"flat Beta(1,1)": (1, 1), "skeptical Beta(10,90) (rates near 10%)": (10, 90)}.items():
    post = stats.beta(a0 + 300, b0 + 700)
    print(f"{name:40s} posterior mean {post.mean():.3f}")`, "flat Beta(1,1)                           posterior mean 0.333\nskeptical Beta(10,90) (rates near 10%)   posterior mean 0.118\nflat Beta(1,1)                           posterior mean 0.300\nskeptical Beta(10,90) (rates near 10%)   posterior mean 0.282"],
    ["table", ["", "Frequentist", "Bayesian"], [
      ["Unknown parameter", "fixed number", "random variable with a distribution"],
      ["Probability means", "long-run frequency over repeated experiments", "degree of belief given the data"],
      ["Main outputs", "p-value, confidence interval, point estimate (e.g. [MLE](math.html#mle))", "posterior distribution, credible interval, P(B > A)"],
      ["Interval meaning", "95% of such intervals contain the truth (a statement about the method)", "95% probability the parameter is inside, given data and prior"],
      ["Prior knowledge", "none in the test itself", "required, explicit (and can be criticised)"],
      ["Computation", "closed-form formulas, simple", "conjugate formulas or sampling ([MCMC](classical.html#mcmc))"],
      ["Small data", "wide, often uninformative", "a sensible prior stabilises estimates"],
      ["Stopping early", "peeking inflates false positives", "posterior is valid at any time (but decision rules still need care)"],
    ]],
    ["note", "MAP estimation and regularisation are the Bayesian view hiding inside ML: L2 regularisation is a Gaussian prior on the weights, L1 a Laplace prior (see [Maximum Likelihood Estimation](math.html#mle) and [Bayesian Linear Regression](classical.html#bayes-linreg)). When there is no neat conjugate formula, sample the posterior with [MCMC](classical.html#mcmc)."],
    ["tip", "Practical stance: both are tools. With plenty of data and a flat prior they usually agree. Use frequentist tests for standard, pre-registered comparisons, and the Bayesian approach when you want direct probabilities, have real prior information, have little data, or need to update continuously. Whichever you use, the earlier lessons stand: decide the question first, look at effect sizes, correct for multiple looks, and do not confuse a model's output with a causal truth."],
  ],
},
);
