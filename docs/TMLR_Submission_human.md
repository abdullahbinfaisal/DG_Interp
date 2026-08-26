## When Invariance Is Not Enough: Sparse Concept Diagnostics for Domain Generalization

## Anonymous authors Paper under double-blind review

## Abstract

Domain generalization methods are commonly motivated by the search for invariant representations. Analyzing a frozen ERM ResNet-50 on PACS, we find that activation invariance is weakly informative about whether an internal concept is harmful and uninformative about whether it is useful. Among class-concept pairs with enough support to estimate, conditioning on invariance reduces the probability that a pair damages the correct class by about 40%, from 13.3% to 7.6% ($p = 0.002$), and leaves the probability that it supports the class unchanged; 83.5% of invariant pairs have no effect exceeding our magnitude threshold. That invariant harm is invisible to a criterion built on activation statistics follows from the definition of invariance. What we measure is what it costs. Thirty-two class-concept pairs that are both invariant and consistent in their harm, 0.028% of the class-concept grid and 0.50% of the pairs that fire at all, account for 22.6% of the model's image-weighted target-domain error.

These measurements come from sparse concept diagnostics, a post-hoc framework that decomposes internal vision representations into sparse autoencoder candidate concepts and scores each class-concept pair on three axes estimated from source domains only: activation invariance across domains, ablation-based discriminative effect on the ground-truth class, and cross-domain consistency of that effect. Splitting the invariant population by the sign of that effect reveals an asymmetry, in that invariant support is distributed across domains while invariant harm concentrates in one or two ($p < 10^{-6}$), and distributed harm proves several times more damaging per unit of effect mass than concentrated harm. The framework also indicates why a trained model contains harmful concepts at all: 78% of harmful pairs involve a concept that supports a different class, so harm is largely not a spurious feature but a genuine feature attached to the wrong class. Retaining 0.07% of the grid raises held-out accuracy above that of the unablated model, so the shortfall is partly one of selectivity rather than of representational capacity. These are **oracle diagnostics**, since the mask is addressed by each image's ground-truth label, and they are not deployable methods; the signed buckets are compared against size- and magnitude-matched random controls that move accuracy in the opposite direction. Domain-generalization representations should therefore be evaluated not only by whether concepts are invariant, but by whether they are discriminatively aligned and class-isolated.

## 1 Introduction

Domain generalization (DG) addresses degradation under distribution shift by learning from several source domains so as to generalize to unseen target domains, and the standard protocol reports target-domain accuracy. That protocol is necessary but incomplete, because it does not reveal which internal cues a model relies on. Two models with equal target accuracy may depend on different correlations, one robust and one fragile, and accuracy on the test domain will not separate them. The limitation is visible in modern benchmarks, where carefully tuned empirical risk minimization (ERM) matches the accuracy of explicit DG algorithms (Gulrajani & Lopez-Paz, 2021).

A dominant class of DG methods attempts to learn domain-invariant representations, meaning features that persist across changes in style, background, or acquisition condition. Enforcing invariance too aggressively, however, can suppress class-discriminative variation and blur the boundaries between classes. This is the discrimination-invariance tradeoff. Prior work characterizes the tradeoff theoretically and demonstrates it at the level of aggregate accuracy, showing that stronger invariance can raise target error (Zhao et al., 2019; Akuzawa et al., 2019).

What remains underexplored is how the tradeoff manifests inside the representation. Existing invariance diagnostics operate on global feature distributions, domain separability, or classifier risk, all of which are quantities averaged over classes. But invariance is not a property a feature has in isolation. The same internal feature can be reliable evidence for one class and a source of interference for another, and any score averaged over classes reports approximately zero for exactly that case. The question we ask is therefore class-conditional and concept-level.

We train a sparse autoencoder (SAE) on a frozen model's feature maps to obtain an overcomplete basis of candidate concepts, and score every class-concept pair on three axes: whether the concept activates evenly across domains for that class, whether ablating it raises or lowers the model's confidence in the correct class, and whether that effect is consistent across domains. All three are estimated on source domains only. Computing them per pair rather than per concept is the distinguishing move of this work, and it is what makes class conflict visible at all. The result is a post-hoc diagnosis of where a trained model's held-out error comes from, expressed in terms of identifiable internal concepts rather than aggregate accuracy.

## Contributions.

- 1. We introduce sparse concept diagnostics: three class-conditional scores over SAE candidate concepts, namely activation invariance H, ablation-based discriminative effect D, and cross-domain consistency R of that effect, all estimated on source domains only.

- 2. We measure that activation invariance is weakly informative about the absence of harm and uninformative about the presence of usefulness, and that invariant support distributes across domains while invariant harm concentrates in one or two. The asymmetry is present at every invariance threshold we examine and widens as the threshold tightens.

- 3. We show that harm is largely misdirected support: 78.3% of harmful class-concept pairs involve a concept that supports some other class, so the appropriate target is class isolation rather than feature removal.

- 4. We validate the resulting categories with **oracle** interventions, meaning masks addressed by the ground-truth label and compared against size- and magnitude-matched random controls. These establish that the categories are functional, and that a source-identified 0.07% of the class-concept grid suffices to exceed the model's own held-out accuracy.

## 2 Related Work

## 2.1 Domain generalization and the invariance hypothesis

DG studies learning under domain shift, typically by training on multiple source domains and evaluating on held-out target domains (Li et al., 2017; Gulrajani & Lopez-Paz, 2021). Many DG algorithms are motivated by the hypothesis that stable predictors should rely on invariant features. Domain-adversarial training attempts to remove domain information from the representation through an adversarial domain classifier (Ganin et al., 2016). CORAL aligns second-order statistics across domains (Sun & Saenko, 2016). MMD-based methods penalize kernel discrepancies between domain feature distributions (Li et al., 2018). Invariant Risk Minimization formalizes a related objective by seeking representations for which the optimal classifier is invariant across environments (Arjovsky et al., 2019).

Invariance alone, however, is not sufficient for robust prediction. Zhao et al. show that domain-invariant representations with low source error need not guarantee target performance under conditional shift, and characterize a tradeoff between learning invariant representations and achieving low joint error across domains (Zhao et al., 2019); subsequent work gives an information-theoretic account of the accuracy-invariance tradeoff (Zhao et al., 2022). In DG specifically, Galstyan et al. decompose generalization error into failure modes and show that the contribution of invariance-related failures varies across methods, datasets, regularization strengths, and training stages (Galstyan et al., 2022). Rationale-invariance methods offer another perspective, asking whether same-category examples are classified using similar decision rationales rather than merely similar feature distributions (Chen et al., 2023). Benchmark studies complicate the picture further: DomainBed showed that under standardized model selection, carefully tuned ERM is competitive with many specialized DG algorithms (Gulrajani & Lopez-Paz, 2021), and common evaluation protocols may leak test-domain information through supervised pretraining or oracle model selection (Yu et al., 2024).

Together these results shift the question from whether a representation is invariant to whether the invariant information is useful for the task. A feature can be domain-invariant but irrelevant to the class, consistently harmful to the correct class, or useful only in a subset of domains. Existing invariance diagnostics operate at the level of global feature distributions, domain separability, or classifier risk, all of which are properties of a concept averaged over classes. **Our research question is the class-conditional, concept-level version: for a concept c and a class k, does c activate across domains, does ablating it raise or lower the model's confidence in k, and is that effect consistent across domains?** Rather than proposing a new training objective, we analyze frozen models after training and ask whether their learned concepts are invariant, discriminative, and class-isolated.

## 2.2 Concept-based interpretability

Concept-based interpretability explains model behavior in terms of higher-level abstractions rather than individual pixels or dense activation vectors. Concept Bottleneck Models predict human-defined concepts before predicting labels, making the intermediate representation directly interpretable when concept annotations are available (Koh et al., 2020). TCAV probes trained models with user-specified concept directions and measures sensitivity through directional derivatives (Kim et al., 2018). Spatial attribution methods such as Grad-CAM produce class-discriminative localization maps (Selvaraju et al., 2017). These either require predefined concepts or provide local saliency rather than a reusable concept basis for analyzing representation structure across domains.

A related line of work discovers visual concepts automatically. Network Dissection evaluates alignment between hidden units and a vocabulary of semantic concepts (Bau et al., 2017); ACE clusters image segments and evaluates their importance to predictions (Ghorbani et al., 2019); CRAFT combines concept discovery with spatial attribution (Fel et al., 2023); completeness-aware methods study whether a concept set suffices to explain predictions and introduce importance measures such as ConceptSHAP (Yeh et al., 2020). Our goal is related but distinct. We do not aim to name every concept or produce human-semantic explanations. We use sparse candidate concepts as a functional diagnostic basis for asking whether learned internal features are invariant, discriminative, and class-consistent across domains.

## 2.3 Sparse autoencoders

Sparse autoencoders are motivated by the possibility that networks represent more features than they have neurons. Work on superposition argues that models store many sparse features in distributed directions, making individual neurons polysemantic and hard to interpret (Elhage et al., 2022). Dictionary-learning and SAE methods attempt to recover a sparse feature basis from dense activations: work on monosemanticity decomposes language-model activations into more interpretable features (Bricken et al., 2023), and Cunningham et al. show that SAEs recover highly interpretable language-model features supporting finer-grained causal analysis (Cunningham et al., 2024). Recent work extends SAEs to vision. Lim et al. introduce PatchSAE for CLIP vision transformers, extracting patch-level visual concepts and studying how adaptation changes the association between images and learned concepts (Lim et al., 2025).

Both literatures converge on the same methodological point: **SAE quality must be validated by intervention, not by unsupervised proxies or qualitative inspection.** Gao et al. study scaling laws and evaluation metrics for k-sparse autoencoders, and report the reconstruction-sparsity tradeoff and the problem of dead latents (Gao et al., 2025). Makelov et al. argue that feature dictionaries should be evaluated in terms of approximation, control and interpretability on specific tasks (Makelov et al., 2024). Karvonen et al. evaluate SAEs through targeted concept-erasure tasks (Karvonen et al., 2024). In vision, Stevens et al. argue that visual feature interpretations should be validated through controlled interventions rather than qualitative inspection alone (Stevens et al., 2025), and Han et al. caution that naive top-activation visualizations can mislead, because a feature's activated patch may co-occur with rather than cause the activation (Han et al., 2025).

We inherit two design choices from this literature. First, we report SAE reconstruction fidelity before interpreting any concept ablation, and treat top-activation grids as qualitative aids that never carry a quantitative claim. Second, we treat our interventions as diagnostic tests of concept functionality rather than as a deployable inference algorithm, and we validate them against matched random controls, since an ablation that helps a weak domain establishes nothing unless a comparable arbitrary ablation does not.

## 3 Sparse Concept Diagnostics

Figure 1 gives the framework in one picture: the path from an image to a per-concept ablation effect, and the three scores that path yields for a single class-concept pair.

*Figure 1: Left, the diagnostic pipeline. An image passes through the frozen feature extractor to a $7 \times 7 \times 2048$ feature map, which is normalized and rearranged into 49 spatial tokens. The SAE encoder maps each token to a 16,384-dimensional code with at most 16 nonzero entries. Ablating concept $c$ zeroes that column across all tokens; the code is then decoded, denormalized, pooled and classified, and the change in the true-class posterior is $\delta_i(c)$. Right, the three scores for one class-concept pair $(k,c)$: three bars of per-domain mean activation $\bar{a}_d$ giving $H$, one bar of $\Delta p(y)$ under ablation giving $D$, and three bars of per-domain $|D_d|$ giving $R$. $H$ records where the concept fires and $R$ records where it matters, and these are not the same thing.*

## 3.1 Setup

Let $\mathcal{D}$ be a set of domains, of which $\mathcal{D}_s \subseteq \mathcal{D}$ are source domains and the remainder are held out. Write $M = |\mathcal{D}_s|$. Let $\mathcal{Y} = \{1, \dots, K\}$ be the class labels. Each image $x_i$ carries a class label $y_i \in \mathcal{Y}$ and a domain label $d_i \in \mathcal{D}$.

A trained vision model is decomposed into a feature extractor $f_\theta$ and a classifier head $f_{\text{cls}}$. The extractor maps an image to an intermediate feature map

$$z_i \;=\; f_\theta(x_i) \;\in\; \mathbb{R}^{C \times h \times w},$$

and the head maps a pooled feature vector to logits over the $K$ classes. Throughout, $f_\theta$ and $f_{\text{cls}}$ are **frozen**. The SAE is trained only to reconstruct intermediate feature maps and is used purely as an analysis tool; no gradient reaches the model under study.

In our main experiments $\mathcal{D}$ is the set of four PACS domains with sketch held out, so $M = 3$, and $K = 7$ for the PACS object classes (Li et al., 2017). The primary model is an ERM-trained ResNet-50 (He et al., 2016; Vapnik, 1999), for which $C = 2048$ and $h = w = 7$.

## 3.2 Sparse autoencoder concept space

The SAE operates on normalized spatial feature tokens. Let $\mu_{\text{SAE}}$ and $\sigma_{\text{SAE}}$ denote the normalizer statistics estimated during SAE training. The feature map is normalized as

$$\tilde{z}_i \;=\; \frac{z_i - \mu_{\text{SAE}}}{\sigma_{\text{SAE}}},$$

then rearranged into $T = hw$ spatial tokens, giving $\tilde{Z}_i \in \mathbb{R}^{T \times C}$ whose rows are individual tokens. An SAE encoder $E$ maps each token to a sparse code, stacked as

$$S_i \;=\; E(\tilde{Z}_i) \;\in\; \mathbb{R}_{\geq 0}^{T \times N}, \qquad \|S_i^{(t)}\|_0 \leq \kappa \;\; \forall t,$$

where $N$ is the dictionary size and $\kappa$ the per-token sparsity. In the main implementation $N = 16{,}384$ and $\kappa = 16$, so each of the $T = 49$ spatial tokens activates at most 16 concepts. The code is nonnegative, and column $c$ of $S_i$ is treated as the spatial activation map of candidate concept $c$.

The decoder $\text{Dec}$ reconstructs the normalized feature map, which is then denormalized before classification:

$$\hat{\tilde{Z}}_i \;=\; \text{Dec}(S_i), \qquad \hat{z}_i \;=\; \sigma_{\text{SAE}} \cdot \hat{\tilde{Z}}_i + \mu_{\text{SAE}},$$

with $\hat{\tilde{Z}}_i$ reshaped back to $C \times h \times w$. For ResNet models, adaptive average pooling is applied before the linear head, so the reconstructed class posterior is

$$\hat{p}_i \;=\; \mathrm{softmax}\big(f_{\text{cls}}(\mathrm{pool}(\hat{z}_i))\big) \;\in\; \Delta^{K-1}.$$

**Scope of the SAE representation.** We use SAE latents as a sparse analysis basis, not as a canonical set of ground-truth semantic concepts. SAE dictionaries are not identifiable in a strict sense: independently trained SAEs may recover different individual latents, and changes in initialization, dictionary size, sparsity level, or optimization can split, merge, or obscure features (Paulo & Belrose, 2025; Chanin et al., 2024; Makelov et al., 2024). We therefore do not interpret a latent index $c$ as an invariant object across independently trained SAEs, nor require every latent to be human-nameable or perfectly monosemantic. Our quantitative claims concern aggregate distributions and bucket masses over class-concept pairs, not the semantic interpretation of isolated latents; top-activating example grids appear only as qualitative aids, never as evidence for a claim. The purpose of the SAE is not to recover the model's unique true features but to provide a sparse, reconstructive basis in which candidate concepts can be tested for activation invariance, class support, and domain consistency.

## 3.3 Concept activation

For concept $c$, define its total activation on image $i$ by summing over spatial tokens,

$$a_i(c) \;=\; \sum_{t=1}^{T} S_i^{(t,c)},$$

and say that $c$ is **active** on image $i$ if it fires at least once, $a_i(c) > 0$. For class $k$ and domain $d$ write

$$\mathcal{I}_k = \{i : y_i = k\}, \qquad \mathcal{I}_{k,d} = \{i : y_i = k,\; d_i = d\},$$

and let $\mathcal{I}_k(c) = \{i \in \mathcal{I}_k : a_i(c) > 0\}$ and $\mathcal{I}_{k,d}(c) = \{i \in \mathcal{I}_{k,d} : a_i(c) > 0\}$ denote the corresponding **active** subsets. The support count $n(k,c) = |\mathcal{I}_k(c)|$ is used in Section 4.3 as an inclusion criterion for estimation.

All diagnostic scores below are class-conditional. The same SAE concept can therefore play different roles for different classes, and Section 5.5 shows that it frequently does.

## 3.4 Activation invariance H

The activation-invariance score $H(k,c)$ measures whether concept $c$ activates evenly across source domains for examples of class $k$. For each domain $d \in \mathcal{D}_s$, compute the mean activation over all images of that class and domain,

$$\bar{a}_d(k,c) \;=\; \frac{1}{|\mathcal{I}_{k,d}|} \sum_{i \in \mathcal{I}_{k,d}} a_i(c) \;\geq\; 0 ,$$

which yields a nonnegative domain-wise activation vector $\big(\bar{a}_d(k,c)\big)_{d \in \mathcal{D}_s}$. Normalizing by its sum gives

$$q_d(k,c) \;=\; \frac{\bar{a}_d(k,c)}{\sum_{d' \in \mathcal{D}_s} \bar{a}_{d'}(k,c)}$$

whenever the denominator is nonzero. If the denominator is zero the concept never activates for class $k$, and we set $H(k,c) = 0$. Otherwise $H(k,c)$ is the normalized entropy

$$H(k,c) \;=\; -\frac{1}{\log M} \sum_{d \in \mathcal{D}_s} q_d(k,c) \log q_d(k,c) \;\in\; [0,1].$$

High $H$ means activation is spread evenly across source domains; low $H$ means it is concentrated in a small number of them. $H$ measures where a concept appears, not whether it helps classification.

## 3.5 Ablation-based discriminative effect D

To test whether a concept functionally supports a class, we ablate it from the SAE code and measure the change in the model's probability for the ground-truth label. For concept $c$, construct a masked code $S_i^{(-c)}$ by zeroing that concept's column across all spatial locations while leaving every other activation unchanged:

$$S_i^{(-c)(t,c)} = 0 \;\; \forall t, \qquad S_i^{(-c)(t,c')} = S_i^{(t,c')} \;\; \forall t,\; \forall c' \neq c .$$

The masked code is decoded, denormalized and passed through the classifier exactly as in Section 3.2, yielding $\hat{p}_i^{(-c)}$. The per-image ablation effect is the drop in the true-class posterior,

$$\delta_i(c) \;=\; \hat{p}_{i,y_i} - \hat{p}_{i,y_i}^{(-c)} .$$

If $\delta_i(c) > 0$, removing $c$ decreases confidence in the correct class, so the concept supports the prediction. If $\delta_i(c) < 0$, removing it increases confidence, so the concept is harmful or distracting for that example.

The class-conditional discriminative effect averages this over the images of class $k$ **on which $c$ is active**,

$$D(k,c) \;=\; \frac{1}{|\mathcal{I}_k(c)|} \sum_{i \in \mathcal{I}_k(c)} \delta_i(c) .$$

Positive $D(k,c)$ means $c$ supports class $k$, negative means it hurts $k$, and near-zero means it is approximately neutral for $k$. Note that $H$ averages over all images of a class while $D$ averages only over those on which the concept fires: $H$ asks how often and how strongly a concept appears, and $D$ asks what it does when it does appear. We say "ablation-based effect" rather than "causal effect" throughout, because the intervention occurs in SAE reconstruction space and may not capture every causal dependency in the original network.

## 3.6 Discriminative consistency R

$D(k,c)$ measures average discriminative effect but not whether that effect is distributed evenly across domains. A concept may help class $k$ in photographs but not in sketches, or help in one domain and hurt in another. We therefore compute domain-specific effects on the same per-image quantity,

$$D_d(k,c) \;=\; \frac{1}{|\mathcal{I}_{k,d}(c)|} \sum_{i \in \mathcal{I}_{k,d}(c)} \delta_i(c), \qquad d \in \mathcal{D}_s .$$

A direct entropy over the signed $D_d(k,c)$ is not well defined when effects have mixed signs, since the normalized values need not be a distribution. We therefore define $R$ over effect **magnitudes**:

$$r_d(k,c) \;=\; \frac{|D_d(k,c)|}{\sum_{d' \in \mathcal{D}_s} |D_{d'}(k,c)|}, \qquad
R(k,c) \;=\; -\frac{1}{\log M} \sum_{d \in \mathcal{D}_s} r_d(k,c) \log r_d(k,c) ,$$

with $R(k,c) = 0$ when the denominator vanishes, that is, when the concept has no measured effect in any source domain.

High $R(k,c)$ means the magnitude of the concept's discriminative effect is spread across domains; low $R(k,c)$ means it is concentrated in one or two.

Because entropy is scale-insensitive, $R$ is interpreted only for pairs satisfying $|D(k,c)| > \tau_D$. A concept with tiny effects in every domain has high entropy while remaining unimportant, and we treat such pairs as neutral rather than robust. The framework uses exactly these three scores. Note that $R$ does not distinguish an effect confined to one domain from an effect whose sign varies across domains, since it is computed over magnitudes. Sign-flipping is rare in our data, affecting 376 pairs or 0.33% of the full class-concept grid, though we report that share against the grid rather than against the gated population on which $R$ is actually interpreted, and the latter is the more informative denominator.

## 3.7 Concept typology and bucket definitions

Continuous scores become categories through three thresholds: $\tau_H$ on invariance, $\tau_R$ on consistency, and $\tau_D$ on effect magnitude. A pair is **neutral** if $|D(k,c)| \leq \tau_D$, **supportive** if $D(k,c) > \tau_D$, and **harmful** if $D(k,c) < -\tau_D$. Neutral pairs are excluded before $R$ is interpreted, for the reason just given. Table 1 gives the full cross-tabulation.

Every bucket used in the remainder of the paper is named once here, with its defining predicate, and referred to by the same name and symbol thereafter. All are additionally restricted to pairs clearing the support floor of Section 4.3, which is an estimation criterion and not part of the concept definition.

| Symbol | Name | Predicate |
| --- | --- | --- |
| $S^+_{\text{hi}}$ | distributed support | $D > \tau_D \;\wedge\; R \geq \tau_R$ |
| $S^+_{\text{lo}}$ | concentrated support | $D > \tau_D \;\wedge\; R < \tau_R$ |
| $S^-_{\text{hi}}$ | distributed harm | $D < -\tau_D \;\wedge\; R \geq \tau_R$ |
| $S^-_{\text{lo}}$ | concentrated harm | $D < -\tau_D \;\wedge\; R < \tau_R$ |
| $S^+_{\text{inv}}$ | **robust support** | $D > \tau_D \;\wedge\; R \geq \tau_R \;\wedge\; H \geq \tau_H$ |
| $S^-_{\text{inv}}$ | **harmful invariant** | $D < -\tau_D \;\wedge\; R \geq \tau_R \;\wedge\; H \geq \tau_H$ |

Two points about these definitions matter for reading the tables. First, the four $S^{\pm}_{\text{hi/lo}}$ buckets are defined by $R$ alone and partition the non-neutral population, while the two $\text{inv}$ buckets additionally require $H$ and are therefore **strict subsets** of their $\text{hi}$ counterparts: $S^-_{\text{inv}} \subset S^-_{\text{hi}}$ and $S^+_{\text{inv}} \subset S^+_{\text{hi}}$. Rows of Table 3 that appear side by side are consequently not disjoint, and their contributions must not be added. Second, since consistency almost implies invariance in our data (Section 5.3), the nesting is tight: at $\tau = 0.7$ the harm buckets differ by three pairs and the support buckets by five.

*Table 1: Concept typology induced by activation invariance $H$, discriminative consistency $R$, and the sign of the discriminative effect $D$. Counts are the populated cells at $\tau_H = \tau_R = 0.7$, $\tau_D = 10^{-4}$ and support floor 30, over the 273 non-neutral class-concept pairs. The 1,269 neutral pairs are excluded, since $R$ is not interpretable without a magnitude gate. The two low-$H$, high-$R$ rows hold 8 pairs between them, because consistent effect across three domains requires nonzero activation in all three, so such pairs are rarely low-$H$ (Section 5.3).*

| $H$ | $R$ | sign of $D$ | Bucket | Pairs | Interpretation |
| --- | --- | --- | --- | --- | --- |
| High | High | Positive | $S^+_{\text{inv}}$ | 76 | Robust class-supporting concept |
| High | High | Negative | $S^-_{\text{inv}}$ | 32 | **Harmful invariant concept** |
| High | Low | Positive | in $S^+_{\text{lo}}$ | 29 | Domain-contingent supporting concept |
| High | Low | Negative | in $S^-_{\text{lo}}$ | 57 | Domain-contingent harmful concept |
| Low | High | Positive | in $S^+_{\text{hi}}$ | 5 | Domain-specific but consistently useful when active |
| Low | High | Negative | in $S^-_{\text{hi}}$ | 3 | Domain-specific but consistently harmful when active |
| Low | Low | Positive | in $S^+_{\text{lo}}$ | 25 | Local or unstable supporting cue |
| Low | Low | Negative | in $S^-_{\text{lo}}$ | 46 | Local or unstable harmful cue |

This typology is the main interpretive object in the paper. It separates concepts that are merely invariant from concepts that are invariant and useful. The harmful invariant bucket $S^-_{\text{inv}}$ is central: these are concepts that activate evenly across source domains and consistently damage the correct class, which is precisely the combination an activation-based criterion scores well and cannot correct.

**Class conflict.** Because $D$ is class-conditional, one further category is definable without reference to $H$ or $R$. A concept $c$ is **conflicting** if it supports at least one class and harms at least one other:

$$\exists\, k : D(k,c) > \tau_D \quad \text{and} \quad \exists\, k' \neq k : D(k',c) < -\tau_D ,$$

with both pairs above the support floor. No co-activity assumption is required, since $D(k,c)$ accumulates only over images of class $k$ on which $c$ actually fires, so a conflict certifies that the concept fires on images of both classes. Section 5.5 shows this category accounts for the large majority of harm in the model.

## 4 Experimental Setup

## 4.1 Dataset and model

The main experiments use PACS (Li et al., 2017), which contains four visual domains: art painting, cartoon, photo, and sketch. The task contains seven object classes. We designate the sketch domain as the target holdout, using the remaining three as source domains. Unless otherwise stated, models are trained with the DomainBed protocol (Gulrajani & Lopez-Paz, 2021), employing non-oracle checkpoint selection based on the highest average accuracy across all source domains. The primary analysis uses an ERM-trained ResNet-50 model at a single checkpoint, selected without reference to the target domain.

We analyze an ERM model deliberately, for two reasons. First, under standardized model selection ERM is competitive with specialized DG algorithms (Gulrajani & Lopez-Paz, 2021), so it is a representative rather than a weak subject. Second, our object of study is a criterion rather than a training objective. Activation invariance can be computed post hoc on any model, and the question of whether it identifies useful concepts is well posed regardless of how the model was trained. Section 7 states what this single setting does and does not establish.

## 4.2 SAE training

The SAE is trained to reconstruct the feature maps from the final layer before the classification head, which for ResNet-50 have $7 \times 7$ spatial resolution and 2048 channels. We use a Top-K SAE with a dictionary size of 16,384 and enforce top-16 sparsity per spatial token. Training proceeds for 250 epochs using the Adam optimizer with an initial learning rate of $3 \times 10^{-4}$ and a cosine decay schedule. The backbone and classifier remain frozen throughout.

The SAE dictionary is trained on source-domain features, and the diagnostic scores $H$, $D$, $R$ and their thresholds are estimated on source-domain data. Target-domain examples are used only for post-hoc evaluation of whether source-estimated concept categories explain held-out behavior. One qualification is worth stating precisely: the two scalar statistics used to normalize feature maps before the SAE encoder were estimated from a single batch spanning all four domains rather than the three source domains. This concerns a mean and a standard deviation only; no dictionary element, score or threshold is estimated with target-domain data.

*Table 2: SAE reconstruction fidelity on PACS for ERM ResNet-50, checkpoint 3300. Micro-averaged accuracy is image-weighted, macro-averaged is class-weighted; the two diverge on sketch because that domain is heavily class-imbalanced. Sketch is the held-out target domain. Evaluation covers every image of each domain.*

| Domain | Original micro | Recon. micro | Δ micro | Original macro | Recon. macro | Δ macro |
| --- | --- | --- | --- | --- | --- | --- |
| Art painting | 99.37 | 99.32 | −0.05 | 99.38 | 99.35 | −0.03 |
| Cartoon | 99.19 | 99.15 | −0.04 | 99.23 | 99.19 | −0.04 |
| Photo | 99.82 | 99.82 | 0.00 | 99.78 | 99.78 | 0.00 |
| Sketch (target) | 80.25 | 80.22 | −0.03 | 83.56 | 83.63 | **+0.07** |

## 4.3 Evaluation protocol

Scores and accuracies are computed over every image in each domain, with no train/test subsampling, no shuffling and no dropped final batch, so all reported figures are exactly reproducible. Accuracy is reported both micro-averaged and macro-averaged over classes throughout, and source and target domains are never pooled into a single average, since three of the four domains are in distribution and pooling would obscure the only number that carries the argument.

When computing statistics over class-concept pairs we restrict attention to pairs active on at least 30 images of the class in question, written $n(k,c) \geq 30$. This support floor is an inclusion criterion for estimation only, and is never used to mask pairs during a forward pass. Applying such a filter class-conditionally at inference would leak label information, since a concept that fires rarely for class $k$ but often for class $k'$ is a confusion signal, and suppressing it on $k$-labeled images deletes evidence for the competing class using the ground-truth label.

The floor exists for $R$ specifically. $R$ is an entropy over three per-domain effect estimates, and at low support it is biased toward zero rather than simply noisy: a source domain in which the concept never fires contributes an exact zero to the distribution, so a pair observed in only one domain scores $R = 0$ by construction rather than by measurement, and is indistinguishable from a genuinely domain-concentrated one. The bias is directly visible in the data. Among pairs active on 1 to 4 images, 84.6% score exactly $R = 0$ and the mean number of source domains in which any effect is observed is 1.17; by 30 to 49 images those figures are 2.6% and 2.79, and above 100 images no pair scores zero and the mean is 2.96. Low $R$ at low support is therefore a sampling artifact rather than evidence of domain-contingency, and admitting such pairs would populate the concentrated buckets with pairs that are merely unobserved. $H$ and $D$ are pooled across domains and do not suffer this, so the floor constrains the interpretability of $R$ rather than expressing a view about whether thinly supported concepts matter.

Most of the grid falls below the floor, 113,146 pairs against 1,542 above, but that figure overstates what is being set aside: 108,290 of those pairs have $n(k,c) = 0$, meaning the concept never fires for that class at all and there is nothing to exclude. **The floor therefore separates 1,542 estimable pairs from 4,856 thinly supported ones**, not from 113,146. We measure what that exclusion costs rather than argue about it. Among the excluded pairs, 282 are harmful with an effect clearing $\tau_D$, and **they carry more aggregate discriminative effect than all 138 harmful pairs above the floor combined**, 0.341 against 0.250. Ablating them recovers 7.2% of target-domain error, while ablating the 138 above-floor harmful pairs recovers 45.6% (Table 3). The 45 excluded supportive pairs are null, costing 0.05 points. The excluded population is therefore functionally real but roughly six times less consequential than a smaller, better-supported population carrying less effect. This is the same dissociation between aggregate effect and consequence that Section 5.4 documents for the $R = 0$ majority, observed here on a population where the effect is genuinely concentrated rather than thinly spread.

Our claims are scoped accordingly: they characterize the pairs whose per-domain effects can be estimated and are silent about those they cannot. Lowering the floor is not the remedy, since it would leave $R$ uninterpretable for the pairs it admitted; raising per-concept exposure would be. Excluded pairs are left intact by every mask, and Figure 2 shows the full population.

*Figure 2: Every class-concept population named in the paper, in one place. Each level splits on exactly one criterion, applied where it is meaningful: support, then effect magnitude, then sign, then consistency. The full $7 \times 16{,}384$ grid divides by support into the 113,146 pairs below the floor and the 1,542 above it; those divide by magnitude into neutral and non-neutral, the non-neutral by sign, and each sign by $R$ and $H$ into the buckets of Table 1. Rows at equal depth are disjoint, and $S^{\pm}_{\text{inv}}$ is a subset of $S^{\pm}_{\text{hi}}$ rather than a sibling of it. Counts are on a log axis because the populations span 32 to 114,688. Only counts are shown. Aggregate discriminative effect is reported in Table 3, where the accuracy change, the share of target error recovered and the matched control appear beside it, because in this model aggregate effect and consequence come apart sharply: the 111,530 $R = 0$ pairs and the 282 harmful pairs below the support floor each carry more effect, several-fold, than the buckets that outperform them.*

## 4.4 Thresholds and sensitivity

The main typology uses thresholds $\tau_H$, $\tau_R$ and $\tau_D$. We set $\tau_H = \tau_R = 0.7$ and $\tau_D = 10^{-4}$.

The choice of 0.7 is not arbitrary. With three source domains, the largest normalized entropy attainable while only two domains are active is $\log 2 / \log 3 = 0.63$, achieved by an even split over those two. Any threshold above 0.63 therefore guarantees that a pair clearing it has nonzero activation, or nonzero effect, in **all three** source domains. We take 0.7 as the lowest such value, because it retains that guarantee while keeping bucket populations as large as possible. Stricter thresholds keep the guarantee too, but shrink the central harmful invariant category: at 0.9 it holds six pairs, too few to carry a claim. Figure 8 sweeps $\tau_H = \tau_R$ continuously from 0.50 to 0.95 and shows the qualitative conclusion holding at every value, so it does not depend on this particular choice.

$\tau_D$ has no comparable principled anchor. It is a magnitude gate, and $10^{-4}$ sits at the 82nd percentile of $|D|$ among pairs with any measured effect, inside a smooth region of that distribution rather than at a natural break, so it cannot be justified from the data's shape and we do not attempt to. What can be shown is that the conclusions do not turn on it: sweeping $\tau_D$ over $\{10^{-5}, 10^{-4}, 10^{-3}\}$ changes the non-neutral population from 658 pairs to 273 to 70, while the ratio between the two fractions in Section 5.3's asymmetry moves only from 2.04 to 2.01 to 1.80.

The support floor of Section 4.3 is a third free parameter, and unlike $\tau_D$ it is not a nuisance one. Swept over $\{10, 20, 30, 50, 100\}$, the fraction of invariant supportive pairs whose effect is also consistent runs from 68.4% to 83.3%, and the harmful fraction from 27.4% to 63.6%. The **direction** of the asymmetry is unchanged at every floor, but the gap narrows as the floor tightens. The mechanism is the one Section 4.3 describes: thin support depresses $R$, and harmful pairs are systematically thinner than supportive ones, because a harmful pair is typically a concept belonging to some other class that fires on a class it does not describe. Part of the gap at permissive floors is therefore attributable to support rather than to harm. We report the asymmetry at a floor of 30 with explicit denominators throughout, and state plainly that its magnitude, though not its direction, depends on that choice.

## 5 Results

Throughout this section we report micro-averaged (image-weighted) and macro-averaged (class-weighted) accuracy separately. The two differ materially on the target domain, where PACS sketch is heavily imbalanced: 772 dog images against 80 house images, and dog is simultaneously the largest and the weakest class. Reporting a single unqualified accuracy for sketch would conceal a discrepancy of more than three percentage points. All diagnostic scores are estimated on the three source domains only; sketch is used exclusively to evaluate whether source-estimated categories explain held-out behavior.

## 5.1 SAE reconstructions preserve classifier behavior

Before using SAE latents for analysis we verify that the reconstruction preserves the classifier's behavior, since a poor reconstruction would render every downstream ablation an artifact of the autoencoder rather than a perturbation of the model. Table 2 reports accuracy for the original model and for the same classifier applied to SAE-reconstructed features, evaluated on every image of each domain.

Reconstruction changes accuracy by at most 0.07 percentage points on any domain and either average, including the weakest domain. The largest movement is a 0.07-point increase in sketch macro accuracy, which is well within the granularity of a 3,929-image domain whose smallest class has 80 members. Concept ablations reported below are therefore interpretable as perturbations of the model's own representation, subject to the caveat of Section 3.2 that they are ablations in reconstruction space rather than causal interventions in the original network.

## 5.2 Activation invariance is uninformative about usefulness

Many domain generalization algorithms rest on a common premise: an attribute that persists across the source domains is likely to be causal and to persist on unseen target domains, whereas an attribute confined to a few domains is likely a spurious correlation (Arjovsky et al., 2019). Under this view, suppressing domain-specific features improves out-of-distribution generalization by forcing the model to rely on the causal, invariant ones.

We find this premise necessary but far from sufficient. Activation invariance $H(k,c)$ measures whether a concept fires uniformly across domains. Discriminative effect $D(k,c)$ measures the signed change in the model's confidence in the true class when the concept is ablated. If invariance alone were sufficient, high-$H$ pairs would concentrate at positive $D$. Instead the high-$H$ region spans positive, near-zero and negative $D$ (Figure 3).

Quantitatively, of the 1,542 class-concept pairs that clear the support floor, 1,269 (82.3%) are discriminatively neutral, 135 (8.8%) support the correct class and 138 (8.9%) hurt it. Restricting to the invariant population barely changes this: among the 1,173 pairs with $H \geq 0.7$, 83.5% are neutral, 9.0% supportive and 7.6% harmful. Conditioning on invariance moves the probability that a pair is harmful from 13.3% (49/369) among low-$H$ pairs to 7.6% (89/1,173) among high-$H$ pairs, a relative reduction of about 40% that is unlikely to be chance (Fisher exact, $p = 0.002$). But it moves the probability that a pair is supportive from 8.1% to 9.0%, which is no change at all. Activation invariance is weakly informative about the absence of harm and uninformative about the presence of usefulness, and five-sixths of what an invariance criterion would preserve does not separate classes at all.

The three regimes have distinct interpretations, and each is visually identifiable. A **neutral** concept (Figure 4) responds to the feet of four-legged animals and activates across every domain, since feet are feet whether painted, drawn or photographed, but it fires on dogs, horses and elephants alike, so it contributes nothing to discriminating among them and ablating it changes the class posterior negligibly for any of them. Stability is not usefulness. A **supportive** concept (Figure 5) is the canonical case that invariance-based reasoning implicitly assumes, being both invariant and genuine evidence for the class it fires on. A **harmful** concept (Figure 6) is domain-invariant yet actively displaces probability from the correct class. Concept 1637 fires on the chest and forelimb region of four-legged mammals in every domain, and because the model has learned that region as evidence for horse, it argues for horse on dogs as well, costing the dog class $9.50 \times 10^{-3}$ of posterior. It is examined in Section 5.5, where it proves to be an instance of the mechanism behind most harm in this model rather than an isolated curiosity.

Class-conditionality is what makes the third regime visible at all. A single global score for that chest concept would average its positive effect on one class against its negative effect on another and report approximately zero, filing a genuine failure mechanism into the neutral majority.

*Figure 3: Activation invariance does not imply discriminative usefulness. Each point is one of the 1,542 class-concept pairs above the support floor, plotting $H(k,c)$ against $D(k,c)$ on a symmetric-log axis that is linear within $\pm\tau_D$, coloured by discriminative regime, with the neutral band shaded and $\tau_H$ marked. High-$H$ pairs span positive, near-zero and negative $D$ rather than concentrating at positive $D$, contradicting the assumption that cross-domain invariance alone yields useful features.*

*Figure 4: The neutral regime. A concept responding to the feet of four-legged animals, shown as top-activating patches across domains: high $H$ with $D \approx 0$, firing on dog, horse and elephant alike and therefore separating none of them. Figures 4 to 6 present one concept from each discriminative regime and are read together; they are shown separately because their source grids differ in aspect ratio. All scores are estimated on the three source domains, and any sketch panel is illustrative only, since sketch is the held-out target and contributes to no score.*

*Figure 5: The supportive regime. Concept 4501 on person: $H = 0.999$, $R = 0.985$, $D = +2.08 \times 10^{-2}$ over 1,283 active images. This is a member of the 76-pair robust-support bucket that alone reaches 94% on the held-out domain, and it is the case invariance-based reasoning implicitly assumes.*

*Figure 6: The harmful regime. Concept 1637 on dog: $H = 0.989$, $R = 0.983$, $D = -9.50 \times 10^{-3}$, a member of the 32-pair harmful invariant bucket. The same latent is robust support for horse ($H = 0.986$, $R = 0.987$, $D = +2.34 \times 10^{-3}$, 722 active images against dog's 63), which is the mechanism of Section 5.5. Where both classes are shown, the horse panels are where the concept lives and the dog panels are where it misfires.*

## 5.3 Discriminative consistency matters beyond activation invariance

A concept can fire evenly across all domains and still only matter in one of them. Activation invariance $H$ cannot detect this, because it only records where a concept appears. The consistency score $R$ records where a concept's discriminative effect appears, and these turn out to be different things.

Figure 9 shows a real instance: concept 11005 evaluated on elephant, which responds to the animal's trunk. It is active on 848 of the 914 elephant images across the three source domains, and its strongest activations land on trunks in art painting, cartoon and photograph alike, so the same detector is doing the same thing in every domain. Mean activation is correspondingly high in all three, giving $H = 0.926$, and a criterion based on activation statistics would score it as a textbook invariant feature.

Its per-domain effect magnitudes are not flat at all. Most of the concept's discriminative influence sits in art painting, and $R = 0.626$ falls below $\tau_R$. The asymmetry is not an artifact of exposure: **cartoon has the most active images of any source domain, 398 against art painting's 249, and carries almost none of the effect.** More exposure yields less consequence. Activation invariance sees an invariant concept, and consistency sees a domain specialist. The distinction is between where a concept fires and where it matters, and 86 pairs in this model are of that kind, being invariant, non-neutral, and inconsistent.

We do not explain the asymmetry, and the figure is not offered as evidence for a mechanism. What the grids establish is narrower and worth stating precisely: because the concept lands on the same structure in all three domains, the effect asymmetry cannot be attributed to the detector behaving differently under different rendering styles. Why the classifier declines to use it in photographs is a separate question the diagnostic does not answer. One possibility, consistent with Section 6.2, is that texture and colour carry the elephant decision in photographs so the trunk is redundant there, while art painting strips texture and makes shape load-bearing. We record that as a conjecture, not a finding.

The two scores are not independent. Both are entropies over the three source domains, so a pair that clears $\tau_R = 0.7$ must have nonzero effect in every source domain: an even split across only two domains gives $\log 2 / \log 3 = 0.63$, which falls below the threshold. The same bound applies to $H$. High consistency therefore implies nonzero activation in every source domain, though not, in itself, that the activation is evenly enough spread to clear $\tau_H$, so consistent pairs are rarely low-$H$ without being required to be high-$H$. The data bear this out: only 8 of the 273 pairs with non-negligible effect are low-$H$ but high-$R$ (Figure 7). $R$ acts largely as a filter applied inside the invariant population, not as a rival measure of invariance.

But it is a filter that removes a great deal. Of the 194 invariant pairs with non-negligible effect, only 108, or **55.7%**, also clear $\tau_R$. Knowing that a concept is activation-invariant tells us little about whether its effect is consistent. Across all 1,173 high-$H$ pairs the figure is 50.7%, but 83.5% of that population is discriminatively neutral and $R$ is not interpretable there for the reason given below, so the gated figure is the one that carries the claim.

The central observation of this section emerges when the gated invariant population is split by the sign of $D$. Among invariant pairs that support the correct class, 72.4% (76/105) are also consistent. Among invariant pairs that hurt the correct class, only 36.0% (32/89) are, an odds ratio of 4.7 that is not attributable to chance (Fisher exact, $p = 4 \times 10^{-7}$). **Invariant support is usually spread across domains; invariant harm is usually concentrated in one or two.** The asymmetry widens as the invariance threshold tightens, reaching 65.2% against 27.3% at $\tau = 0.8$ and 49.4% against 15.4% at $\tau = 0.9$. Figure 8 sweeps $\tau$ continuously: the supportive fraction exceeds the harmful one at every value, and the gap widens across the range in which both populations stay large enough to estimate. The qualitative conclusion therefore does not depend on a particular choice of $\tau$; its magnitude, though not its direction, depends on the support floor, and Section 4.4 reports that sweep.

This matters for how the harmful invariant category should be read. Most of those concepts are not harmful everywhere. They fire in every domain and do their damage in a subset. A criterion that equalizes activation statistics cannot address them, because their activations are already balanced, and that balance is what qualifies them as invariant in the first place. The asymmetry that makes them harmful is invisible to any activation-level criterion.

The neutral panel of Figure 7 makes a separate point: 38.4% of near-zero-$D$ pairs land in the high-$H$, high-$R$ cell. Entropy is scale-insensitive, so a concept with negligible influence spread evenly across domains scores as perfectly consistent. Consistency of a non-effect is meaningless, which is why $R$ is interpreted only for pairs satisfying $|D(k,c)| > \tau_D$.

One limit bounds what this section claims about low $R$. A low value admits two readings: the effect may be genuinely confined to one domain, or merely skewed across three, since a split of $(0.8, 0.15, 0.05)$ gives $R = 0.56$ with all domains active. We therefore call low-$R$ pairs domain-contingent rather than spurious. $R$ measures where an effect concentrates and says nothing about whether the underlying visual cue is causally unrelated to the label.

**Reconciling this with Section 5.2.** These two sections are easily read as opposing each other, and the reconciliation is the argument of the paper. Invariance reduces the rate at which concepts are harmful, by about 40%. But the harm that survives the invariance filter is, by construction, harm that no activation-level criterion can detect, and Section 5.4 shows that within it the consistent half is several times more damaging per pair and per unit of effect mass than the concentrated half. A criterion can be genuinely protective on average and still leave the most expensive failures untouched.

*Figure 7: High activation invariance does not imply a consistent discriminative effect. Pairs are split by the sign and magnitude of $D(k,c)$, then cross-tabulated by $H$ and $R$ at threshold 0.7. Each panel is shaded in its own regime hue and shows counts and shares of that regime. Among invariant pairs that support the correct class, 72.4% are also consistent, the 76 of the supportive panel's top-right cell. Among invariant pairs that hurt it, only 36.0% are, the 32 in the harmful panel's top-right cell, which is the harmful invariant bucket $S^-_{\text{inv}}$. The neutral panel shows why $R$ needs a magnitude gate, since 38.4% of pairs with negligible effect still score as perfectly consistent. Pairs active on fewer than 30 images for the class are excluded throughout.*

*Figure 8: The support-harm asymmetry is not an artifact of the invariance threshold $\tau_H = \tau_R$. Among invariant pairs with non-negligible effect, the figure plots the fraction whose effect is also consistent, swept over that threshold from 0.50 to 0.95. The support floor is held at 30 throughout, and its separate effect is reported in Section 4.4. Both curves fall as the threshold tightens, but the harmful curve falls faster, so the gap between them widens rather than closing. The shaded region below $\log 2 / \log 3 = 0.63$ is where clearing $\tau$ no longer implies activation in all three source domains, so the buckets change meaning there. The chosen value 0.7 is the lowest round threshold above it. This figure is the threshold-sensitivity analysis promised in Section 4.4.*

*Figure 9: A high-$H$, low-$R$ pair, showing invariant activation with domain-contingent effect. Concept 11005 evaluated on elephant, which responds to the trunk: $H = 0.926$, $R = 0.626$, $D = +1.01 \times 10^{-4}$, active on 848 of the 914 elephant images across the three source domains. **Top:** the eight strongest activations in each source domain, unselected. The concept lands on trunks in all three, so these grids act as a control: whatever explains the effect asymmetry below, it is not that the detector behaves differently across rendering styles. **Bottom, left to right:** mean activation per domain (180.2, 62.1, 136.7), high in all three, giving $H = 0.926$; active-image count per domain (249, 398, 201), showing cartoon has the most exposure of any domain; and per-domain effect magnitude $|D_d|$, concentrated in art painting, giving $R = 0.626$, below $\tau_R$. The middle panel forecloses the exposure objection, since the concept fires on more cartoons than art paintings and moves the class posterior in almost none of them. All quantities are estimated on source domains only.*

## 5.4 The typology corresponds to functional prediction behavior

We next test whether the diagnostic categories describe real model behavior by ablating each bucket in SAE code space and re-evaluating. For an image with true label $y$, the concepts masked are those paired with $y$ in the bucket. These are oracle interventions: they use the ground-truth label and are not deployable test-time methods. Their role is that of any controlled ablation. If removing a category changes prediction behavior in the direction the diagnostic predicts, the category captures something functional. Table 3 and Figure 10 report the results.

| | Source domains (macro) | | | Target: sketch | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intervention | Art | Cartoon | Photo | macro | Δ macro | micro | Δ micro | Errors rec. | per unit Σ\|D\| | Control Δ |
| Baseline (SAE reconstruction) | 99.35 | 99.19 | 99.78 | 83.63 | | 80.22 | | | | |
| Mask all harmful (138 pairs, Σ\|D\| 0.250) | 99.75 | 99.84 | 99.86 | 90.87 | +7.24 | 89.23 | +9.01 | 45.6% | 36.1 | −20.98 |
| Mask $S^-_{\text{hi}}$, distributed harm (35, 0.087) | 99.49 | 99.44 | 99.86 | 87.32 | +3.69 | 85.19 | +4.97 | 25.1% | **57.2** | −3.12 |
| ⤷ Mask $S^-_{\text{inv}}$, harmful invariant (32, 0.083) | 99.49 | 99.41 | 99.86 | 86.96 | +3.33 | 84.70 | +4.48 | 22.6% | **54.3** | −2.10 |
| Mask $S^-_{\text{lo}}$, concentrated harm (103, 0.163) | 99.63 | 99.69 | 99.78 | 86.17 | +2.54 | 82.87 | +2.65 | 13.4% | 16.3 | −5.40 |
| *Mask harmful pairs below the support floor (282, 0.341)* | 99.48 | 99.47 | 99.78 | 85.04 | +1.41 | 81.65 | +1.43 | 7.2% | 4.2 | |
| Mask all $R = 0$ pairs (111,530, 0.069) | 99.35 | 99.19 | 99.78 | 83.71 | +0.08 | 80.33 | +0.11 | 0.6% | 1.6 | |
| Mask $S^+_{\text{lo}}$, concentrated support (54, 0.025) | 99.29 | 99.12 | 99.78 | 82.98 | −0.65 | 79.56 | −0.66 | | 26.6 | −1.07 |
| Mask $S^+_{\text{hi}}$, distributed support (81, 0.237) | 74.39 | 80.83 | 77.45 | 44.55 | −39.08 | 39.17 | −41.05 | | **173.1** | +1.06 |
| Mask all supportive (135, 0.262) | 56.91 | 63.11 | 59.72 | 23.63 | −60.00 | 18.53 | −61.69 | | 235.5 | +2.50 |

*Table 3: Concept interventions, ordered by target-domain effect. All masks are class-conditional and use the true label. Δ is against the SAE-reconstruction baseline. **"Errors rec."** is the share of the baseline's 19.78-point target-domain error that the intervention removes, computed on micro-averaged accuracy. **"per unit Σ|D|"** is $|\Delta\,\text{micro}|$ divided by the bucket's aggregate effect mass, so the two columns are consistent with each other and with the Δ micro column beside them. **"Control Δ"** is the mean change on sketch macro under three size-matched random masks binned by the target's $|D|$ quantiles; its composition is constrained by the eligible pool, as discussed below, and two rows have no control because their populations are defined outside the typology. The indented row is a subset of the row above it, $S^-_{\text{inv}} \subset S^-_{\text{hi}}$ at 32 of 35 pairs, and their error shares must not be added. Disjoint rows are not additive either, but in the opposite direction: masking two buckets together recovers more than the sum of their separate recoveries. Source and target accuracy are reported separately rather than pooled, since three of the four domains are in distribution.*

**Validity of the design.** Two things must be established before any row can be read as a result. First, the direction of the sign effect is **tautological**: $D$ is defined as the drop in the true-class posterior under ablation, so masking negative-$D$ pairs must raise that posterior. Only the magnitude, the asymmetry between source and target domains, and the margin over a matched control carry information.

We therefore pair each signed bucket with a size-matched random control drawn per class from the support-floored pairs the target did not select, binned by the target's own $|D|$ quantiles and averaged over three seeds. The matching is necessarily partial and we state its limit precisely. For a target drawn entirely from above $\tau_D$, the eligible pool contains only 135 supportive pairs against 1,269 neutral ones, so the upper $|D|$ bins cannot be filled from like-for-like candidates and the shortfall is drawn from the closest available magnitudes. The control is consequently a mixture whose composition is dictated by the pool, and the sign of each control in Table 3 follows from that composition rather than from anything about the diagnostic. **We therefore do not rest the specificity claim on the matched controls alone.** The two rows that no pool-composition objection touches are the $R = 0$ row and the below-floor row: both carry more aggregate effect than buckets that beat them several-fold, and both are ablations of populations selected without reference to any bucket. Masking the 138 harmful pairs raises sketch accuracy by 7.24 points macro while its control lowers it by 20.98, a margin of 28 points in the opposite direction, but it is the $R = 0$ and below-floor rows that establish the result is not a generic consequence of ablating effect-bearing pairs.

Second, the gains are not dictionary denoising, and aggregate effect is not a proxy for consequence. The first of those two rows is the $R = 0$ population: 111,530 of the 114,688 pairs, of which 109,974 have exactly zero measured effect and 108,290 never fire for that class at all. We label the row by its predicate rather than calling these concepts inert, since Section 4.3 shows that $R = 0$ at low support often reflects absence of observation rather than absence of effect. Masking all of them moves sketch by 0.08 points macro, recovering 0.6% of the error, despite their aggregate $|D|$ of 0.069 exceeding that of the 35-pair distributed-harm bucket at 0.087. The second is the below-floor row, which makes the same point where the effect is genuinely concentrated rather than thinly spread: 282 harmful pairs below the support floor carry 0.341, more than all 138 harmful pairs above it, and recover 7.2% of target error against their 45.6%. **The two rows are disjoint by construction**, since the below-floor bucket requires $R \neq 0$, so their error shares do not overlap.

Two populations therefore out-mass the buckets that beat them, by factors of forty and six. Aggregate $|D|$ measures how much effect a population contains, not how much the model's decisions rest on it, and it should not be used as an importance measure. We return to this in Section 7. One clarification on the below-floor row: Section 4.3 rules out using the support count as a selection criterion for masking, because a class-conditional count filter suppresses concepts that are rare for the true class and thereby deletes competing-class evidence. That row does not do this. It masks a population defined by the sign and magnitude of $D$, restricted to the below-floor region so as to measure what the floor excludes, and like every other row in this table it is an oracle diagnostic rather than a procedure we propose. Finally, every intervention leaves the three source domains within one point of baseline while moving sketch by up to 60, so the diagnostic is estimated only on source domains yet its consequences appear almost entirely in the domain it never saw.

**Finding 1: harm that is invariant and consistent is disproportionately expensive.** The 32 pairs of $S^-_{\text{inv}}$, which are invariant, consistent in their harm, and constitute 0.028% of the grid and 0.50% of the pairs that fire at all, account for **22.6% of the model's target-domain error**, measured on micro-averaged accuracy. The slightly larger $S^-_{\text{hi}}$, which drops the invariance requirement and adds three low-$H$ pairs, accounts for 25.1%. Because their activation statistics are already balanced across source domains, a criterion built on activation alignment has no gradient to apply to them, and that much follows from the definition of invariance rather than from any measurement. What the interventions add is the price.

**Finding 2: consistency, not invariance, is what separates the expensive harm from the cheap.** The comparison that carries this is $S^-_{\text{hi}}$ against $S^-_{\text{lo}}$, both harmful, both drawn from the same invariance-eligible population, differing only in $R$. Per pair, distributed harm is 5.5 times more damaging than concentrated harm, at 0.72% against 0.13% of target error each. Because the two buckets carry different effect mass, 0.087 against 0.163, that could in principle be a magnitude effect wearing a consistency label, so a mass normalization is required. Distributed harm recovers 57.2 points of micro accuracy per unit of effect mass against concentrated harm's 16.3, a factor of **3.5**, or 2.7 on macro. The supportive side agrees more strongly, at 173.1 against 26.6, a factor of **6.5**, or 6.3 on macro. Two cautions apply. **This is a normalization, not a mass-matched intervention, and we do not claim it as one**, since matching the distribution of per-pair magnitudes rather than their sum would be needed to settle it, and Section 7 states this as an open limitation. And the contrast is on the consistency axis: we do not separately ablate the 49 harmful pairs that fail the invariance threshold, so we make no per-unit claim about harm that invariance screens out.

**Finding 3: distributed rather than concentrated support is what the model runs on.** Masking the 81 pairs of $S^+_{\text{hi}}$ costs 39 points macro, while masking the 54 pairs of $S^+_{\text{lo}}$ costs 0.65 points, which falls within the range spanned by its own matched random control at −1.07. Domain-contingent support is close to redundant, a conclusion Section 5.7 reaches independently from the opposite direction.

**A note on additivity.** The signed sub-buckets do not sum to their parent. $S^-_{\text{hi}}$ and $S^-_{\text{lo}}$ are disjoint and recover 25.1% and 13.4% of target error, while masking all 138 harmful pairs together recovers 45.6%, more than the 38.5% their parts suggest. The supportive side behaves the same way, at −41.7 points against −61.7 when combined. Removing concepts jointly is more consequential than removing them separately, which is expected if the model's evidence for a class is partly redundant, since a surviving harmful pair can continue to carry the wrong vote once its partner is gone. We report this rather than interpret it further, and note that it means bucket-wise error shares are lower bounds on their joint effect.

*Figure 10: Target-domain effect of each intervention in Table 3, plotted as sketch macro Δ on horizontal diverging bars, each paired with the mean Δ of its size- and |D|-matched random control as a ghost bar, averaged over three seeds. Every harmful bucket separates from its control, and the control for the full harmful set moves in the opposite direction: +7.24 against −20.98.*

## 5.5 Harm is largely misdirected support

The preceding sections establish that harmful concepts exist, that invariance does not remove them, and that they are expensive. This section asks why a trained model contains them at all, and the answer changes what remediation would mean.

Because $D$ is class-conditional, the same latent can support one class and hurt another; Section 3.7 calls such a concept **conflicting**. Of the 533 concepts with at least one class above the support floor, 139 are non-neutral for at least one class, and 56 of those (40.3%) are conflicting. More tellingly, **108 of the 138 harmful class-concept pairs, or 78.3%, involve a concept that supports some other class.**

Negative discriminative effect is therefore, four times out of five, not a spurious or defective feature. It is a genuine feature attached to the wrong class.

The archetype is concept 1637, which responds to the chest and forelimb region of four-legged mammals and appears in Figure 6. For horse it is robust support, with $H = 0.986$, $R = 0.987$ and $D = +2.34 \times 10^{-3}$ over 722 active images, so the model has learned this region as evidence for horse. For dog the same latent has $H = 0.989$, $R = 0.983$ and $D = -9.50 \times 10^{-3}$: it fires on dogs too, since the underlying shape is similar and especially so in the flatter, lower-texture renderings of cartoons and paintings, but the model reads the pattern as horse-evidence and displaces probability away from the true class. Both pairs are invariant and consistent, so **the dog pair is a member of the 32-pair harmful invariant bucket** of Section 5.4, which makes the archetype an instance of the paper's central category rather than an illustration adjacent to it. Concept 2979 is the same chest-and-forelimb story at three times the magnitude, at $D = +2.92 \times 10^{-2}$ for horse against $-2.37 \times 10^{-2}$ for dog, which is what makes this a pattern rather than an anecdote. Dog is also the model's weakest class on the target domain at 48.06%, and it is the class most harmed by conflicting concepts.

Nothing about the detector is defective. It detects what it detects, reliably, in every domain. The defect lies in the mapping from that detector to a class decision. The mechanism is shared visual structure that the representation never separated, disambiguated in distribution by cues that weaken under shift, after which the shared detector still fires and votes for the wrong class.

Ranking conflicting concepts by the contrast $\max_k D(k,c) - \min_k D(k,c)$ isolates the clearest individual cases: concept 52 supports giraffe while harming person, concept 3128 supports dog while harming elephant, and concept 2979 supports horse while harming dog.

Figure 11 examines the first, which is the highest-contrast conflict in the model and the most legible instance of the mechanism. Concept 52 responds to the patterned skin of a giraffe's neck. On people it fires where the same visual signature recurs, in the regular blocked pattern of a tie and in similar checked regions, and the model reads that pattern as giraffe evidence at the expense of the person class. Neither the detector nor the images it selects are arbitrary. It reliably finds a texture that happens to occur on two classes, and only the read-out treats that texture as decisive. This is the chest-and-forelimb story in a second, visually distinct form, which is what distinguishes it from an anecdote.

The consequence for remediation is that "remove the spurious feature" is the wrong prescription, because the feature is not spurious and removing it would cost the class it legitimately serves. The problem is **class isolation**: the same evidence must be read differently depending on what else is present. That is a property of the read-out rather than of the feature set, and it suggests that objectives targeting class separation in concept space may be more appropriate than objectives targeting domain alignment.

The per-class distribution of this burden is descriptive rather than predictive, and we report it as such in Table 4. Rows are ordered by target-domain accuracy rather than alphabetically, and the harmful rate is given alongside the raw count, since floor populations range from 144 to 326 pairs and the counts are not comparable across classes without it. The weakest class on the target domain, dog at 48.06% micro, carries both the most harmful pairs and the most conflicting concepts acting against it, while the strongest, guitar at 97.37%, carries almost none. But person carries 32 harmful pairs and 26 conflicting concepts and still reaches 95.62%, so conflict burden alone does not predict per-class failure. The resolution is visible in the collapse behavior of the degenerate masks in Section 5.4: surviving predictions consistently pile onto person, which is the model's default class and therefore wins ties. Class-level robustness depends on at least two quantities, the conflict burden a class carries and whether it wins or loses the resulting ties, and a one-dimensional per-class risk score cannot express their interaction. With seven classes we do not attempt a correlation.

*Table 4: Per-class diagnostic counts at $\tau_D = 10^{-4}$ with support floor 30, ordered by target accuracy. The final column counts concepts that harm this class while supporting another. The neutral column is omitted, being determined by the other three. Reading down the accuracy ordering shows the non-monotonicity directly: `dog` is worst with the highest harmful rate, but `person` is second-best while carrying the second-highest harmful count, and `giraffe` at 76.89% is not explained by either quantity.*

| Class | Sketch acc. (micro) | Pairs above floor | Supportive | Harmful | Harmful rate | Harmed by conflicting concept |
| --- | --- | --- | --- | --- | --- | --- |
| dog | 48.06 | 286 | 33 | 50 | 17.5% | 34 |
| giraffe | 76.89 | 185 | 15 | 10 | 5.4% | 10 |
| horse | 83.21 | 243 | 24 | 24 | 9.9% | 18 |
| house | 88.75 | 144 | 9 | 3 | 2.1% | 3 |
| elephant | 95.54 | 208 | 23 | 16 | 7.7% | 15 |
| person | 95.62 | 326 | 19 | 32 | 9.8% | 26 |
| guitar | 97.37 | 150 | 12 | 3 | 2.0% | 2 |

*Figure 11: Harm as misdirected support. Concept 52 is the highest-contrast conflicting concept in the model. On giraffe it is close to an ideal robust-support pair, with $H = 0.9998$, $R = 0.992$ and $D = +6.38 \times 10^{-2}$ over 813 active images, while on person it has $H = 0.602$, $R = 0.105$ and $D = -6.72 \times 10^{-3}$ over 82. The contrast between those two rows of scores is the point: the concept is invariant, consistent and strongly useful where it belongs, and none of those things where it misfires. **Top row:** its strongest activations on giraffe images, which land on the patterned skin of the neck. **Bottom row:** its strongest activations on person images, which land on ties and similarly blocked or checked regions carrying the same visual signature. Both rows are the top four activations for that class, unselected, pooled over the three source domains. The detector is neither defective nor firing arbitrarily, since it reliably finds a texture that occurs on two classes, and the failure lies in a read-out that treats that texture as evidence for giraffe. A single concept is shown rather than the top three, because the quantitative claim is carried by the 56 conflicting concepts and the 78.3% figure in the text, and this figure exists only to make the mechanism legible.*

## 5.6 Domain-contingent effects concentrate in the stylized source domains

The mechanism of Section 5.5 raises a further question about the low-$R$ population: when an effect is concentrated, where is it concentrated? Because $R$ is an entropy over per-domain effects, for every low-$R$ pair we can read this off directly as $\arg\max_d |D_d(k,c)|$, with no intervention required.

A raw table of those counts would be confounded, since a domain in which all effects are systematically larger wins the argmax regardless of concentration. Mean $|D_d|$ over non-neutral pairs is indeed uneven, at $2.13 \times 10^{-3}$ for art painting, $1.92 \times 10^{-3}$ for cartoon and $1.15 \times 10^{-3}$ for photo, and active-image exposure differs as well, at 26.6k, 35.4k and 19.9k images respectively. We therefore compare the low-$R$ distribution against the same distribution for high-$R$ pairs, which are by construction not concentrated and so provide the appropriate null (Table 5).

| | art painting | cartoon | photo |
| --- | --- | --- | --- |
| Concentrated support $S^+_{\text{lo}}$ (n = 54) | 48.1% | 46.3% | 5.6% |
| Distributed support $S^+_{\text{hi}}$, null (n = 81) | 58.0% | 19.8% | 22.2% |
| *enrichment* | 0.83× | **2.34×** | **0.25×** |
| Concentrated harm $S^-_{\text{lo}}$ (n = 103) | 37.9% | 55.3% | 6.8% |
| Distributed harm $S^-_{\text{hi}}$, null (n = 35) | 22.9% | 54.3% | 22.9% |
| *enrichment* | **1.66×** | 1.02× | **0.30×** |

*Table 5: Which source domain carries the largest per-domain effect, $\arg\max_d |D_d(k,c)|$, as a share of each row. Low-$R$ (concentrated) buckets are compared against the corresponding high-$R$ bucket, which is not concentrated by construction and therefore supplies the null. Enrichment is the ratio of the two rows above it. Bold marks departures from the null of more than 1.5× in either direction. The null rows are small, at n = 81 and n = 35, so these ratios should be read as indicative rather than precise.*

Three readings survive the null comparison. Concentrated support is enriched 2.3-fold in cartoon, at 46.3% against 19.8%. Concentrated harm is enriched 1.7-fold in art painting, at 37.9% against 22.9%. And photo is depleted three- to four-fold in both, at 5.6% and 6.8% against a null of roughly 22%, so effects that concentrate almost never concentrate in photographs. Cartoon's dominance of the harmful column at 55.3% is by contrast **not** a concentration effect, since the null is 54.3%. Cartoon supplies most harmful effect in this model whether or not that effect is domain-contingent, and only the supportive column shows genuine cartoon-specific concentration. Reading the raw counts without the null would have produced the opposite conclusion.

The pattern is that domain-contingent effect resides in the stylized source domains rather than in photographs. Since the held-out target is itself a stylized, texture-poor domain, the concepts carrying domain-contingent effect are the style-sensitive ones, which is consistent with their influence failing to transfer to a different style. Concept 11005 of Figure 9 is one instance of the depletion side of this: it is concentrated support whose effect sits 77% in art painting and essentially none in photographs, despite firing on photographs as readily as anywhere else. We stress that a single concept illustrates the pattern without evidencing it, since an enrichment ratio is a property of a population and the claim rests on Table 5. We record the pattern as a coherent reading rather than a demonstrated mechanism: establishing it would require per-domain interventions we do not perform, and part of photo's depletion is mechanical given its smaller mean effect and lower exposure.

## 5.7 Retaining 0.07% of the grid exceeds the model's own accuracy

The interventions so far remove concepts. Inverting the mask, by retaining a single bucket and ablating everything else, produces the result reported in Table 6.

| Keep only | Pairs kept | Sketch macro | Sketch micro |
| --- | --- | --- | --- |
| (baseline) | | 83.63 | 80.22 |
| 76 uniformly random pairs (3 seeds) | 76 | 14.29 / 14.25 / 14.82 | 4.07 / 4.05 / 4.84 |
| $S^+_{\text{lo}}$, concentrated support | 54 | 25.00 | 18.25 |
| $S^+_{\text{inv}}$, robust support | 76 | **94.02** | **93.56** |
| All supportive | 135 | 95.00 | 94.25 |

*Table 6: Keep-only interventions. Every row retains the concepts paired with the true label and therefore injects label information; the random row measures how much of the effect that injection accounts for, and it collapses to chance. Its control is uniform per class rather than $|D|$-matched, deliberately, since stratifying would preferentially retain high-$|D|$ pairs, which are mostly supportive for some class, and would reintroduce the very effect being controlled for. The three seeds are shown individually rather than averaged because two of them land exactly at 1/7 = 14.29%, which an average would obscure.*

Retaining only the 76 robust-support pairs and ablating the remaining 114,612, or 99.93% of the grid, yields **94.02% macro and 93.56% micro on sketch, above the unablated model**, with source domains at 99.85% to 100%. The frozen model already contains a concept subset sufficient for 94% accuracy on a domain it never saw, and that subset is identifiable from source-domain statistics alone. Its failure to reach that accuracy is a matter of selectivity rather than of representational capacity.

**The objection this invites, stated plainly.** Retaining only concepts selected because they support class $y$, on images whose true label is $y$, deletes every competing-class direction from the reconstruction. That is label-informative in a way a random retention is not, and it is why source accuracy rises to between 99.85% and 100% after 99.93% of the grid is removed. Any reading of these rows must account for that component rather than treat the accuracy as achieved.

Two comparisons do so. Retaining 76 arbitrary pairs, size-matched per class, collapses the model to exactly chance at 14.29% macro, with every prediction becoming person, so a degenerate reconstruction does not by itself recover the label and the retained bucket is doing work. More informatively, all three keep-only rows share the same label-injection component, so differences between them isolate what the buckets contribute: concentrated support alone reaches 25.00%, robust support alone reaches 94.02%, and adding the concentrated pairs on top of robust support moves it only to 95.00%. That 69-point gap between two buckets under identical label injection is the result, and the absolute level is not. Domain-contingent support is close to redundant, which is Finding 3 of Section 5.4 arrived at from the opposite direction.

As with the harmful buckets, robust support carries roughly ten times the $|D|$ mass of concentrated support, so these rows conflate consistency with effect magnitude. The normalization of Section 5.4 applies here too, and the limitation is stated once in Section 7.

## 6 Discussion

## 6.1 What an invariance criterion cannot reach

A concept that is invariant and consistent in its harm has, by construction, balanced activation statistics across source domains, and that balance is precisely what qualifies it as invariant. A criterion defined on cross-domain divergence in feature distributions therefore cannot separate it from a useful concept, and its pathology resides entirely in the sign of its effect on the class posterior, which no activation-level statistic measures. That part is definitional. The empirical content is the cost: 32 class-concept pairs, 0.028% of the grid and 0.50% of the pairs that fire at all, account for 22.6% of the model's target-domain error, and the consistent half of invariant harm is several times more damaging per unit of effect mass than the concentrated half.

The relevant unit is therefore not the globally invariant feature but the class-conditional concept whose activation and effect are both stable. This is not a claim that invariance is useless as a criterion, since our own measurements show it reduces the rate of harm by about 40%. It is a claim about what remains after it has done its work, and about the fact that what remains is disproportionately expensive. We make the claim about the criterion rather than about any particular training objective: we analyze a model trained without one, and Section 7 records that the corresponding prediction for explicitly aligned models is untested.

## 6.2 Failures of generalization can be failures of selectivity

Redundancy, rather than invariance, may be what degrades under shift. Ablations that are no-ops in distribution turn out to be decisive out of distribution: removing 99.93% of the class-concept grid leaves source accuracy unchanged, because in distribution the model carries enough redundant evidence that no single concept determines the decision, yet the same removal is worth ten points on the target domain. We have not measured margins or calibration and so do not claim to know why the target domain is the more sensitive of the two. The asymmetry itself is what the observation establishes.

This complicates the conventional reading of a domain-generalization failure as a deficit, in which the model did not learn features that transfer. Retaining 0.07% of the grid, namely the pairs that are invariant, consistent and supportive, raises held-out accuracy above that of the unablated model while leaving source accuracy at or above its original level. The representation already contains evidence sufficient for substantially better target-domain accuracy than the model achieves.

This is an oracle bound and not a method, since applying the mask requires each image's label. But the bound is informative about diagnosis rather than remedy: when a model underperforms out of distribution, one should ask not only whether the required features are absent, but also whether they are present and diluted. The two diagnoses imply different interventions, and the framework distinguishes them.

## 7 Limitations

**First, SAE latents are candidate concepts, not guaranteed human-semantic units.** Some correspond to recognizable object parts or styles, while others remain difficult to name. Our quantitative claims depend on sparse reconstruction and ablation behavior, not on human interpretability, and reconstruction fidelity is a precondition for the whole analysis: were the reconstruction to change model predictions substantially, every downstream concept effect would have to be described as an effect in SAE reconstruction space rather than in the original model. Section 5.1 establishes that it does not, to within 0.07 percentage points.

**Second, the framework is post-hoc and label-using, and its interventions are diagnostics rather than methods.** It uses class and domain labels to analyze a trained model, which is appropriate for auditing but is not a deployable DG training or test-time adaptation procedure. The interventions of Sections 5.4 and 5.7 additionally use the ground-truth class to decide which concepts to mask or keep, so they validate the concept categories without implying that the same accuracy changes are achievable without labels at test time. Whether any of it transfers to a label-free setting remains open. The label-free variants we examined, which collapse the mask across classes, are null, but they also discard the class-conditionality our analysis identifies as the locus of the signal, so they are not a decisive test. Relatedly, the per-class profiles of Section 5.5 are descriptive: conflict burden does not predict per-class target accuracy, and validating any such quantity as a selection criterion would require additional datasets, seeds, algorithms and pre-registered protocols.

**Third, the empirical scope is one dataset, one checkpoint, one autoencoder.** The main experiments cover PACS with a single ERM ResNet-50 checkpoint and a single SAE, and we have not run a seed analysis. What this setting establishes is that the phenomena exist and are measurable: a model can carry invariant concepts that are neutral, useful, or costly; the costly ones can be identified from source domains alone; and ablating them moves held-out accuracy in the predicted direction against controls. What it cannot establish is how the proportions vary across datasets, architectures, training objectives or dictionaries. In particular, our argument predicts that a model trained with an explicit alignment objective should retain comparably many invariant-and-harmful concepts, since such an objective operates on the statistics that qualify them as invariant. That prediction is untested here, and testing it is the most direct way to falsify the paper's central claim. Threshold choices likewise affect bucket assignments. We report sensitivity to $\tau_H$ and $\tau_R$ in Figure 8, where the qualitative conclusion holds at every value swept and the asymmetry widens as the threshold tightens, so the main conclusion does not depend on a single threshold value. The support floor is a more consequential exclusion than the thresholds are. As Section 4.3 records, it removes most of the grid, and ablating the excluded harmful pairs recovers 7.2% of target error, which is small beside the 45.6% recovered by the above-floor harmful pairs but not zero, and is obtained from more aggregate effect rather than less. Our claims are scoped to the well-supported population and we make none about the excluded one. A lower floor is not the remedy, since it would leave $R$ uninterpretable for exactly the pairs it admitted; higher per-concept exposure would be.

**Fourth, we do not fully separate consistency from effect magnitude.** Harmful pairs are markedly more likely to be low-consistency than supportive ones. Among invariant pairs with non-negligible effect, 36.0% (32 of 89) of harmful pairs clear $\tau_R$ against 72.4% (76 of 105) of supportive pairs, and the keep-only comparisons of Section 5.7 conflate the two, since robust support carries roughly ten times the effect mass of concentrated support. The mass-normalized comparison in Section 5.4 indicates the effect is not purely one of magnitude, since distributed harm recovers 3.5 times more error per unit of effect mass than concentrated harm and distributed support costs 6.5 times more, but a mass-matched intervention would be required to settle it. Such a comparison would need to match the distribution of per-pair magnitudes rather than their sum, because aggregate effect mass proves to be a poor proxy for consequence in our data: 111,530 pairs whose total mass exceeds that of a 35-pair bucket recover forty times less error.

## 8 Conclusion

This paper introduced sparse concept diagnostics for domain generalization. The central argument is that DG should not be diagnosed by activation invariance alone. A useful invariant concept must also be class-supporting, consistent in its discriminative effect across domains, and isolated from competing classes. By decomposing trained vision models into SAE-derived candidate concepts and scoring each concept class-conditionally, the framework distinguishes robust support from harmful invariance, domain-contingent cues, and class conflict.

Applied to a frozen ERM ResNet-50 on PACS, the framework yields three findings that target-domain accuracy alone cannot express. First, invariance is weakly informative about the absence of harm and uninformative about the presence of usefulness. Second, invariant support is distributed across domains while invariant harm concentrates, and the concentrated half is the cheaper one, so the harm that is both invariant and consistent is disproportionately costly and is by definition beyond the reach of any criterion defined on activation statistics. Third, harmful concepts are in the large majority not spurious features but genuine features attached to the wrong class, which makes class isolation rather than feature removal the appropriate target.

Rather than proposing a new DG algorithm, sparse concept diagnostics offer a post-hoc audit of what a trained model has learned, how much of its held-out error is attributable to identifiable concepts, and whether its shortfall reflects missing evidence or evidence that is present but diluted. The last of those distinctions carries a conjecture worth stating plainly: since removing 99.93% of the class-concept grid leaves source accuracy untouched while moving target accuracy by ten points, **redundancy, rather than invariance, may be what degrades under shift.**

## References

- Kei Akuzawa, Yusuke Iwasawa, and Yutaka Matsuo. Adversarial invariant feature learning with accuracy constraint for domain generalization. In European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases, 2019.

- Martin Arjovsky, Leon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. In arXiv preprint arXiv:1907.02893, 2019.

- David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6541-6549, 2017.

- Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023.

- David Chanin et al. A is for absorption: Studying feature splitting and absorption in sparse autoencoders, 2024.

- Liang Chen, Yong Zhang, Yibing Song, Anton van den Hengel, and Lingqiao Liu. Domain generalization via rationale invariance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1751-1760, 2023.

- Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In International Conference on Learning Representations, 2024.


- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. Transformer Circuits Thread, 2022.

- Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux, Julien Colin, R’emi Cad‘ene, and Thomas Serre. CRAFT: Concept recursive activation factorization for explainability. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2711-2721, 2023.

- Tigran Galstyan, Hrayr Harutyunyan, Hrant Khachatrian, Greg Ver Steeg, and Aram Galstyan. Failure modes of domain generalization algorithms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19077-19086, 2022.

- Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. Journal of Machine Learning Research, 17(59):1-35, 2016.

- Leo Gao, Tom Dupr’e la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. In International Conference on Learning Representations, 2025.

- Amirata Ghorbani, James Wexler, James Y. Zou, and Been Kim. Towards automatic concept-based explanations. In Advances in Neural Information Processing Systems, volume 32, 2019.

- Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In International Conference on Learning Representations, 2021.

- Sangyu Han, Yearim Kim, and Nojun Kwak. Causal interpretation of sparse autoencoder features in vision, 2025.

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, 2016.

- Adam Karvonen, Can Rager, Samuel Marks, and Neel Nanda. Evaluating sparse autoencoders on targeted concept erasure tasks, 2024.

- Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory Sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In International Conference on Machine Learning, 2018.

- Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International Conference on Machine Learning, 2020.

- Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M. Hospedales. Deeper, broader and artier domain generalization. In IEEE International Conference on Computer Vision, 2017.

- Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C. Kot. Domain generalization with adversarial feature learning. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5400-5409, 2018. doi: 10.1109/CVPR.2018.00566.

- Hyesu Lim, Jinho Choi, Jaegul Choo, and Steffen Schneider. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. In International Conference on Learning Representations, 2025.

- Aleksandar Makelov, George Lange, and Neel Nanda. Towards principled evaluations of sparse autoencoders for interpretability and control, 2024.

- Gonçalo Paulo and Nora Belrose. Sparse autoencoders trained on the same data learn different features, 2025.

- Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In IEEE International Conference on Computer Vision, 2017.


- Samuel Stevens, Wei-Lun Chao, Tanya Berger-Wolf, and Yu Su. Sparse autoencoders for scientifically rigorous interpretation of vision models, 2025.

- Baochen Sun and Kate Saenko. Deep CORAL: Correlation alignment for deep domain adaptation. In European Conference on Computer Vision Workshops, 2016.

- Vladimir N. Vapnik. An overview of statistical learning theory. IEEE Transactions on Neural Networks, 10 (5):988-999, 1999.

- Chih-Kuan Yeh, Been Kim, Sercan O. Arik, Chun-Liang Li, Tomas Pfister, and Pradeep Ravikumar. On completeness-aware concept-based explanations in deep neural networks. In Advances in Neural Information Processing Systems, volume 33, pp. 20554-20565, 2020.

- Han Yu, Xingxuan Zhang, Renzhe Xu, Jiashuo Liu, Yue He, and Peng Cui. Rethinking the evaluation protocol of domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21897-21908, 2024.

- Han Zhao, Remi Tachet Des Combes, Kun Zhang, and Geoffrey J. Gordon. On learning invariant representa- tions for domain adaptation. In International Conference on Machine Learning, 2019.

- Han Zhao, Chen Dan, Bryon Aragam, Tommi S. Jaakkola, Geoffrey J. Gordon, and Pradeep Ravikumar. Fundamental limits and tradeoffs in invariant representation learning. Journal of Machine Learning Research, 23(340):1-56, 2022.
