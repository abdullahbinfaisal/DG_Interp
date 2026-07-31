## When Invariance Is Not Enough: Sparse Concept Diagnostics for Domain Generalization

## Anonymous authors Paper under double-blind review

## Abstract

Domain generalization methods are often motivated by the search for invariant representations, but activation invariance alone does not reveal whether a feature is useful for classification. We propose sparse concept diagnostics, a post-hoc framework for analyzing trained domain-generalization models by decomposing internal vision representations into sparse autoencoder candidate concepts. For each class–concept pair we compute three scores on source domains only: activation invariance across domains, ablation-based discriminative effect on the ground-truth class, and cross-domain consistency of that effect. Applied to a frozen ERM ResNet-50 on PACS, the framework shows that conditioning on invariance nearly halves the chance that a concept is harmful yet leaves the chance that it is useful unchanged, and that 83.5% of invariant class–concept pairs have no measurable discriminative effect. The concepts that damage held-out accuracy most are those an activation-alignment criterion scores best: 35 pairs that are both invariant and consistent in their harm, 0.03% of the class–concept grid, account for a quarter of all target-domain errors. We further find that 78% of harmful pairs involve a concept that supports a different class, so harmful concepts are largely not spurious features but genuine features attached to the wrong class, and that retaining 0.07% of the grid raises held-out accuracy above that of the unablated model — indicating that the failure is partly one of selectivity rather than of representational capacity. Interventions are validated against size- and magnitude-matched random controls, which move accuracy in the opposite direction. Our results suggest that domain-generalization representations should be evaluated not only by whether concepts are invariant, but by whether they are discriminatively aligned and class-isolated.

## 1 Introduction

Neural networks can perform poorly when evaluated on data that differs from their training distribution. Domain generalization (DG) addresses this by learning from several source domains in order to generalize to unseen target domains. The standard protocol reports target-domain accuracy. This is necessary but incomplete because it does not reveal which internal cues a model relies on. Two models with equal target accuracy may depend on different correlations, one robust, one fragile, and accuracy on the test domain may not separate them. The limitation is prominent in modern benchmarks, where carefully tuned empirical risk minimization (ERM) matches the accuracy of explicit DG algorithms Gulrajani & Lopez-Paz (2021). An interpretable analysis of the feature representation addresses what accuracy cannot. It can expose spurious cues, distinguish models with similar accuracy, and trace which internal features a decision rests on. Most importantly, it lets us examine how representation structure changes under DG objectives, especially objectives that encourage domain-invariant features. A dominant class of DG methods attempts to learn domain invariant representations: features that persist across changes in style, background, or acquisition condition. However, enforcing invariance too aggressively can suppress class-discriminative variation and blur the boundaries between classes. This is referred to as the discrimination-invariance tradeoff. Prior work characterizes this tradeoff theoretically and demonstrates it at the level of aggregate accuracy, showing that stronger invariance can raise target error Zhao et al. (2019); Akuzawa et al. (2019). However, how this tradeoff manifests at the level of class-conditional internal features remains underexplored. To address these issues, we propose a diagnostic framework for domain generalization algorithms. We train a sparse autoencoder (SAE) on a model’s feature maps to obtain an overcomplete sparse basis of candidate [URL 🔗](#page-0)

concepts. For each concept, we first quantify whether it activates evenly across domains for a given class.


Second, we measure whether ablating the concept raises or lowers the model’s confidence in the correct class. Third, we test whether that effect is consistent across domains. This analysis reveals model behavior that is invisible from target-domain accuracy alone, enabling a post-hoc diagnosis of DG failure modes rather than merely ranking methods by accuracy.

## Contributions.

- 1. We introduce sparse concept diagnostics, a post-hoc framework for analyzing trained DG models using SAE-derived candidate concepts.

- 2. We define three class-conditional scores: activation invariance H, ablation-based discriminative effect D, and cross-domain consistency R of the discriminative effect.

- 3. We use these scores to construct a concept typology that separates robust class-supporting concepts from harmful invariant, domain-contingent, and class-conflicting concepts.

- 4. We aggregate concept categories into class-level and model-level diagnostic profiles and test their functional relevance using oracle concept interventions.

## 2 Related Work

## 2.1 Domain generalization and invariant representations

DG studies learning under domain shift, typically by training on multiple source domains and evaluating on held-out target domains (Li et al., 2017; Gulrajani & Lopez-Paz, 2021). Many DG algorithms are motivated by the hypothesis that stable predictors should rely on invariant features. Domain-adversarial training attempts to remove domain information from the representation through an adversarial domain classifier (Ganin et al., 2016). CORAL aligns second-order statistics across domains (Sun & Saenko, 2016). MMD-based methods penalize kernel discrepancies between domain feature distributions (Li et al., 2018). Invariant Risk Minimization (IRM) formalizes a related objective by seeking representations for which the optimal classifier is invariant across environments (Arjovsky et al., 2019). [URL 🔗](#page-0)

However, benchmark studies have complicated the interpretation of progress in DG. DomainBed showed that, under standardized model selection and hyperparameter search, carefully tuned empirical risk minimization can be competitive with many specialized DG algorithms (Gulrajani & Lopez-Paz, 2021). More recent work has further argued that common DG evaluation protocols may leak test-domain information through supervised pretraining or oracle model selection (Yu et al., 2024). These findings motivate a shift from asking only whether a method improves target-domain accuracy to asking what kinds of representations trained models actually learn. Our work follows this diagnostic perspective: rather than proposing a new DG training objective, we analyze frozen DG models after training and examine whether their learned concepts are invariant, discriminative, and class-isolated. [URL 🔗](#page-0)

## 2.2 Why invariance alone can fail

The motivation for invariant representations is strong, but invariance alone is not sufficient for robust prediction. Zhao et al. show that domain-invariant representations with low source error need not guarantee target performance under conditional shift, and they characterize a fundamental tradeoff between learning invariant representations and achieving low joint error across domains (Zhao et al., 2019). Subsequent work gives an information-theoretic account of the accuracy–invariance tradeoff and characterizes feasible regions for representations that must preserve task-relevant information while discarding domain or nuisance information (Zhao et al., 2022). In DG specifically, Galstyan et al. propose decomposing generalization error into failure modes and show that the contribution of invariance-related failures can vary across methods, datasets, regularization strengths, and training stages (Galstyan et al., 2022). Rationale-invariance methods provide another perspective by asking whether same-category examples are classified using similar decision rationales rather than merely similar feature distributions (Chen et al., 2023). [URL 🔗](#page-0)

These results suggest that the relevant question is not simply whether a representation is invariant, but whether the invariant information is useful for the task. A feature can be domain-invariant but irrelevant to


the class, consistently harmful to the correct class, or useful only in a subset of domains. Existing invariance diagnostics usually operate at the level of global feature distributions, domain separability, or classifier risk. In contrast, our work asks a class-conditional concept-level question: for a concept c and class k, does the concept activate across domains, does it support or hurt the model’s confidence in class k, and is that effect consistent across domains?

## 2.3 Concept-based and visual concept interpretability

Concept-based interpretability explains model behavior in terms of higher-level abstractions rather than individual pixels or dense activation vectors. Concept Bottleneck Models predict human-defined concepts before predicting labels, thereby making the intermediate representation directly interpretable when concept annotations are available (Koh et al., 2020). TCAV probes trained models with user-specified concept directions and measures sensitivity to those concepts through directional derivatives (Kim et al., 2018). Spatial attribution methods such as Grad-CAM produce class-discriminative localization maps that highlight image regions important for a prediction (Selvaraju et al., 2017). These methods are useful, but they either require predefined concepts or provide local saliency rather than a reusable concept basis for analyzing representation structure across domains. [URL 🔗](#page-0)

A related line of work automatically discovers visual concepts. Network Dissection evaluates the alignment between hidden units and a vocabulary of semantic concepts such as objects, parts, textures, scenes, and colors (Bau et al., 2017). ACE extracts visual concepts automatically by clustering image segments and evaluates their importance to predictions (Ghorbani et al., 2019). CRAFT combines concept discovery with spatial attribution, aiming to explain both what the model sees and where it sees it (Fel et al., 2023). Completeness-aware concept methods study whether a concept set is sufficient to explain model predictions and introduce concept-importance measures such as ConceptSHAP (Yeh et al., 2020). Our goal is related but distinct. We do not primarily aim to name every concept or produce human-semantic explanations. Instead, we use sparse candidate concepts as a functional diagnostic basis for asking whether learned internal features are invariant, discriminative, and class-consistent across domains. [URL 🔗](#page-0)

## 2.4 Sparse autoencoders for representation analysis

Sparse autoencoders (SAEs) are motivated by the possibility that neural networks represent more features than they have individual neurons. Work on superposition argues that models can store many sparse features in distributed directions, making individual neurons polysemantic and difficult to interpret (Elhage et al., 2022). Dictionary-learning and SAE methods attempt to recover a sparse feature basis from dense activations. Anthropic’s work on monosemanticity uses sparse autoencoders to decompose language-model activations into more interpretable features (Bricken et al., 2023), and Cunningham et al. show that SAEs can recover highly interpretable language-model features and support finer-grained causal analysis of model behavior (Cunningham et al., 2024). [URL 🔗](#page-0)

Recent work has also emphasized that SAE quality should be evaluated carefully. Gao et al. study scaling laws and evaluation metrics for k-sparse autoencoders, highlighting the reconstruction–sparsity tradeoff and the problem of dead latents (Gao et al., 2025). Makelov et al. argue that feature dictionaries should be evaluated in terms of approximation, control, and interpretability on specific tasks (Makelov et al., 2024). Karvonen et al. evaluate SAEs through targeted concept-erasure tasks, connecting SAE quality to downstream control rather than only unsupervised reconstruction proxies (Karvonen et al., 2024). These works motivate two design choices in our paper: first, we report SAE reconstruction fidelity before interpreting concept ablations; second, we treat our interventions as diagnostic tests of concept functionality rather than as a deployable inference algorithm. [URL 🔗](#page-0)

## 2.5 Sparse autoencoders in vision

Although much SAE interpretability work began in language models, recent work has extended SAEs to vision representations. Stevens et al. apply SAEs to vision models and emphasize that visual feature interpretations should be validated through controlled interventions rather than only qualitative inspection (Stevens et al., [URL 🔗](#page-0)


2025). Lim et al. introduce PatchSAE for CLIP vision transformers, extracting patch-level visual concepts and studying how model adaptation changes associations between images and learned concepts (Lim et al., 2025). Recent work on causal interpretation of vision-SAE features further cautions that naive top-activation visualizations can be misleading in attention-based vision models because a feature’s activated patch may co-occur with, but not cause, the feature activation (Han et al., 2025). [URL 🔗](#page-0)

These findings support our cautious use of SAE latents as candidate concepts. We do not assume that every SAE feature is perfectly monosemantic or human-nameable. Instead, we require that the SAE reconstruction preserve classifier behavior and then use ablation-based diagnostics to measure whether each candidate concept functionally supports or hurts a class. This distinguishes our work from prior vision-SAE studies: rather than studying semantic feature discovery or model adaptation in general, we use SAE concepts to diagnose domain generalization failure modes.

## 2.6 Positioning of our work

Our work sits at the intersection of DG diagnostics, concept-based interpretability, and SAE-based represen- tation analysis. Prior DG work studies whether models generalize and whether representations align across domains. Prior concept methods study whether internal features can be made human-interpretable. Prior SAE work studies whether dense activations can be decomposed into sparse features. We combine these threads to ask a different question: when a trained DG model contains a sparse candidate concept, what role does that concept play for a particular class across domains?

The key distinction is class-conditionality. We do not assign a single global score to a concept. Instead, we evaluate H(k, c), D(k, c), and R(k, c) for each class–concept pair. This lets us distinguish robust class- supporting concepts from harmful invariant concepts, domain-contingent concepts, and class-conflicting concepts. As a result, our framework diagnoses why activation invariance may fail to support generalization: invariant concepts must also be discriminatively aligned with the correct class and isolated from competing classes.

## 3 Sparse Concept Diagnostics

## 3.1 Setup

Let denote a set of domains and

di

fcls. The feature extractor maps an image to an intermediate feature map

The classifier head maps the feature representation to logits over classes. Throughout, the model is frozen. The SAE is trained only to reconstruct intermediate feature maps and is used as an analysis tool.

In our main experiments, is the set of PACS domains,

and = 7 for the PACS object classes (Li et al., 2017). The primary model is an ERM-trained ResNet backbone (He et al., 2016; Vapnik, 1999). [URL 🔗](#page-0)

a set of class labels. For each image xi, let yi

be its class label and

Y

its domain label. A trained vision model is decomposed into a feature extractor fθ and classifier head

## 3.2 Sparse autoencoder concept space

The SAE operates on normalized spatial feature tokens. Let µSAE and σSAE denote the scalar or channel-wise normalizer statistics used during SAE training. We normalize the feature map as


The normalized feature map is rearranged into T = HW spatial tokens,

An SAE encoder E maps tokens to a sparse code,

where K is the number of SAE latent concepts. In the main implementation, K = 16,384 and top-16 sparsity is used, so each spatial token activates at most 16 concepts. The code is nonnegative, Si 0, and each column c of Si is treated as the spatial activation map of candidate concept c.

The decoder DSAE reconstructs the normalized feature map,

and the reconstructed feature is denormalized before classification:

For ResNet models, adaptive average pooling is applied before the linear classifier head. The reconstructed class probability is

Scope of the SAE representation. We use SAE latents as a sparse analysis basis, not as a canonical

set of ground-truth semantic concepts. This distinction is important because SAE dictionaries are not identifiable in a strict sense: independently trained SAEs may recover different individual latents, and changes in initialization, dictionary size, sparsity level, or optimization can split, merge, or obscure features (Paulo & Belrose, 2025; Chanin et al., 2024; Makelov et al., 2024). We therefore do not interpret a latent index c as an invariant object across independently trained SAEs, nor do we require every latent to be human-nameable or perfectly monosemantic. Our claims are instead about the behavior of a trained model when its representation is projected into a sparse reconstruction space and analyzed through class-conditional activation and ablation statistics. [URL 🔗](#page-0)

We report reconstruction fidelity before interpreting any concept-level ablations, since a poor reconstruction would make downstream effects artifacts of the SAE rather than meaningful perturbations of the model representation; this follows recent work emphasizing reconstruction quality, sparsity, dead-latent behavior, and task-grounded evaluation when using SAEs for interpretability (Gao et al., 2025; Makelov et al., 2024; Karvonen et al., 2024). Our main quantitative claims are based on aggregate distributions and bucket masses over class–concept pairs, not on the semantic interpretation of isolated individual latents. Moreover, we treat visualizations of top-activating examples as qualitative aids only; they are not used as proof that a latent has a particular semantic meaning, especially because top-activation visualizations in vision SAEs may conflate local activation with broader contextual causes (Han et al., 2025). Additionally, our discriminative score D(k, c) is an ablation-based effect in SAE reconstruction space, not a claim of causal identification in the original network. The purpose of the SAE is therefore not to recover the unique true features of the model, but to provide a sparse, reconstructive basis in which we can test whether candidate concepts are activation-invariant, class-supporting, and domain-consistent. [URL 🔗](#page-0)

To reduce dependence on idiosyncratic dictionary choices, we evaluate the robustness of our conclusions at the level at which they are claimed: reconstruction accuracy, score distributions, typology proportions, and class/model-level diagnostic profiles. In particular, we report sensitivity to threshold choices and, where computationally feasible, to SAE hyperparameters or independently trained SAE seeds. We do not require one-to-one matching of individual latents across runs; instead, we ask whether the aggregate conclusions, for example, that high activation invariance does not imply positive discriminative effect, remain stable under reasonable changes to the sparse basis.


## 3.3 Concept activation

For concept c, define its total activation on image i as

We say that concept c is active on image i if it fires at least once across spatial tokens:

For class k, define

and for class k, domain d, and concept c, define the active subset

Similarly,

All diagnostic scores are class-conditional. The same SAE concept can therefore play different roles for different classes.

## 3.4 Activation invariance H

The activation-invariance score H(k, c) measures whether concept c activates evenly across domains for examples of class k. For each domain d, compute the mean activation

where

This yields a nonnegative domain-wise activation vector

We normalize raw mean activations by their sum,

= P

when the denominator is nonzero. If the denominator is zero, the concept never activates for class k and we set H(k, c) = 0. Otherwise, H(k, c) is the normalized entropy

Thus H(k, c) [0, 1] when q(k,c) is a valid distribution. High H means activation is spread evenly across domains; low H means activation is concentrated in a small number of domains. Crucially, H measures where a concept appears, not whether it helps classification.


## 3.5 Ablation-based discriminative effect D

To test whether a concept functionally supports a class, we ablate the concept from the SAE code and measure the change in the model’s probability for the ground-truth label. For active concept c, construct a

masked code S(−c) by setting the concept column to zero across all spatial locations:

i

while leaving all other concept activations unchanged:

The masked reconstruction is

which is denormalized and passed through the classifier to obtain

The per-image ablation effect is

If δ(c) i > 0, removing concept c decreases confidence in the correct class, so the concept supports the prediction. If δ(c) i < 0, removing the concept increases confidence in the correct class, so the concept is harmful or distracting for that example.

The class-conditional discriminative effect is

Positive D(k, c) means concept c supports class k, negative D(k, c) means it hurts class k, and near-zero D(k, c) means it is approximately neutral for class k. We use “ablation-based effect” rather than “causal effect” because the intervention occurs in the SAE reconstruction space and may not capture all causal dependencies in the original network.

## 3.6 Discriminative consistency R

The score D(k, c) measures average discriminative effect, but not whether that effect is distributed consistently across domains. A concept may help class k in photos but not sketches, or may be helpful in one domain and harmful in another. We therefore compute domain-specific discriminative effects,

A direct entropy over signed Dd(k, c) values is not generally well-defined when effects have mixed signs. We therefore define R using effect magnitudes. For concepts with non-negligible discriminative effect, define

The magnitude-consistency score is


*Table 1: Concept typology induced by activation invariance H, discriminative effect D, and discriminative consistency R. Neutral concepts with |D(k, c)| τD are excluded before interpreting R.*

| H R | sign of D | Interpretation |
| --- | --- | --- |
| High High Positive Robust class-supporting concept |   |   |
| High High Negative Harmful invariant concept |   |   |
| High Low Positive Domain-contingent supporting concept |   |   |
| High Low Negative Domain-contingent harmful concept |   |   |
| Low High Positive Domain-specific but consistently useful when active |   |   |
| Low High Negative Domain-specific but consistently harmful when active |   |   |
| Low Low Positive Local or unstable supporting cue |   |   |
| Low Low Negative Local or unstable harmful cue |   |   |

High R(k, c) means that the magnitude of the concept’s discriminative effect is spread across domains. Low R(k, c) means that the effect is concentrated in a small number of domains.

Because entropy is scale-insensitive, R is interpreted only for concepts satisfying |D(k, c)| > τD. A concept with tiny effects in every domain can have high entropy while still being unimportant. We treat such concepts as neutral rather than robust. The framework uses exactly these three scores; R always refers to the magnitude-consistency score in Eq. 27, and concepts near D = 0 are excluded before typology construction. Note that R does not distinguish an effect confined to one domain from an effect whose sign varies across domains, since it is computed over magnitudes; the latter case is rare in our data, affecting 0.33% of pairs. [URL 🔗](#page-0)

## 3.7 Concept typology

We convert continuous scores into concept categories using thresholds τH, τR, and τD. Define

Concepts with |D(k, c)| τD are treated as neutral and excluded from the main truth-table typology. Table 1 summarizes the main categories. [URL 🔗](#page-0)

This typology is the main interpretive object in the paper. It separates concepts that are merely invariant from concepts that are invariant and useful. In particular, the high-H, high-R, negative-D bucket is central: these are invariant concepts that consistently hurt the correct class, directly showing why activation invariance alone is not sufficient.

## 3.8 Class-level and model-level diagnostic profiles

A model should not be reduced to one average invariance score. Instead, we summarize the distribution of its discriminative concept mass across diagnostic categories. For class k, define the total discriminative mass

The robust support mass is


The harmful invariant mass is

The domain-contingent mass is

To measure class conflict, define concept c as conflicting if it supports at least one class and hurts at least one other class:

The class-level conflict mass is

For a model or checkpoint m, we aggregate over classes:

with analogous definitions for HIMm, DCMm, and CMm. We interpret the tuple

as a diagnostic profile, not as a validated model-selection criterion.

## 4 Experimental Setup

## 4.1 Dataset and model

The main experiments use PACS (Li et al., 2017), which contains four visual domains: art painting, cartoon, photo, and sketch. The task contains seven object classes. We designate the sketch domain as the target holdout, utilizing the remaining three as source domains. Unless otherwise stated, models are trained with the DomainBed protocol (Gulrajani & Lopez-Paz, 2021), employing non-oracle checkpoint selection based on the highest average accuracy across all source domains. The primary analysis uses an ERM-trained ResNet-50 model at a single checkpoint, selected without reference to the target domain. We focus on one cleanly controlled setting rather than claiming broad benchmark coverage; Section 7 states the resulting limits on external validity. [URL 🔗](#page-0)

## 4.2 SAE training

The SAE is trained to reconstruct the feature maps from the final layer before the classification head. The ResNet-50 has a (7 7 spatial resolution, and 2048 channels. We use a Top-K SAE with a dictionary size of 16,384 and enforce top-16 sparsity per spatial token. Training proceeds for 250 epochs using the Adam optimizer with an initial learning rate of 3×10−4 and a cosine decay schedule. To strictly preserve the domain generalization setting, the SAE is trained and analyzed exclusively on the source domains, while the target domain is reserved solely for verification. The backbone and classifier remain entirely frozen throughout.

To verify training convergence and ensure that higher dimensional interventions map reliably back to the original feature space, we evaluate the classifier on SAE-reconstructed features. The goal is to confirm that these reconstructions induce minimal deviation from the original model behavior.

Unless otherwise stated, the SAE is trained on source-domain features only. The diagnostic scores H, D, R and thresholds are also estimated on source-domain data. Target-domain examples are used only for post-hoc evaluation of whether source-estimated concept categories explain held-out behavior. One qualification is worth stating precisely: the two scalar statistics used to normalize feature maps before the SAE encoder were estimated from a single batch spanning all four domains rather than the three source domains. This concerns a mean and a standard deviation only, and no dictionary element, score or threshold is estimated with target-domain data.


*Table 2: SAE reconstruction fidelity on PACS for ERM ResNet-50, checkpoint 3300. Micro-averaged accuracy is image-weighted, macro-averaged is class-weighted; the two diverge on sketch because that domain is heavily class-imbalanced. Sketch is the held-out target domain. Evaluation covers every image of each domain.*

| Domain | Original micro | Recon. micro | Δ micro | Original macro | Recon. macro |
| --- | --- | --- | --- | --- | --- |
| Art painting | 99.37 | 99.32 | 0.05 | 99.38 | 99.35 |
| Cartoon | 99.19 | 99.15 | 0.04 | 99.23 | 99.19 |
| Photo | 99.82 | 99.82 | 0.00 | 99.78 | 99.78 |
| Sketch (target) | 80.25 | 80.22 | 0.03 | 83.56 | 83.63 |

## 4.3 Evaluation protocol

Scores and accuracies are computed over every image in each domain, with no train/test subsampling, no shuffling and no dropped final batch, so all reported figures are exactly reproducible. Accuracy is reported both micro-averaged and macro-averaged over classes throughout, and source and target domains are never pooled into a single average, since three of the four domains are in distribution and pooling would obscure the only number that carries the argument.

When computing statistics over class–concept pairs we restrict attention to pairs active on at least 30 images of the class in question. This support floor is an inclusion criterion for estimation only: R is an entropy over three per-domain effect estimates, and a cell supported by a handful of images yields a meaningless Dd. It is never used to mask pairs during a forward pass. Applying such a filter class-conditionally would leak label information, since a concept that fires rarely for class k but often for class k′ is a confusion signal, and suppressing it on k-labeled images deletes evidence for the competing class using the ground-truth label.

## 4.4 Thresholds and sensitivity

The main typology uses thresholds τH, τR, and τD. We set τH = τR = 0.7 and τD = 10−4. The choice of 0.7 is not arbitrary: with three source domains, log 2/ log 3 = 0.63, so a threshold of 0.7 is the largest value at which clearing the threshold guarantees nonzero activation, or nonzero effect, in all three source domains. Higher thresholds lose that guarantee and also reduce the central harmful invariant category to six pairs, too few to support a claim. We report sensitivity for τH and τR in {0.7, 0.8, 0.9} in Section 5.3, where the qualitative conclusion strengthens monotonically with the threshold.

## 5 Results

Throughout this section we report micro-averaged (image-weighted) and macro-averaged (class-weighted) accuracy separately. The two differ materially on the target domain, where PACS sketch is heavily imbalanced: 772 dog images against 80 house images, and dog is simultaneously the largest and the weakest class. Reporting a single unqualified accuracy for sketch would conceal a discrepancy of more than three percentage points. All diagnostic scores are estimated on the three source domains only; sketch is used exclusively to evaluate whether source-estimated categories explain held-out behavior.

## 5.1 SAE reconstructions preserve classifier behavior

Before using SAE latents for analysis we verify that the reconstruction preserves the classifier's behavior, since a poor reconstruction would render every downstream ablation an artifact of the autoencoder rather than a perturbation of the model. Table 2 reports accuracy for the original model and for the same classifier applied to SAE-reconstructed features, evaluated on every image of each domain.

Reconstruction changes accuracy by at most 0.05 percentage points on any domain, including the weakest. Concept ablations reported below are therefore interpretable as perturbations of the model's own representation, subject to the caveat of Section 3.2 that they are ablations in reconstruction space rather than causal interventions in the original network.

## 5.2 Activation-invariant concepts are not necessarily useful

Many domain generalization algorithms rest on a common premise: an attribute that persists across the source domains is likely to be causal and to persist on unseen target domains, whereas an attribute confined to a few domains is likely a spurious correlation Arjovsky et al. (2019). Under this view, suppressing domain-specific features improves out-of-distribution generalization by forcing the model to rely on the causal, invariant ones. [URL 🔗](#page-0)

We find this premise necessary but far from sufficient. Activation invariance H(k, c) measures whether a concept fires uniformly across domains. Discriminative effect D(k, c) measures the signed change in the model's confidence in the true class when the concept is ablated. If invariance alone were sufficient, high-H concepts would concentrate at positive D. Instead the high-H region spans positive, near-zero and negative D (Figure 1).

The three regimes have distinct interpretations, and each is visually identifiable. A neutral concept (D ≈ 0) activates across all domains yet does not separate classes. The clearest instance is a body-plan concept responding to the four-legged animal silhouette (Figure 3): legs and torso are legs and torso whether painted, drawn or photographed, so H is near one, but the concept fires on dogs, horses and elephants alike and therefore contributes nothing to discriminating among them. Ablating it changes the class posterior negligibly for any of them. Stability is not usefulness.

A negative-D concept is domain-invariant yet actively harmful. We observe a chest-and-forelimb concept (Figure 2) that fires on the front quarters of four-legged mammals in every domain. On dogs it is genuine evidence and its ablation lowers confidence in dog, so D(dog, c) > 0. On horses the same region fires — the underlying shape is similar, particularly in the flatter, lower-texture renderings of cartoons and paintings — but the model reads the pattern as dog-evidence, so it displaces probability away from horse and D(horse, c) < 0. Nothing about the detector is defective: it detects what it detects, reliably, in every domain. The defect lies in the mapping from that detector to a class decision. A positive-D concept is the canonical case that invariance-based reasoning implicitly assumes, for instance the same chest concept evaluated on dog.

This example also motivates class-conditionality directly. A single global score for the chest concept would average a positive effect on dog against a negative effect on horse and report approximately zero, filing a genuine failure mechanism into the neutral majority. Only the class-conditional formulation exposes it.

Quantitatively, of the 1542 class–concept pairs that clear the support floor, 1269 (82.3%) are discriminatively neutral, 135 (8.8%) support the correct class and 138 (8.9%) hurt it. Restricting to the invariant population barely changes this: among the 1173 pairs with H ≥ 0.7, 83.5% are neutral, 9.0% supportive and 7.6% harmful. Conditioning on invariance moves the probability that a concept is harmful from 13.3% among low-H pairs to 7.6% among high-H pairs, so invariance does nearly halve the incidence of harm. But it moves the probability that a concept is supportive from 8.1% to 9.0%, which is no change at all. Activation invariance is weakly informative about the absence of harm and uninformative about the presence of usefulness, and five-sixths of what an alignment objective would stabilize does not separate classes at all.


*Figure 1: Activation invariance does not imply discriminative usefulness. Each point is a class-concept pair (k, c), plotting activation invariance H(k, c) against discriminative strength D(k, c). High-H concepts span positive, near-zero, and negative D, contradicting the assumption that cross-domain invariance alone yields useful features.*

## 5.3 Discriminative consistency matters beyond activation invariance

A concept can fire evenly across all domains and still only matter in one of them. Activation invariance H cannot detect this, because it only looks at where a concept appears. The consistency score R looks at where a concept’s discriminative effect appears, and these turn out to be different things.

A concrete case separates the two. Consider a concept responding to outline and line weight. Such a concept fires on essentially every image, since photographs contain edges as readily as drawings, so H is high. Yet its discriminative influence may be concentrated almost entirely in cartoons, where outline is the dominant class cue, and be negligible in photographs, where texture and color carry the decision. Activation invariance sees a textbook invariant concept. Consistency sees a cartoon specialist. The distinction is between where a concept *fires* and where it *matters*.

The two scores are not independent. Both are entropies over the three source domains, so a pair that clears τR = 0.7 must have nonzero effect in every source domain. An even split across only two domains gives log 2/ log 3 = 0.63, which falls below the threshold. The same bound applies to H. High consistency therefore implies broad activation, and the data confirms this: only 8 of the 273 pairs with non-negligible effect are low-H but high-R (Figure 4). R acts as a filter applied inside the invariant population, not as a rival measure of invariance. But it is a filter that removes a great deal. Of the 1173 high-H pairs, only 50.7% also clear τR. Knowing that a concept is activation-invariant tells us almost nothing about whether its effect is consistent.

The central observation of this section emerges when the invariant population is split by the sign of D. Among invariant concepts that support the correct class, 72.4% (76/105) are also consistent. Among invariant concepts that hurt the correct class, only 36.0% (32/89) are. Invariant support is usually spread across domains; invariant harm is usually concentrated in one or two. The asymmetry strengthens monotonically as the threshold tightens, reaching 65.2% against 27.3% at τ = 0.8 and 49.4% against 15.4% at τ = 0.9, so the qualitative conclusion does not depend on a particular choice of threshold.

This matters for how the harmful invariant category should be read. Most of those concepts are not harmful everywhere; they fire in every domain and do their damage in a subset. An alignment objective that equalizes activation statistics cannot address them, because their activations are already balanced — that balance is what qualifies them as invariant in the first place. The asymmetry that makes them harmful is invisible to any activation-level criterion.

The neutral panel of Figure 4 makes a separate point: 38.4% of near-zero-D pairs land in the high-H, high-R cell. Entropy is scale-insensitive, so a concept with negligible influence spread evenly across domains scores as perfectly consistent. Consistency of a non-effect is meaningless, which is why R is interpreted only for pairs satisfying |D(k, c)| > τD.

Two limits bound what this section claims. First, a low R admits two readings: the effect may be genuinely confined to one domain, or merely skewed across three, since a split of (0.8, 0.15, 0.05) gives R = 0.56 with all domains active. We therefore call low-R concepts domain-contingent rather than spurious; R measures where an effect concentrates and says nothing about whether the underlying visual cue is causally unrelated to the label. Second, and more importantly, we do not claim that R carries information independent of the sign and magnitude of D. The harmful population is enriched approximately twofold in low-R pairs relative to the supportive population, so in this model the two are correlated, and separating them would require mass-matched interventions that we do not perform. The results of Section 5.4 are accordingly presented as validation that the diagnostic categories correspond to functional behavior, not as evidence that consistency is causally prior to sign.


*Figure 2: A single concept responding to the chest and forelimb region of four-legged mammals, shown across domains for dog (top) and horse (bottom). The concept is invariant and consistent, supports dog (D > 0) and harms horse (D < 0). Nothing about the detector is defective; the same visual evidence is read as dog-evidence on both animals.*

*Figure 3: A highly invariant concept with near-zero discriminative effect on all four classes shown. It responds to the four-legged body plan, which is shared across dog, horse and elephant and therefore separates none of them.*


*Figure 4: High activation invariance does not imply a consistent discriminative effect. Pairs are split by the sign and magnitude of D(k, c), then cross-tabulated by H and R at threshold 0.7. Among invariant pairs that support the correct class, 72.4% are also consistent; among invariant pairs that hurt it, only 36.0% are. The neutral panel shows why R needs a magnitude gate. Pairs active on fewer than 30 images for the class are excluded.*

## 5.4 The typology corresponds to functional prediction behavior

We next test whether the diagnostic categories describe real model behavior by ablating each bucket in SAE code space and re-evaluating. For an image with true label y, the concepts masked are those paired with y in the bucket. These are oracle interventions: they use the ground-truth label and are not deployable test-time methods. Their role is that of any controlled ablation. If removing a category changes prediction behavior in the direction the diagnostic predicts, the category captures something functional.

Two considerations frame every row of Table 3. First, the direction of the sign effect is tautological. D is defined as the drop in the true-class posterior under ablation, so masking negative-D pairs must raise that posterior. Only the magnitude, the asymmetry between source and target domains, and the margin over a matched control carry information. Second, we therefore pair every intervention with a size-matched, |D|-histogram-matched random control drawn from the support-floored pairs the target did not select, averaged over three seeds. Without such a control the objection that ablating any comparable set of concepts helps a weak domain would be unanswerable. Stratification is essential rather than cosmetic: 1269 of the 1542 support-floored pairs are neutral, so a uniformly sampled control is trivially easy to outperform.

Because the baseline leaves 19.78 percentage points of error on sketch, we also express each intervention as the fraction of that error it recovers, which makes effect sizes comparable across buckets of very different size.

Four results follow. First, the categories are functional and the controls are decisive. Masking the 138 harmful pairs raises sketch accuracy by 7.24 points macro, recovering 45.6% of the model's target-domain error. Masking 138 random pairs with a closely matched effect-mass profile lowers it by 20.98 points. The two move in opposite directions, a margin of 28 points. Every harmful bucket outperforms its control by 5 to 28 points; every supportive bucket underperforms its control by 40 to 63.

Second, the gains are not dictionary denoising. Of the 114,688 class–concept pairs, 111,530 have zero measured effect in every source domain. Masking all of them moves sketch by 0.08 points, recovering 0.6% of the error, despite their aggregate |D| mass (0.069) exceeding that of the 35-pair distributed-harm bucket (0.087). Effect mass spread thinly over a hundred thousand pairs cannot move a decision; the same mass concentrated in 35 pairs flips predictions. This also cautions against using aggregate |D| as an importance measure.

Third, the concepts that cost the most are the ones that score best on invariance. The 35 pairs that are both invariant and consistent in their harm — 0.03% of the grid — account for 25.1% of all target-domain errors, and the 32-pair harmful invariant bucket accounts for 22.6%. Per pair, distributed harm is 5.5 times more damaging than concentrated harm (0.72% against 0.13% of errors each). These are precisely the concepts an activation-alignment penalty has no gradient to act on.

Fourth, distributed rather than concentrated support is what the model runs on. Masking the 81 pairs of distributed support costs 39 points; masking the 54 pairs of concentrated support costs 0.65 points and is indistinguishable from its random control at −1.07. Across every intervention the three source domains move by less than one point while sketch moves by as much as 60, so the diagnostic is estimated only on source domains yet its consequences appear almost entirely in the domain it never saw.

|   | Source domains (macro) |   |   | Target |   |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Intervention | Art | Cartoon | Photo | Sketch macro | Δ | Errors recovered | Control Δ |
| Baseline (SAE reconstruction) | 99.35 | 99.19 | 99.78 | 83.63 | — | — | — |
| Mask all harmful (138 pairs) | 99.75 | 99.84 | 99.86 | 90.87 | +7.24 | 45.6% | −20.98 |
| Mask distributed harm (35) | 99.49 | 99.44 | 99.86 | 87.32 | +3.69 | 25.1% | −3.12 |
| Mask harmful invariant (32) | 99.49 | 99.41 | 99.86 | 86.96 | +3.33 | 22.6% | −2.10 |
| Mask concentrated harm (103) | 99.63 | 99.69 | 99.78 | 86.17 | +2.54 | 13.4% | −5.40 |
| Mask inert pairs only (111,530) | 99.35 | 99.19 | 99.78 | 83.71 | +0.08 | 0.6% | — |
| Mask concentrated support (54) | 99.29 | 99.12 | 99.78 | 82.98 | −0.65 | — | −1.07 |
| Mask distributed support (81) | 74.39 | 80.83 | 77.45 | 44.55 | −39.08 | — | +1.06 |
| Mask all supportive (135) | 56.91 | 63.11 | 59.72 | 23.63 | −60.00 | — | +2.50 |

*Table 3: Concept interventions. All masks are class-conditional and use the true label. Δ is against the SAE-reconstruction baseline. "Errors recovered" is the share of the baseline's 19.78 point target-domain error (micro) that the intervention removes. "Control Δ" is the mean change under three size-matched, |D|-histogram-matched random masks. Source and target accuracy are reported separately rather than pooled, since three of the four domains are in distribution.*

**Sufficiency of robust support.** We also invert the mask, retaining a single bucket and ablating everything else (Table 4). Retaining only the 76 robust-support pairs and ablating the remaining 114,612 — 99.93% of the grid — yields 94.02% macro and 93.56% micro on sketch, above the unablated model, with source domains at 99.85% to 100%. The frozen model therefore already contains a concept subset sufficient for 94% accuracy on a domain it never saw, and that subset is identifiable from source-domain statistics alone. Its failure to reach that accuracy is a matter of selectivity rather than of representational capacity.

Two comparisons keep this interpretable rather than circular. Retaining 76 arbitrary pairs, size-matched per class, collapses the model to exactly chance (14.29% macro, with every prediction becoming person), so the accuracy reflects which concepts were retained and not the label used to address the mask. And since all three keep-only rows share the same label-injection component, differences between them are informative: concentrated support alone reaches 25.00%, robust support alone reaches 94.02%, and adding the concentrated pairs on top of robust support moves it only to 95.00%. Domain-contingent support is close to redundant.

This comparison does, however, conflate consistency with effect magnitude: robust support carries roughly ten times the |D| mass of concentrated support. We therefore do not claim that distributed support is sufficient because it is distributed rather than because it is larger.

| Keep only | Pairs kept | Sketch macro | Sketch micro |
| --- | --- | --- | --- |
| — (baseline) | — | 83.63 | 80.22 |
| 76 uniformly random pairs (3 seeds) | 76 | 14.29 / 14.25 / 14.82 | 4.07 / 4.05 / 4.84 |
| Concentrated support | 54 | 25.00 | 18.25 |
| Robust support (H, R ≥ 0.7, D > τD) | 76 | 94.02 | 93.56 |
| All supportive | 135 | 95.00 | 94.25 |

*Table 4: Keep-only interventions. Every row retains concepts paired with the true label and therefore injects label information; the random row measures how much of the effect that injection accounts for. It collapses to chance, so the robust-support result reflects the retained bucket.*

## 5.5 Domain-contingent effects concentrate in the stylized source domains

Because R is an entropy over per-domain effects, for every low-R pair we can ask which source domain carries the effect, as argmax over d of |Dd(k, c)|. This requires no intervention.

A raw table of those counts would be confounded, since a domain in which all effects are systematically larger wins the argmax regardless of concentration. Mean |Dd| over non-neutral pairs is indeed uneven, at 2.13 × 10−3 for art painting, 1.92 × 10−3 for cartoon and 1.15 × 10−3 for photo, and active-image exposure differs as well. We therefore compare the low-R distribution against the same distribution for high-R pairs, which are by construction not concentrated and so provide the appropriate null.

Concentrated support is enriched 2.3-fold in cartoon relative to the null (46.3% against 19.8%). Concentrated harm is enriched 1.7-fold in art painting (37.9% against 22.9%). Photo is depleted three to four-fold in both, at 5.6% and 6.8% against a null of roughly 22%. Cartoon's dominance of the harmful column, at 55.3%, is not a concentration effect, since the null is 54.3%: cartoon supplies most harmful effect in this model whether or not that effect is domain-contingent, and only the supportive column shows genuine cartoon-specific concentration.

The pattern is that domain-contingent effect resides in the stylized source domains rather than in photographs. Since the held-out target is itself a stylized, texture-poor domain, the concepts carrying domain-contingent effect are the style-sensitive ones, which is consistent with their influence failing to transfer to a different style. We record this as a coherent reading rather than a demonstrated mechanism; establishing it would require per-domain interventions we do not perform, and part of photo's depletion is mechanical given its smaller mean effect and lower exposure.

## 5.6 Concepts are not class-isolated, and harm is largely misdirected support

Because D is class-conditional, the same latent can support one class and hurt another. We call a concept conflicting if it satisfies D(k, c) > τD for at least one class and D(k′, c) < −τD for at least one other, both above the support floor. No co-activity assumption is needed: D(k, c) accumulates only over images of class k on which c actually fires, so a conflict certifies that the concept fires on images of both classes.

Of the 533 concepts with at least one class above the support floor, 139 are non-neutral for at least one class. Of those, 56 (40.3%) are conflicting. More tellingly, 108 of the 138 harmful class–concept pairs, or 78.3%, involve a concept that supports some other class.

This is the paper's answer to why a trained model contains harmful concepts at all. Negative discriminative effect is, four times out of five, not a spurious or defective feature. It is a genuine feature attached to the wrong class. The chest-and-forelimb concept of Figure 2 is the archetype: shared visual structure that the representation never separated, disambiguated in distribution by cues that weaken under shift, after which the shared detector still fires and votes for the wrong class. The most conflicted concepts by contrast max_k D − min_k D support giraffe while harming person, support dog while harming elephant, and support horse while harming dog.

The consequence for remediation is that "remove the spurious feature" is the wrong prescription, because the feature is not spurious. The problem is class isolation: the same evidence must be read differently depending on what else is present. That is a property of the read-out rather than of the features.

Table 5 gives the per-class picture. The weakest class on the target domain, dog at 48.06%, carries both the most harmful pairs and the most conflicting concepts acting against it, while the strongest, guitar at 97.37%, carries almost none. But person carries 32 harmful pairs and 26 conflicting concepts and still reaches 95.62%, so conflict burden alone does not predict per-class failure. The resolution is visible in the collapse behavior of the degenerate masks in Section 5.4: surviving predictions consistently pile onto person, which is the model's default class and therefore wins ties. Class-level robustness depends on at least two quantities, the conflict burden a class carries and whether it wins or loses the resulting ties. A one-dimensional per-class risk score cannot express this, and with seven classes we do not attempt a correlation. Giraffe, at 76.89% with only 10 conflicting concepts against it, is not explained by either quantity.

| Class | Sketch acc. (micro) | Pairs above floor | Supportive | Harmful | Neutral | Harmed by conflicting concept |
| --- | --- | --- | --- | --- | --- | --- |
| dog | 48.06 | 286 | 33 | 50 | 203 | 34 |
| elephant | 95.54 | 208 | 23 | 16 | 169 | 15 |
| giraffe | 76.89 | 185 | 15 | 10 | 160 | 10 |
| guitar | 97.37 | 150 | 12 | 3 | 135 | 2 |
| horse | 83.21 | 243 | 24 | 24 | 195 | 18 |
| house | 88.75 | 144 | 9 | 3 | 132 | 3 |
| person | 95.62 | 326 | 19 | 32 | 275 | 26 |

*Table 5: Per-class diagnostic counts at τD = 10−4 with support floor 30. The final column counts concepts that harm this class while supporting another.*

## 6 Discussion

## 6.1 Invariance must be discriminatively aligned

The main lesson is that activation invariance is not sufficient, and our measurements let us say how insufficient. Conditioning on invariance nearly halves the probability that a concept is harmful but leaves the probability that it is useful unchanged, and 83.5% of invariant class–concept pairs have no measurable discriminative effect at all. Alignment objectives may therefore preserve a great deal of stable but unhelpful information. The relevant unit is not the globally invariant feature but the class-conditional concept whose activation and effect are both stable.

The sharper point concerns what alignment cannot reach. A concept that is invariant and consistent in its harm has, by construction, balanced activation statistics across source domains; that balance is precisely what qualifies it as invariant. A penalty on cross-domain divergence in feature distributions therefore has no gradient to apply to it. Its pathology resides entirely in the sign of its effect on the class posterior, which no activation-level statistic measures. Empirically these are the most expensive concepts in the model: 35 class–concept pairs, 0.03% of the grid, account for a quarter of all target-domain errors.

## 6.2 Failures of generalization can be failures of selectivity

The conventional reading of a domain-generalization failure is a deficit: the model did not learn features that transfer. Our keep-only results complicate that reading. Retaining 0.07% of the class–concept grid — the pairs that are invariant, consistent and supportive — raises held-out accuracy above that of the unablated model while leaving source accuracy at or above its original level. The representation already contains evidence sufficient for substantially better target-domain accuracy than the model achieves.

This is an oracle bound and not a method, since applying the mask requires each image's label. But the bound is informative about diagnosis rather than remedy: when a model underperforms out of distribution, one should ask not only whether the required features are absent but also whether they are present and diluted. The two diagnoses imply different interventions, and the framework distinguishes them.

A related observation is that ablations which are no-ops in distribution can be decisive out of distribution. Removing 99.93% of the grid leaves source accuracy unchanged, because in distribution the model carries enough redundant evidence that no single concept determines the decision. The same removal is worth ten points on the target domain, where the model operates closer to its decision boundary. Redundancy, rather than invariance, may be what degrades under shift.

## 6.3 Class-conditional analysis is necessary, and harm is misdirected support

A concept cannot be assigned a single global role without reference to class. The same latent can support one class and interfere with another. This is why all scores are defined as H(k, c), D(k, c), and R(k, c) rather than H(c), D(c), and R(c). Class-conditionality is not a technical detail; it is the mechanism by which the framework detects class conflict and lack of class isolation.

Our measurements give this a stronger form. Of the harmful class–concept pairs, 78.3% involve a concept that supports some other class. Negative discriminative effect is therefore, in the large majority of cases, not a spurious or defective feature but a genuine feature attached to the wrong class. A concept detecting the chest and forelimb region of a four-legged mammal is correct evidence for dog and, on a horse, becomes an argument for dog anyway. A global per-concept score would average these opposing effects toward zero and classify the concept as inert, concealing the mechanism entirely.

This reframes what remediation would involve. The instruction "remove the spurious feature" is misdirected, because the feature is not spurious and removing it would cost the class it legitimately serves. The problem is class isolation: the same evidence must be read differently depending on what else is present. That is a property of the read-out rather than of the feature set, and it suggests that objectives targeting class separation in concept space may be more appropriate than objectives targeting domain alignment.

## 6.4 Class-level profiles should be diagnostic, not prescriptive

Aggregating concept categories yields per-class summaries, but these should not be oversold, and our own results show why. Conflict burden alone does not predict per-class target accuracy: the class carrying the second-heaviest burden is also the second most accurate, because it is the model's default prediction and therefore wins the ties that conflict produces. Class-level robustness depends on at least two quantities, and a one-dimensional risk score cannot express their interaction. Validating any such quantity as a model-selection criterion would require additional datasets, seeds, algorithms, and pre-registered selection protocols. We leave that to future work and report per-class counts descriptively.

## 6.5 Diagnostic interventions are not deployment methods

The interventions use the true class to decide which concepts to mask or keep. These are oracle interventions and should be interpreted as tests of the diagnostic categories, not as test-time algorithms. Their role is analogous to a controlled ablation: if removing a bucket changes prediction behavior in the expected direction, then the bucket captures functional model behavior.

Two properties of our design make this more than a formality. The direction of any sign-based effect is guaranteed by the definition of D, so we rely on matched random controls rather than on direction: the harmful buckets outperform size- and magnitude-matched random masks by 5 to 28 points, and the matched control for the full harmful set moves target accuracy in the opposite direction. And the keep-only results are bounded by a control that retains an equal number of arbitrary pairs, which collapses the model to chance, establishing that those results reflect the retained concepts rather than the label used to address the mask.

Whether any of this transfers to a label-free setting remains open. The label-free variants we examined, which collapse the mask across classes, are null — but they also discard the class-conditionality that our analysis identifies as the locus of the signal, so they are not a decisive test.

## 7 Limitations

First, SAE latents are candidate concepts, not guaranteed human-semantic units. Some latents may correspond to recognizable object parts or styles, while others may remain difficult to name. The quantitative claims depend on sparse reconstruction and ablation behavior, not on perfect human interpretability.

Second, the framework is post-hoc. It may use class labels and domain labels to analyze a trained model. This is appropriate for auditing and understanding, but it should not be confused with a deployable DG training or test-time adaptation method.

Third, oracle interventions use the true class, and possibly target-domain statistics, to define masks. They validate concept categories but do not imply that the same accuracy changes can be achieved without labels at test time.

Fourth, the main experiments cover PACS with a single ERM ResNet-50 checkpoint and a single sparse autoencoder, so the scope of the empirical claim is correspondingly limited. Because SAE dictionaries are not identifiable across independently trained autoencoders, we would compare only aggregate quantities across seeds — typology proportions, the support and harm asymmetry, and the fraction of target error attributable to each bucket — and we have not done so. Additional datasets, algorithms, backbones and seeds would strengthen external validity but are not necessary for the central diagnostic contribution. In particular, the prediction that an explicitly alignment-trained model should retain comparably many invariant-and-harmful concepts follows from our argument and remains untested.

Fifth, threshold choices affect bucket assignments. We report sensitivity to τH and τR in Section 5.3, where the qualitative conclusion strengthens monotonically as the threshold tightens. The main conclusion should not depend on a single threshold value.

Sixth, we do not establish that consistency carries information independent of the sign and magnitude of discriminative effect. Harmful pairs are enriched approximately twofold in low-consistency pairs, and the keep-only comparisons conflate consistency with effect magnitude. Establishing independence would require mass-matched bucket comparisons. Relatedly, aggregate effect mass proves to be a poor proxy for consequence in our data — 111,530 pairs whose total mass exceeds that of a 35-pair bucket recover forty times less error — so such comparisons would need to match the distribution of per-pair magnitudes rather than their sum.

Finally, reconstruction fidelity is a necessary precondition for interpreting ablations. If an SAE reconstruction substantially changes model predictions, downstream concept effects must be described as effects in SAE reconstruction space rather than direct effects in the original model.

## 8 Conclusion

This paper introduced sparse concept diagnostics for domain generalization. The central argument is that DG should not be diagnosed by activation invariance alone. A useful invariant concept must also be class-supporting, consistent in its discriminative effect across domains, and isolated from competing classes. By decomposing trained vision models into SAE-derived candidate concepts and scoring each concept class-conditionally, the framework distinguishes robust support from harmful invariance, domain-contingent cues, and class conflict.

Applied to a frozen ERM ResNet-50 on PACS, the framework yields three findings that target-domain accuracy alone cannot express. Invariance is weakly informative about the absence of harm and uninformative about the presence of usefulness. The concepts most costly to held-out accuracy are those that score best on invariance, and are therefore precisely the concepts an activation-alignment objective cannot act upon. And harmful concepts are, in the large majority, not spurious features but genuine features attached to the wrong class, which makes class isolation rather than feature removal the appropriate target.

Rather than proposing a new DG algorithm, sparse concept diagnostics offer a post-hoc audit of what a trained model has learned, how much of its held-out error is attributable to identifiable concepts, and whether its shortfall reflects missing evidence or evidence that is present but diluted.

## Broader Impact Statement

This work is primarily diagnostic and interpretability-oriented. Better understanding of DG failure modes may help practitioners identify spurious, harmful, or class-conflicting cues before deploying models under distribution shift. However, interpretability tools can also create false confidence if their outputs are treated as complete explanations. We therefore emphasize reconstruction fidelity, threshold sensitivity, and the distinction between oracle diagnostics and deployment-time methods. The framework should be used as one component of a broader evaluation pipeline that includes standard accuracy, calibration, robustness, and domain-specific safety analysis.

## References

- Kei Akuzawa, Yusuke Iwasawa, and Yutaka Matsuo. Adversarial invariant feature learning with accuracy constraint for domain generalization. In European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases, 2019.

- Martin Arjovsky, Leon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. In arXiv preprint arXiv:1907.02893, 2019.

- David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6541–6549, 2017.

- Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023.

- David Chanin et al. A is for absorption: Studying feature splitting and absorption in sparse autoencoders, 2024.

- Liang Chen, Yong Zhang, Yibing Song, Anton van den Hengel, and Lingqiao Liu. Domain generalization via rationale invariance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1751–1760, 2023.

- Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In International Conference on Learning Representations, 2024.


- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. Transformer Circuits Thread, 2022.

- Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux, Julien Colin, R’emi Cad‘ene, and Thomas Serre. CRAFT: Concept recursive activation factorization for explainability. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2711–2721, 2023.

- Tigran Galstyan, Hrayr Harutyunyan, Hrant Khachatrian, Greg Ver Steeg, and Aram Galstyan. Failure modes of domain generalization algorithms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19077–19086, 2022.

- Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. Journal of Machine Learning Research, 17(59):1–35, 2016.

- Leo Gao, Tom Dupr’e la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. In International Conference on Learning Representations, 2025.

- Amirata Ghorbani, James Wexler, James Y. Zou, and Been Kim. Towards automatic concept-based explanations. In Advances in Neural Information Processing Systems, volume 32, 2019.

- Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In International Conference on Learning Representations, 2021.

- Sangyu Han, Yearim Kim, and Nojun Kwak. Causal interpretation of sparse autoencoder features in vision, 2025.

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, 2016.

- Adam Karvonen, Can Rager, Samuel Marks, and Neel Nanda. Evaluating sparse autoencoders on targeted concept erasure tasks, 2024.

- Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory Sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In International Conference on Machine Learning, 2018.

- Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International Conference on Machine Learning, 2020.

- Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M. Hospedales. Deeper, broader and artier domain generalization. In IEEE International Conference on Computer Vision, 2017.

- Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C. Kot. Domain generalization with adversarial feature learning. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5400–5409, 2018. doi: 10.1109/CVPR.2018.00566.

- Hyesu Lim, Jinho Choi, Jaegul Choo, and Steffen Schneider. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. In International Conference on Learning Representations, 2025.

- Aleksandar Makelov, George Lange, and Neel Nanda. Towards principled evaluations of sparse autoencoders for interpretability and control, 2024.

- Gonçalo Paulo and Nora Belrose. Sparse autoencoders trained on the same data learn different features, 2025.

- Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In IEEE International Conference on Computer Vision, 2017.


- Samuel Stevens, Wei-Lun Chao, Tanya Berger-Wolf, and Yu Su. Sparse autoencoders for scientifically rigorous interpretation of vision models, 2025.

- Baochen Sun and Kate Saenko. Deep CORAL: Correlation alignment for deep domain adaptation. In European Conference on Computer Vision Workshops, 2016.

- Vladimir N. Vapnik. An overview of statistical learning theory. IEEE Transactions on Neural Networks, 10 (5):988–999, 1999.

- Chih-Kuan Yeh, Been Kim, Sercan O. Arik, Chun-Liang Li, Tomas Pfister, and Pradeep Ravikumar. On completeness-aware concept-based explanations in deep neural networks. In Advances in Neural Information Processing Systems, volume 33, pp. 20554–20565, 2020.

- Han Yu, Xingxuan Zhang, Renzhe Xu, Jiashuo Liu, Yue He, and Peng Cui. Rethinking the evaluation protocol of domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21897–21908, 2024.

- Han Zhao, Remi Tachet Des Combes, Kun Zhang, and Geoffrey J. Gordon. On learning invariant representa- tions for domain adaptation. In International Conference on Machine Learning, 2019.

- Han Zhao, Chen Dan, Bryon Aragam, Tommi S. Jaakkola, Geoffrey J. Gordon, and Pradeep Ravikumar. Fundamental limits and tradeoffs in invariant representation learning. Journal of Machine Learning Research, 23(340):1–56, 2022.
