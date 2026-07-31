## When Invariance Is Not Enough: Sparse Concept Diagnostics for Domain Generalization

## Anonymous authors Paper under double-blind review

## Abstract

[TODO: Revise abstract in end] Domain generalization methods are often motivated by the search for invariant representations, but activation invariance alone does not reveal whether a feature is useful for classification. We propose sparse concept diagnostics, a post-hoc framework for analyzing trained domain-generalization models by decomposing internal vision representations into sparse autoencoder candidate concepts. For each concept, we compute class-conditional scores measuring activation invariance across domains, ablation-based discriminative effect on the ground-truth class, and cross-domain consistency of that effect. These scores separate robust class-supporting concepts from invariant distractors, domain- contingent cues, and class-conflicting concepts. On PACS models, our analysis shows that domain-generalization failures are not explained simply by an absence of invariant activations; rather, failures arise when invariant concepts are not consistently class-supporting or when concepts support one class while interfering with another. Aggregating these categories yields class-level and model-level diagnostic profiles, while oracle concept interventions validate that the identified categories correspond to functional prediction behavior. Our results suggest that domain-generalization representations should be evaluated not only by whether concepts

are invariant, but by whether they are discriminatively aligned and class-isolated.

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

and = 7 for the PACS object classes (Li et al., 2017). The primary model is an ERM-trained ResNet backbone (He et al., 2016; Vapnik, 1999). [TODO: Additional models or checkpoints are included only when the corresponding SAE diagnostics are complete.] [URL 🔗](#page-0)

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

Because entropy is scale-insensitive, R is interpreted only for concepts satisfying |D(k, c)| > τD. A concept with tiny effects in every domain can have high entropy while still being unimportant. We treat such concepts as neutral rather than robust. Optionally, we also compute a sign-consistency score,

which is high when domain-wise effects share the same sign and low when positive and negative effects cancel across domains. Unless otherwise stated, R refers to the magnitude-consistency score in Eq. 27, and concepts near D = 0 are excluded before typology construction. [URL 🔗](#page-0)

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

The main experiments use PACS (Li et al., 2017), which contains four visual domains: art painting, cartoon, photo, and sketch. The task contains seven object classes. We designate the sketch domain as the target holdout, utilizing the remaining three as source domains. Unless otherwise stated, models are trained with the DomainBed protocol (Gulrajani & Lopez-Paz, 2021), employing non-oracle checkpoint selection based on the highest average accuracy across all source domains. The primary analysis uses an ERM-trained ResNet-50 model. We focus on the cleanest completed setting rather than claiming broad benchmark coverage. Additional DG algorithms, backbones, or datasets may be included in Appendix ?? if their diagnostics are fully completed. [URL 🔗](#page-0)

## 4.2 SAE training

The SAE is trained to reconstruct the feature maps from the final layer before the classification head. The ResNet-50 has a (7 7 spatial resolution, and 2048 channels. We use a Top-K SAE with a dictionary size of 16,384 and enforce top-16 sparsity per spatial token. Training proceeds for 250 epochs using the Adam optimizer with an initial learning rate of 3×10−4 and a cosine decay schedule. To strictly preserve the domain generalization setting, the SAE is trained and analyzed exclusively on the source domains, while the target domain is reserved solely for verification. The backbone and classifier remain entirely frozen throughout.

To verify training convergence and ensure that higher dimensional interventions map reliably back to the original feature space, we evaluate the classifier on SAE-reconstructed features. The goal is to confirm that these reconstructions induce minimal deviation from the original model behavior.

Unless otherwise stated, the SAE is trained on source-domain features only. The diagnostic scores H,D, R and thresholds are also estimated on source-domain data. Target-domain examples are used only for post-hoc evaluation of whether source-estimated concept categories explain held-out behavior.


*Table 2: Overall SAE reconstruction fidelity on PACS for ERM ResNet-50.*

| Domain |   |   | Original Acc. SAE Recon. Acc. Drop |
| --- | --- | --- | --- |
| Art painting | 99.31 | 99.31 | 0.00 |
| Cartoon | 99.03 | 98.98 | 0.05 |
| Photo | 99.77 | 99.77 | 0.00 |
| Sketch | 80.29 | 80.20 | 0.10 |

## 4.3 Thresholds and sensitivity

The main typology uses thresholds τH, τR, and τD. We use τH = [TODO: 0.8 or final value] and τR = [TODO: 0.8 or final value] in the main analysis. The threshold τD is selected as [TODO: 1e-3 or value]. The appendix reports threshold sensitivity for τH {0.7, 0.8, 0.9}, τR {0.7, 0.8, 0.9}, and multiple choices of τD including clustering based approaches and fixed magnitude thresholding.

## 5 Results

## 5.1 SAE reconstructions preserve classifier behavior

Before utilizing SAE latents for analysis, we verify that SAE reconstructions preserve the classifier’s behavior. Table 2 reports domain-level reconstruction fidelity. Across PACS domains, SAE-reconstructed features closely match the original ERM features: the drops are 0.00 percentage points for art painting and photo, 0.05 percentage points for cartoon, and 0.10 percentage points for sketch. Thus, even on the weakest domain, sketch, reconstruction changes overall accuracy by less than 0.1 percentage points. [URL 🔗](#page-0)

## 5.2 Activation-invariant concepts are not necessarily useful

Many domain generalization algorithms rest on a common premise: an attribute that persists across the source domains is likely to be causal and to persist on unseen target domains, whereas an attribute confined to a few domains is likely a spurious correlation Arjovsky et al. (2019). Under this view, suppressing domain-specific features improves out-of-distribution generalization by forcing the model to rely on the causal, invariant ones. [URL 🔗](#page-0)

We find this premise necessary but not sufficient. Activation invariance, the degree to which a concept fires uniformly across domains, captured by H(k, c), does not by itself make a concept useful. Each concept additionally carries a discriminative strength D(k, c), the signed effect of its activation on the model’s confidence in the true class. If invariance alone were sufficient, high-H concepts would concentrate at positive D. Instead, the high-H region contains positive, neutral, and negative discriminative effects (Figure 1). [URL 🔗](#page-0)

These three regimes have distinct interpretations. A neutral concept (D 0) activates across all domains yet does not separate classes. For instance a legs concept (Fig. 2 fires on elephants, dogs, and horses alike and therefore contributes nothing to discriminating among them. Concept (D < 0) is domain invariant yet actively harmful. We observe a dog-chest concept (Fig. 3, that also activates on horses across every domain, owing to their visually similar chests, and consistently pushes the model to incorrectly classify horses as dogs. Its activation invariance is high, but its discriminative power is negative for horses. A positive concept (D > 0) is the canonical case of a feature that supports the correct class, for instance the dog-chest activated on an image of a dog. [URL 🔗](#page-0)

In fact, among high-H concepts, on average 7.2% are positive-D, 85.6% are neutral, and 7.2% are negative-D across all seven classes. This shows that activation invariance alone does not identify robust class evidence, the vast majority of highly active concepts carry no discriminative signal.


*Figure 1: Activation invariance does not imply discriminative usefulness. Each point is a class-concept pair (k, c), plotting activation invariance H(k, c) against discriminative strength D(k, c). High-H concepts span positive, near-zero, and negative D, contradicting the assumption that cross-domain invariance alone yields useful features.*

## 5.3 Discriminative consistency matters beyond activation invariance

A concept can fire evenly across all domains and still only matter in one of them. Activation invariance H cannot detect this, because it only looks at where a concept appears. The consistency score R looks at where a concept’s discriminative effect appears, and these turn out to be different things.

The two scores are not independent. Both are entropies over the three source domains, so a pair that clears τR = 0.7 must have nonzero effect in every source domain. An even split across only two domains gives log 2/ log 3 = 0.63, which falls below the threshold. The same bound applies to H. High consistency therefore implies broad activation, and the data confirms this: only 3 of the 217 pairs with non-negligible effect are low-H but high-R (Fig. 5). R acts as a filter applied inside the invariant population, not as a rival measure of invariance. But it is a filter that removes a lot. Of the 1068 high-H pairs, only 46.4% also clear τR. Knowing that a concept is activation-invariant tells us almost nothing about whether its effect is consistent. [URL 🔗](#page-0)

The pattern becomes clearer when we split the invariant pairs by the sign of D. Among invariant concepts that support the correct class, 72.5% (74/102) are also consistent. Among invariant concepts that hurt the correct class, only 29.7% (19/64) are. In short, invariant support is usually spread across domains, while invariant harm is usually concentrated in one or two. This matters for how we read the harmful invariant category of Section ??. Most of those concepts are not harmful everywhere. They fire in every domain but do their damage in a few. An alignment objective that equalises activation statistics cannot help here, because these concepts already have balanced activations. The asymmetry that makes them harmful is invisible to it. The neutral panel of Fig. 5 makes a separate point: 35.2% of near-zero-D pairs land in the high-H, high-R cell. Entropy is scale-insensitive, so tiny effects spread evenly still score as consistent. This is why R must be read behind the magnitude gate introduced in Section ??. [URL 🔗](#page-0)


*Figure 2: A highly invariant concept that has a high positive discrimination on the dog class, and a high negative discrimination on the horse class.*

*Figure 3: A highly invariant concept that has a near zero discrimination on the 4 classes above.*

We then test whether low-R concepts behave differently as a group. We mask every class–concept pair below a threshold on R and sweep the threshold (Fig. 4, Table 3). The three source domains barely move: +0.15, +0.78, and +0.01 percentage points. Held-out sketch accuracy rises by 7.67 points at τR = 0.8. The gain appears only in the domain the diagnostic never saw. The shape of the curve is also informative. Most of the effect arrives by τR = 0.1 (+3.35 points), where we remove only concepts whose effect sits almost entirely in one domain. Accuracy then climbs gradually and drops back to +6.38 at τR = 0.9. At that point we start removing consistent concepts along with concentrated ones. If ablation were simply good for a weak domain, the curve would keep rising. The turnover suggests the mask is selecting a specific population instead. [URL 🔗](#page-0)


*Figure 4: Held-out sketch accuracy as concepts with concentrated discriminative effect are removed. The horizontal axis is the threshold below which a class–concept pair is masked; zero masks nothing. Accuracy peaks at +7.6 points at τR = 0.8, then declines as consistent concepts start being removed as well.*

Three limits on this result. First, the mask uses each image’s true label to decide what to remove, so it is an oracle diagnostic and not something that could run at test time. [TODO: Report the label-free variant here once available; it is the only version relevant to deployment.] Second, the mask is not gated on so many of the pairs it removes have near-zero effect. [TODO: Rerun the sweep masking only pairs with > τD, and report pair counts. If most of the sketch gain survives, the claim is about concentrated discriminative concepts. If not, part of it is dictionary denoising, and this paragraph should say so.] Third, a low R can mean two things. The effect may be genuinely confined to one domain, or merely skewed across three: a split of (0.8, 0.15, 0.05) gives R = 0.56 with all domains active. For this reason we call low-R concepts domain-contingent rather than spurious. R measures where an effect concentrates. It says nothing about whether the underlying visual cue is causally unrelated to the label. [TODO: Add controls before finalising: a matched-size random mask, and an inverse mask that keeps low-R pairs and removes high-R ones. Without these a reviewer can argue that removing any large set of concepts helps a weak domain.]

|   | Source domains | Target |
| --- | --- | --- |
| Configuration |   | Art painting Cartoon Photo Sketch |
| Original model | 99.31 | 99.03 99.77 80.29 |
| SAE reconstruction | 99.31 | 98.98 99.77 80.20 |
| + rare-concept mask (nk,c < 30) | 99.34 | 99.03 99.81 83.40 |
| + low-R mask (R < 0.8) | 99.48 | 99.81 99.82 91.08 |

*Table 3: Per-domain accuracy under successive concept interventions. Each row adds to the one above it. The two masking rows use the true label and are oracle diagnostics. Source and target accuracy are reported separately rather than pooled, since three of the four domains are in distribution.*


*Figure 5: High activation invariance does not imply a consistent discriminative effect. Pairs are split by the sign and magnitude of D(k, c), then cross-tabulated by H and R at threshold 0.7. Among invariant pairs that support the correct class, 72.5% are also consistent; among invariant pairs that hurt it, only 29.7% are. The neutral panel shows why R needs a magnitude gate. Pairs active on fewer than 30 images for the class are excluded (Section ??).*

## 5.4 The concept typology reveals distinct class-level failure modes

We next count the discriminative concept mass in each typology bucket. Table ?? should report either raw counts or |D|-weighted masses by class. The important comparison is not just how many invariant concepts exist, but how many of them are robust-supporting versus harmful or domain-contingent.

Expected write-up after results are available. [TODO: Insert: Classes with lower target accuracy have lower robust support mass and/or higher harmful invariant or conflict mass. Name the strongest classes and weakest classes.]

## 5.5 Harmful invariant concepts demonstrate why invariance is not enough

A central category is the high-H, high-R, negative-D bucket. These concepts are stable across domains and consistently discriminative, but in the wrong direction: ablating them increases the model’s confidence in the correct class. This bucket directly challenges the assumption that invariant concepts are automatically desirable.

Expected write-up after results are available. [TODO: Insert: Give one or two concrete concept examples. Describe what the concept appears to capture, which class it is evaluated for, its H/D/R scores, and why it is harmful or supportive. Avoid overclaiming semantic labels unless visual evidence is clear.]

## 5.6 Class-conflict profiles reveal lack of class isolation

Because D is class-conditional, the same concept may support one class and hurt another. This motivates conflict mass. A high-conflict concept is not merely non-invariant; it is shared in a way that interferes with class separation. This section should use the M or conflict processor to show how often concepts have positive effect for one class and negative effect for another.

Expected write-up after results are available. [TODO: Insert: Classes with high conflict mass exhibit lower accuracy or more unstable intervention behavior. Identify whether conflict is driven by specific concept families or distributed broadly across many concepts.]


*Table 4: Class-level diagnostic profiles. Higher robust support mass is desirable; higher harmful invariant, domain-contingent, and conflict mass indicate potential failure modes.*

| Class Accuracy RSM | HIM | DCM | Conflict |
| --- | --- | --- | --- |
| Class 0 | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |
| Class 1 | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |
| Class 2 | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |
| Class 3 | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |
| Class 4 | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |
| Class 5 | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |
| Class 6 | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |

*Table 5: Model or checkpoint diagnostic profiles. If only ERM checkpoints are analyzed, label this table as checkpoint-level rather than algorithm-level.*

| Model/checkpoint | Accuracy RSM | HIM | DCM | Conflict |
| --- | --- | --- | --- | --- |
| ERM checkpoint [TODO: ID] | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |   |
| ERM checkpoint [TODO: ID] | [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |   |
| [TODO: Optional additional DG model] [TODO: ] [TODO: ] [TODO: ] [TODO: ] [TODO: ] |   |   |   |   |

## 5.7 Model or checkpoint profiles summarize representational failure modes

We aggregate class-level profiles to produce model or checkpoint profiles. This is the paper’s model-level extension. The goal is not to propose a new model-selection metric, but to summarize how discriminative concept mass is distributed across useful and harmful categories.

Expected write-up after results are available. [TODO: Insert: Higher-performing checkpoints tend to have higher RSM and lower HIM/DCM/conflict, or explain any deviations. Phrase cautiously: profiles summarize failure modes rather than serving as a validated model-selection rule.]

## 5.8 Oracle interventions validate the diagnostic categories

Finally, we test whether concept categories correspond to functional behavior by masking or keeping specific sets of concepts. These interventions are diagnostic. Some use the ground-truth class k and are therefore oracle interventions. We report this explicitly.

Expected write-up after results are available. [TODO: Insert: Removing harmful concepts improves or changes accuracy by X. Removing only harmful invariant concepts changes accuracy by Y, validating that invariant-but-harmful concepts are functionally meaningful. Keeping only robust support yields Z, showing whether robust-support concepts contain sufficient evidence. The no-label global mask is weaker/stronger by W and should be interpreted as a leakage sanity check.]

## 6 Discussion

## 6.1 Invariance must be discriminatively aligned

The main lesson is that activation invariance is not sufficient. A concept that appears across domains may be neutral, harmful, or useful only in a subset of domains. This explains why global alignment objectives can be


*Table 6: Concept interventions. The column “oracle information” explicitly states whether the intervention uses the true class or target-domain labels.*

| Intervention | Rule | Oracle information Purpose |   | Accuracy |
| --- | --- | --- | --- | --- |
| Original model | None | None | Baseline | [TODO: |
| SAE reconstruction |   | None after SAE reconstruction None | Reconstruction baseline | [TODO: |
| Remove negative-D Mask D(k, c) < |   | True class k | Test harmful concepts | [TODO: |
| Remove harmful invariant Mask high H, high R, negative D |   | True class k | Test invariant harm | [TODO: |
| Keep robust support | Keep high H, high R, positive D | True class k | Test robust sufficiency | [TODO: |
| Global no-label mask | Mask globally harmful concepts No true test label Leakage sanity check |   |   | [TODO: |

insufficient: alignment may preserve stable but unhelpful information. The relevant unit is therefore not the globally invariant feature, but the class-conditional concept whose activation and effect are both stable.

## 6.2 Class-conditional analysis is necessary

A concept cannot be assigned a single global role without reference to class. The same latent can support one class and interfere with another. This is why all scores are defined as H(k, c), D(k, c), and R(k, c) rather than H(c), D(c), and R(c). Class-conditionality is not a technical detail; it is the mechanism by which the framework detects class conflict and lack of class isolation.

## 6.3 Model-level profiles should be diagnostic, not prescriptive

Aggregating concept categories yields useful model-level summaries, but these summaries should not be oversold. A model profile can reveal whether a checkpoint has more robust-supporting mass or more harmful invariant mass. It may explain why one checkpoint behaves differently from another. However, validating these quantities as model-selection criteria would require additional datasets, seeds, algorithms, and pre-registered selection protocols. We leave that to future work.

## 6.4 Diagnostic interventions are not deployment methods

The strongest interventions use the true class to decide which concepts to mask or keep. These are oracle interventions and should be interpreted as tests of the diagnostic categories, not as test-time algorithms. Their role is analogous to a controlled ablation: if removing a bucket changes prediction behavior in the expected direction, then the bucket captures functional model behavior.

## 7 Limitations

First, SAE latents are candidate concepts, not guaranteed human-semantic units. Some latents may correspond to recognizable object parts or styles, while others may remain difficult to name. The quantitative claims depend on sparse reconstruction and ablation behavior, not on perfect human interpretability.

Second, the framework is post-hoc. It may use class labels and domain labels to analyze a trained model. This is appropriate for auditing and understanding, but it should not be confused with a deployable DG training or test-time adaptation method.

Third, oracle interventions use the true class, and possibly target-domain statistics, to define masks. They validate concept categories but do not imply that the same accuracy changes can be achieved without labels at test time.

Fourth, if the main experiments are limited to PACS and ERM ResNet checkpoints, the scope of the empirical claim is correspondingly limited. Additional datasets, algorithms, and backbones would strengthen external validity but are not necessary for the central diagnostic contribution.


Fifth, threshold choices affect bucket assignments. We therefore report sensitivity to τH, τR, and τD in Appendix ??. The main qualitative conclusion should not depend on a single threshold value.

Finally, reconstruction fidelity is a necessary precondition for interpreting ablations. If an SAE reconstruction substantially changes model predictions, downstream concept effects must be described as effects in SAE reconstruction space rather than direct effects in the original model.

## 8 Conclusion

This paper introduced sparse concept diagnostics for domain generalization. The central argument is that DG should not be diagnosed by activation invariance alone. A useful invariant concept must also be class- supporting, consistent in its discriminative effect across domains, and isolated from competing classes. By decomposing trained vision models into SAE-derived candidate concepts and scoring each concept class- conditionally, the framework distinguishes robust support from harmful invariance, domain-contingent cues, and class conflict. The resulting class-level and model-level profiles provide a structured way to understand DG failure modes beyond accuracy. Rather than proposing a new DG algorithm, sparse concept diagnostics offer a post-hoc audit of what trained DG models have learned and why their invariant features may or may not support generalization.

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
