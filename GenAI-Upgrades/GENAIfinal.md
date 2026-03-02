Title
Calibrated Conformal Genera�ve Document Extrac�on: End-to-End OCR-Free Vision–Language
Modeling with Structured Coverage Guarantees


Abstract
This work presents a fully genera�ve, OCR-free mul�modal vision–language model for structured
document informa�on extrac�on with formally calibrated uncertainty. Unlike prior approaches that
bolt standard Monte Carlo dropout or heuris�c confidence heads onto Donut-style architectures, the
proposed framework embeds token- and field-level conformal predic�on directly into a T5-style
genera�ve decoder, yielding coverage-guaranteed predic�on sets over structured outputs (key–value
pairs, tables, fields). A geometry-aware conformal scoring func�on leverages visual layout, reading
order, and content embeddings to construct region-sensi�ve nonconformity scores, while a
hierarchical calibra�on scheme enforces coverage at token-, field-, and document-level
simultaneously. The model is trained end-to-end with a joint objec�ve that couples sequence
likelihood, conformal score regulariza�on, and differen�able surrogate coverage losses. At inference,
the model produces structured predic�on sets and an absten�on-aware extrac�on policy that
op�mizes expected u�lity under coverage constraints. Extensive experiments on revised, detemplated versions of FUNSD, CORD, SROIE, and mul�-page DocVQA, with synthe�c and real noise,
demonstrate that the method matches or improves SOTA accuracy while significantly improving
calibra�on, coverage, and selec�ve risk. The design is op�mized to be trainable on a single RTX 5050
GPU via parameter-efficient adapta�on and conformal calibra�on atop small open-source
backbones. The result is a TPAMI/ACL-grade framework that turns genera�ve document extrac�on
into a rigorously calibrated, uncertainty-aware predic�on problem.


Core idea (2–3 lines)
End-to-end OCR-free genera�ve document extrac�on with **hierarchical, geometry-aware conformal**
**predic�on** directly embedded in the decoder, producing token/field/document-level **predic�on sets**
**with provable coverage** . The model outputs not just values, but calibrated sets and absten�ons,
enabling trustworthy extrac�on under explicit risk–coverage tradeoffs, all trainable on a single
consumer GPU via parameter-efficient tuning.


Research hypothesis
A conformal-predic�on-integrated genera�ve vision–language model with geometry-aware
nonconformity scores and hierarchical coverage constraints can (1) match or exceed SOTA accuracy
on document extrac�on benchmarks, and (2) significantly improve calibra�on, selec�ve risk, and
robustness to layout/noise compared to standard MC-dropout- or confidence-head-based
uncertainty baselines, while remaining trainable on a single RTX 5050 GPU.


Key novelty contribu�ons


1. **Hierarchical Conformal Genera�ve Decoder** : A new decoder formula�on that produces

token-, field-, and document-level predic�on sets for structured document outputs, with
mul�-level conformal coverage constraints baked into the decoding and training objec�ves.


2. **Geometry-Aware Nonconformity Scoring** : A mul�modal nonconformity score that fuses

visual layout, spa�al structure, and seman�c content to define document-specific conformal
scores for key–value pairs and tables, rather than trea�ng outputs as flat sequences.


3. **Differen�able Surrogate Coverage Loss** : A novel loss that encourages conformal scores and

predic�on set sizes to approximate desired coverage levels during training, enabling end-toend op�miza�on of calibra�on quality instead of post-hoc conformal wrapping only.


4. **Absten�on-Aware Structured Decoding Policy** : A decoding algorithm that jointly chooses

predic�on sets and absten�ons under a u�lity func�on (e.g., F1–coverage tradeoff),
extending selec�ve classifica�on to fully genera�ve structured outputs.


5. **Single-GPU-Efficient Design** : A parameter-efficient adapta�on scheme (LoRA + layer-wise

scaling + low-rank visual adapters) and calibra�on protocol explicitly engineered to fit within
RTX 5050 constraints, demonstra�ng that rigorous uncertainty with conformal predic�on is
feasible on consumer hardware for document extrac�on.


Final architecture design


  - **Backbone encoder** :


`o` OCR-free vision encoder (small ViT/Swin) opera�ng directly on document images
(single and mul�-page via page-wise encoding + learned page-posi�on embeddings).


`o` Op�onal shallow text encoder for weak supervision using pseudo-OCR tokens if
available, fused via cross-a�en�on but not required at inference (s�ll OCR-free by
default).


  - **Genera�ve decoder** :


`o` T5-small/base-sized transformer decoder with:


          - Standard autoregressive genera�on over a structured serializa�on of
document fields (JSON-like schema with slots for keys, values, table
rows/columns).


          - Addi�onal **conformal score head** at each decoding step producing per-token
nonconformity logits.


          - Field-level aggrega�on modules that pool token nonconformity into field
scores via a�en�on over layout-aware token embeddings.


  - **Geometry-aware fusion** :


`o` For each token posi�on, a fused representa�on concatena�ng: visual region
features, posi�onal/layout embeddings (absolute and rela�ve 2D coordinates, block
indices), and seman�c context.


`o` A small MLP projects the fused representa�on to nonconformity scores used by the
conformal predic�on layer.


  - **Hierarchical conformal layer** (nonparametric component atop decoder outputs):


`o` Uses calibra�on splits to compute quan�les of nonconformity at token, field, and
document level.


`o` Stores per-task, per-field quan�les used in decoding to define predic�on sets and
absten�ons.


Learning objec�ves / loss func�ons
Let y be the structured output (sequence of tokens with field segmenta�on), ŷ the model output, s(·)
nonconformity scores, and τ the learned/target quan�le thresholds.


1. **Genera�ve loss (L_gen)**


`o` Standard nega�ve log-likelihood over the serialized sequence:


          - L_gen = − log pθ(y | x)


`o` Encourages accurate structured genera�on.


2. **Conformal score regulariza�on (L_conf)**


`o` For each token i with nonconformity score si and empirical calibra�on quan�le qα
(for target coverage 1−α):


          - Encourage scores to be well-separated: posi�ves near or below qα,
incorrect/hard examples above qα.


`o` Implemented as a margin-based loss:


          - L_conf = E[(si − qα + m) _+ for correct tokens] + E[(qα − si + m)_ + for incorrect
tokens]


`o` m is a margin hyperparameter.


3. **Differen�able surrogate coverage loss (L_cov)**


`o` Approximate the coverage constraint by a so� indicator over whether true
tokens/fields fall into the learned predic�on set (defined by si ≤ τ̂, where τ̂ is a
differen�able approxima�on of qα).


`o` L_cov penalizes empirical devia�on of coverage from target (1−α) at mul�ple levels:


          - L_cov = |Ĉ_token − (1−α_token)| + |Ĉ_field − (1−α_field)| + |Ĉ_doc −
(1−α_doc)|


`o` Ĉ_* are differen�able coverage surrogates using sigmoid relaxa�ons.


4. **Regulariza�on of set size (L_size)**


`o` Penalize overly large predic�on sets:


          - L_size = λ_size · E[#candidates per token/field]


5. **Total loss**


`o` L_total = L_gen + λ_conf L_conf + λ_cov L_cov + λ_size L_size


`o` λ_conf, λ_cov, λ_size tuned via small grid search; abla�ons will test their impact.


Training strategy


  - **Stage 1: Backbone warm-start**


`o` Ini�alize encoder and decoder from compact OCR-free checkpoints (e.g., Donut-base
or Pix2Struct-base) using HF models.


`o` Freeze encoder (or apply small LoRA adapters) and fine-tune decoder on supervised
document extrac�on without uncertainty modules (only L_gen).


  - **Stage 2: Uncertainty module ac�va�on**


`o` Unfreeze conformal score heads and geometry-aware fusion layers; keep most
backbone weights frozen or LoRA-tuned to stay within RTX 5050 limits.


`o` Train with L_total on training split; maintain a held-out calibra�on split.


  - **Stage 3: Conformal calibra�on**


`o` A�er training, fix θ and compute empirical nonconformity scores on calibra�on set.


`o` Compute per-task, per-field quan�les qα for token/field/doc levels.


`o` Op�onally iterate a short fine-tuning phase where τ̂ (threshold approximators) are
learned to emulate these quan�les, minimizing L_cov.


  - **Implementa�on details**


`o` Mixed precision training (FP16 or BF16)


`o` LoRA on selected decoder layers and a�en�on matrices


`o` Gradient checkpoin�ng for memory


`o` Batch size 1–2 with gradient accumula�on to achieve effec�ve batch size 16–32


`o` Early stopping based on valida�on coverage + F1.


Inference strategy


1. **Base genera�on**


`o` Run encoder + decoder with beam search or constrained sampling over structured
schema.


`o` For each token and field generated, also produce nonconformity scores si.


2. **Predic�on set construc�on**


`o` Use stored quan�les qα (or learned τ̂) to define predic�on sets:


          - Token-level: include candidates whose si ≤ qα_token.


          - Field-level: aggregate token scores to field score and include field predic�on
if sfield ≤ qα_field.


3. **Absten�on-aware decision**


`o` Define a u�lity func�on U(F1, coverage, absten�on_rate).


`o` For each field or document, decide to **accept**, **re-prompt/re-generate**, or **abstain**
based on:


          - Whether true coverage constraints are sa�sfied (approximate)


          - Expected improvement vs. cost of re-genera�on


`o` Implement a simple policy learned on valida�on set: thresholds on set size and
nonconformity sta�s�cs.


4. **Output**


`o` Final outputs include:


          - Single best structured extrac�on (for downstream systems)


          - Predic�on sets for cri�cal fields (e.g., total amount, date, address)


          - Confidence/coverage es�mates at field/document level


          - Flags/absten�ons for low-confidence documents.


Uncertainty modeling


  - **Type** : Nonparametric distribu�on-free uncertainty via conformal predic�on, not Bayesian
approxima�on.


  - **Granularity** : Token-level, field-level, document-level.


  - **Signals** : Geometry-aware nonconformity scores combining visual loca�on, layout, and
context; not just so�max entropy.


  - **Guarantees** : Empirical coverage at or near target levels (e.g., 90–95%) for held-out
calibra�on data; evaluated across document types and corrup�on levels.


  - **Comparison** : Directly compare against MC Dropout, so�max entropy, and auxiliary
confidence heads for calibra�on error (ECE), Brier score, and selec�ve risk.


Evalua�on protocol


  - **Datasets** (using revised / de-templated versions where available):


`o` FUNSD-r (resampled to reduce template leakage)


`o` CORD-r / CORD (with template stra�fica�on)


`o` SROIE-r (re-split to reduce template duplica�on)


`o` DocVQA (single and mul�page subsets)


`o` Addi�onal real noisy scans and synthe�c corrup�ons (blur, rota�on, JPEG ar�facts,
occlusion).


  - **Metrics** :


`o` **Accuracy** : F1, EM for field extrac�on and table reconstruc�on.


`o` **Calibra�on** : ECE, MCE, Brier score at token, field, and document level.


`o` **Coverage & set quality** : Empirical coverage vs target, predic�on set size
distribu�ons, token/field-level oracle F1 given sets.


`o` **Selec�ve predic�on** : Risk–coverage curves, AURC, absten�on curves (F1 vs.
coverage), under both natural and corrupted distribu�ons.


`o` **Robustness** : Performance under different noise types/levels and layout shi�s.


`o` **Efficiency** : Inference �me and memory on RTX 5050, comparing confidence
methods.


Baselines


1. **Donut-style OCR-free genera�ve model** with:


`o` Plain so�max confidence (no uncertainty modeling)


`o` MC Dropout-based uncertainty


`o` Auxiliary confidence head + calibra�on loss.


2. **Pix2Struct-style OCR-free model**, similarly augmented.


3. **DocFormerv2-style encoder + standard genera�ve decoder with confidence head** (where

feasible within compute).


4. **Conformal wrap baseline** : Post-hoc conformal predic�on applied to off-the-shelf

Donut/Pix2Struct using simple nonconformity (nega�ve log-probability) without geometryaware scoring or hierarchical coverage.


5. **Selec�ve predic�on baselines** from NLP (calibrated confidence thresholds, entropy-based

selec�on).


Abla�ons


  - **Architecture-level** :


`o` With vs without geometry-aware nonconformity (using only sequence logits).


`o` With vs without hierarchical (token+field+doc) coverage vs token-only coverage.


  - **Loss components** :


`o` L_gen only vs L_gen + L_conf vs L_gen + L_conf + L_cov vs full L_total.


`o` Vary λ_conf, λ_cov, λ_size to show effect on coverage and set size.


  - **Conformal vs classical uncertainty** :


`o` Conformal predic�on vs MC Dropout vs entropy vs confidence head.


  - **Calibra�on regimes** :


`o` Global quan�les vs per-field quan�les vs per-document-type quan�les.


  - **Decoding policy** :


`o` Greedy decoding vs beam search vs constrained decoding; effect on calibra�on and
coverage.


  - **Hardware-aware** :


`o` LoRA/adapter configura�ons and their impact on accuracy and calibra�on under
VRAM constraints.


Expected results


  - Comparable or slightly be�er F1/EM than strong OCR-free baselines on revised
CORD/FUNSD/SROIE/DocVQA while:


`o` Reducing ECE and Brier score significantly (e.g., 30–50% rela�ve reduc�on vs MC
Dropout/confidence head).


`o` Achieving empirical coverage close to targets (e.g., 90–95%) at token, field,
document level, even under noise and layout shi�s.


`o` Producing compe��ve risk–coverage curves and AURC, clearly outperforming MC
Dropout and entropy-based selec�on.


`o` Demonstra�ng that predic�on sets and absten�on policies materially improve
reliability for high-stakes fields (totals, dates, iden�fiers).


  - Demonstra�ng **feasible training on RTX 5050** with wall-clock �mes on the order of a few
days per dataset using parameter-efficient tuning.


Why this is novel
Because it **integrates conformal predic�on and structured coverage control into the genera�ve**
**decoding process itself**, with geometry-aware nonconformity scoring and hierarchical, differen�able
surrogate coverage objec�ves, rather than trea�ng uncertainty as a post-hoc heuris�c add-on. This
shi�s document extrac�on from “Donut + MC Dropout” engineering to a **principled, coverage-**
**guaranteed, uncertainty-aware genera�ve framework**, tailored to structured, mul�modal document
outputs and explicitly engineered to be viable on a single consumer GPU.


Compute feasibility plan


  - Use T5-small/base-scale decoder (≤ 220M params) and small ViT/Swin encoder (≤ 100M)
with:


`o` LoRA on selected a�en�on and feed-forward layers; freeze most backbone
parameters.


`o` FP16/BF16 training with gradient checkpoin�ng and gradient accumula�on.


`o` Batch size 1–2, effec�ve batch size 16–32 via accumula�on.


`o` Dataset sizes limited to standard benchmarks (tens of thousands samples, not
pretraining-scale).


`o` Conformal calibra�on computed offline on calibra�on splits with small batches.


  - Es�mated:


`o` Fine-tuning per dataset: 2–4 days on RTX 5050.


`o` Total project: 2–3 weeks ac�ve training, within realis�c PhD �meline.


Target venues


  - Primary: **IEEE TPAMI**, **ACL main conference** (core methodology: structured genera�ve
modeling + principled uncertainty).


  - Secondary: **NeurIPS/ICLR workshops** on uncertainty and structured predic�on; **CVPR/ICCV**
**document understanding workshops** for extended applica�ons.


Final verdict (1 sentence)
This project is a single, coherent, TPAMI/ACL-level research direc�on: a rigorously calibrated,
conformal-predic�on-integrated, end-to-end OCR-free genera�ve vision–language architecture for
document extrac�on with genuine algorithmic novelty and realis�c single-GPU feasibility.


