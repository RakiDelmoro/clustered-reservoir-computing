# Knowledge Base Index

| Article | Summary | Compiled From | Updated |
|---------|---------|---------------|---------|
| [[concepts/autoregressive-inference]] | Frame-by-frame inference with reservoir state carry-forward and sliding window aggregation | daily/2026-04-18.md | 2026-04-18 |
| [[concepts/baselines]] | Comprehensive evaluation comparing against architectural ablations and alternative models | daily/2026-04-16.md, daily/2026-04-18.md | 2026-04-18 |
| [[concepts/clustar]] | Clustered Spatio-Temporal Reservoir — structured reservoir computing for visual world models; final single-stage supervised implementation | daily/2026-04-16.md, daily/2026-04-18.md | 2026-04-18 |
| [[concepts/clustered-reservoir]] | 10×100 neurons with small-world topology, functional specialization — core structural innovation | daily/2026-04-16.md, daily/2026-04-18.md | 2026-04-18 |
| [[concepts/deep-echo-state-network]] | Stacked reservoir layers for hierarchical temporal feature extraction | daily/2026-04-16.md, daily/2026-04-18.md | 2026-04-18 |
| [[concepts/device-management]] | GPU/CPU tensor placement fixes: alpha buffer registration, label-device alignment | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/echo-state-network]] | Canonical reservoir computing (Jaeger 2001) — fixed sparse recurrent reservoir with echo state property | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/encoder-bottleneck]] | Spatial encoder produces near-identical embeddings for spinning vs stationary digits — root cause of rotation blindness | daily/2026-04-18.md | 2026-04-18 |
| [[concepts/evolved-reservoir]] | Neuroevolution-optimized reservoir connectivity, spectral radii, or neuron properties | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/fine-tuning]] | Readout adaptation protocol — original plan, not used in simplified architecture | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/import-path-management]] | sys.path fix + clustar.* imports for dev convenience across working directories | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/input-routing]] | Feature streams partitioned to dedicated cluster groups in the reservoir | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/linear-probe]] | Frozen representation evaluation — considered but not used | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/liquid-state-machine]] | Spiking reservoir variant (Maass 2002) with leaky integrate-and-fire neurons | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/moving-mnist]] | Synthetic video benchmark with 4 action classes (moving/spinning/collision/stationary) | daily/2026-04-16.md, daily/2026-04-18.md | 2026-04-18 |
| [[concepts/multi-task-learning]] | Joint multi-objective training — deprecated in simplified CluSTAR | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/multi-timescale-dynamics]] | Heterogeneous leaking rates (fast/medium/slow α) within clusters | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/neuroevolution]] | Evolutionary optimization of reservoir topologies — future work | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/on-the-fly-generation]] | Memory-efficient dataset synthesis without pre-caching | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/parameter-counting]] | Trainable parameter reporting for model transparency | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/progress-tracking]] | tqdm progress bars for feature extraction visibility | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/quick-sanity-test]] | 100-sample smoke test before full training | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/reservoir-computing]] | Fixed recurrent + trainable readout paradigm for temporal dynamics | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/ridge-regression]] | Closed-form readout training (Tikhonov regularization) — no gradient descent | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/script-organization]] | Single vs multi-script trade-offs; finalized to one entry point | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/self-supervised-pretraining]] | 5 pretext tasks on unlabeled data — designed but removed in simplification | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/single-stage-supervised-training]] | Definitive training paradigm — direct action classifier on labeled data, no pre-training | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/spatial-encoder]] | Fixed random orthogonal projection 784→128 | daily/2026-04-16.md | 2026-04-16 |
| [[concepts/temporal-aggregation]] | [x₀, x_T, mean, max] concatenation wins — converts state sequences to fixed-length vectors | daily/2026-04-16.md, daily/2026-04-18.md | 2026-04-18 |
| [[concepts/test-video-generation]] | Synthetic test.mp4 with digit transitioning between action classes for inference demo | daily/2026-04-18.md | 2026-04-18 |
| [[concepts/world-model]] | Predictive dynamics model P(s'\|s,a) for agent planning | daily/2026-04-16.md | 2026-04-16 |
| [[connections/clustar-components]] | Encoder → routing → reservoir → multi-timescale → readout pipeline | daily/2026-04-16.md | 2026-04-16 |
| [[connections/deep-reservoir-experiment]] | 2-layer deep reservoir failed: accuracy 80.4%→75.4%, spinning detection stayed at 0% | daily/2026-04-18.md | 2026-04-18 |
| [[connections/evaluation-protocols]] | Linear probe vs fine-tuning — deprecated by simplified architecture | daily/2026-04-16.md | 2026-04-16 |
| [[connections/multi-task-self-supervision]] | Self-supervised + multi-task combined — original design removed | daily/2026-04-16.md | 2026-04-16 |
| [[connections/reservoir-paradigms]] | Taxonomy: ESN/LSM/Deep ESN/Evolved/CluSTAR | daily/2026-04-16.md | 2026-04-16 |
| [[connections/reservoir-world-models]] | Reservoir as efficient dynamics backbone for embedded RL/robotics | daily/2026-04-16.md | 2026-04-16 |
| [[connections/simplification-path]] | Multi-stage → single-stage rationale with empirical validation | daily/2026-04-16.md | 2026-04-16 |
