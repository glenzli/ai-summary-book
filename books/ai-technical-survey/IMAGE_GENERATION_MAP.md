# Image Generation Mapping

This file maintains the mapping between the images used in the book and the Python scripts used to generate them. Use this reference to quickly find and modify the code for any figure.

## Chapter 01: Neural Networks Basics

| Image File | Cited In | Generation Script | Notes |
| :--- | :--- | :--- | :--- |
| `perceptron_geometry.png` | `1.2_perceptron_and_limits.md` | `chapter_01/scripts/generate_perceptron_plot.py` | |
| `xor_problem.png` | `1.2_perceptron_and_limits.md` | `chapter_01/scripts/generate_xor_plot.py` | New: Visualizes AND, OR, XOR logic gates |
| `xor_mapping_process.png` | `1.3_neural_networks_basics.md` | `chapter_01/scripts/generate_xor_mapping.py` | Visualizes ReLU transformation for XOR problem |
| `svm_vs_perceptron.png` | `1.4_statistical_learning_era.md` | `chapter_01/scripts/generate_svm_vs_perceptron.py` | Renamed from visualize_*.py |
| `universal_approximation_step.png` | `1.3_neural_networks_basics.md` | `chapter_01/scripts/generate_universal_step_plot.py` | Generates PNG now |
| `universal_approximation_relu.png` | `1.3_neural_networks_basics.md` | `chapter_01/scripts/generate_universal_relu_plot.py` | Replaces svg |
| `universal_approximation.png` | `1.3_neural_networks_basics.md`, `2.1_foundations_and_math.md` | `chapter_01/images/generate_approximation_plot.py` | |
| `activation_functions/*.png` | `1.3_neural_networks_basics.md` | `chapter_01/scripts/generate_activation_plot.py` | Generates: sigmoid, tanh, relu, leaky_relu, gelu, swish |

## Chapter 02: Foundations

| Image File | Cited In | Generation Script | Notes |
| :--- | :--- | :--- | :--- |
| `cnn_spatial.png` | `2.2_cnn_architectures.md` | `chapter_02/scripts/generate_cnn_plots.py` | Visualizes Conv, Stride, Padding |
| `cnn_feature_hierarchy.png` | `2.2_cnn_architectures.md` | `chapter_02/scripts/generate_cnn_features.py` | Visualizes feature hierarchy (edges -> textures -> objects) |
| `resnet_flow.png` | `2.2_cnn_architectures.md` | `chapter_02/scripts/generate_resnet_plot.py` | Visualizes Residual Block and Gradient Flow |
| `optimizer_landscape_3d.png` | `2.1_foundations_and_math.md` | `chapter_02/scripts/generate_optimizer_comparison.py` | 3D Loss Landscape |
| `optimizer_trajectory_2d.png` | `2.1_foundations_and_math.md` | `chapter_02/scripts/generate_optimizer_comparison.py` | 2D Trajectory with fine-grained contour |
| `weight_initialization.png` | `2.1_foundations_and_math.md` | `chapter_02/scripts/generate_init_plot.py` | Histograms of activations across layers for different init methods |
| `vanishing_gradient.png` | `2.3_rnn_dynamics.md` | `chapter_02/scripts/generate_rnn_plots.py` | Visualizes lambda^t decay |
| `activation_gradients.png` | `2.3_rnn_dynamics.md` | `chapter_02/scripts/generate_rnn_plots.py` | Visualizes derivatives of Tanh/Sigmoid/ReLU |
| `lstm_forget_gate_retention.svg` | `2.4_lstm_and_seq2seq.md` | `chapter_02/scripts/generate_lstm_retention_plot.py` | Dependency-free SVG showing multiplicative forget-gate retention |
| `bias_variance_tradeoff.png` | `2.1_foundations_and_math.md` | `chapter_02/scripts/generate_theory_plots.py` | Function: `plot_bias_variance` |
| `regularization_geometry.png` | `2.1_foundations_and_math.md` | `chapter_02/scripts/generate_regularization_plot.py` | Standalone script |
| `dropout_ensemble.png` | `2.1_foundations_and_math.md` | `chapter_02/scripts/generate_dropout_ensemble.py` | |
| `backprop_node.png` | `appendix/a.6_backpropagation.md` | `appendix/images/generate_backprop_details.py` | New split view (Node) |
| `backprop_gates.png` | `appendix/a.6_backpropagation.md` | `appendix/images/generate_backprop_details.py` | New split view (Gates) |
| `universal_approximation_bump.png` | `appendix/a.5_universal_approximation.md` | `chapter_02/images/generate_theory_plots.py` | Function: `plot_universal_approximation` |

## Chapter 03: Transformers

| Image File | Cited In | Generation Script | Notes |
| :--- | :--- | :--- | :--- |
| `positional_encoding.png` | `3.3_positional_encoding_and_norm.md` | `chapter_03/scripts/generate_pe_plot.py` | |
| `attention_alignment_heatmap.png` | `3.1_attention_mechanisms.md` | `chapter_03/scripts/generate_attention_alignment_heatmap.py` | Toy alignment heatmap for α_tj |
| `causal_mask_demo.png` | `3.4_training_and_inference.md` | `chapter_03/scripts/generate_causal_mask_demo.py` | Visualizes causal mask and masked softmax |
| `kv_cache_memory_curve.png` | `3.4_training_and_inference.md` | `chapter_03/scripts/generate_kv_cache_memory_curve.py` | Approx KV cache memory growth vs sequence length |
| `attention_efficiency_frontier.svg` | `3.5_frontier_attention_architectures.md` | `chapter_03/scripts/generate_attention_efficiency_frontier.py` | Dependency-free SVG comparing dense, sparse, latent-cache, and hybrid long-context routes |
| `mamba_ssm_flow.svg` | `3.6_state_space_models_and_mamba.md` | `chapter_03/scripts/generate_mamba_ssm_flow.py` | Dependency-free SVG comparing explicit attention lookup with selective SSM state updates |

## Chapter 04: Pretraining

| Image File | Cited In | Generation Script | Notes |
| :--- | :--- | :--- | :--- |
| `scaling_law_plot.png` | `4.3_gpt_generative_models.md` | `chapter_04/scripts/generate_scaling_law.py` | |
| `mlm_masking_strategy.png` | `4.2_bert_architecture.md` | `chapter_04/scripts/generate_mlm_masking_strategy.py` | BERT 80/10/10 masking breakdown |
| `bert_vs_gpt_attention_mask.png` | `4.3_gpt_generative_models.md` | `chapter_04/scripts/generate_bert_vs_gpt_attention_mask.py` | Bidirectional vs causal visibility matrix |
| `span_corruption_example.png` | `4.4_unified_frameworks_t5.md` | `chapter_04/scripts/generate_span_corruption_example.py` | T5 span corruption input/target example |

## Chapter 05: Fine-tuning

| Image File | Cited In | Generation Script | Notes |
| :--- | :--- | :--- | :--- |
| `lora_diagram.png` | `5.3_peft_methods.md` | `chapter_05/scripts/generate_lora_plot.py` | |
| `lora_rank_tradeoff.png` | `5.3_peft_methods.md` | `chapter_05/scripts/generate_lora_rank_tradeoff.py` | Toy rank vs params/returns curve |
| `kl_anchor_tradeoff.png` | `5.2_rlhf_and_alignment.md` | `chapter_05/scripts/generate_kl_anchor_tradeoff.py` | Toy β tradeoff: reward vs KL |
| `post_training_pipeline.svg` | `5.2_rlhf_and_alignment.md` | `chapter_05/scripts/generate_post_training_pipeline.py` | Dependency-free SVG comparing SFT, RLHF/PPO, and DPO routes |
| `post_training_taxonomy.svg` | `5.5_modern_post_training.md` | `chapter_05/scripts/generate_post_training_taxonomy.py` | Dependency-free SVG of the modern post-training stack and evaluation loop |
| `inference_speed_stack.svg` | `5.7_inference_speed_and_serving.md` | `chapter_05/scripts/generate_inference_speed_stack.py` | Dependency-free SVG of latency/throughput optimization layers |
| `open_model_ecosystem.svg` | `5.8_open_weight_model_ecosystem.md` | `chapter_05/scripts/generate_open_model_ecosystem.py` | Dependency-free SVG showing how one base model branches into LoRA, distillation, merge, quantization, and runtime variants |
| `quantization_tradeoff.png` | `5.4_quantization_and_optimization.md` | `chapter_05/scripts/generate_quantization_tradeoff.py` | Weight memory vs precision (illustrative) |

## Chapter 06: Future Trends

| Image File | Cited In | Generation Script | Notes |
| :--- | :--- | :--- | :--- |
| `vector_search_plot.png` | `6.3_rag_and_context.md` | `chapter_06/scripts/generate_vector_search.py` | |
| `agent_control_loop.svg` | `6.2_agents_and_reasoning.md` | `chapter_06/scripts/generate_agent_control_loop.py` | Dependency-free SVG of the controlled agent loop |
| `rag_long_context_tradeoff.svg` | `6.3_rag_and_context.md` | `chapter_06/scripts/generate_rag_context_tradeoff.py` | Dependency-free SVG of RAG vs long-context tradeoffs |
| `agent_system_stack.svg` | `6.5_agent_system_engineering.md` | `chapter_06/scripts/generate_agent_system_stack.py` | Dependency-free SVG of Agent runtime/protocol/tool/observability layers |
| `agent_trust_boundaries.svg` | `6.5_agent_system_engineering.md` | `chapter_06/scripts/generate_agent_trust_boundaries.py` | Dependency-free SVG of trusted control plane vs untrusted data/effects |
| `world_model_media_stack.svg` | `6.6_world_models_and_generative_media.md` | `chapter_06/scripts/generate_world_model_media_stack.py` | Dependency-free SVG connecting generative media, latent world models, and agent loops |

## 附录 (Appendix)

| Image File | Cited In | Generation Script | Notes |
| :--- | :--- | :--- | :--- |
| `lagrange_geometric.png` | `a.4_regularization.md` | `appendix/scripts/generate_appendix_plots.py` | |
| `bayesian_priors.png` | `a.4_regularization.md` | `appendix/scripts/generate_appendix_plots.py` | |
| `bayesian_update.png` | `a.4_regularization.md` | `appendix/scripts/generate_appendix_plots.py` | |
| `vc_dimension.png` | `a.3_statistical_learning_theory.md` | `appendix/scripts/generate_vc_plot.py` | |
| `gradient_descent_1d.png` | `a.1_optimization_basics.md` | `appendix/scripts/generate_optimization_plots.py` | |
| `sgd_trajectory_3d.png` | `a.1_optimization_basics.md` | `appendix/scripts/generate_optimization_plots.py` | |
| `learning_rate_comparison.png` | `a.1_optimization_basics.md` | `appendix/scripts/generate_optimization_plots.py` | |
| `momentum_vs_sgd_a8.png` | `a.8_advanced_optimization.md` | `appendix/scripts/generate_evolution_plots.py` | New: Visualizes SGD vs Momentum on noisy loss (Morandi style) |
| `margin_gamma_comparison.png` | `a.2_perceptron_convergence.md` | `appendix/scripts/generate_margin_gamma.py` | New: Visualizes difficulty based on margin size |
| `data_radius_normalization.png` | `a.2_perceptron_convergence.md` | `appendix/scripts/generate_radius_normalization.py` | New: Visualizes impact of data radius R on stability |
| `convergence_speed_gamma.png` | `a.2_perceptron_convergence.md` | `appendix/scripts/generate_convergence_speed.py` | New: Visualizes impact of margin gamma on convergence steps |
| `momentum_badcase_scale.png` | `a.8_advanced_optimization.md` | `appendix/scripts/generate_evolution_plots.py` | Visualizes Momentum failure on scale imbalance |
| `rmsprop_vs_sgd.png` | `a.8_advanced_optimization.md` | `appendix/scripts/generate_evolution_plots.py` | Visualizes RMSProp advantage on scale imbalance |
| `rmsprop_badcase_noise.png` | `a.8_advanced_optimization.md` | `appendix/scripts/generate_evolution_plots.py` | Visualizes RMSProp failure on noisy gradients |
| `adam_vs_scale.png` | `a.8_advanced_optimization.md` | `appendix/scripts/generate_evolution_plots.py` | Visualizes Adam success on scale imbalance |
| `adam_vs_noise.png` | `a.8_advanced_optimization.md` | `appendix/scripts/generate_evolution_plots.py` | Visualizes Adam success on noisy gradients |
