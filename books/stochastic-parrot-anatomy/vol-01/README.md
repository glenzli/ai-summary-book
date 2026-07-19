# 卷一 模型与系统

本卷承担全书唯一的模型设计与工程基础：先沿技术谱系说明神经网络、Transformer、预训练和后训练怎样形成，再把多模态、生成媒体、世界模型、检索、Agent 和部署生命周期分别展开。它回答“模型和系统怎样建成”，不提前承担一次运行的逐事件剖析、概率解释或内部机制解释。

## 技术谱系与共同结构

0. [人工智能演进路线](ch00_ai_evolution_roadmap.md)
1. [AI 范式、感知机与早期连接主义](ch01_early_ai_perceptron_connectionism.md)
2. [深度学习基础：训练、CNN 与序列模型](ch02_deep_learning_cnn_sequence_models.md)
3. [注意力与 Transformer](ch03_attention_transformer_sequence_architectures.md)
4. [预训练语言模型：ELMo、BERT、GPT 与 T5](ch04_pretrained_language_models.md)
5. [后训练、对齐与模型适配](ch05_post_training_alignment.md)
6. [推理效率与模型服务](ch06_inference_efficiency_serving.md)

## 模型家族与系统工程

7. [多模态模型](ch07_multimodal_models.md)
8. [生成媒体：从自回归到 Diffusion 与 Flow](ch08_generative_media_diffusion_flow.md)
9. [世界模型、具身智能与 VLA](ch09_world_models_embodied_ai.md)
10. [上下文、检索与记忆](ch10_context_retrieval_memory.md)
11. [Agent、工具与行动系统](ch11_agents_tools_systems.md)
12. [模型、数据与部署生命周期](ch12_model_data_lifecycle.md)

## 卷内资料

- [技术史来源与版本注记](SOURCE_NOTES.md)
- [卷内术语表](GLOSSARY.md)
- [数学与模型推导](../appendices/README.md)：只保留正文实际使用且能增加理解的材料

## 范围与时效边界

卷一面向希望理解技术谱系和模型工程的读者。公式用于解释模型结构、训练目标或资源消耗；更一般的线性代数、优化和概率背景由附录及标准教材承担。

动态技术条目最后系统校准于 **2026 年 7 月**。产品名称、API、价格和排行榜不构成稳定主线；新研究只在它改变模型家族、训练目标、执行机制或系统边界时进入正文。

卷末只得到模型与系统的静态设计图。下一卷将选择一次具体输入，从 token 化、prefill、logit、解码和 streaming 一直跟踪到输出或工具调用。
