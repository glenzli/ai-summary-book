# 卷一 模型、训练与系统

本卷沿技术演进展开 AI 的模型基础：注意力与序列架构进入第三章，预训练进入第四章，后训练、检索与服务进入第五章，生成媒体与世界模型进入第六章。卷末把模型工件、训练记录、数据谱系与部署系统边界放进同一审计对象。

## 技术演进与模型基础

0. [人工智能演进路线](ch00_ai_evolution_roadmap.md)
1. [AI 范式、感知机与早期连接主义](ch01_early_ai_perceptron_connectionism.md)
2. [深度学习基础：训练、CNN 与序列模型](ch02_deep_learning_cnn_sequence_models.md)
3. [注意力、Transformer 与高效序列架构](ch03_attention_transformer_sequence_architectures.md)
4. [预训练语言模型：ELMo、BERT、GPT 与 T5](ch04_pretrained_language_models.md)
5. [后训练、对齐与模型效率](ch05_post_training_alignment_efficiency.md)
6. [多模态、Agent 与世界模型](ch06_multimodal_agents_world_models.md)

## 卷内综合

7. [模型工件、数据谱系与系统边界](ch07_model_artifacts_audit.md)

## 卷内资料

- [技术史来源与版本注记](SOURCE_NOTES.md)
- [卷内术语表](GLOSSARY.md)
- [学习理论扩展讲义](../appendices/learning-notes/)：逐步推导、数值例子与图示
- [学习与优化证明内核](../appendices/app-c_learning_optimization_kernel.md)：严格命题、完整假设与反查 locator

## 范围与时效边界

卷一面向希望理解技术谱系的读者，也允许工程读者复查机制。正文保留历史脉络、核心架构、训练范式、系统工程和关键数学直觉；较长推导进入学习理论讲义与附录 C。稳定主线优先于产品编年史或模型排名。

动态技术条目校准截至 **2026 年 7 月 12 日**。2024 年以后模型、产品、API、协议和系统条目只用于说明长上下文、多模态生成、推理后训练、开放权重、Agent 运行时和世界模型等路线，不保证之后的名称、价格、区域、上下文长度、能力或策略不变。

阅读动态条目时依次检查：机制和训练/推理接口是否有公开证据；论文、协议与 API 版本是否仍有效；数据、成本、延迟、隐私、安全和可维护性是否与当前任务匹配。公开发布页、系统卡、技术报告、独立复现实验和协议规范承担不同证据责任；功能相似不能反推出未公开训练配方。
