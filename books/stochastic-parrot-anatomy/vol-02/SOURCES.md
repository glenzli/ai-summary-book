# 资料源

本表给出本卷执行机制的主要技术来源。正文中的接口与伪代码是跨实现抽象；具体产品的 chat template、调度器和工具协议仍须以其版本化文档为准。

## Token、Transformer 与解码

- Taku Kudo and John Richardson, [*SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*](https://aclanthology.org/D18-2012/), EMNLP 2018。用于 tokenizer 与可逆文本处理的具体实现背景。
- Ashish Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017。用于 scaled dot-product attention、multi-head attention 和 encoder–decoder Transformer。
- Ari Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751), ICLR 2020。用于 nucleus sampling 的提出与开放式文本解码背景。
- Woosuk Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), SOSP 2023。用于 KV cache 分页管理与连续批处理的系统背景。

## 扩散、潜变量与流

- Jonathan Ho, Ajay Jain, and Pieter Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239), NeurIPS 2020。用于高斯前向扰动与学习逆过程。
- Robin Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752), CVPR 2022。用于潜空间扩散与 cross-attention 条件化。
- Jonathan Ho and Tim Salimans, [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598), 2022。用于有条件与无条件预测的组合。
- Yaron Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747), ICLR 2023。用于向量场和 ODE 采样的基本接口。
- Shen Nie et al., [*Large Language Diffusion Models*](https://arxiv.org/abs/2502.09992), 2025。作为离散掩码扩散语言模型的实例；本卷不据此断言它已经取代自回归语言模型。

## 工具与外部执行

- JSON Schema, [*JSON Schema Specification*](https://json-schema.org/specification)。用于结构化参数验证的接口背景。
- Martin Kleppmann, [*Designing Data-Intensive Applications*](https://dataintensive.net/), O'Reilly, 2017。用于重试、幂等性、消息交付与分布式失败语义的工程背景。

上述来源支持生成与执行机制。最终输出是否真实仍需外部证据，不能由执行轨迹单独推出。
