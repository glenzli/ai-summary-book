# 书籍状态

当前版本是教材内容收口稿。第 0--17 章和附录 A--BO 已按连续教材口径组织；核心链覆盖依赖类型论、恒等类型、等价、单值性、HIT、圆的基本群、单值范畴论、Yoneda、模型语义和研究边界。

本书的收口标准不是把所有高级主题内部证明，而是让每个非平凡结论具有明确身份：书内定理、条件化推导、精确外部输入或研究边界。证明身份规则见 [B_proof_status_blueprints.md](B_proof_status_blueprints.md)，范围门槛见 [CLOSURE_SCOPE.md](CLOSURE_SCOPE.md)，依赖分层见 [DEPENDENCY_LAYERS.md](DEPENDENCY_LAYERS.md)，出版审计见 [PUBLICATION_CLOSURE_AUDIT.md](PUBLICATION_CLOSURE_AUDIT.md)。

后续维护边界：

- 不横向扩张正文主线；新增高级材料应优先进入边界附录，并标明对象语言、假设和外部来源。
- 不把 Rezk 泛性质、Blakers--Massey、Freudenthal、模型存在性、谱序列收敛等外部输入改写成书内证明，除非同时补齐完整证明核和来源说明。
- 不默认使用选择、公理化 resizing、排中律、额外 HIT、cubical judgmental computation 或外部对象语言规则；每次使用都必须在正文或附录中显式标注。
- 维护入口文件时同步 [SOURCES.md](SOURCES.md)、[NOTATION.md](NOTATION.md)、[S_source_locator_index.md](S_source_locator_index.md) 和 [K_remaining_obligations.md](K_remaining_obligations.md)。

本目录当前没有独立的 `TERM_INDEX.md` 或独立答案手册入口；不要为统一命名而伪造不存在的术语索引或解答文件。
