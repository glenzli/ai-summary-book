# 外部输入与资料源

本表只把正文明确使用的接口作为外部输入。本轮审计基线为 2026-07-14：Unicode 17.0.0、UAX #15 revision 57、UAX #29 revision 47。链接仍指向标准维护的稳定入口；具体实现记录必须另存实际采用的版本、profile、勘误状态与访问日期。

## 文本、Unicode 与 tokenizer

- **外部输入 1.A：UTF-8 唯一编码与合法域。** Unicode Consortium, [*The Unicode Standard*](https://www.unicode.org/standard/standard.html)，编码形式章节；F. Yergeau, [RFC 3629: *UTF-8, a transformation format of ISO 10646*](https://www.rfc-editor.org/rfc/rfc3629)。正文使用：Unicode 标量值有唯一合法 UTF-8 表示；合法串可唯一解码；surrogate、overlong 和超范围序列非法。本书不重证逐字节编码表。
- **外部输入 1.B：Unicode 规范化。** Unicode Consortium, [UAX #15: *Unicode Normalization Forms*](https://unicode.org/reports/tr15/)，本轮采用 revision 57。正文使用：NFC/NFD/NFKC/NFKD 的定义、规范/兼容等价边界与幂等性；不忽略早期实现勘误与版本记录。
- **外部输入 1.C：扩展字素簇。** Unicode Consortium, [UAX #29: *Unicode Text Segmentation*](https://unicode.org/reports/tr29/)，本轮采用 revision 47，尤其是 UAX29-C1-1/C1-2。正文使用：在固定 Unicode 版本及默认规则或明确 profile 后，扩展字素簇分段作为确定算法；它不把字素簇等同于标量值、字形或无条件的用户感知字符。
- **序列化实例。** T. Bray, [RFC 8259: *The JavaScript Object Notation (JSON) Data Interchange Format*](https://www.rfc-editor.org/rfc/rfc8259)。正文只使用 JSON 数据模型与多种字节序列化可表达同一结构化值的边界。
- **tokenizer 实例。** Taku Kudo and John Richardson, [*SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*](https://aclanthology.org/D18-2012/), EMNLP 2018。它是具体实现实例，不承担“所有 tokenizer 可逆”的一般结论。

## 函数与操作语义

- Benjamin C. Pierce et al., [*Software Foundations, Smallstep*](https://softwarefoundations.cis.upenn.edu/plf-current/Smallstep.html) 与 [*Equiv*](https://softwarefoundations.cis.upenn.edu/plf-current/Equiv.html)。正文使用带标签小步语义、确定性、多步关系和程序等价的标准接口。
- Gordon D. Plotkin, *A Structural Approach to Operational Semantics*. 正文使用结构操作语义的规则化方法，不引用未声明的语言特定定理。
- Glynn Winskel, *The Formal Semantics of Programming Languages*. 正文使用状态、轨迹和并发语义的一般背景。

## 概率核与轨迹测度

- **外部输入 5.A：有限核迭代。** Olav Kallenberg, [*Foundations of Modern Probability*, 3rd ed.](https://link.springer.com/book/10.1007/978-3-030-61871-1), Springer, 2021，`Kernels, Disintegration, and Invariance`（pp. 55--77）。正文使用概率测度与随机核逐次积分给出有限乘积路径测度的存在与唯一性。
- **外部输入 5.B：Ionescu--Tulcea 扩张。** 同上及本仓库《概率的边界》第三章。正文使用历史相关可测核在无限乘积 $\sigma$-代数上唯一扩张路径测度的定理。
- **外部输入 5.C：标准 Borel 随机化引理。** 同上及本仓库《概率的边界》第八章。正文只在下一步值域为标准 Borel 空间时使用单一 $[0,1]$ 均匀变量的可测实现。
- 总书卷三的[独立性、随机核与条件信息](../vol-03/ch04_independence_kernels_conditioning.md)给出 Ionescu--Tulcea 定理的同口径完整陈述；[随机算法与语言模型概率](../vol-03/ch07_randomized_algorithms_lm_probability.md)给出标准 Borel 随机化引理和实现映射口径。
- Dexter Kozen, *Semantics of Probabilistic Programs*. 正文以其概率程序测度语义作为历史与方法背景。
- Henning Kerstan and Barbara König, [*Coalgebraic Trace Semantics for Continuous Probabilistic Transition Systems*](https://arxiv.org/abs/1310.7417), *Logical Methods in Computer Science* 9(4), 2013。正文使用连续概率转移系统上轨迹测度的研究定位，不直接调用其深层煤代数定理。

本书不重证一般乘积测度扩张、标准 Borel 空间编码或随机化引理；第 5 章完整证明可数离散有限路径归一化和统一终止下界。

## 分布式事件与流式协议

- Leslie Lamport, [*Time, Clocks, and the Ordering of Events in a Distributed System*](https://lamport.azurewebsites.net/pubs/time-clocks.pdf), *Communications of the ACM* 21(7), 1978。正文采用论文中的不可反自反 happens-before 关系，并在需要通常偏序定义时显式取自反闭包；不把墙钟先后当作该关系的定义。
- 流式 offset、确认、取消和 exactly-once 的具体保证属于协议级设计；正文自行定义抽象状态机，不把某个厂商 API 行为作为普遍外部定理。

## Provenance、标识与签名

- W3C, [*PROV-DM: The PROV Data Model*](https://www.w3.org/TR/prov-dm/)、[*PROV-CONSTRAINTS*](https://www.w3.org/TR/prov-constraints/) 与 [*PROV-O*](https://www.w3.org/TR/prov-o/)，W3C Recommendations, 2013。正文使用 entity、activity、agent、generation、usage、derivation、association、attribution 及其有效性约束；并采用 PROV-CONSTRAINTS 明示的边界：没有一般 `wasDerivedFrom` 传递推理。PROV 的 responsibility 词汇不被升级为法律或道德责任。
- Open Container Initiative, [*Image Specification*](https://github.com/opencontainers/image-spec)。内容寻址与 descriptor 是工程实例，不承担哈希无碰撞结论。
- Software Heritage, [*Persistent Identifiers*](https://www.softwareheritage.org/save-and-reference-research-software/)。软件制品持久标识的工程实例。
- NIST, [FIPS 186-5: *Digital Signature Standard*](https://csrc.nist.gov/pubs/fips/186-5/final)。正文只使用“签名验证把密钥、算法和消息字节联系起来”的接口，不由签名推出消息事实为真。

## 形式语义、指称与真值

- Alfred Tarski, *The Concept of Truth in Formalized Languages*. 正文采用形式语言中相对于结构递归定义满足关系的经典接口。
- Stanford Encyclopedia of Philosophy, [*Reference*](https://plato.stanford.edu/entries/reference/) 与 [*Truth*](https://plato.stanford.edu/entries/truth/)。用于自然语言指称、真值理论与语义欠定的边界定位，不作为单一自然语言语义定理。
- 本书不建立完整自然语言解析器，也不假设自动核验器是无误真值 oracle。

## 代理、治理与规范边界

- NIST, [*Artificial Intelligence Risk Management Framework (AI RMF 1.0)*](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf), 2023。正文只把角色、责任记录和风险治理作为可采用的组织实践；不把框架当作法律结论。
- 作者资格、信用、道德责任与法律责任在第 10 章均由显式规范体系参数化。本书不引用任何来源来给出具体法域的最终法律判断。
