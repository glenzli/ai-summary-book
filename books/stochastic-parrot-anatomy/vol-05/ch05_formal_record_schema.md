# 第五章 形式记录 schema（参考）

这一参考章给出全书使用的最小形式 schema。字段并不试图穷尽某个具体平台，而是固定跨卷反复调用的对象类型：输入、配置、轨迹、token、文本、字节、制品、来源图、主张记录、规范元数据与运行状态。它只保留能够支持验证器、审计包和贯穿案例的公共核。

## 5.1 类型化字段

固定输入集合 $\mathcal I$、配置集合 $\mathcal C$、事件 schema $\Sigma$ 下的轨迹集合 $\operatorname{Tr}_\Sigma$、token 集 $\mathcal V$、Unicode 标量集合 $\mathbb U$、字节集合 $\mathbb B$、制品集合 $\mathsf{Art}$、来源图集合 $\mathsf{Prov}$、主张记录集合 $\mathsf{ClaimRec}$、证据引用集合 $\mathsf{EvRef}$、说明文本集合 $\mathsf{Text}$ 和运行状态集合

$$
\mathsf{Status}=\{\operatorname{running},\operatorname{succeeded},\operatorname{failed},\operatorname{cancelled},\operatorname{unknown}\}.
$$

缺失原因集合为

$$
\mathsf{AbsenceReason}=
\{\operatorname{NotYetProduced},\operatorname{NotApplicable},\operatorname{DecodeError},\operatorname{SerializationError},\operatorname{Redacted},\operatorname{Unknown}\}.
$$

对任意集合 $X$，定义

$$
\operatorname{Field}(X)=
(\{\operatorname{Value}\}\times X)
\sqcup
(\{\operatorname{Absent}\}\times \mathsf{AbsenceReason}).
$$

记 $f\downarrow$ 表示 $f=(\operatorname{Value},x)$，记 $f\uparrow_r$ 表示 $f=(\operatorname{Absent},r)$。因此空字符串、空列表与缺失不是同一件事：前两者是某个 $X$ 中的值，后者必须携带缺失原因。`Field` 只保证结构上没有无类型空洞，不保证现有值已经足以支持任何主张。

## 5.2 规范记录与验收剖面

令规范条件索引集为

$$
\mathsf K=\{\operatorname{standing},\operatorname{informed},
\operatorname{voluntary},\operatorname{scoped},
\operatorname{third\_party},\operatorname{minimization},
\operatorname{risk\_distribution},\operatorname{contestability},
\operatorname{remediation}\}.
$$

条件状态集合为

$$
\mathsf{CondState}=\{\operatorname{satisfied},\operatorname{violated},
\operatorname{unknown},\operatorname{not\_applicable}\},
$$

单个条件记录为

$$
\mathsf{CondRec}=\mathsf{CondState}
\times\operatorname{Field}(\mathsf{EvRef}^*)
\times\operatorname{Field}(\mathsf{Text}).
$$

风险级别和规范决策分别取值于

$$
\mathsf{Risk}=\{\operatorname{low},\operatorname{moderate},\operatorname{high}\},
\qquad
\mathsf{NormDecision}=\{\operatorname{approved},\operatorname{dry\_run},
\operatorname{human\_review},\operatorname{rejected}\}.
$$

固定可寻址的策略版本集合 $\mathsf{PolicyVersion}$。规范记录集合定义为

$$
\mathsf{NormRec}=\mathsf{Risk}\times\mathsf{CondRec}^{\mathsf K}
\times\mathcal P(\mathsf K)\times\mathsf{PolicyVersion}
\times\mathsf{NormDecision}.
$$

一条规范记录写作 $n=(r,\nu,R,\pi,d)\in\mathsf{NormRec}$。其中 $R$ 是验收剖面 $\pi$ 为该次调用指定的必需条件集合。谓词 $\operatorname{WF}_{\mathsf N}(n)$ 要求：状态为 `satisfied` 或 `violated` 时证据字段有值；状态为 `unknown` 时说明字段有值；状态为 `not_applicable` 时说明字段有值且该条件不属于 $R$；策略版本 $\pi$ 可寻址。只要 $d=\operatorname{approved}$，就必须有

$$
\forall k\in R,\qquad \operatorname{state}(\nu(k))=\operatorname{satisfied}.
$$

风险级别决定 $R$ 的范围和所需证据强度。特别地，任何必需条件处于 `violated` 或 `unknown` 时，高风险调用都不能进入 `approved`；允许的终态只能是缩小动作、dry-run、转人工或拒绝。这是工程验收门，不是一般法律结论。

## 5.3 输出记录

一次输出记录是十二元组

$$
\mathcal O=(i,c,t,v_g,v_c,u,b,\mathbf a,p,\mathbf s,n,q)
$$

其中

$$
\begin{aligned}
\mathcal O\in{}&
\operatorname{Field}(\mathcal I)
\times\operatorname{Field}(\mathcal C)
\times\operatorname{Field}(\operatorname{Tr}_\Sigma)
\times\operatorname{Field}(\mathcal V^*)
\times\operatorname{Field}(\mathcal V^*)
\times\operatorname{Field}(\mathbb U^*)
\times\operatorname{Field}(\mathbb B^*)\\
&\times\operatorname{Field}(\mathsf{Art}^*)
\times\operatorname{Field}(\mathsf{Prov})
\times\operatorname{Field}(\mathsf{ClaimRec}^*)
\times\operatorname{Field}(\mathsf{NormRec})
\times\mathsf{Status}.
\end{aligned}
$$

这里 $\mathsf{NormRec}$ 是满足上一节载体类型的规范记录集合，良构性另由 $\operatorname{WF}_{\mathsf N}$ 检查。状态 $q$ 不允许缺失；即使提交状态未知，也必须显式记录为 `unknown`。

## 5.4 良构谓词

$\operatorname{WF}(\mathcal O)$ 是以下条件的合取。凡条件提到两个字段的值，均只在这两个字段有值时施加；所需字段缺失是否可接受，留给下一节的用途剖面判断。

1. 每个字段都恰为 `Value(x)` 或 `Absent(reason)`，不使用无说明空位；
2. 若 $t\downarrow$，则其值是 $\Sigma$ 下的合法轨迹；若 $t,v_g\downarrow$，则 $v_g$ 等于轨迹的生成 token 投影；
3. 若 $t,v_c\downarrow$，则 $v_c$ 等于轨迹的提交 token 投影；
4. 若 $c,v_c,u\downarrow$，则 $u$ 与配置指定的 tokenizer 解码和 redaction 规则一致；
5. 若 $c,u,b\downarrow$，则 $b$ 与配置指定的序列化规则一致；
6. 若 $\mathbf a,p\downarrow$，则每个制品在 provenance 图中连接到产生它的 activity 和上游 entity；
7. 若 $p\downarrow$，则图中引用的 entity、activity 和 agent 标识可解析；
8. 若 $\mathbf s\downarrow$，则每条主张记录链接到文本跨度或制品、固定语境和证据协议；
9. 若 $n\downarrow$，则 $\operatorname{WF}_{\mathsf N}(n)$ 成立，且规范记录不把因果来源误写为作者身份或责任的充分证明；
10. $q$ 与轨迹中的成功、失败、取消或未知提交证据不冲突；若响应丢失而提交无法判定，$q$ 必须为 `unknown`，不得猜测为 `failed` 或 `succeeded`。

该谓词是审计规范，不是现实世界的全知描述。

## 5.5 三种不得合并的终态

给定用途剖面 $\Gamma$，它指定必需字段、允许的缺失原因、证据协议和可接受风险。定义：

1. $\operatorname{SchemaComplete}(\mathcal O)$：记录能按十二元组解析且 $\operatorname{WF}(\mathcal O)$ 成立；
2. $\operatorname{EvidenceSufficient}_\Gamma(\mathcal O)$：$\Gamma$ 要求的字段均有值，相关证据引用可解析，指定检查器返回接受；
3. $\operatorname{ClaimSupported}(s,\mathcal O)$：主张 $s$ 的证据协议在固定语境中返回 `Supported`。

主张核验状态集合取为

$$
\mathsf{ClaimStatus}=\{\operatorname{Supported},\operatorname{Refuted},
\operatorname{Inconclusive},\operatorname{Malformed}\}.
$$

三种终态互不蕴含。记录可以用 `Absent(Unknown)` 填满所有槽位而结构完整，却不具备足够证据；字段可以全部有值，证据仍可能反驳正文主张；某一主张也可能已由外部证书支持，而整个运行包仍因轨迹缺失而不完整。状态页必须分别报告三者，禁止用一个 `complete` 覆盖它们。

## 5.6 合同接口

复现合同写作

$$
C=(\mathcal D,N,P,M,A),
$$

其中 $\mathcal D$ 是合法输入域，$N$ 是规范化函数，$P$ 是前置条件，$M$ 是比较规则，$A$ 是判定算法。对合法输入，$A$ 返回 Pass、Fail 或 Inconclusive；对 schema 错误，返回结构错误。

## 5.7 责任接口

调用责任记录写作

$$
\mathsf{Resp}=(\operatorname{actor},\operatorname{permission},\operatorname{action},
\operatorname{approval},\operatorname{commit},\operatorname{rollback\_or\_compensation},
\operatorname{contest\_route},\operatorname{evidence},\operatorname{norm\_ref}).
$$

其中 `norm_ref` 指向上一节规范记录，`contest_route` 指向通知、复核、暂停或纠正程序。任何高风险提交都必须同时引用批准证据、规范验收剖面和提交回执。该元组不是法律责任的充分定义，只是工程审计所需的最小结构。

形式 schema 到这里停止。它规定记录怎样良构，却不决定一项具体主张是否已有足够证据；下一章把这些类型填入贯穿案例，并展示缺失、未定和已提交状态如何同时保留。
