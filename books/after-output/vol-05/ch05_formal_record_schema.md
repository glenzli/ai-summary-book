# 第五章 形式记录 schema

这一参考章给出全书使用的最小形式 schema。字段并不试图穷尽某个具体平台，而是固定跨卷反复调用的对象类型：输入、配置、轨迹、token、文本、字节、制品、来源图、主张记录、规范元数据与运行状态。schema 来自《一次输出的存在论》和《随机鹦鹉的自传》的形式模型，经合并后保留能够支持验证器、审计包和贯穿案例的公共核。

## A.1 字段

固定输入集合 $\mathcal I$、配置集合 $\mathcal C$、事件 schema $\Sigma$ 下的轨迹集合 $\operatorname{Tr}_\Sigma$、token 集 $\mathcal V$、Unicode 标量集合 $\mathbb U$、字节集合 $\mathbb B$、制品集合 $\mathsf{Art}$、来源图集合 $\mathsf{Prov}$、主张记录集合 $\mathsf{ClaimRec}$、规范元数据集合 $\mathsf{Norm}$ 和运行状态集合

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

## A.2 输出记录

一次输出记录是十二元组

$$
\mathcal O=(i,c,t,v_g,v_c,u,b,\mathbf a,p,\mathbf s,n,q)
$$

其中

$$
\begin{aligned}
\mathcal O\in{}&
\mathcal I\times\mathcal C\times\operatorname{Tr}_\Sigma
\times\mathcal V^*\times\mathcal V^*
\times\operatorname{Field}(\mathbb U^*)
\times\operatorname{Field}(\mathbb B^*)\\
&\times\mathsf{Art}^*
\times\mathsf{Prov}
\times\mathsf{ClaimRec}^*
\times\mathsf{Norm}
\times\mathsf{Status}.
\end{aligned}
$$

## A.3 良构谓词

$\operatorname{WF}(\mathcal O)$ 是以下条件的合取：

1. 轨迹 $t$ 合法；
2. $v_g$ 等于生成 token 投影；
3. $v_c$ 等于提交 token 投影；
4. 文本字段 $u$ 与 tokenizer 解码和 redaction 规则一致；
5. 字节字段 $b$ 与序列化规则一致；
6. 制品 $\mathbf a$ 与字节或上游输入有记录关系；
7. provenance 图 $p$ 覆盖关键 entity、activity 和 agent；
8. 主张记录 $\mathbf s$ 链接到文本跨度、语境和证据协议；
9. 规范元数据 $n$ 不把因果来源误写为作者或责任的充分证明；
10. 状态 $q$ 与成功、失败、取消或未知提交证据一致。

该谓词是审计规范，不是现实世界的全知描述。

## A.4 合同接口

复现合同写作

$$
C=(\mathcal D,N,P,M,A),
$$

其中 $\mathcal D$ 是合法输入域，$N$ 是规范化函数，$P$ 是前置条件，$M$ 是比较规则，$A$ 是判定算法。对合法输入，$A$ 返回 Pass、Fail 或 Inconclusive；对 schema 错误，返回结构错误。

## A.5 责任接口

调用责任记录写作

$$
\mathsf{Resp}=(\operatorname{actor},\operatorname{permission},\operatorname{action},\operatorname{approval},\operatorname{commit},\operatorname{rollback},\operatorname{evidence}).
$$

它不是法律责任的充分定义，只是工程审计所需的最小结构。
