# 第十一章：一次输出的完整分解

我们已经在不同章节中见过 `SP404` 最终消息 $u_\star$ 的字节、token、生成轨迹、工具提交、流片段、来源图、事实主张和规范分类。如果这些材料仍散落在互不相连的日志里，审计者只能靠猜测把它们拼回一次运行。最后一步是给每个分量定型，并把转换关系写成可检查的一致性谓词。

本章以贯穿案例的完整记录 $\mathcal O_\star$ 收束全书。它不会把所有层压回一个字段；相反，十二个分量分别保存输入、配置、轨迹、表示、制品、语义与归属。记录的“完整”始终相对于事件 schema，而不是对现实中每个物理事件的全知描述。

## 11.1 带缺失原因的字段与记录类型

单独的空值不能区分“尚未产生”“不适用”和“处理失败”。固定缺失原因集合

$$
\mathsf{AbsenceReason}=
\{\operatorname{NotYetProduced},
\operatorname{NotApplicable},
\operatorname{DecodeError},
\operatorname{SerializationError},
\operatorname{Redacted},
\operatorname{Unknown}\}.
$$

对任意集合 $X$，定义带标签联合

$$
\operatorname{Field}(X)=
(\{\operatorname{Value}\}\times X)
\sqcup
(\{\operatorname{Absent}\}\times\mathsf{AbsenceReason}).
$$

记有值分支为 $\operatorname{Value}(x)$，缺失分支为
$\operatorname{Absent}(r)$。这两个构造子互斥；缺失不表示数学发散，也不与空串混同。固定输入集合
$\mathcal I$、配置集合 $\mathcal C$、事件 schema $\Sigma$ 下的轨迹集合
$\operatorname{Tr}_\Sigma$、token 集 $\mathbb V$、制品集合
$\mathsf{Art}$、来源图集合 $\mathsf{Prov}$、主张记录集合
$\mathsf{ClaimRec}$、规范元数据集合 $\mathsf{Norm}$ 与运行状态集

$$
\mathsf{Status}=
\{\operatorname{running},\operatorname{succeeded},
\operatorname{failed},\operatorname{cancelled},
\operatorname{unknown}\}.
$$

## 11.2 输出记录

令 $\mathsf{OutputRec}_\Sigma$ 为下列乘积类型，并把一次输出记录定义为其中的十二元组

$$
\mathcal O=
(i,c,t,v_g,v_c,u,b,\mathbf a,p,\mathbf s,n,q)
$$

：

$$
\begin{aligned}
\mathsf{OutputRec}_\Sigma={}&
\mathcal I\times\mathcal C\times\operatorname{Tr}_\Sigma
\times\mathbb V^*\times\mathbb V^*
\times\operatorname{Field}(\mathbb U^*)
\times\operatorname{Field}(\mathbb B^*)\\
{}\times\mathsf{Art}^*
\times\mathsf{Prov}
\times\mathsf{ClaimRec}^*
\times\mathsf{Norm}
\times\mathsf{Status}.
\end{aligned}
$$

各分量含义为：

- $i$：输入实体和语境；
- $c$：模型、tokenizer、解码、执行与记录 schema 配置；
- $t$：相对于 $\Sigma$ 记录的事件轨迹；
- $v_g$：生成事件投影得到的 token 序列，包括尚未提交候选；
- $v_c$：commit 事件投影得到的已提交 token 序列；
- $u$：Unicode 标量序列的 Value，或带原因的 Absent；
- $b$：传输或持久化字节的 Value，或带原因的 Absent；
- $\mathbf a$：零个或多个保存、发布或外部副作用制品；
- $p$：有根 provenance 图；
- $\mathbf s$：文本主张、语境和核验状态记录；
- $n$：署名、许可、信用和责任声明及其规范依据；
- $q$：当前或最终运行状态。

“完整”指这些字段足以满足本书规定的审计接口，不表示记录了现实中的所有物理微状态。

## 11.3 良构谓词

定义
$\operatorname{WF}(\mathcal O)$ 为下列条件的合取。

1. **轨迹合法。** $t$ 从与 $i,c$ 对应的初态出发，并逐步满足 $c$ 所声明 LTS 的转移规则。
2. **生成投影。**
   $v_g=\operatorname{genTok}(t)$，其中投影及特殊 token 规则属于 $\Sigma$。
3. **提交投影。**
   $v_c=\operatorname{commitTok}(t)$。对没有撤回或分支候选的 append-only 协议，还要求 $v_c\preceq v_g$。若 $v_g$ 记录了后来被拒绝的候选，该前缀式不再合法，schema 必须改为“候选 ID 到 commit 事件”的显式关联；patch 协议则使用文档操作解释器。
4. **解码。** 配置与 schema 给出互斥谓词
   $\operatorname{TextReady}_c(t)$、
   $\operatorname{TextRedacted}_c(t)$。若 redacted 成立，则
   $u=\operatorname{Absent}(\operatorname{Redacted})$ 并保留授权记录。否则，若尚未 ready，则按 schema 取
   $\operatorname{Absent}(\operatorname{NotYetProduced})$ 或
   $\operatorname{Absent}(\operatorname{NotApplicable})$。若 ready 且
   $v_c\in\operatorname{AdmTok}_\Theta$，则
   $u=\operatorname{Value}(\operatorname{Dec}_\Theta(v_c))$；若 ready 但不在域内，则
   $u=\operatorname{Absent}(\operatorname{DecodeError})$，且轨迹或状态记录失败证据。任何
   $\operatorname{Absent}(\operatorname{Unknown})$ 都必须链接到记录缺口，不能替代已知的 decode error。
5. **序列化。** 当 $u=\operatorname{Value}(u_0)$ 且 serializer 在
   $(u_0,\mathsf{Envelope})$ 上成功时，
   $b=\operatorname{Value}(b_0)$，其中 $b_0$ 是其唯一字节结果。若 serializer 被调用但失败，则
   $b=\operatorname{Absent}(\operatorname{SerializationError})$ 并保留错误；尚未调用、不适用、被隐去或证据缺失分别使用相应 Absent 原因，不得都压成同一空值。
6. **制品关联。** 若 $b=\operatorname{Value}(b_0)$，则每个直接序列化制品
   $a\in\mathbf a$ 的原始或规范字节与 $b_0$ 一致；其他制品必须经一个已记录变换活动与 $b_0$ 或更早输入关联。若 $b$ 在 Absent 分支，制品关联只能依靠已记录的其他输入或活动，不能把 Absent 构造子当作字节。任何情形都不能仅凭相同文件名建立关联。
7. **来源覆盖。** $p$ 至少包含生成这些制品的 activity、关键输入 entity、工具响应和已知 agent 关联，并满足所采用 PROV 约束。
8. **主张链接。** 当 $u=\operatorname{Value}(u_0)$ 时，每个
   $s\in\mathbf s$ 链接到 $u_0$ 的有效跨度、完整语境、证据协议和核验状态；当 $u$ 在 Absent 分支时，不得伪造文本跨度。
9. **规范分层。** $n$ 记录所用规范体系，不把 $p$ 或 $t$ 形式上误写为作者或责任关系的充分证明。
10. **状态一致。** succeeded 需要所声明成功谓词成立；failed、cancelled 与 unknown 分别保留失败、取消 cut 或未知提交状态，不能仅由最终自然语言自述决定。

这是记录完整性规范，不是自然界“输出”唯一可能的形而上定义。

## 11.4 记录轨迹与实际轨迹

设实际系统事件集合为 $E_{\mathrm{real}}$，记录器是部分观察

$$
L:E_{\mathrm{real}}^*\rightharpoonup
\operatorname{Tr}_\Sigma.
$$

日志相对于 schema 完整，只表示每个 $\Sigma$ 要求的事件都被记录并通过一致性检查。它不蕴含 $L$ 单射：多个硬件、网络或组织层实际轨迹可以有同一日志。审计结论只能量化日志可见信息，除非另有遥测或证明扩大观察。

## 11.5 投影身份

对记录分量的任意总投影
$\pi:\mathsf{OutputRec}_\Sigma\to X$，定义

$$
\mathcal O_1\equiv_\pi\mathcal O_2
\Longleftrightarrow
\pi(\mathcal O_1)=\pi(\mathcal O_2).
$$

由命题 8.1，它是等价关系。可选投影包括：

- $\pi_b$：同序列化字段，包括 Value 的字节或 Absent 原因；
- $\pi_u$：同 Unicode 字段，包括 Value 的文本或 Absent 原因；
- $\pi_{v_c}$：同已提交 token 序列；
- $\pi_t$：同记录轨迹；
- $\pi_q$：同任务状态。

两个 $\operatorname{Absent}(\operatorname{Unknown})$ 字段相等，只说明记录采用同一缺失标签，不说明现实中的未知值相同。若用途要区分不同缺口，还应把缺失事件 ID 或证据范围并入投影。

provenance 比较通常使用第八章的带签名图同构而非原始图对象相等；语义主张比较使用第九章相对于模型类或语境的等价；规范元数据比较依赖规范版本。

## 11.6 不存在用途无关的唯一身份

这里的结论是概念论证：

- 存在 UTF-8 与 UTF-16 两个字节制品，Unicode 文本相同而字节不同；
- 存在两种 token 分解，解码文本相同而 token 序列不同；
- 存在两个独立运行，最终字节相同而 provenance 不同；
- 存在最终文本相同、用户呈现历史不同的流式轨迹。

因此上述合理等价关系在某些记录对上给出不同答案。若不先指定用途与投影，“是否同一次输出”没有唯一判据。

## 11.7 把贯穿案例填入十二个字段

取贯穿案例最终时点的记录 $\mathcal O_\star$。第五章固定的随机输入为
$U_1=0.73$，所以实际 token 路径是
$v^{(b)}=(201,103,104,105,106,107)$。十二个字段可以逐项填写如下：

| 分量 | $\mathcal O_\star$ 中的值 |
|---|---|
| $i_\star$ | 用户请求、航班运营日期 $d_\star$、工作目录与授权语境 |
| $c_\star$ | 模型与实现版本、$\Theta_\star$、随机输入记录、工具和流协议版本、schema $\Sigma_\star$ |
| $t_\star$ | 查询、两次写入 attempt、一次文件 commit、token 生成、序列化、乱序 recv 和三次消息 commit 的完整记录轨迹 |
| $v_g$ | $v^{(b)}$，即生成事件投影 |
| $v_c$ | $v^{(b)}$，因为本次没有 token 撤回或拒绝 |
| $u$ | $\operatorname{Value}(u_\star)$ |
| $b$ | $\operatorname{Value}(E_8(u_\star))$；JSON envelope 字节另存为制品字段 |
| $\mathbf a$ | 航班快照 $e_f$、文件制品 $e_t$、幂等记录 $e_k$、消息制品 $e_u$ 及 envelope |
| $p$ | 第八章以 $e_u$ 为根并连接 $a_q,a_{w1},a_{w2},a_g$ 的来源图 $p_\star$ |
| $\mathbf s$ | 第九章的 $(\varphi_f,\operatorname{Supported})$ 与 $(\varphi_w,\operatorname{Supported})$，连同语境和证据链接 |
| $n$ | 适用的署名、贡献和运维责任规则版本 $n_\star$，以及按这些规则作出的分类 |
| $q$ | $\operatorname{succeeded}$；依据是幂等重试确认与提交后文件核验，不是消息中的自述 |

这个表不是摘要性的“运行元数据”。每个值都落在 11.2 节声明的对应类型中；例如 $u$ 与 $b$ 必须使用 `Value` 构造子，不能直接把文本塞入
$\operatorname{Field}(\mathbb B^*)$，而 $\mathbf s$ 中保存的是带语境的主张记录，不是两个孤立真值。

## 11.8 沿良构谓词重放这次运行

现在逐段核对 $\operatorname{WF}(\mathcal O_\star)$。轨迹从
$(i_\star,c_\star)$ 指定的初态开始；查询返回 $e_f$ 后，首次写入 attempt 产生
$e_t,e_k$，确认丢失只改变控制器状态，第二次 attempt 读取 $e_k$ 并恢复到已知提交状态。随后第五章的实现映射选择 $v^{(b)}$，所以生成投影和提交投影都等于表中的值。

由于 $v^{(b)}\in\operatorname{AdmTok}_{\Theta_\star}$，解码条件给出

$$
u=\operatorname{Value}
(\operatorname{Dec}_{\Theta_\star}(v^{(b)}))
=\operatorname{Value}(u_\star).
$$

严格 UTF-8 serializer 再给出唯一负载
$b=\operatorname{Value}(E_8(u_\star))$。第七章的三个字节片段虽然按
$m_2,m_1,m_3$ 到达，却按序号提交为这个 $b$；到达历史和最终字节因此同时保存在不同投影中。

制品关联由 $p_\star$ 中的生成与使用边覆盖。$\varphi_f$ 的跨度连接“SP404 已取消”，证据连接 $e_f$；$\varphi_w$ 的跨度连接“已写入 trip.md”，证据连接 $e_k$ 与提交后的文件对象。$n_\star$ 只按其规则分类描述事实。最后，状态 success 的见证是任务谓词、文件核验和消息提交均成立。十项条件由此分别得到证据，没有任何一项由“最终文本看起来合理”替代。

## 11.9 同一次运行的三个截面

记录在运行中不是一次性出现的。下表截取三个时点，展示缺失原因和状态怎样随事件推进：

| 截面 | 工具状态 | 文本字段 $u$ | 任务状态 $q$ |
|---|---|---|---|
| 首次写入确认丢失后 | $\operatorname{UnknownCommitState}$ | $\operatorname{Absent}(\operatorname{NotYetProduced})$ | $\operatorname{unknown}$ |
| token 已生成、片段 2 先到后 | 写入已由重试确认 | 若 schema 只在全消息 commit 后解码，则仍为 $\operatorname{Absent}(\operatorname{NotYetProduced})$ | $\operatorname{running}$ |
| 三片按序提交并完成核验后 | committed 且已读回 | $\operatorname{Value}(u_\star)$ | $\operatorname{succeeded}$ |

第一截面中，文件在虚构世界里已经写入，但记录不能把 unknown 冒充 committed；第二截面中，候选字节已经存在，却还没有 schema 所定义的最终文本；第三截面才是 11.7 节的 $\mathcal O_\star$。若写入最终失败而生成器仍输出“已写入”，语言生成子系统可以正常终止，任务状态却应为 failed，$\varphi_w$ 的核验状态也应为 Refuted。一个状态字段无法代替这三种结论。

## 11.10 机器可读 schema 的最低约束

实现 schema 至少应：

1. 对每个字段给出类型、版本与可选原因；
2. 为事件指定 stream/run ID、序号和时间语义；
3. 对 token、Unicode 与字节转换保存 tokenizer 和 serializer 哈希；
4. 对制品保存 exact byte hash、schema 和变换 activity；
5. 对 claims 保存原文跨度、语境、证据与状态；
6. 对 normative 字段保存规则集与决策者；
7. 允许 unknown、cancelled、partial 和 conflicting，而不强行压成 success/failure。

schema 验证只能证明结构和局部一致性；不能证明外部事实、日志完整捕获现实或规范判断正确。

## 11.11 最小审计程序

对记录 $\mathcal O$，依次检查：

1. 当前结论引用哪个分量和哪个等价关系？
2. 转换函数、定义域、版本与失败值是否保存？
3. 错误、重试、撤回、取消和 commit 是否在 $t$ 中？
4. 工具事实与副作用是否链接到世界状态证据？
5. provenance 图是否满足所采用约束和比较签名？
6. 每个文本主张是否补全语境并给出协议相对状态？
7. 署名、信用与责任是否显式引用规范体系？
8. 任何“完整”是否只在 $\Sigma$ 的记录边界内声称？

沿着 $\mathcal O_\star$ 回看全书，所谓“一次输出”已经不再是一个含混名词。它可以是
$v^{(b)}$、$u_\star$、$E_8(u_\star)$、$t_\star$、$e_u$、
$p_\star$、两个事实主张或一组规范分类；每个对象都有自己的类型和同一关系。
$\operatorname{WF}$ 所做的不是选出其中一个“真正输出”，而是检查它们之间的转换、证据和归属是否彼此一致。改变审计用途就会改变投影；保留这个选择，才是本书最终得到的输出存在论。

## 练习

**练习 11.1.** 为一次普通聊天回答填写十二个分量，并逐条核对 $\operatorname{WF}$。

**练习 11.2.** 构造 $u$ 相同但 $v_c$、$t$ 和 $p$ 均不同的两个良构输出记录。

**练习 11.3.** 为一个 unknown commit state 的工具调用分别标注自然语言主张状态、任务状态和世界状态证据。

**练习 11.4.** 设计一个机器可读输出记录 schema，使错误、取消、缺失值及其原因不被混淆。

**练习 11.5.** 审查“这句话就是模型的思想”至少包含哪些类型跃迁，并为每一步写出所缺桥接理论。


## 完整记录审计接口

### S5.7 输出记录


本书使用十二分量记录：

$$
\mathcal O=(i,c,t,v_g,v_c,u,b,\mathbf a,p,\mathbf s,n,q).
$$

其中 $i$ 是输入，$c$ 是配置，$t$ 是轨迹，$v_g$ 是生成 token，$v_c$ 是提交 token，$u$ 是文本，$b$ 是字节，$\mathbf a$ 是制品，$p$ 是 provenance，$\mathbf s$ 是主张记录，$n$ 是规范元数据，$q$ 是状态。

完整不表示全知，只表示相对于声明 schema 已足以审计。

### 练习


**练习 S5.1.** 给出文本相同但 provenance 不同的两个输出。

**练习 S5.2.** 给出工具调用已提交但最终文本未显示的例子。

**练习 S5.3.** 对一段回答列出至少三个主张，并为每个主张指定核验证据。

### S6.12 从运行对象到不确定性


一次输出不是字符串，而是部分函数、关系、轨迹、工具、外部世界、并发事件、artifact、provenance、主张核验和规范归属的组合。这个类型化运行对象是后续概率、复现、解释和责任分析的共同基底。

### 练习


**练习 S6.1.** 构造一个 UTF-8 字节解码失败但原始字节仍可被完整保存和校验的输出记录。

**练习 S6.2.** 给出一个工具调用返回超时但实际副作用已经提交的轨迹，并说明结果为何不能直接标为 `Fail`。

**练习 S6.3.** 为两个并发工具调用画出 happens-before 偏序，并给出两个合法线性化。

**练习 S6.4.** 说明相同内容摘要为何不能单独证明两个 artifact 具有相同 provenance 或规范归属。
