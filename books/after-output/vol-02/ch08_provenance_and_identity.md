# 第八章：制品身份、provenance 与可验证声明

用户看到“SP404 已取消；已写入 trip.md。”后，系统保存了至少两个数字制品：最终消息和被修改的
`trip.md`。此外还有航班查询快照、幂等键记录、两次写入 attempt 的日志和流片段。复制最终消息会产生新存储对象；重新运行可能产生相同字节；忽略 attempt ID 后，两张来源图又可能看起来相同。因此，“这是同一个输出吗”必须先说明比较的是内容、运行还是生成史。

本章从贯穿案例的保存阶段进入制品身份与 provenance。第一章给出字节和规范化接口，第三、六、七章提供事件史；W3C PROV-DM 与 PROV-CONSTRAINTS 用作外部标准。哈希、签名和来源图会各自支持一种有限结论，而不会被合并成一个笼统的“真实性证明”。

## 8.1 制品与观察函数

设 $\mathsf{Art}$ 为制品集合。固定以下部分或总函数：

- 存储标识
  $\operatorname{sid}:\mathsf{Art}\to\mathsf{StoreID}$；
- 逻辑运行标识
  $\operatorname{rid}:\mathsf{Art}\rightharpoonup\mathsf{RunID}$；
- 原始字节
  $\operatorname{raw}:\mathsf{Art}\rightharpoonup\mathbb B^*$；
- 规范字节
  $\operatorname{canon}:\mathsf{Art}\rightharpoonup\mathbb B^*$。

$\operatorname{canon}$ 只有在 schema、规范化版本和序列化规则固定时才有定义。它不应被当作所有制品天然拥有的“真实内容”。

对任意函数 $f:X\to Y$，其核关系定义为

$$
x\equiv_f x'
\Longleftrightarrow f(x)=f(x').
$$

**命题 8.1（观察核是等价关系）.** $\equiv_f$ 在 $X$ 上是等价关系。

**证明.** 由 $Y$ 上等号的自反、对称与传递性逐项得到。证毕。

部分函数的核只在其定义域上构成等价关系。因而以下关系都有明确域：

- 存储身份 $\equiv_{\operatorname{sid}}$；
- 原始字节身份 $\equiv_{\operatorname{raw}}$；
- 规范内容身份 $\equiv_{\operatorname{canon}}$；
- 运行身份 $\equiv_{\operatorname{rid}}$。

它们一般互不等价。复制产生不同存储 ID 而保持字节；重跑产生不同 run ID 而可能保持内容；一次运行可导出多个不同字节制品。

## 8.2 PROV 数据模型接口

本书使用 W3C PROV-DM 的下列外部接口：

- **Entity**：具有某些固定方面、可被描述的物理、数字或概念对象；
- **Activity**：在时间区间内使用、生成或改变实体的过程；
- **Agent**：在 PROV 模型中与活动或实体的某种责任归属相关的对象，可为 person、organization 或 software agent；
- 关系包括 `used`、`wasGeneratedBy`、`wasDerivedFrom`、
  `wasAssociatedWith` 和 `wasAttributedTo`。

PROV 中“responsibility”是该来源数据模型中的关联语义。它允许记录某实体被归于某 agent，**不等同于**法律责任、道德责任或著作权作者资格。后者需要第十章的规范前提。

有效 PROV 记录还受生成先于使用等约束；只列出节点和边不保证记录满足 PROV-CONSTRAINTS。

## 8.3 来源图

为便于本书内部推理，把一个有限来源记录表示为带类型和属性的有向多重图

$$
P=(N_E,N_A,N_G,E,s,t,\ell,\alpha),
$$

其中 $N_E,N_A,N_G$ 是两两不交的 entity、activity、agent 节点集，$E$ 是边实例集合，

$$
s,t:E\to N_E\sqcup N_A\sqcup N_G
$$

分别给出每条有向边的源与目标；$\ell$ 给节点和边赋类型标签，$\alpha$ 给出所选不可变属性。每个输出制品指定一个根 entity。

“来源图相同”必须声明哪些属性参与比较。固定标签签名 $\Lambda$ 后，两个有根来源图 **$\Lambda$-同构**，若存在各节点种类上的双射和边集上的双射；它们保持根，并与 $s,t$ 交换，同时保持边类型及 $\Lambda$ 中的属性。

贯穿案例可以生成下列最小来源记录。实体
$e_i,e_f,e_k,e_t,e_u$ 分别表示用户请求、航班状态快照、幂等提交记录、
`trip.md` 制品和最终消息制品；活动
$a_q,a_{w1},a_{w2},a_g$ 分别表示查询、首次写入、幂等重试和消息生成。关键关系为：

| 来源事实 | 在运行中的含义 |
|---|---|
| $a_q$ `used` $e_i$，$e_f$ `wasGeneratedBy` $a_q$ | 查询请求产生带时点的 `Cancelled` 快照 |
| $a_{w1}$ `used` $e_f$，$e_t$ `wasGeneratedBy` $a_{w1}$ | 第一次 attempt 提交文件制品 |
| $e_k$ `wasGeneratedBy` $a_{w1}$ | 服务保存 $k_\star$ 的提交记录 |
| $a_{w2}$ `used` $e_k$ | 第二次 attempt 读取既有记录而不重复写入 |
| $a_g$ `used` $e_f,e_k$，$e_u$ `wasGeneratedBy` $a_g$ | 生成最终句子并把两个主张连接到证据实体 |

每个活动还可用 `wasAssociatedWith` 连接到执行它的软件或组织 agent。若根取
$e_u$，这是最终消息的来源图；若根取 $e_t$，则得到文件制品的来源图。两图共享节点不意味着根实体相同，也不意味着两项制品具有相同内容。

**命题 8.2（来源图同构是等价关系）.** 在固定签名 $\Lambda$ 的有限有根来源图集合上，$\Lambda$-同构是等价关系。

**证明.** 恒等映射给出自反性；图同构的逆映射仍保持根、类型、端点和属性，给出对称性；两个保持这些结构的双射复合后仍保持这些结构，给出传递性。证毕。

改变 $\Lambda$ 会改变关系。例如忽略时间可把两次重跑视为同构，保留 attempt ID 则会区分它们。

## 8.4 内容相同不蕴含来源相同

**命题 8.3.** 存在制品 $a,b$ 满足
$a\equiv_{\operatorname{raw}}b$，但其有根来源图在保留生成 activity ID 的签名下不同构。

**证明.** 令活动 $g_1$ 与 $g_2$ 在不同运行中都写入同一字节串 `hello`，分别生成实体 $a,b$。于是
$\operatorname{raw}(a)=\operatorname{raw}(b)$。若签名保留 activity ID，则任何有根图同构都必须把与 $a$ 相邻的生成活动 $g_1$ 映到具有同一 ID 的活动；$b$ 的图只有不同 ID 的 $g_2$，故不存在该同构。证毕。

若签名忽略 activity ID，两张最小图可能同构。这说明来源身份不是脱离比较签名的绝对关系。

## 8.5 哈希身份不是精确内容身份

固定哈希函数 $H:\mathbb B^*\to D$。定义摘要身份

$$
a\equiv_H b
\Longleftrightarrow
H(\operatorname{raw}(a))=H(\operatorname{raw}(b))
$$

于 raw 有定义的制品域上。

**命题 8.4（内容相同蕴含摘要相同）.** 若
$a\equiv_{\operatorname{raw}}b$，则 $a\equiv_H b$。

**证明.** 对相等字节应用函数 $H$ 得相等摘要。证毕。

逆命题成立当且仅当 $H$ 在所考虑字节集合上单射。若 $D$ 是固定有限长度摘要的有限值域，则 $H$ 不可能在全部有限字节串上单射，因为定义域 $\mathbb B^*$ 无限而 $D$ 有限。密码学 collision resistance 是关于受限攻击者找到碰撞之困难性的计算假设，不是“碰撞不存在”的数学证明。

因此摘要匹配是工程证据或内容寻址键；需要逐字节相等结论时，协议还必须说明信任的哈希假设或执行直接比较。

## 8.6 派生不是同一

`wasDerivedFrom` 是有方向的来源关系。PROV-CONSTRAINTS 没有给出
`wasDerivedFrom` 的一般传递推理规则；该关系也不要求自反或对称，因而不是身份等价。即使应用另取其传递闭包，闭包也只表达派生链可达性，不自动保存字节、指称或信息量。复制、转码、摘要、翻译与人工改写都可被记录为派生，但信息损失和变换语义不同。

可复现来源包应保存：

- 输入 entity 的稳定标识或内容；
- activity 的程序、参数和环境；
- 生成、使用与提交时间；
- 变换的确定性或随机输入；
- 人工编辑与批准事件；
- 使用的 provenance schema 与约束版本。

只写一条泛化派生边不足以重建活动。

## 8.7 logical、attempt 与 event 身份

一个用户意图可有 logical request ID $l$；每次物理执行有 attempt ID
$a_i$；每个流片段有 event ID $e_{ij}$。良好记录至少满足：

$$
\operatorname{attemptOf}(a_i)=l,
\qquad
\operatorname{eventOf}(e_{ij})=a_i.
$$

重试时 attempt ID 必须改变，幂等逻辑操作键可以保持。把 session ID 当作 run ID 会把多个执行混为一体；把 request ID 当作 event ID 又会无法去重流片段。

## 8.8 签名与可验证声明

数字签名验证的是：在指定算法、密钥和验证输入下，签名与某字节串匹配。它本身不证明：

- 签署者现实身份，除非证书或治理链另行建立；
- 字节中的事实为真；
- provenance 记录完整；
- 签署者具有作者资格或承担法律责任。

可信时间戳同样只支持相对于时间戳服务协议的“某摘要在某时已被见证”声明。每项结论都必须保留验证算法、证书状态、时间和信任根。

## 8.9 编辑与粒度

最终文本可经历模型草稿、人类删除、自动格式化、翻译和发布审批。token、字符、句子、段落和文档级来源分配可能不同。操作日志能提高粒度，却不能自动给共同改写分配唯一作者比例。

这是欠定性而非数据缺陷：若规范规则没有指定“保留结构”“改写措辞”和“最终批准”各自如何计入作者或信用，相同 provenance 事实可与多种分配相容。

现在可以准确比较贯穿案例的几个“相同”：复制 $e_u$ 可保持原始字节而改变存储 ID；用不同 attempt 重建同一句话可保持内容而改变运行 ID；若 $\Lambda$ 保留
$a_{w1},a_{w2}$，省去重试的运行与原运行不会来源图同构。哈希能紧凑绑定某组字节，PROV 能记录它们怎样形成，却还没有解释句中“SP404”指什么，也没有判断“已取消”和“已写入”是否为真。下一章把来源实体当作证据输入，而把指称、真值与核验状态定义在另一层。

## 练习

**练习 8.1.** 为一次带搜索工具的回答构造满足基本时间约束的有根 PROV 图，并标出 entity、activity 与 agent。

**练习 8.2.** 分别写出内容哈希、运行 UUID 和数字签名所支持结论的前提与反例。

**练习 8.3.** 给出同一 `wasDerivedFrom` 边型下两个信息损失不同的变换，并说明为何派生关系不是等价关系。

**练习 8.4.** 为重试请求设计 logical、attempt 和 event 三层 ID，并写出完整性约束。

**练习 8.5.** 为人类重写 AI 草稿的案例分别定义字符来源、结构来源和规范作者关系；说明为什么三者可以给出不同分配。


## 身份与来源审计接口

### S5.4 Artifact 与 provenance


artifact 是可保存、复制、签名或引用的数据制品。provenance 是关于制品如何产生的图。两个文件内容相同，不表示来源相同；同一来源图也不表示内容未被后续编辑。

最小 provenance 应记录：

1. 输入 entity；
2. 模型和运行 activity；
3. 工具 activity；
4. 输出 artifact；
5. 参与 agent；
6. 时间、版本和配置；
7. 用于校验的 digest 或签名。

### S6.9 provenance 与身份


Provenance 图包含 entity、activity 和 agent。简化关系包括：

- entity wasGeneratedBy activity；
- activity used entity；
- activity wasAssociatedWith agent；
- entity wasDerivedFrom entity。

内容相同不推出来源相同。两个 `trip.md` 文件字节相同，一个可能由模型写入，另一个可能由人工复制；它们的内容身份相同，来源身份不同。

哈希也不是全部身份。哈希证明的是相对于算法的摘要匹配，不证明来源、授权、语义或版权。
