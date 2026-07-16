# 第一章 字节、字符串、Unicode、字形与 token

客户端把最终消息保存下来时，磁盘上没有一个叫作“句子”的原子对象。它得到的是字节；JSON 解析器把其中一部分解释成字符串值；Unicode 规则把字符串组织为标量序列和字素簇；渲染器再把它变成屏幕上的字形。模型一侧保存的却可能是 token ID。贯穿案例的可见句子是

> SP404 已取消；已写入 trip.md。

本章沿着这句话在各层的表示逐步向内追踪。所需预备知识只有集合、有限序列与函数复合；Unicode 标量值、规范化、扩展字素簇和 UTF-8 的标准定义作为外部输入列在 [SOURCES.md](SOURCES.md) 中。

## 1.1 一句话落盘以后

对任意集合 $A$，记

$$
A^*=\coprod_{n\in\mathbb N}A^n
$$

为 $A$ 上有限序列的集合，空序列记为 $\epsilon$，连接记为 $xy$。固定：

- 八位字节集合 $\mathbb B=\{0,\ldots,255\}$；
- Unicode 标量值集合
  $$
  \mathbb U=\{0,\ldots,\mathtt{10FFFF}_{16}\}\setminus
  \{\mathtt{D800}_{16},\ldots,\mathtt{DFFF}_{16}\};
  $$
- UTF-16 编码单元集合 $\mathbb C_{16}=\{0,\ldots,2^{16}-1\}$；
- 某 tokenizer 的有限 token ID 集合 $\mathbb V$。

“字符串”不是无类型名词。字节 API 的 string 可属于 $\mathbb B^*$，许多运行时 string 属于 $\mathbb C_{16}^*$，本书的规范文本则属于 $\mathbb U^*$。同一符号改变这些类型时必须显式经过编码或解码函数。

记贯穿案例的 Unicode 标量序列为 $u_\star\in\mathbb U^*$。它以 ASCII 标量
`S`、`P`、`4`、`0`、`4` 开始，随后含空格、汉字、全角分号、拉丁字母和句号。UTF-8 编码后的字节串记为 $b_\star=E_8(u_\star)$；若 API 又把它嵌入 JSON 响应，则网络上传输的是更长的 envelope 字节串，而不是 $b_\star$ 本身。已经出现了三个对象：规范文本 $u_\star$、其 UTF-8 负载 $b_\star$，以及包含字段名、引号和转义的响应字节。

## 1.2 UTF-8 的合法域

RFC 3629 与 Unicode 标准确定一个总编码函数

$$
E_8:\mathbb U^*\longrightarrow \mathbb B^*.
$$

令

$$
\operatorname{UTF8}=E_8(\mathbb U^*)\subsetneq\mathbb B^*
$$

为合法 UTF-8 字节串集合。严格解码是总函数

$$
D_8:\operatorname{UTF8}\longrightarrow\mathbb U^*,
$$

等价地也可写为部分函数 $D_8:\mathbb B^*\rightharpoonup\mathbb U^*$，其定义域恰为 $\operatorname{UTF8}$。

**外部输入 1.A（UTF-8 唯一编码）.** 对每个 Unicode 标量值，UTF-8 有且仅有一个合法字节表示；合法字节串能唯一分解并解码为标量序列。这里排除 surrogate code point、overlong encoding 和超出 U+10FFFF 的序列。

由该标准接口得到

$$
D_8\circ E_8=\operatorname{id}_{\mathbb U^*},
\qquad
E_8\circ D_8=\operatorname{id}_{\operatorname{UTF8}}.
$$

**定理 1.1（双向 round-trip）.** $E_8$ 是 $\mathbb U^*$ 到 $\operatorname{UTF8}$ 的双射，$D_8=E_8^{-1}$。

**证明.** 第一式说明 $E_8$ 有左逆，故单射：若 $E_8(u)=E_8(v)$，应用 $D_8$ 得 $u=v$。按 $\operatorname{UTF8}=E_8(\mathbb U^*)$ 的定义，$E_8$ 满射到该集合。第二式与第一式说明 $D_8$ 是其双侧逆。证毕。

宽容解码器通常定义为总函数 $\widetilde D_8:\mathbb B^*\to\mathbb U^*$，并以 U+FFFD 替换错误片段。它不是 $E_8$ 在全部 $\mathbb B^*$ 上的逆；多个非法字节串可以得到同一替换文本。

对贯穿案例，严格解码给出 $D_8(b_\star)=u_\star$。如果传输途中截断了“已”的三字节编码，只保留其前两个字节，那么所得串不在 $\operatorname{UTF8}$ 中，$D_8$ 没有函数值。宽容解码器可以显示替换字符，但那是另一个总函数的结果，不能作为原句被完整收到的证据。

## 1.3 标量值、字素簇与字形

Unicode 标量值不等于用户感知的“字符”。

**外部输入 1.C（Unicode 文本分段）.** UAX #29 在固定 Unicode 版本下给出默认扩展字素簇边界规则，也允许通过明确 profile 修改规则或属性。本书只使用“版本与 profile 固定后，规则确定分段结果”这一接口；不重证规则表，也不把扩展字素簇升级为语言学上无条件的字符定义。

固定 Unicode 版本 $\nu$ 与一个符合 UAX #29 的 profile $\gamma$。记
$\mathbb U^+=\mathbb U^*\setminus\{\epsilon\}$，令
$\mathcal G_{\nu,\gamma}\subseteq\mathbb U^+$ 为该规则可能产生的非空标量块集合，则分段算法可表示为总函数

$$
\operatorname{grapheme}_{\nu,\gamma}:
\mathbb U^*\to\mathcal G_{\nu,\gamma}^*.
$$

若
$\operatorname{grapheme}_{\nu,\gamma}(u)=(g_1,\ldots,g_m)$，则
$u=g_1\cdots g_m$；空输入映到空块序列。一个扩展字素簇可含多个标量值，不同 profile 也可能给出不同边界。

渲染还依赖字体、fallback、文字整形、语言、方向、尺寸和布局。把所有这些参数收进配置集合 $\mathcal R$ 后，可写部分函数

$$
\operatorname{shape}:\mathbb U^*\times\mathcal R
\rightharpoonup \operatorname{GlyphRun}.
$$

缺失字体或不受支持的布局可使其失败。截图相同不蕴含标量序列相同；标量序列相同也不蕴含像素相同。

## 1.4 Unicode 规范化

固定 Unicode 版本与一种规范化形式，规范化为总函数

$$
N:\mathbb U^*\to\mathbb U^*.
$$

**外部输入 1.B（规范化接口）.** NFC、NFD、NFKC、NFKD 在其标准定义下幂等，即 $N(N(u))=N(u)$。canonical normalization 保持规范等价类；compatibility normalization 还会折叠某些兼容差异。

幂等不蕴含单射。若 $u\ne v$ 而 $N(u)=N(v)$，规范化已经丢失区分信息。因而在标识符、安全检查或数字签名之前使用 NFKC，必须把所选版本和策略写进协议，而不能只说“先规范化”。

## 1.5 Tokenizer 制品

一个 tokenizer 配置不是只有词表。记

$$
\Theta=(\mathbb V,N_\Theta,A_\Theta,P_\Theta,\delta_\Theta),
$$

分别表示 token ID 集、预处理、分词算法、每个 token 的负载以及特殊 token 规则。只有固定 $\Theta$ 和所有并列选择后，编码过程才可能是函数。

编码的输入域可能是 Unicode 或字节。为统一记号，令 $X_\Theta$ 为输入集合，$\operatorname{AdmIn}_\Theta\subseteq X_\Theta$ 为可接受输入；令 $\operatorname{AdmTok}_\Theta\subseteq\mathbb V^*$ 为可解码 token 序列。于是

$$
\operatorname{Enc}_\Theta:
\operatorname{AdmIn}_\Theta\to\operatorname{AdmTok}_\Theta,
\qquad
\operatorname{Dec}_\Theta:
\operatorname{AdmTok}_\Theta\to X_\Theta.
$$

若算法保留未指定的随机选择或并列选择，所谓“编码器”首先只是关系；固定选择规则后才右唯一。若某些特殊 token 只控制协议而没有文本负载，则 $\operatorname{Dec}_\Theta$ 未必在整个 $\mathbb V^*$ 上有定义。

为了让后文可以逐步计算，固定一个仅用于贯穿案例的玩具配置
$\Theta_\star$。它把 $u_\star$ 的规范编码取为

$$
v_\star=(101,102,103,104,105,106,107),
$$

七个 ID 的负载依次为 `SP`、`404`、` 已取消`、`；`、`已写入`、
` trip.md` 与 `。`。于是

$$
\operatorname{Enc}_{\Theta_\star}(u_\star)=v_\star,
\qquad
\operatorname{Dec}_{\Theta_\star}(v_\star)=u_\star.
$$

这不是对现实 tokenizer 边界的经验声称，而是一个完整指定的有限例子。若另一 admissible 序列把 `SP404` 合成一个 token，它仍可解码为 $u_\star$；token 身份已经改变，文本身份没有改变。

## 1.6 可逆性与规范像

设 $\operatorname{Enc}:X\to V^*$、$\operatorname{Dec}:A\to X$，其中 $\operatorname{Enc}(X)\subseteq A\subseteq V^*$。

**定理 1.2（编码 round-trip 的精确后果）.** 若

$$
\operatorname{Dec}(\operatorname{Enc}(x))=x
\quad\text{对所有 }x\in X,
$$

则：

1. $\operatorname{Enc}$ 单射；
2. $\operatorname{Dec}$ 在 $\operatorname{Enc}(X)$ 上满射到 $X$；
3. 对每个 $v\in\operatorname{Enc}(X)$，
   $\operatorname{Enc}(\operatorname{Dec}(v))=v$。

**证明.** 第一项由左逆蕴含单射。第二项中，任取 $x\in X$，$x=\operatorname{Dec}(\operatorname{Enc}(x))$，故 $x$ 在限制映射的像中。第三项令 $v=\operatorname{Enc}(x)$，则
$\operatorname{Enc}(\operatorname{Dec}(v))
=\operatorname{Enc}(\operatorname{Dec}(\operatorname{Enc}(x)))
=\operatorname{Enc}(x)=v$。证毕。

第三项只对编码器的规范像成立。若 $A$ 还含其他能解码成同一文本的 token 序列，则 $\operatorname{Enc}\circ\operatorname{Dec}$ 在 $A$ 上不必为恒等。

**反例 1.3（解码可多对一）.** 设词表负载含 `ab`、`a`、`b`，解码按负载连接。则序列 $[\mathtt{ab}]$ 与 $[\mathtt a,\mathtt b]$ 都解码为 `ab`。一个确定编码器可以选前者作为规范表示，解码器仍非单射。

若 $X_\Theta=\mathbb U^*$，预处理包含总函数
$N_\Theta:X_\Theta\to X_\Theta$，且编码器实际编码
$N_\Theta(x)$，则常见性质是

$$
\operatorname{Dec}_\Theta(\operatorname{Enc}_\Theta(x))=N_\Theta(x),
$$

而不是 $x$。此时编码器对原始输入也不可能单射。

## 1.7 特殊 token 与文本投影

模型 token 轨迹可包含 BOS、EOS、padding、控制 token 或工具边界。定义文本投影前必须指定：

- 哪些 token 有字节或 Unicode 负载；
- 特殊 token 是删除、转义还是报错；
- token 负载连接后采用何种严格解码；
- 流式片段是否允许切开一个多字节 UTF-8 序列。

因此“token 解码”可以是部分函数，也可以是返回
$\operatorname{Result}(\mathbb U^*,E)$ 的总函数；两种接口不能混写。

## 1.8 序列化与呈现

API 常把文本放入 JSON 等外壳。固定 schema、字段顺序约定、字符转义和数值序列化后，serializer 才是确定函数。JSON 对象在数据模型中可相等而传输字节不同；直接字符和 Unicode 转义也可表示同一字符串值。

取证或复现至少区分：

1. 原始传输字节；
2. 解析后的结构化值；
3. tokenizer 制品及 token ID；
4. 规范 Unicode 序列；
5. 渲染配置和截图。

贯穿案例现在有了一条类型正确的静态链：

$$
v_\star
\xrightarrow{\operatorname{Dec}_{\Theta_\star}}
u_\star
\xrightarrow{E_8}
b_\star
\xrightarrow{\operatorname{serialize}}
b_{\mathrm{env}}.
$$

这条链说明最终材料如何转换，却没有说明 $v_\star$ 是怎样产生的，也没有记录查询、写入或重试。把整个系统写成一个从 prompt 到 $u_\star$ 的函数会立即丢掉失败、选择与交互。下一章先澄清什么时候这种函数写法成立，什么时候只能写成部分函数或关系。

## 练习

**练习 1.1.** 给出一个幂等但非单射的规范化函数，并指出其商掉的等价类。

**练习 1.2.** 证明若 $E:X\to Y$ 有右逆 $R:Y\to X$，则 $E$ 满射；给出它不单射的例子。

**练习 1.3.** 构造同一文本的两种 admissible token 分解，并说明确定编码器为何仍只返回一种。

**练习 1.4.** 构造两个不同非法 UTF-8 字节串，使某个替换式宽容解码器把它们映到同一 Unicode 序列，并说明 round-trip 在哪一步失败。

**练习 1.5.** 为聊天响应设计同时保存传输字节、结构化对象、token、Unicode 文本和渲染证据的制品格式，并写出每条转换边的版本字段。
