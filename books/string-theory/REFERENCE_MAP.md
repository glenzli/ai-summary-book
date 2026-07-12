# 逐章资料源映射

本文档把 [SOURCES.md](SOURCES.md) 中的资料源映射到章节。它不是逐句脚注系统，而是教材内容层面的引用闭合表。

## 0-5：世界面、CFT 和 BRST

| 章节 | 主要资料源 |
|---|---|
| 0 | Polchinski, Green-Schwarz-Witten, Becker-Becker-Schwarz |
| 1 | Polchinski Vol. 1, Zwiebach, Becker-Becker-Schwarz |
| 2 | Polchinski Vol. 1, Zwiebach, Green-Schwarz-Witten Vol. 1 |
| 3 | Di Francesco-Mathieu-Senechal, Ginsparg, Polchinski Vol. 1 |
| 4 | Polchinski Vol. 1, Green-Schwarz-Witten Vol. 1, Zwiebach；`GT72` |
| 5 | Polchinski Vol. 1, Green-Schwarz-Witten Vol. 1, Ginsparg；`POLY81` |

## 6-8：振幅、D-branes 和 RNS

| 章节 | 主要资料源 |
|---|---|
| 6 | Polchinski Vol. 1, Green-Schwarz-Witten Vol. 1, Zwiebach |
| 7 | Polchinski Vol. 1-2, Johnson, Becker-Becker-Schwarz；`BUS87/88`、`NAR86` |
| 8 | Polchinski Vol. 1, Green-Schwarz-Witten Vol. 1, Becker-Becker-Schwarz；`GSO77` |

## 9-12：超弦、heterotic、低能作用和 D-branes

| 章节 | 主要资料源 |
|---|---|
| 9 | Green-Schwarz-Witten Vol. 1-2, Polchinski Vol. 2, Becker-Becker-Schwarz |
| 10 | Polchinski Vol. 2, Green-Schwarz-Witten Vol. 2, Becker-Becker-Schwarz；`NAR86`、`GS84` |
| 11 | Polchinski Vol. 1-2, Becker-Becker-Schwarz, Blumenhagen-Lust-Theisen；`CFMP85`、`KOS61`、`GW86` |
| 12 | Johnson, Polchinski Vol. 2, Becker-Becker-Schwarz |

## 13-16：紧化、对偶性、高 genus 和拓扑弦

| 章节 | 主要资料源 |
|---|---|
| 13 | Becker-Becker-Schwarz, Blumenhagen-Lust-Theisen, Huybrechts；`YAU78`、`BTT`、`DUY` |
| 14 | Polchinski Vol. 2, Becker-Becker-Schwarz, Green-Schwarz-Witten Vol. 2 |
| 15 | Polchinski Vol. 1, Di Francesco-Mathieu-Senechal, Ginsparg |
| 16 | Hori et al., Mirror Symmetry, Blumenhagen-Lust-Theisen |

## 17-20：非微扰、AdS/CFT、flux 和接口

| 章节 | 主要资料源 |
|---|---|
| 17 | Polchinski Vol. 2, Becker-Becker-Schwarz, Johnson；`CARDY86`、`SV96`、`FKS95`、`WALD93/94` |
| 18 | Maldacena, Gubser-Klebanov-Polyakov, Witten, Polchinski Vol. 2；`BF82`、`HREN00`、`MAL-W98` |
| 19 | Becker-Becker-Schwarz, Blumenhagen-Lust-Theisen, Polchinski Vol. 2 |
| 20 | Hori et al., Maldacena, Witten, Di Francesco-Mathieu-Senechal |

## 附录

| 附录 | 主要资料源 |
|---|---|
| A | Huybrechts, Becker-Becker-Schwarz |
| B | Di Francesco-Mathieu-Senechal, Ginsparg |
| C | Green-Schwarz-Witten Vol. 2, Becker-Becker-Schwarz |
| D | Polchinski Vol. 1, Di Francesco-Mathieu-Senechal |
| E | Green-Schwarz-Witten Vol. 1-2, Becker-Becker-Schwarz |

## 外部输入和猜想的引用原则

1. `E` 类外部输入优先回指本表中对应章节的资料源。
2. `C` 类物理猜想优先回指原始论文或标准综述；AdS/CFT 使用 Maldacena、Gubser-Klebanov-Polyakov、Witten。
3. 本书不复制资料源原文，只使用标准定义、定理名称、公式和教材化推导。

## 主线输入闭合表

| 正文编号 | 状态 | 来源定位 | 本书不承担的部分 |
|---|---|---|---|
| 4.10 | `E` | `GT72` | no-ghost theorem 的完整表示论证明 |
| 5.12、6.16 | `E` | `POLY81`；Polchinski Vol. 1 | determinant line、moduli measure 与 sewing/factorization theorem |
| 7.8、7.18 | `E` | `BUS87/88`、`NAR86`；Polchinski Vol. 1 | cocycle/mutual-locality 与完整 Narain CFT 构造 |
| 8.16A | `E` | `GSO77`；Green-Schwarz-Witten Vol. 1 | spin-structure modular sum 与 higher-genus supermoduli |
| 10.11 | `E` | `GS84` | chiral determinant、index/descent 与 trace identities 的全计算 |
| 11.2、11.6B、11.13 | `E` | `CFMP85`、`KOS61`、`GW86` | sigma-model renormalization、equivalence theorem、RNS 四-graviton correlator |
| 13.2、13.7、13.12 | `E` | `YAU78`、`BTT`、`DUY` | Monge-Ampere、unobstructedness、Hermitian-Yang-Mills existence |
| 13.4、13.16、13.18 | `E` | `LM89`；Huybrechts；Becker-Becker-Schwarz；附录 A/E | holonomy principle、quintic deformation theory、compact Hodge theorem |
| 17.7--17.12 | `E/C` | `CARDY86`、`SV96`、`FKS95`、`WALD93/94` | Cardy theorem、black-hole solution、attractor flow、Noether-charge theorem 与 quantum match |
| 18.1、18.6 | `C` | `MAL97`、`GKP98/WIT98` | 完整非微扰 bulk/boundary 等价 |
| 18.4、18.11、18.12 | `E/S` | Polchinski Vol. 2；`HREN00`、`MAL-W98` | D3 solution、holographic counterterms、Wilson-loop dictionary |
