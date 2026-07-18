#!/usr/bin/env python3
"""Generate and embed the book's selected scientific explainer diagrams."""

from __future__ import annotations

import html
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "images" / "explainers"
MONTHS = sorted(ROOT.glob("[01][0-9]_*.md"))
EXPECTED_DIAGRAMS_PER_MONTH = 15


@dataclass(frozen=True)
class Diagram:
    day: int
    kind: str
    caption: str
    nodes: tuple[tuple[str, str], ...]


def d(day: int, kind: str, caption: str, *nodes: tuple[str, str]) -> Diagram:
    return Diagram(day, kind, caption, nodes)


# Fifteen diagrams per month. Topics are selected when a picture can explain a
# mechanism, internal structure, force, cycle, or spatial relationship better
# than the daily subject illustration alone.
DIAGRAMS = [
    # January: sky, light, time, and heat.
    d(1, "flow", "氢原子核在高温高压下结合，释放的能量逐步传到太阳表面。", ("atom", "氢原子核"), ("fusion", "高温高压下聚变"), ("sun", "能量向外传递")),
    d(2, "flow", "阳光进入大气后，短波长的蓝光更容易被空气分子散向四周。", ("sun", "多种颜色的阳光"), ("air", "空气分子散射"), ("eye", "四周蓝光进入眼睛")),
    d(5, "flow", "物体挡住直线传播的光，在背后形成暗影和较模糊的半影。", ("lamp", "光从光源出发"), ("block", "身体挡住一部分光"), ("shadow", "暗影与半影")),
    d(7, "flow", "脸上反射的光按相等角度离开镜面，眼睛把光路倒推到镜后。", ("face", "脸反射光"), ("mirror", "镜面改变光的方向"), ("eye", "眼睛看见虚像")),
    d(9, "flow", "阳光在雨滴中折射、反射并分色，特定方向的彩光进入眼睛。", ("sun", "白色阳光"), ("drop-rainbow", "雨滴折射与色散"), ("eye", "彩色光到达眼睛")),
    d(10, "flow", "太阳总照亮约半个地球；地球自转让同一地点轮流经过明暗两面。", ("sun", "阳光照向地球"), ("half-earth", "一半明、一半暗"), ("rotation", "自转带来昼夜")),
    d(11, "flow", "阳光照到月面后向许多方向反射，其中一部分进入眼睛，让我们看见月亮。", ("sun", "阳光照向月面"), ("moon", "月面反射阳光"), ("eye", "反射光进入眼睛")),
    d(12, "orbit", "月亮绕地球移动时，我们看见的受光部分不同，于是出现月相。", ("earth", "地球"), ("moon", "新月附近"), ("half-moon", "弦月附近"), ("full-moon", "满月附近")),
    d(13, "flow", "星光穿过密度不断变化的空气，传播方向和亮度会快速微调。", ("star", "遥远星光"), ("wavy-air", "流动大气不断折射"), ("eye", "亮度看起来闪动")),
    d(17, "orbit", "地轴保持倾斜；一个半球朝向太阳时，阳光更直、白天更长。", ("sun", "太阳"), ("tilted-earth", "北半球朝向太阳"), ("tilted-earth", "南半球朝向太阳"), ("orbit", "一年绕行一周")),
    d(20, "forces", "冰排开水获得向上的浮力；浮力与重量平衡时，冰便浮着。", ("ice", "浮在水面的冰"), ("force-up", "水的浮力向上"), ("force-down", "冰的重量向下")),
    d(21, "flow", "水分子结冰时形成六角结构，雪晶因而常沿六个对称方向生长。", ("water-particles", "水汽分子靠近晶体"), ("hex-ice-lattice", "冰里形成六角结构"), ("snow", "沿六个方向生长")),
    d(23, "flow", "金属把皮肤的热量传走得较快，因此同温度下摸起来更凉。", ("hand", "温暖的手"), ("metal", "金属快速导热"), ("heat-out", "皮肤热量较快离开")),
    d(26, "flow", "钟用稳定的周期运动作节拍，再把许多周期累计成时间读数。", ("oscillator", "规律重复的振动"), ("counter", "逐次计数"), ("clock", "换算成秒、分、时")),
    d(31, "flow", "太阳粒子受地球磁场引导到极区，撞击高层大气并使气体发出彩光。", ("sun", "太阳释放带电粒子"), ("earth-field", "地球磁场引向极区"), ("earth-air", "高层气体受撞击发光")),

    # February: air, weather, and water.
    d(32, "flow", "气体分子持续运动并撞击容器内壁，大量撞击共同形成压力。", ("air", "不停运动的分子"), ("collision", "撞击容器内壁"), ("pressure", "许多撞击形成压力")),
    d(33, "cycle", "地面受热不均使空气密度和压力不同，空气的整体运动形成风。", ("sun-ground", "地面受热不均"), ("warm-air", "暖空气上升"), ("cool-air", "较冷空气补来"), ("wind", "形成循环与风")),
    d(35, "layers", "肥皂分子帮助水膜稳定；薄膜前后表面的反射光会发生干涉。", ("bubble", "肥皂泡薄膜"), ("soap", "两层表面活性分子"), ("light", "两面反射光相互叠加")),
    d(36, "network", "气球内部压力向外推，橡胶的回缩力和外部空气共同抵抗。", ("balloon", "被拉伸的气球"), ("air", "内部气体向外推"), ("spring", "橡胶回缩向内"), ("air", "外界空气也向内压")),
    d(37, "forces", "嘴里降低气压后，大气压推动杯中水面，水沿吸管上升。", ("straw", "吸管里的水柱"), ("force-up", "大气压推动水面"), ("force-down", "水柱重量向下")),
    d(38, "flow", "潮湿空气上升、膨胀并冷却，水汽依附凝结核形成小云滴。", ("warm-air", "潮湿空气上升"), ("cool-air", "膨胀并冷却"), ("cloud", "水汽凝成云滴")),
    d(39, "flow", "小云滴或冰晶不断碰并长大，重到上升气流托不住时便落下。", ("tiny-drops", "许多微小云滴"), ("merge-drops", "碰并或结冰长大"), ("rain", "重力使降水落下")),
    d(40, "flow", "表面张力把小雨滴拉向球形，空气阻力会把下落雨滴底部压平。", ("drop", "小滴趋向球形"), ("wind-wave", "下落时受到空气阻力"), ("beading-drop", "较大雨滴底部变平")),
    d(41, "flow", "雪花穿过整层大气；沿途温度决定它保持冰晶还是融成雨。", ("snow", "云中形成冰晶"), ("air-layers", "穿过不同冷暖层"), ("rain-snow", "地面收到雨或雪")),
    d(42, "flow", "小冰粒在强上升气流附近移动，碰到过冷水滴后一层层冻结成冰雹。", ("tiny-drops", "上升气流托住小冰粒"), ("merge-drops", "碰到过冷水滴结冰"), ("hailstone", "长大变重后落下")),
    d(44, "flow", "云内电荷分离产生强电场，空气被电离后形成快速放电通道。", ("charge-cloud", "正负电荷分离"), ("ionized-air", "强电场击穿空气"), ("lightning", "电荷沿通道放电")),
    d(47, "flow", "液面上能量较高的水分子逃进空气，剩余水的平均能量下降。", ("water", "液态水表面"), ("fast-molecule", "快分子逃离"), ("cool-water", "留下的水变凉")),
    d(50, "flow", "冰吸收能量后，较整齐的晶体排列被破坏，水分子便能彼此换位流动。", ("ice-lattice", "冰中较整齐的排列"), ("heat", "吸收熔化所需能量"), ("liquid-water", "分子换位成为液体")),
    d(52, "cycle", "水可在海洋、空气、陆地和生物之间循环，但每滴水的路径不同。", ("ocean", "海洋与湖泊"), ("cloud", "蒸发、蒸腾与云"), ("rain", "雨雪回到陆地"), ("river", "河流和地下水回海")),
    d(56, "flow", "表面活性剂一端靠近油、一端靠近水，许多分子包住油污后随水离开。", ("oil-molecules", "油污黏在表面"), ("micelle", "肥皂分子包围油污"), ("clean-water", "小油滴分散并被冲走")),

    # March: plants and gardens.
    d(60, "layers", "种皮保护内部；子叶或胚乳储存养料，胚会长成新的根和芽。", ("seed", "一粒种子"), ("seed-food", "储存养料的部分"), ("embryo", "会继续生长的胚")),
    d(61, "flow", "种子得到适量水、氧气和合适温度后，代谢启动并开始发芽。", ("water-air", "水、氧气与温度"), ("seed", "种子恢复代谢"), ("sprout", "胚根和幼芽生长")),
    d(62, "flow", "胚根先伸出，帮助固定幼苗并吸收水；随后幼芽向光处生长。", ("seed", "种皮吸水裂开"), ("root", "胚根先向下伸"), ("sprout", "幼芽随后向上长")),
    d(63, "network", "根毛扩大吸收面积，水和矿物质进入根后由木质部向上运输。", ("root", "根与根毛"), ("water", "吸收水"), ("salt", "吸收矿物质"), ("stem", "沿木质部向上")),
    d(65, "flow", "叶片薄而平，能接收较多光并缩短二氧化碳进出细胞的距离。", ("broad-leaf", "宽阔的叶面"), ("light-air", "接光并交换气体"), ("vein", "叶脉支撑和运输")),
    d(66, "flow", "叶绿素吸收较多红光和蓝光，更多绿光被反射进眼睛，叶子便显绿色。", ("light", "多种颜色的阳光"), ("leaf", "叶绿素选择性吸收"), ("eye", "较多绿光进入眼睛")),
    d(67, "flow", "叶绿体利用光能，把水和二氧化碳转成糖，并释放氧气。", ("light-water-air", "光、水和二氧化碳"), ("leaf", "叶绿体进行光合作用"), ("sugar-oxygen", "糖和氧气")),
    d(68, "cycle", "光合作用储存能量，细胞呼吸再用氧气分解有机物释放可用能量。", ("leaf", "光合作用制造有机物"), ("sugar", "有机物储存能量"), ("cell", "细胞呼吸释放能量"), ("air", "气体在环境中交换")),
    d(71, "flow", "蜜蜂身上沾到花粉，再把花粉带到另一朵同种花的柱头。", ("flower", "花药产生花粉"), ("bee", "蜜蜂携带花粉"), ("fruit", "授粉后可能形成种子")),
    d(73, "flow", "蒲公英冠毛增大空气阻力，让种子下落较慢并被风带远。", ("seed-parachute", "带冠毛的种子"), ("wind", "空气阻力托住冠毛"), ("far-seed", "慢慢落到远处")),
    d(74, "flow", "果实上的钩刺先挂住毛或布料，同行一段后又在摩擦和碰撞中掉落。", ("hook-seed", "带钩刺的果实"), ("body", "挂在毛或衣物上同行"), ("seed", "途中掉下、到达新地方")),
    d(76, "layers", "形成层每年增加新组织；生长快慢不同会留下可辨认的年轮。", ("tree", "树干横切面"), ("growth-ring", "每年增加的新组织"), ("xylem", "不同年份留下圆环")),
    d(81, "layers", "荷叶表面的微小凸起和蜡质层减少水的贴附，水便聚成圆珠滚走。", ("lotus", "荷叶表面"), ("micro-bumps", "微小凸起与蜡层"), ("beading-drop", "水珠接触少、容易滚")),
    d(83, "flow", "单侧光照使茎两边伸长程度不同，生长差异让茎弯向光源。", ("window-light", "光从一侧照来"), ("stem-growth", "背光侧伸长较多"), ("bent-plant", "茎逐渐弯向光")),
    d(84, "cycle", "孢子在合适基质中长成菌丝网络；条件合适时长出蘑菇，再释放新孢子。", ("spores", "孢子落到合适基质"), ("mycelium", "菌丝生长并吸收养分"), ("mushroom", "条件合适长出蘑菇"), ("spores", "蘑菇释放新孢子")),

    # April: small animals and habitats.
    d(91, "layers", "昆虫有头、胸、腹三部分，三对足都连接在胸部。", ("insect", "一只昆虫"), ("insect-body", "头、胸、腹"), ("six-legs", "胸部连接六条腿")),
    d(92, "compare", "昆虫通常三段身体、六条腿；蜘蛛两大体区、八条腿。", ("insect", "昆虫：六条腿"), ("spider", "蜘蛛：八条腿")),
    d(93, "cycle", "找到食物的蚂蚁留下信息素，更多蚂蚁沿路走又会加强气味。", ("ant-food", "蚂蚁找到食物"), ("pheromone", "回巢留下气味"), ("ant-line", "同伴沿气味前进"), ("strong-trail", "路线被反复加强")),
    d(95, "flow", "蜜蜂翅膀和飞行肌肉振动，使周围空气形成传播的压力波。", ("bee", "翅与肌肉快速振动"), ("sound-wave", "空气压力波传播"), ("ear", "耳朵听见嗡声")),
    d(97, "cycle", "完全变态把生命史分成卵、幼虫、蛹和成虫四个阶段。", ("egg", "卵"), ("caterpillar", "毛毛虫"), ("chrysalis", "蛹内重建身体"), ("butterfly", "成虫蝴蝶")),
    d(100, "flow", "发光细胞里的物质在酶和氧气参与下反应，把化学能变成光。", ("molecule", "发光物质与氧气"), ("enzyme", "酶帮助反应"), ("firefly-light", "化学能变成冷光")),
    d(102, "network", "四片翅膀分别调节角度和拍动，持续改变升力、推力和转向力矩。", ("dragonfly", "独立调节的四片翅"), ("wing", "升力托住身体"), ("motion-arrow", "推力改变速度"), ("vortex", "不对称拍翅帮助转向")),
    d(103, "flow", "后腿肌肉先让弹性结构储能，突然释放时把身体推离地面。", ("grasshopper", "后腿弯曲蓄力"), ("spring", "外骨骼储存弹性能"), ("jump", "快速伸腿起跳")),
    d(106, "cycle", "较像树枝的个体更容易避开捕食；许多代后相关特征可能更常见。", ("varied-insects", "个体外形有差异"), ("bird", "捕食造成筛选"), ("camouflage", "伪装者较易留下后代"), ("generations", "特征逐代变常见")),
    d(108, "flow", "足底肌肉波推动身体，黏液受力时改变流动性质，既保护身体又帮助传力。", ("snail", "足底产生肌肉波"), ("wave", "收缩波沿足部移动"), ("motion-arrow", "黏液帮助身体向前")),
    d(109, "cycle", "环形肌和纵向肌轮流收缩，再用刚毛固定一段身体，推动身体前进。", ("long-worm", "环形肌收缩：变细变长"), ("anchor", "刚毛固定身体一段"), ("short-worm", "纵向肌收缩：变粗变短"), ("forward", "收缩波向后、身体向前")),
    d(113, "network", "蛛网的辐射丝和螺旋丝分担拉力；不同蛛丝兼顾强度与延展。", ("web", "整张蛛网"), ("radial", "辐射丝传递拉力"), ("spiral", "螺旋丝吸收运动"), ("anchor", "锚点把力传到周围")),
    d(118, "layers", "脚趾上的刚毛继续分叉成极细末端，增大与墙面的近距离接触。", ("gecko-foot", "壁虎脚趾"), ("setae", "大量细小刚毛"), ("molecular-contact", "微弱分子吸引相加")),
    d(119, "flow", "尾巴从预设薄弱处断开后，局部神经和肌肉仍能短时活动，吸引捕食者。", ("lizard-tail", "尾部在特定位置断开"), ("muscle", "局部神经肌肉继续活动"), ("motion-arrow", "蜥蜴趁机逃离")),
    d(120, "flow", "皮肤中色素与纳米晶体共同改变反射光；晶体间距不同会加强不同波长。", ("chameleon-skin", "变色龙皮肤受到信号"), ("crystal-spacing", "纳米晶体间距改变"), ("light", "反射光颜色随之改变")),

    # May: animal structures and adaptations.
    d(121, "layers", "羽轴两侧长出羽枝，羽枝上的小钩互相扣住，组成轻而连续的羽片。", ("feather", "一根羽毛"), ("barbs", "羽轴与许多羽枝"), ("hooklets", "小钩把羽枝扣在一起")),
    d(122, "forces", "鸟翼改变周围空气的速度和方向，空气对翅膀产生向上的合力。", ("wing", "有弧度和角度的翅膀"), ("force-up", "空气给翅膀向上合力"), ("force-down", "翅膀把空气推向下方")),
    d(123, "compare", "普通鸟翼适合推动空气；企鹅较扁硬的鳍状翼适合推动密度更大的水。", ("bird-wing", "飞行翼推动空气"), ("penguin-flipper", "鳍状翼推动海水")),
    d(124, "layers", "鸭羽互相扣合并带有油脂，水难以深入贴近皮肤，常在表面形成水珠。", ("duck", "鸭子的外层羽毛"), ("oily-feather", "整齐羽片与油脂"), ("beading-drop", "水珠留在外面")),
    d(125, "flow", "啄击产生的力沿喙、头骨、颈部和身体传递；适应结构和动作共同减小风险。", ("woodpecker", "喙快速碰到木头"), ("skull", "力沿头颈结构传递"), ("body", "姿势和肌肉共同稳定")),
    d(126, "flow", "猫头鹰飞羽边缘的细小结构减弱较大的气流涡旋，使飞行声更分散。", ("owl-wing", "带细缘的飞羽"), ("small-vortices", "气流被分成较小扰动"), ("quiet-wave", "较少强烈飞行噪声")),
    d(130, "flow", "海豚发出短促声波，接收物体反射的回声，再由大脑判断方向和距离。", ("dolphin", "发出点击声"), ("echo", "声波碰到物体返回"), ("brain", "比较回声时间和方向")),
    d(132, "network", "象鼻没有骨头，纵向、环向和斜向肌肉互相配合，便能伸缩、弯曲和扭转。", ("trunk", "没有骨头的象鼻"), ("muscle", "多方向肌肉束"), ("motion-arrow", "伸缩、弯曲与扭转")),
    d(133, "network", "长颈鹿的心脏、弹性血管和调节结构共同维持抬头、低头时的血流。", ("giraffe", "长长的颈部"), ("heart", "有力泵血"), ("vessel", "血管调节压力"), ("valve", "静脉瓣帮助回流")),
    d(135, "layers", "驼峰主要储存脂肪；需要时脂肪可供能，代谢过程中也会产生少量水。", ("camel", "骆驼的驼峰"), ("fat", "储存脂肪而非水袋"), ("energy", "分解后提供能量")),
    d(141, "cycle", "温暖血液流经兔耳，把热传给空气；较凉的血再流回身体。", ("heart", "温暖血液流出"), ("rabbit-ear", "大耳朵增加散热面积"), ("heat-out", "热量传到周围空气"), ("vessel", "较凉血液流回")),
    d(142, "flow", "胡须弯曲时像杠杆带动毛囊，毛囊感受器把细小变化变成神经信号。", ("hair", "胡须受到轻微触碰"), ("follicle", "毛囊感受弯曲"), ("brain", "神经信号送往大脑")),
    d(145, "flow", "马腿的肌腱和韧带能帮助锁住关节，让站立休息时不必持续强烈收缩肌肉。", ("horse-leg", "腿部关节排列"), ("spring", "肌腱和韧带保持张力"), ("balance", "少用肌肉也能站稳")),
    d(146, "flow", "水流过鳃丝和鳃小片，水中的氧跨过薄膜进入血液。", ("fish", "水从口部进入"), ("gill", "流过许多薄鳃片"), ("blood-oxygen", "氧气扩散进血液")),
    d(147, "forces", "鱼改变鳔内气体体积来调节平均密度，使浮力接近自身重量。", ("fish-bladder", "鱼体内的鳔"), ("force-up", "排水产生浮力"), ("force-down", "鱼的重量向下")),

    # June: body and senses.
    d(152, "layers", "表皮负责外层屏障，真皮含血管、神经和腺体，更深组织连接并缓冲。", ("skin", "皮肤横切面"), ("epidermis", "表皮：外层屏障"), ("dermis", "真皮与更深组织")),
    d(153, "flow", "指纹脊线改变接触和摩擦，也会把滑动变成细小振动，帮助触觉感受器读出表面。", ("finger-ridges", "指腹脊线接触物体"), ("sound-wave", "滑动产生细小振动"), ("brain", "触觉信号送往大脑")),
    d(154, "layers", "毛囊底部细胞分裂并角化，被逐渐推到皮肤外形成发丝。", ("hair", "露出皮肤的发丝"), ("follicle", "皮肤里的毛囊"), ("dividing-cells", "底部细胞不断分裂")),
    d(155, "layers", "骨外层较致密，内部海绵骨沿受力方向排列，空隙中可有骨髓。", ("bone", "一段骨头"), ("compact-bone", "致密的外层"), ("spongy-marrow", "海绵骨与骨髓")),
    d(156, "compare", "肩关节的球窝结构换来大活动范围；膝关节更像受引导的铰链。", ("shoulder", "肩：灵活的球窝关节"), ("knee", "膝：较稳定的铰链结构")),
    d(157, "compare", "肌肉只能主动收缩拉动骨；弯肘和伸肘由不同肌群轮流主导。", ("muscle", "一组肌肉收缩弯肘"), ("muscle", "另一组肌肉收缩伸肘")),
    d(158, "flow", "起搏细胞产生电信号，信号按顺序传过心脏，让各腔室协调收缩。", ("pacemaker", "起搏细胞发出信号"), ("heart-signal", "电信号依次传开"), ("heart", "心房与心室协调泵血")),
    d(159, "cycle", "右心把血送到肺交换气体，左心再把富氧血送往全身。", ("heart", "心脏"), ("lung", "肺部交换氧和二氧化碳"), ("body", "全身细胞使用氧气"), ("vessel", "血液沿血管返回")),
    d(160, "layers", "气道不断分叉到肺泡；肺泡壁很薄，并紧贴毛细血管，便于气体扩散。", ("lung", "不断分叉的气道"), ("alveoli", "许多微小肺泡"), ("gas-exchange", "氧和二氧化碳跨薄壁交换")),
    d(161, "compare", "吸气时膈肌下降、胸腔变大、压力降低；呼气时过程大致相反。", ("inhale", "吸气：膈肌下降"), ("exhale", "呼气：膈肌上升")),
    d(164, "flow", "危险热信号先经感觉神经到脊髓，再由运动神经命令肌肉迅速缩手。", ("hot-hand", "皮肤感受危险热度"), ("spinal", "脊髓快速连接信号"), ("muscle", "手臂肌肉收缩避开")),
    d(165, "flow", "物体反射的光经角膜和晶状体聚焦到视网膜，再变成神经信号。", ("object-light", "物体反射光"), ("eye-lens", "角膜和晶状体聚焦"), ("brain", "视网膜与大脑解释信号")),
    d(168, "flow", "声波振动鼓膜和听小骨，再在耳蜗中推动液体、弯曲感受细胞。", ("sound-wave", "空气中的声波"), ("ear", "鼓膜与听小骨振动"), ("cochlea", "耳蜗把振动变成神经信号")),
    d(169, "network", "大脑比较视觉、内耳和肌肉关节信号，再不断调整姿势来保持平衡。", ("brain", "大脑比较多路信号"), ("balance-senses", "视觉、内耳和关节线索"), ("muscle", "肌肉不断修正姿势")),
    d(175, "flow", "食物先被弄碎，再由胃肠和消化液继续分解；小肠把许多养分吸收入血。", ("teeth", "口腔咀嚼并混合唾液"), ("stomach", "胃和消化液继续处理"), ("intestine", "小肠吸收许多养分")),

    # July: tools and machines.
    d(183, "flow", "杠杆绕支点转动；离支点更远处用力，可以用较小的力产生足够力矩。", ("lever", "远处施加较小力"), ("seesaw", "支点传递转动作用"), ("weight", "近处承受较大负载")),
    d(184, "compare", "跷跷板是否平衡取决于两边的重量和它们到支点的距离。", ("seesaw", "一边：重量 × 距离"), ("seesaw", "另一边：重量 × 距离")),
    d(185, "flow", "定滑轮主要改变拉力方向；多股绳共同承重时，每股绳分担一部分负载。", ("pulley", "绳绕过滑轮"), ("rope-segments", "多股绳分担负载"), ("lift", "少些力、拉更长距离")),
    d(186, "compare", "物体滑动时接触面持续摩擦；装上轮子后主要变成滚动，通常阻力较小。", ("sliding-box", "箱子在地面滑动"), ("wheel-cart", "轮子在接触处滚动")),
    d(187, "flow", "相邻齿轮用齿传力并反向转动；齿数比例会改变转速和力矩。", ("small-gear", "小齿轮转得较快"), ("gear-contact", "齿与齿连续接触"), ("large-gear", "大齿轮较慢、力矩较大")),
    d(189, "flow", "螺纹像绕在圆柱上的斜面；转动较长距离会产生沿轴方向的移动和力。", ("incline", "一条斜面"), ("screw", "斜面绕成螺纹"), ("forward", "旋转换成前进")),
    d(190, "flow", "同样的力集中在较小刃口面积上会产生较大压力，使材料更容易分开。", ("hand", "手施加同样的力"), ("wedge", "窄刃口产生较大压力"), ("motion-arrow", "材料被推向两侧")),
    d(191, "compare", "斜坡把同样高度的提升分散到更长距离，所以通常需要较小的沿坡力。", ("steep-lift", "直抬：距离短、力较大"), ("ramp", "斜坡：距离长、力较小")),
    d(192, "flow", "弹簧变形时储存弹性势能；约束松开后，恢复力把能量转成运动。", ("pressed-spring", "压缩或拉伸"), ("spring", "储存弹性势能"), ("motion", "恢复形状并推动物体")),
    d(194, "flow", "磁铁周围存在磁场；铁磁材料内部的小磁区受影响后产生明显吸引。", ("magnet", "磁铁产生磁场"), ("field", "磁场穿过空间"), ("iron", "铁磁材料受到吸引")),
    d(196, "flow", "导线中的电流产生磁场；绕成线圈并加入软铁芯后，许多磁场共同增强。", ("coil", "线圈中有电流"), ("iron", "软铁芯加强磁场"), ("field", "形成可开关的电磁铁")),
    d(197, "flow", "电池、导线和灯形成闭合路径后，电荷可以持续移动并传递能量。", ("battery", "电池提供电势差"), ("closed-circuit", "导线形成闭合路径"), ("lamp", "灯把电能变成光和热")),
    d(199, "flow", "线圈中的电流产生磁场，与另一磁场相互作用形成力矩，使转子转动。", ("coil", "线圈通电产生磁场"), ("magnet", "两个磁场相互作用"), ("motor", "力矩推动转子")),
    d(200, "compare", "电动机把电能转成机械运动；发电机用机械运动改变磁通并产生电压。", ("motor", "电动机：电 → 运动"), ("generator", "发电机：运动 → 电")),
    d(203, "flow", "把手移动阀门里的密封件，改变水流通道大小；通道关紧后水便停止。", ("faucet-handle", "把手带动内部零件"), ("valve", "阀门改变通道大小"), ("water", "水流变大、变小或停止")),

    # August: Earth and geography.
    d(213, "flow", "大型天体的引力把物质拉向共同中心；足够大时，整体会接近球形。", ("matter", "许多物质聚在一起"), ("gravity-center", "引力朝共同中心"), ("earth", "形成接近球形的地球")),
    d(215, "forces", "无论站在地球哪一边，重力都大致指向地心，所以脚下都是当地的“下”。", ("earth-person", "站在球形地球各处"), ("force-up", "地面支持身体"), ("force-down", "重力指向地心")),
    d(216, "layers", "地球由薄地壳、厚地幔和金属核心等层次组成，越深条件越极端。", ("earth-cutaway", "地球内部剖面"), ("crust-mantle", "薄地壳与厚地幔"), ("core", "外核与内核")),
    d(217, "flow", "岩石圈板块缓慢移动；在边界处碰撞、拉开或错动，塑造地表。", ("plates", "许多岩石圈板块"), ("plate-motion", "碰撞、拉开或错动"), ("mountain-quake", "山脉、火山和地震")),
    d(218, "flow", "断层突然滑动释放弹性能，地震波向外传播并使地面摇动。", ("fault", "岩石受力并卡住"), ("slip", "断层突然滑动"), ("seismic-wave", "地震波向外传播")),
    d(219, "flow", "岩浆中的气体膨胀并增加压力；岩浆沿裂缝上升，可能到达地表喷发。", ("magma", "含气体的岩浆"), ("volcano-conduit", "沿裂缝和通道上升"), ("eruption", "熔岩、碎屑和气体喷出")),
    d(220, "flow", "板块碰撞或火山堆积能抬高地表，风、水、冰和重力又持续侵蚀山体。", ("colliding-plates", "板块挤压或火山堆积"), ("rising-mountain", "地表逐渐抬高成山"), ("eroding-mountain", "侵蚀同时削低山体")),
    d(221, "flow", "岩石先被风化，流水或冰再搬走碎屑；长期侵蚀会加深和拓宽山谷。", ("weathering", "岩石破碎与改变"), ("river", "流水搬运碎屑"), ("valley", "山谷逐渐被刻出")),
    d(222, "cycle", "岩石会在冷却、风化沉积、受热受压和熔融中转变，路径不只有一种。", ("igneous-rock", "冷却形成火成岩"), ("sediment-layers", "风化沉积形成沉积岩"), ("metamorphic-rock", "受热受压形成变质岩"), ("magma", "熔融后再次成为岩浆")),
    d(226, "flow", "制图者选择比例尺、方向和符号，把真实空间关系缩小到平面。", ("landscape", "真实的大地方"), ("scale-symbol", "选择比例尺与符号"), ("map", "缩小后的地图")),
    d(227, "compare", "球面无法无变形地铺成平面；不同投影会分别改变面积、形状或距离。", ("globe", "球形地球"), ("flat-map", "平面地图必有变形")),
    d(229, "orbit", "多颗卫星广播精确时间，接收器比较信号到达时间来估计距离和位置。", ("receiver", "地面接收器"), ("satellite", "卫星一"), ("satellite", "卫星二"), ("satellite", "还需要更多卫星")),
    d(232, "cycle", "冰雪减少会露出深色海水，海水吸收更多阳光并升温，又会促进更多融化。", ("ice-sheet", "冰雪反射较多阳光"), ("water", "融化后露出深色海水"), ("sun", "海水吸收更多阳光"), ("heat", "升温促进继续融化")),
    d(237, "flow", "河水进入较平静水域后流速降低，搬运能力下降，泥沙逐渐沉积成三角洲。", ("fast-river", "河流携带泥沙"), ("slow-water", "入海后流速减慢"), ("delta", "泥沙分流并沉积")),
    d(241, "flow", "多年积雪被压成冰；冰在重力作用下变形和滑动，缓慢流向低处。", ("snow", "多年积雪堆积"), ("glacier", "压实成为厚冰"), ("ice-flow", "重力使冰缓慢流动")),

    # September: oceans and coasts.
    d(244, "cycle", "水和岩石作用带走溶解物，经河流进入海洋；水蒸发后盐分大多留下。", ("rain-rock", "雨水溶解少量矿物"), ("river", "河流把离子带入海洋"), ("ocean", "盐分在海水中积累"), ("vapor", "蒸发带走水、不带走盐")),
    d(245, "flow", "风把能量交给海面；水粒子多在原地附近绕动，而波形和能量向前传播。", ("wind-wave", "风给海面能量"), ("particle-circle", "水粒子附近绕动"), ("shore-wave", "波能传到岸边")),
    d(246, "orbit", "月球引力是潮汐主因，太阳也会参与；地球自转让海岸经过潮汐隆起。", ("earth", "地球与海洋"), ("moon", "月球引力"), ("sun", "太阳引力也参与"), ("tide", "海岸经历涨落")),
    d(247, "cycle", "风推动表层海水，温度和盐度造成密度差，又驱动深层海水缓慢循环。", ("warm-current", "风驱动暖表层流"), ("cooling", "冷却或增盐后变密"), ("deep-current", "密水下沉形成深层流"), ("upwelling", "别处海水上升补充")),
    d(249, "network", "越深处上方水柱越高、重量越大，因此各方向的水压都更高。", ("deep-object", "深处的水下物体"), ("water", "四周水都施加压力"), ("ocean-light", "深度增加"), ("deep-object", "外壳各面承受更大压力")),
    d(250, "layers", "海水会逐步吸收和散射阳光；不同颜色穿透不同，足够深处几乎全黑。", ("ocean-light", "阳光进入海面"), ("color-depth", "红光先减弱、蓝光较深"), ("dark-depth", "深处剩下很少光")),
    d(251, "network", "珊瑚虫给共生藻提供栖身处和原料，共生藻用光合作用提供部分养分。", ("coral", "珊瑚虫群体"), ("algae", "细胞内的共生藻"), ("sun", "藻利用阳光"), ("nutrient", "双方交换物质")),
    d(252, "flow", "持续热压力会破坏珊瑚与共生藻的关系；藻离开后，白色骨骼透过组织显现。", ("healthy-coral", "含共生藻的健康珊瑚"), ("heat", "海水过热造成压力"), ("bleached-coral", "失去藻后显出白色骨骼")),
    d(255, "flow", "水母伞部收缩把水推向后方，水的反作用力使身体向前移动。", ("jellyfish-open", "伞部张开进水"), ("jellyfish-close", "肌肉收缩挤水"), ("forward", "水向后、身体向前")),
    d(258, "compare", "多数螃蟹的腿关节更适合向身体两侧弯曲，因此横走时步幅通常更大。", ("crab-front", "向前走：关节活动受限"), ("crab-side", "横着走：腿更易摆动")),
    d(260, "layers", "色素细胞伸展或收缩改变色块，反射细胞还会选择性加强某些波长。", ("octopus-skin", "章鱼皮肤"), ("chromatophore", "色素囊伸缩"), ("reflector", "反射细胞改变亮色")),
    d(261, "flow", "鱿鱼把水吸进外套膜腔，再从漏斗高速喷出，反作用力推动身体。", ("squid", "外套膜腔吸入水"), ("water", "水从漏斗向后喷"), ("motion-arrow", "身体受到反向推力")),
    d(262, "flow", "鲸鲨让含浮游生物的水流入口中，过滤结构留下食物，水从鳃处流出。", ("plankton-water", "含小生物的海水"), ("filter-mouth", "过滤结构截留食物"), ("clean-water", "水从鳃裂流出")),
    d(263, "flow", "雌海马把卵放入雄海马育儿袋，胚胎在袋内发育，最后由雄海马释放。", ("egg", "雌海马把卵放入"), ("seahorse-pouch", "雄海马育儿袋调节环境"), ("baby-seahorse", "小海马发育后离开")),
    d(265, "network", "海龟会综合地磁、阳光、海浪和气味等多种线索导航，不靠单一地图。", ("turtle", "迁徙中的海龟"), ("magnetic-field", "地球磁场"), ("sun-wave", "太阳与海浪方向"), ("smell", "近岸气味等线索")),

    # October: space.
    d(275, "layers", "恒星中心通过核聚变供能，能量穿过内部后从表面以辐射形式离开。", ("star", "一颗恒星"), ("fusion", "中心核聚变区"), ("radiation", "能量逐层向外传递")),
    d(277, "orbit", "太阳的引力把行星的惯性运动弯成轨道，许多天体共同组成太阳系。", ("sun", "太阳"), ("planet", "内侧行星轨道"), ("planet", "外侧行星轨道"), ("comet", "小天体也绕太阳")),
    d(280, "compare", "水星白天受强烈日照，夜面又因大气极稀薄而快速散热，温差很大。", ("mercury-day", "向阳面很热"), ("mercury-night", "背阳面很冷")),
    d(281, "layers", "金星厚大气让阳光进入，却强烈吸收地表发出的红外辐射，造成高温。", ("venus", "金星地表"), ("thick-air", "厚厚的二氧化碳大气"), ("trapped-heat", "红外能量较难逃出")),
    d(282, "network", "地表液态水能否长期存在，取决于恒星、轨道、行星质量、大气和气候反馈。", ("sun", "恒星提供能量"), ("orbit", "轨道决定受光变化"), ("earth-air", "大气调节压力与温度"), ("water", "条件合适时存在液态水")),
    d(285, "layers", "土星环由无数冰和岩石颗粒组成，各自沿轨道运行，并非一整张固体圆盘。", ("saturn", "土星与光环"), ("ring-pieces", "许多大小不同的颗粒"), ("small-orbits", "颗粒各自绕土星运行")),
    d(290, "flow", "彗星接近太阳时冰升华，气体带出尘埃；太阳风和辐射把尾巴推向背日方向。", ("icy-comet", "含冰和尘埃的彗核"), ("sun-heat", "接近太阳后升华"), ("comet-tail", "气体和尘埃形成尾巴")),
    d(291, "flow", "小天体高速进入大气，压缩并加热前方空气而发光；少数残块能落到地面。", ("meteoroid", "太空中的流星体"), ("meteor", "进入大气形成亮迹"), ("meteorite", "落地残块叫陨石")),
    d(292, "flow", "高速撞击产生冲击波，挖出物质并向外抛射，留下近圆形撞击坑。", ("impactor", "高速天体撞来"), ("impact", "冲击和挖掘地面"), ("crater", "圆坑与周围抛射物")),
    d(294, "flow", "日食发生在太阳、月球、地球接近排成一线，月影落到地球局部区域时。", ("sun", "太阳"), ("moon", "月球挡住部分阳光"), ("earth-shadow", "月影落到地球")),
    d(295, "flow", "月食时地球挡住直射阳光，大气把剩余红橙光折进影子，照到月面。", ("sun", "太阳光"), ("earth-air", "地球遮挡并由大气折光"), ("red-moon", "红橙光照进月影")),
    d(296, "forces", "月球不断向前运动，同时被地球引力拉弯路径，于是持续绕地球下落。", ("orbiting-moon", "向前运动的月球"), ("force-up", "惯性让它继续向前"), ("force-down", "地球引力把路径拉弯")),
    d(297, "forces", "火箭把高速气体向后喷出；火箭受到大小相等、方向相反的推力。", ("rocket", "火箭与发动机"), ("force-up", "推力让火箭向前"), ("force-down", "高速气体向后喷")),
    d(299, "network", "空间站和宇航员一起绕地球自由落体；没有地板持续托住身体，便呈现失重。", ("earth", "地球引力仍然存在"), ("satellite", "空间站沿轨道下落"), ("body", "宇航员与空间站同落"), ("motion-arrow", "共同绕地球前进")),
    d(304, "network", "黑洞本身不发出可逃逸的光，但恒星轨道、热吸积气体和引力波能暴露它。", ("black-hole", "看不见的黑洞本身"), ("orbit", "附近恒星绕行"), ("hot-drop", "吸积气体发出辐射"), ("wave", "合并产生引力波")),

    # November: materials, food, and the home.
    d(305, "compare", "固体粒子多在固定位置附近振动；液体能换位流动；气体相距更远。", ("solid-particles", "固体：紧密而有序"), ("liquid-particles", "液体：靠近但能换位"), ("gas-particles", "气体：分散并自由运动")),
    d(307, "flow", "冰在熔点附近吸收能量，较整齐的晶体结构被破坏，分子仍是水分子。", ("ice-lattice", "冰的晶体排列"), ("heat", "吸收熔化潜热"), ("liquid-water", "变成流动的液态水")),
    d(308, "flow", "液面分子受力不对称，表面张力倾向缩小表面积，小水滴因此接近球形。", ("water-particles", "液面分子受力不对称"), ("molecule", "表面张力向内收拢"), ("drop", "水滴趋向较小表面积")),
    d(309, "compare", "水分子彼此吸引，油分子也更愿与油相聚；密度差又决定哪层在上。", ("water-molecules", "水分子聚在水层"), ("oil-molecules", "油分子聚在油层")),
    d(310, "flow", "盐晶体进入水后分成带电离子，水分子包围离子并把它们分散开。", ("salt-crystal", "盐的离子晶体"), ("hydration", "水分子包围离子"), ("salt-solution", "离子分散在水中")),
    d(312, "compare", "温水中的分子通常运动更快，碰撞和混合更频繁，许多固体便溶得更快。", ("cold-water", "冷水：分子运动较慢"), ("warm-water", "温水：运动和混合较快")),
    d(314, "flow", "酵母分解面团里的糖，产生二氧化碳；有弹性的面筋网络把气泡留住。", ("yeast-sugar", "酵母利用糖"), ("gas-bubble", "产生二氧化碳气泡"), ("bread", "面筋留住气体形成小洞")),
    d(316, "flow", "加热使蛋白质展开，展开的分子互相连接成网络，蛋液便由流动变为凝固。", ("folded-protein", "折叠的蛋白质"), ("heat", "热使结构展开"), ("protein-network", "重新连接成固体网络")),
    d(318, "flow", "玉米粒内水分受热成为高压水汽，撑破外壳后，热软淀粉快速膨胀。", ("corn", "坚硬外壳包住水和淀粉"), ("pressure", "加热形成高压水汽"), ("popcorn", "外壳破开、淀粉膨胀")),
    d(319, "cycle", "制冷剂在冰箱内蒸发吸热，在外部压缩并冷凝放热，再循环回来。", ("evaporator", "箱内蒸发并吸热"), ("compressor", "压缩机提高压力"), ("condenser", "箱外冷凝并放热"), ("valve", "降压后重新进入箱内")),
    d(321, "flow", "变化的电场让极性分子不断调整方向，吸收的电磁能转成无规则分子运动。", ("microwave", "微波电场变化"), ("polar-molecule", "极性分子反复转向"), ("heat", "随机运动增加、食物变热")),
    d(323, "layers", "双层杯壁、真空或隔热层和严密杯盖分别减弱传导、对流和蒸发。", ("thermos", "保温杯整体"), ("vacuum-wall", "双层杯壁与低气压夹层"), ("lid", "杯盖减少空气交换")),
    d(325, "flow", "水对纸纤维的附着和细孔中的表面张力共同作用，把水拉进纸内。", ("paper-fibers", "交错的纸纤维"), ("capillary", "细孔产生毛细作用"), ("wet-paper", "水沿纤维间隙扩散")),
    d(327, "flow", "玻璃表面小缺口会集中应力；裂纹一旦开始扩展，脆硬材料便迅速断开。", ("glass", "玻璃表面有微小缺口"), ("crack", "缺口尖端集中应力"), ("fragments", "裂纹扩展使玻璃破碎")),
    d(330, "flow", "橡胶长分子链被拉直排列后，热运动和交联会促使它们回到卷曲状态。", ("coiled-chain", "原本卷曲的长分子链"), ("stretched-chain", "拉伸后链条较整齐"), ("spring", "交联帮助恢复形状")),

    # December: transport, energy, and scientific practice.
    d(335, "flow", "身体重心越过支撑脚后，另一只脚向前接住；感官和肌肉持续修正。", ("step-one", "重心移向支撑脚前方"), ("step-two", "另一只脚向前迈出"), ("balance", "视觉、内耳和肌肉修正")),
    d(336, "network", "骑行者通过细小转向把轮胎接触点移到重心下方，持续恢复平衡。", ("bicycle", "前进中的自行车"), ("eye", "感官报告身体倾斜"), ("motion-arrow", "车把做细小转向"), ("train-wheel", "接触点回到重心下方")),
    d(337, "flow", "发动机或电动机产生转矩，传动系统把转矩送到驱动轮，轮胎再推动车辆。", ("energy", "燃料或电能进入"), ("motor", "发动机或电动机转动"), ("train-wheel", "驱动轮通过摩擦推地")),
    d(339, "compare", "急停时身体因惯性继续向前；安全带延长减速时间，并把力分散到较强部位。", ("no-belt", "没有约束：身体继续前冲"), ("seatbelt", "安全带：较缓地减速并分力")),
    d(341, "flow", "钢轮在钢轨上滚动阻力较小；轮缘和略带锥度的轮面帮助列车沿轨道。", ("train-wheel", "钢轮在钢轨上滚动"), ("motion-arrow", "滚动阻力较小"), ("train-wheel", "轮面和轮缘帮助导向")),
    d(343, "forces", "空心船体排开大量水；排水产生的浮力等于船重时，大船便稳定漂浮。", ("boat", "空心而宽大的船体"), ("force-up", "排开水产生浮力"), ("force-down", "船和货物的重量")),
    d(344, "network", "斜着的帆受到空气合力，水中的龙骨抵抗侧滑，剩下的分力推动船前进。", ("sailboat", "帆和水下龙骨"), ("wind", "风给帆斜向作用力"), ("water", "龙骨抵抗侧滑"), ("motion-arrow", "合力留下前进分量")),
    d(345, "forces", "机翼改变空气的速度和方向；压力分布与向下偏转的气流共同产生升力。", ("airplane-wing", "有迎角的机翼"), ("force-up", "空气对机翼的升力"), ("force-down", "机翼使气流向下偏转")),
    d(346, "forces", "旋翼像不断旋转的机翼，把大量空气加速向下，反作用力托起直升机。", ("rotor", "高速旋转的旋翼"), ("force-up", "空气给旋翼向上作用力"), ("force-down", "旋翼把空气推向下")),
    d(347, "forces", "球囊里的热空气密度较低；排开冷空气产生的浮力超过总重量时便上升。", ("hot-balloon", "装着热空气的球囊"), ("force-up", "周围空气产生浮力"), ("force-down", "球囊、篮子与空气的重量")),
    d(348, "cycle", "压载舱进水会增加潜艇平均密度使其下沉；排水充气则使它上浮。", ("ballast-air", "压载舱多为空气"), ("surface-sub", "平均密度小、浮在水面"), ("ballast-water", "压载舱进水"), ("deep-sub", "平均密度增大、潜艇下沉")),
    d(350, "flow", "半导体吸收光子后分离电荷，电极收集电荷并让电流流过外部电路。", ("sun", "太阳光子到达面板"), ("solar-cell", "半导体分离电荷"), ("circuit", "电流进入外部电路")),
    d(351, "flow", "风经过叶片产生升力并转动转子，转轴带动发电机把机械能转换成电能。", ("wind", "流动空气经过叶片"), ("rotor", "叶片受力带动转子"), ("generator", "发电机产生电能")),
    d(353, "network", "电网先升高电压远距离输电，再逐级降压，把电能分配到不同用户。", ("power-plant", "发电站"), ("transformer-up", "升压后远距离输送"), ("substation", "变电站逐级降压"), ("home", "家庭和其他用户")),
    d(359, "flow", "地表吸收阳光后发出红外辐射，温室气体吸收并向各方向再发射其中一部分。", ("sun", "阳光加热地表"), ("earth-air", "地表向外发出红外"), ("trapped-heat", "温室气体吸收并再发射")),
]


PALETTES = [
    ("#eef7ff", "#2d6fbb", "#f2b84b", "#18324a"),
    ("#edf8f7", "#187e88", "#ef8d62", "#173b46"),
    ("#f3f8ea", "#4c8a4b", "#d9a63c", "#28452d"),
    ("#fff6e9", "#d0713d", "#4e8a72", "#4b3326"),
    ("#f7f1fb", "#7b5da7", "#d96b67", "#3b2f51"),
    ("#f4f8f9", "#258097", "#e16b68", "#263e51"),
    ("#fff5ea", "#cf663f", "#397e8d", "#4b342d"),
    ("#f2f5ed", "#667d42", "#c47a45", "#35402f"),
    ("#edf7f8", "#217d92", "#e07955", "#244354"),
    ("#f1f2fb", "#5268ad", "#d48b49", "#293451"),
    ("#fbf5ed", "#a86a3d", "#477c84", "#49372c"),
    ("#f1f7f3", "#3c7c63", "#dc7655", "#2b463b"),
]

FALLBACK_KEYS: set[str] = set()


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 7, marker: bool = False, dash: str = "") -> str:
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"{marker_attr}{dash_attr}/>'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "none", width: float = 0) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def ellipse(cx: float, cy: float, rx: float, ry: float, fill: str, stroke: str = "none", width: float = 0) -> str:
    return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", width: float = 0, radius: float = 8) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def path(data: str, fill: str, stroke: str = "none", width: float = 0, marker: bool = False) -> str:
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    return f'<path d="{data}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{marker_attr}/>'


def polygon(points: list[tuple[float, float]], fill: str, stroke: str = "none", width: float = 0) -> str:
    data = " ".join(f"{x},{y}" for x, y in points)
    return f'<polygon points="{data}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>'


def wrap_label(label: str, limit: int = 8) -> list[str]:
    if "｜" in label:
        return label.split("｜")
    if len(label) <= limit:
        return [label]
    split = min(limit, math.ceil(len(label) / 2))
    return [label[:split], label[split:]]


def text_block(x: float, y: float, label: str, fill: str, size: int = 27, anchor: str = "middle", weight: int = 600) -> str:
    lines = wrap_label(label)
    tspans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else size + 8}">{esc(part)}</tspan>' for i, part in enumerate(lines))
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-family="system-ui, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{weight}">{tspans}</text>'


def icon(key: str, cx: float, cy: float, scale: float, primary: str, accent: str, dark: str, bg: str) -> str:
    """Draw a compact, label-supported scientific symbol."""
    s = scale
    parts: list[str] = []

    aliases = {
        "rain-rock": "water",
        "river": "water",
        "fast-river": "water",
        "slow-water": "water",
        "ocean": "water",
        "upwelling": "water",
        "plankton-water": "water",
        "clean-water": "water",
        "magma": "hot-drop",
        "fat": "fat-drop",
        "algae": "leaf",
        "nutrient": "molecule",
        "smell": "wave",
        "quiet-wave": "sound-wave",
        "echo": "sound-wave",
        "forward": "motion-arrow",
        "jump": "motion-arrow",
        "far-seed": "seed-parachute",
        "generations": "varied-insects",
        "core": "earth-cutaway",
        "dark-depth": "ocean-light",
        "color-depth": "ocean-light",
        "reflector": "chromatophore",
        "oil-molecules": "oil-particles",
        "water-molecules": "water-particles",
        "cold-water": "cold-particles",
        "warm-water": "warm-particles",
        "motion": "motion-arrow",
        "small-vortices": "vortex",
        "particle-circle": "wave-particles",
        "shore-wave": "wind-wave",
        "warm-current": "current",
        "deep-current": "current",
        "cooling": "cold-particles",
        "magnetic-field": "field",
        "sun-wave": "light",
        "rain-snow": "snow-rain",
        "full-moon": "moon",
        "sugar": "molecule",
        "sugar-oxygen": "molecule",
    }
    key = aliases.get(key, key)

    if key == "hex-ice-lattice":
        points = [(0, 0)] + [(math.cos(math.radians(60 * i)) * 58, math.sin(math.radians(60 * i)) * 58) for i in range(6)]
        for index in range(1, 7):
            next_index = 1 + index % 6
            parts.append(line(cx + points[index][0] * s, cy + points[index][1] * s, cx + points[next_index][0] * s, cy + points[next_index][1] * s, primary, 4 * s))
            parts.append(line(cx, cy, cx + points[index][0] * s, cy + points[index][1] * s, primary, 3 * s))
        for index, (dx, dy) in enumerate(points):
            parts.append(circle(cx + dx * s, cy + dy * s, 11 * s, accent if index % 2 else primary, dark, 2 * s))
        return "".join(parts)

    if key == "earth-field":
        parts.append(circle(cx, cy, 42 * s, primary, dark, 4 * s))
        parts.append(path(f"M{cx-12*s},{cy-39*s} Q{cx+12*s},{cy-5*s} {cx+35*s},{cy-25*s} Q{cx+20*s},{cy+20*s} {cx+32*s},{cy+34*s}", accent, "none"))
        for spread in (68, 92):
            parts.append(path(f"M{cx-8*s},{cy-spread*s} Q{cx-spread*s},{cy-55*s} {cx-spread*s},{cy} Q{cx-spread*s},{cy+55*s} {cx-8*s},{cy+spread*s}", "none", accent, 4 * s))
            parts.append(path(f"M{cx+8*s},{cy-spread*s} Q{cx+spread*s},{cy-55*s} {cx+spread*s},{cy} Q{cx+spread*s},{cy+55*s} {cx+8*s},{cy+spread*s}", "none", accent, 4 * s))
        return "".join(parts)

    if key == "hailstone":
        parts.append(circle(cx, cy, 70 * s, bg, dark, 4 * s))
        parts.append(circle(cx, cy, 50 * s, primary, dark, 3 * s))
        parts.append(circle(cx, cy, 30 * s, bg, dark, 3 * s))
        parts.append(circle(cx, cy, 13 * s, accent, dark, 2 * s))
        for dx in (-55, 0, 55):
            parts.append(line(cx + dx * s, cy - 98 * s, cx + dx * s, cy - 80 * s, accent, 4 * s))
        return "".join(parts)

    if key == "micelle":
        parts.append(circle(cx, cy, 42 * s, accent, dark, 4 * s))
        for index in range(10):
            angle = math.radians(index * 36)
            inner_x, inner_y = cx + math.cos(angle) * 48 * s, cy + math.sin(angle) * 48 * s
            outer_x, outer_y = cx + math.cos(angle) * 78 * s, cy + math.sin(angle) * 78 * s
            parts.append(line(inner_x, inner_y, outer_x, outer_y, primary, 4 * s))
            parts.append(circle(outer_x, outer_y, 9 * s, primary, dark, 2 * s))
        return "".join(parts)

    if key == "crystal-spacing":
        for origin, gap in [(-48, 19), (48, 30)]:
            for row in (-1, 0, 1):
                for col in (-1, 0, 1):
                    parts.append(circle(cx + (origin + col * gap) * s, cy + row * gap * s, 7 * s, accent if (row + col) % 2 else primary, dark, 1.5 * s))
        parts.append(line(cx - 10 * s, cy - 55 * s, cx + 12 * s, cy - 55 * s, dark, 4 * s, marker=True))
        return "".join(parts)

    if key == "finger-ridges":
        for inset in (0, 14, 28, 42):
            parts.append(path(f"M{cx-(74-inset)*s},{cy+58*s} Q{cx-(58-inset)*s},{cy-(72-inset)*s} {cx},{cy-(72-inset)*s} Q{cx+(58-inset)*s},{cy-(72-inset)*s} {cx+(74-inset)*s},{cy+58*s}", "none", primary if inset % 28 == 0 else accent, 5 * s))
        return "".join(parts)

    if key == "balance-senses":
        parts.append(path(f"M{cx-38*s},{cy-36*s} Q{cx},{cy-70*s} {cx+38*s},{cy-36*s} Q{cx},{cy-2*s} {cx-38*s},{cy-36*s} Z", bg, dark, 3 * s))
        parts.append(circle(cx, cy - 36 * s, 10 * s, primary, dark, 2 * s))
        parts.append(path(f"M{cx-70*s},{cy+42*s} q{-15*s},{-35*s} {18*s},{-42*s} q{28*s},{-4*s} {18*s},{24*s} q{-5*s},{15*s} {-20*s},{16*s}", "none", accent, 5 * s))
        parts.append(line(cx + 36 * s, cy + 13 * s, cx + 68 * s, cy + 48 * s, primary, 8 * s))
        parts.append(circle(cx + 36 * s, cy + 13 * s, 11 * s, accent, dark, 2 * s))
        parts.append(circle(cx + 68 * s, cy + 48 * s, 11 * s, primary, dark, 2 * s))
        return "".join(parts)

    if key == "faucet-handle":
        parts.append(line(cx, cy - 16 * s, cx, cy + 70 * s, dark, 13 * s))
        parts.append(rect(cx - 70 * s, cy - 42 * s, 140 * s, 28 * s, accent, dark, 4 * s, 8 * s))
        parts.append(circle(cx, cy - 28 * s, 16 * s, bg, dark, 4 * s))
        parts.append(line(cx - 28 * s, cy + 70 * s, cx + 28 * s, cy + 70 * s, primary, 8 * s))
        return "".join(parts)

    if key == "colliding-plates":
        parts.append(polygon([(cx - 92 * s, cy + 34 * s), (cx - 12 * s, cy + 34 * s), (cx - 12 * s, cy - 8 * s), (cx - 92 * s, cy + 8 * s)], primary, dark, 3 * s))
        parts.append(polygon([(cx + 92 * s, cy + 34 * s), (cx + 12 * s, cy + 34 * s), (cx + 12 * s, cy - 8 * s), (cx + 92 * s, cy + 8 * s)], accent, dark, 3 * s))
        parts.append(line(cx - 95 * s, cy - 45 * s, cx - 25 * s, cy - 45 * s, primary, 6 * s, marker=True))
        parts.append(line(cx + 95 * s, cy - 45 * s, cx + 25 * s, cy - 45 * s, accent, 6 * s, marker=True))
        parts.append(polygon([(cx - 25 * s, cy + 5 * s), (cx, cy - 48 * s), (cx + 25 * s, cy + 5 * s)], bg, dark, 3 * s))
        return "".join(parts)

    if key in {"rising-mountain", "eroding-mountain"}:
        parts.append(polygon([(cx - 88 * s, cy + 62 * s), (cx - 8 * s, cy - 62 * s), (cx + 32 * s, cy + 3 * s), (cx + 58 * s, cy - 35 * s), (cx + 92 * s, cy + 62 * s)], primary, dark, 4 * s))
        if key == "rising-mountain":
            parts.append(line(cx, cy + 20 * s, cx, cy - 90 * s, accent, 7 * s, marker=True))
        else:
            for dx in (-55, -12, 36, 68):
                parts.append(line(cx + dx * s, cy - 92 * s, cx + (dx - 14) * s, cy - 68 * s, accent, 4 * s))
            parts.append(path(f"M{cx-5*s},{cy-45*s} Q{cx+12*s},{cy} {cx+65*s},{cy+62*s}", "none", bg, 7 * s))
        return "".join(parts)

    if key == "igneous-rock":
        parts.append(path(f"M{cx-78*s},{cy+35*s} Q{cx-65*s},{cy-58*s} {cx-5*s},{cy-72*s} Q{cx+72*s},{cy-60*s} {cx+78*s},{cy+22*s} Q{cx+42*s},{cy+76*s} {cx-38*s},{cy+68*s} Z", primary, dark, 4 * s))
        for dx, dy in [(-42, -20), (-15, 20), (18, -32), (44, 14), (3, 48)]:
            parts.append(circle(cx + dx * s, cy + dy * s, 8 * s, accent, dark, 1.5 * s))
        return "".join(parts)

    if key == "sediment-layers":
        for index, (color, width) in enumerate([(primary, 150), (accent, 138), (primary, 128), (accent, 118)]):
            parts.append(path(f"M{cx-width/2*s},{cy+(-52+index*34)*s} Q{cx},{cy+(-68+index*34)*s} {cx+width/2*s},{cy+(-52+index*34)*s} L{cx+width/2*s},{cy+(-28+index*34)*s} Q{cx},{cy+(-44+index*34)*s} {cx-width/2*s},{cy+(-28+index*34)*s} Z", color, dark, 2 * s))
        return "".join(parts)

    if key == "metamorphic-rock":
        for offset, color in [(-38, primary), (-12, accent), (14, primary), (40, accent)]:
            parts.append(path(f"M{cx-76*s},{cy+offset*s} Q{cx-25*s},{cy+(offset-20)*s} {cx+10*s},{cy+offset*s} T{cx+76*s},{cy+offset*s}", "none", color, 9 * s))
        parts.append(line(cx - 105 * s, cy, cx - 82 * s, cy, dark, 5 * s, marker=True))
        parts.append(line(cx + 105 * s, cy, cx + 82 * s, cy, dark, 5 * s, marker=True))
        return "".join(parts)

    if key == "ice-sheet":
        parts.append(path(f"M{cx-92*s},{cy+42*s} L{cx-76*s},{cy-42*s} Q{cx-24*s},{cy-68*s} {cx+28*s},{cy-48*s} L{cx+82*s},{cy-8*s} L{cx+92*s},{cy+42*s} Z", bg, dark, 4 * s))
        parts.append(path(f"M{cx-92*s},{cy+42*s} Q{cx-32*s},{cy+25*s} {cx+18*s},{cy+42*s} T{cx+92*s},{cy+42*s}", primary, primary, 8 * s))
        parts.append(line(cx - 65 * s, cy - 18 * s, cx - 20 * s, cy - 35 * s, accent, 4 * s))
        return "".join(parts)

    if key == "coiled-chain":
        parts.append(path(f"M{cx-82*s},{cy+35*s} C{cx-92*s},{cy-50*s} {cx-22*s},{cy-58*s} {cx-15*s},{cy+4*s} C{cx-8*s},{cy+65*s} {cx+64*s},{cy+58*s} {cx+68*s},{cy-2*s} C{cx+72*s},{cy-48*s} {cx+35*s},{cy-62*s} {cx+18*s},{cy-28*s}", "none", primary, 8 * s))
        for dx, dy in [(-76, 30), (-40, -42), (-13, 4), (28, 48), (68, -2), (22, -32)]:
            parts.append(circle(cx + dx * s, cy + dy * s, 9 * s, accent, dark, 2 * s))
        return "".join(parts)

    if key == "hook-seed":
        parts.append(circle(cx, cy, 42 * s, accent, dark, 4 * s))
        for index in range(10):
            angle = math.radians(index * 36)
            x1, y1 = cx + math.cos(angle) * 38 * s, cy + math.sin(angle) * 38 * s
            x2, y2 = cx + math.cos(angle) * 78 * s, cy + math.sin(angle) * 78 * s
            parts.append(line(x1, y1, x2, y2, dark, 4 * s))
            hook_x = x2 + math.cos(angle + 0.7) * 13 * s
            hook_y = y2 + math.sin(angle + 0.7) * 13 * s
            parts.append(line(x2, y2, hook_x, hook_y, dark, 3 * s))
        return "".join(parts)

    if key == "mushroom":
        parts.append(path(f"M{cx-75*s},{cy-5*s} Q{cx-58*s},{cy-76*s} {cx},{cy-82*s} Q{cx+58*s},{cy-76*s} {cx+75*s},{cy-5*s} Z", accent, dark, 4 * s))
        parts.append(path(f"M{cx-24*s},{cy-4*s} Q{cx-18*s},{cy+48*s} {cx-38*s},{cy+74*s} L{cx+38*s},{cy+74*s} Q{cx+18*s},{cy+48*s} {cx+24*s},{cy-4*s} Z", bg, dark, 4 * s))
        parts.append(path(f"M{cx-58*s},{cy-5*s} Q{cx},{cy+20*s} {cx+58*s},{cy-5*s}", "none", primary, 4 * s))
        return "".join(parts)

    if key == "mycelium":
        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            x2, y2 = cx + math.cos(radians) * 78 * s, cy + math.sin(radians) * 62 * s
            parts.append(path(f"M{cx},{cy} Q{cx+math.cos(radians+0.5)*38*s},{cy+math.sin(radians+0.5)*30*s} {x2},{y2}", "none", primary, 5 * s))
            parts.append(line(x2, y2, x2 + math.cos(radians + 0.55) * 24 * s, y2 + math.sin(radians + 0.55) * 24 * s, accent, 3 * s))
            parts.append(line(x2, y2, x2 + math.cos(radians - 0.55) * 24 * s, y2 + math.sin(radians - 0.55) * 24 * s, accent, 3 * s))
        parts.append(circle(cx, cy, 10 * s, accent, dark, 2 * s))
        return "".join(parts)

    if key == "spores":
        for dx, dy, radius in [(-55, 25, 13), (-26, -36, 10), (8, 18, 15), (42, -24, 12), (60, 38, 9), (-2, -66, 8)]:
            parts.append(circle(cx + dx * s, cy + dy * s, radius * s, accent if dx > 0 else primary, dark, 2 * s))
        return "".join(parts)

    if key == "snail":
        parts.append(path(f"M{cx-80*s},{cy+42*s} Q{cx-20*s},{cy+18*s} {cx+74*s},{cy+35*s} Q{cx+90*s},{cy+48*s} {cx+64*s},{cy+58*s} L{cx-70*s},{cy+58*s} Z", primary, dark, 4 * s))
        parts.append(circle(cx - 22 * s, cy - 8 * s, 47 * s, accent, dark, 4 * s))
        parts.append(path(f"M{cx-22*s},{cy-8*s} q{28*s},{-25*s} {31*s},{9*s} q{2*s},{25*s} {-23*s},{24*s}", "none", dark, 4 * s))
        for dx in (48, 70):
            parts.append(line(cx + dx * s, cy + 30 * s, cx + (dx + 8) * s, cy - 12 * s, dark, 3 * s))
            parts.append(circle(cx + (dx + 8) * s, cy - 15 * s, 5 * s, dark))
        return "".join(parts)

    if key == "lizard-tail":
        parts.append(ellipse(cx - 30 * s, cy, 48 * s, 25 * s, primary, dark, 4 * s))
        parts.append(circle(cx - 78 * s, cy - 4 * s, 21 * s, accent, dark, 4 * s))
        parts.append(path(f"M{cx+15*s},{cy} Q{cx+55*s},{cy-22*s} {cx+82*s},{cy+12*s} Q{cx+100*s},{cy+34*s} {cx+112*s},{cy+12*s}", "none", primary, 13 * s))
        for dx, dy in [(-45, 18), (-10, 18)]:
            parts.append(line(cx + dx * s, cy + dy * s, cx + (dx - 18) * s, cy + 52 * s, dark, 4 * s))
            parts.append(line(cx + dx * s, cy - dy * s, cx + (dx - 18) * s, cy - 52 * s, dark, 4 * s))
        parts.append(line(cx + 42 * s, cy - 17 * s, cx + 49 * s, cy + 18 * s, dark, 3 * s, dash="6 5"))
        return "".join(parts)

    if key == "chameleon-skin":
        parts.append(rect(cx - 82 * s, cy - 64 * s, 164 * s, 128 * s, bg, dark, 4 * s, 18 * s))
        for row in range(3):
            for col in range(4):
                x = cx + (-57 + col * 38 + (row % 2) * 10) * s
                y = cy + (-38 + row * 38) * s
                color = primary if (row + col) % 2 else accent
                parts.append(circle(x, y, 15 * s, color, dark, 2 * s))
        return "".join(parts)

    if key == "trunk":
        parts.append(path(f"M{cx-65*s},{cy-58*s} Q{cx-25*s},{cy-74*s} {cx-10*s},{cy-35*s} Q{cx+4*s},{cy+4*s} {cx+48*s},{cy+30*s} Q{cx+75*s},{cy+47*s} {cx+58*s},{cy+75*s}", "none", primary, 28 * s))
        parts.append(path(f"M{cx-65*s},{cy-58*s} Q{cx-25*s},{cy-74*s} {cx-10*s},{cy-35*s} Q{cx+4*s},{cy+4*s} {cx+48*s},{cy+30*s} Q{cx+75*s},{cy+47*s} {cx+58*s},{cy+75*s}", "none", dark, 4 * s))
        for offset in (-32, -6, 20, 43):
            parts.append(line(cx + offset * s, cy + (offset + 18) * 0.55 * s, cx + (offset + 18) * s, cy + (offset + 3) * 0.55 * s, accent, 3 * s))
        return "".join(parts)

    if key == "horse-leg":
        points = [(cx - 32 * s, cy - 82 * s), (cx - 8 * s, cy - 28 * s), (cx + 8 * s, cy + 18 * s), (cx + 42 * s, cy + 72 * s)]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            parts.append(line(x1, y1, x2, y2, primary, 16 * s))
            parts.append(line(x1, y1, x2, y2, dark, 3 * s))
        for x, y in points[1:-1]:
            parts.append(circle(x, y, 13 * s, accent, dark, 3 * s))
        parts.append(path(f"M{cx-48*s},{cy-65*s} Q{cx+18*s},{cy-10*s} {cx+28*s},{cy+61*s}", "none", accent, 6 * s))
        parts.append(line(cx + 28 * s, cy + 72 * s, cx + 68 * s, cy + 72 * s, dark, 7 * s))
        return "".join(parts)

    if key == "teeth":
        for index in range(4):
            x = cx + (-60 + index * 40) * s
            parts.append(path(f"M{x-16*s},{cy-54*s} Q{x},{cy-70*s} {x+16*s},{cy-54*s} L{x+13*s},{cy+25*s} L{x},{cy+65*s} L{x-13*s},{cy+25*s} Z", bg, dark, 4 * s))
        return "".join(parts)

    if key == "stomach":
        parts.append(path(f"M{cx-24*s},{cy-78*s} Q{cx+22*s},{cy-45*s} {cx+4*s},{cy-5*s} Q{cx-14*s},{cy+34*s} {cx+50*s},{cy+42*s} Q{cx+20*s},{cy+88*s} {cx-34*s},{cy+58*s} Q{cx-80*s},{cy+28*s} {cx-54*s},{cy-15*s} Q{cx-34*s},{cy-45*s} {cx-24*s},{cy-78*s} Z", accent, dark, 5 * s))
        return "".join(parts)

    if key == "intestine":
        parts.append(rect(cx - 76 * s, cy - 72 * s, 152 * s, 144 * s, bg, dark, 6 * s, 26 * s))
        parts.append(path(f"M{cx-46*s},{cy-42*s} C{cx+58*s},{cy-62*s} {cx+58*s},{cy-8*s} {cx-30*s},{cy-12*s} C{cx-72*s},{cy-13*s} {cx-70*s},{cy+25*s} {cx+20*s},{cy+15*s} C{cx+70*s},{cy+8*s} {cx+62*s},{cy+55*s} {cx-42*s},{cy+48*s}", "none", primary, 11 * s))
        return "".join(parts)

    if key in {"sliding-box", "wheel-cart"}:
        parts.append(rect(cx - 66 * s, cy - 48 * s, 132 * s, 92 * s, accent, dark, 5 * s, 6 * s))
        ground_y = cy + 72 * s
        parts.append(line(cx - 92 * s, ground_y, cx + 92 * s, ground_y, dark, 5 * s))
        if key == "wheel-cart":
            for dx in (-42, 42):
                parts.append(circle(cx + dx * s, cy + 57 * s, 20 * s, bg, dark, 6 * s))
                parts.append(circle(cx + dx * s, cy + 57 * s, 5 * s, primary))
        else:
            for dx in (-48, -15, 18, 51):
                parts.append(line(cx + dx * s, cy + 48 * s, cx + (dx - 12) * s, cy + 62 * s, primary, 3 * s))
        return "".join(parts)

    if key in {"healthy-coral", "bleached-coral"}:
        color = accent if key == "healthy-coral" else bg
        stroke = dark if key == "healthy-coral" else primary
        branches = [(-55, 55, -45, -28), (-22, 58, -15, -72), (12, 60, 18, -48), (48, 58, 60, -18)]
        for x1, y1, x2, y2 in branches:
            parts.append(line(cx + x1 * s, cy + y1 * s, cx + x2 * s, cy + y2 * s, stroke, 16 * s))
            parts.append(circle(cx + x2 * s, cy + y2 * s, 15 * s, color, stroke, 4 * s))
        parts.append(line(cx - 75 * s, cy + 62 * s, cx + 76 * s, cy + 62 * s, stroke, 10 * s))
        if key == "healthy-coral":
            for dx, dy in [(-42, 4), (-8, -28), (25, 2), (54, 22)]:
                parts.append(circle(cx + dx * s, cy + dy * s, 6 * s, primary))
        return "".join(parts)

    if key in {"crab-front", "crab-side"}:
        parts.append(ellipse(cx, cy, 55 * s, 38 * s, accent, dark, 5 * s))
        parts.append(circle(cx - 20 * s, cy - 18 * s, 5 * s, dark))
        parts.append(circle(cx + 20 * s, cy - 18 * s, 5 * s, dark))
        for side in (-1, 1):
            parts.append(path(f"M{cx+side*42*s},{cy-12*s} Q{cx+side*82*s},{cy-55*s} {cx+side*95*s},{cy-26*s}", "none", dark, 6 * s))
            for row in (-18, 0, 18):
                reach = 88 if key == "crab-side" else 70
                parts.append(line(cx + side * 40 * s, cy + row * s, cx + side * reach * s, cy + (row + (18 if row >= 0 else -18)) * s, dark, 5 * s))
        if key == "crab-side":
            parts.append(line(cx - 38 * s, cy + 70 * s, cx + 48 * s, cy + 70 * s, primary, 5 * s, marker=True))
        else:
            parts.append(line(cx, cy + 80 * s, cx, cy + 48 * s, primary, 5 * s, marker=True))
        return "".join(parts)

    if key == "seahorse-pouch":
        parts.append(path(f"M{cx-18*s},{cy-76*s} Q{cx+48*s},{cy-58*s} {cx+18*s},{cy-8*s} Q{cx-8*s},{cy+28*s} {cx+32*s},{cy+55*s} Q{cx+62*s},{cy+77*s} {cx+26*s},{cy+82*s} Q{cx-22*s},{cy+78*s} {cx-36*s},{cy+28*s} Q{cx-58*s},{cy-18*s} {cx-18*s},{cy-76*s} Z", primary, dark, 4 * s))
        parts.append(ellipse(cx + 4 * s, cy + 24 * s, 34 * s, 42 * s, bg, accent, 5 * s))
        for dx, dy in [(-10, 10), (12, 4), (5, 28), (-8, 36)]:
            parts.append(circle(cx + dx * s, cy + dy * s, 6 * s, accent))
        return "".join(parts)

    if key == "baby-seahorse":
        for offset in (-38, 28):
            color = primary if offset < 0 else accent
            parts.append(circle(cx + offset * s, cy - 50 * s, 18 * s, color, dark, 3 * s))
            parts.append(path(f"M{cx+(offset+12)*s},{cy-54*s} l{18*s},{7*s} l{-18*s},{7*s} Z", color, dark, 2 * s))
            parts.append(path(f"M{cx+offset*s},{cy-32*s} q{28*s},{26*s} {4*s},{58*s} q{-22*s},{20*s} {7*s},{42*s} q{20*s},{15*s} {-4*s},{27*s}", "none", color, 10 * s))
            parts.append(path(f"M{cx+(offset+7)*s},{cy+93*s} q{20*s},{-8*s} {8*s},{-23*s}", "none", dark, 3 * s))
            parts.append(circle(cx + (offset + 4) * s, cy - 54 * s, 3.5 * s, dark))
        return "".join(parts)

    if key == "black-hole":
        parts.append(ellipse(cx, cy, 96 * s, 35 * s, "none", accent, 12 * s))
        parts.append(ellipse(cx, cy, 76 * s, 24 * s, "none", primary, 8 * s))
        parts.append(circle(cx, cy, 44 * s, dark, primary, 4 * s))
        parts.append(path(f"M{cx-92*s},{cy+18*s} Q{cx},{cy+62*s} {cx+92*s},{cy+18*s}", "none", accent, 5 * s))
        return "".join(parts)

    if key == "corn":
        parts.append(ellipse(cx, cy, 48 * s, 76 * s, accent, dark, 5 * s))
        for row in range(5):
            for col in range(3):
                parts.append(circle(cx + (-22 + col * 22) * s, cy + (-48 + row * 24) * s, 8 * s, primary, dark, 1.5 * s))
        parts.append(path(f"M{cx-48*s},{cy+18*s} Q{cx-88*s},{cy+42*s} {cx-52*s},{cy+82*s}", "none", primary, 14 * s))
        parts.append(path(f"M{cx+48*s},{cy+18*s} Q{cx+88*s},{cy+42*s} {cx+52*s},{cy+82*s}", "none", primary, 14 * s))
        return "".join(parts)

    if key == "popcorn":
        for dx, dy, radius in [(-42, 5, 32), (-18, -30, 35), (18, -36, 38), (45, 2, 34), (8, 18, 43)]:
            parts.append(circle(cx + dx * s, cy + dy * s, radius * s, bg, dark, 4 * s))
        parts.append(circle(cx + 5 * s, cy + 8 * s, 20 * s, accent, dark, 3 * s))
        return "".join(parts)

    if key in {"crack", "fragments"}:
        if key == "crack":
            parts.append(rect(cx - 78 * s, cy - 74 * s, 156 * s, 148 * s, bg, primary, 5 * s, 4 * s))
            parts.append(path(f"M{cx-12*s},{cy-74*s} L{cx+8*s},{cy-28*s} L{cx-18*s},{cy+2*s} L{cx+20*s},{cy+30*s} L{cx-2*s},{cy+74*s}", "none", dark, 7 * s))
            parts.append(line(cx + 8 * s, cy - 28 * s, cx + 52 * s, cy - 44 * s, dark, 4 * s))
        else:
            for points in [
                [(-78, -62), (-12, -78), (-25, -8), (-72, 2)],
                [(2, -72), (76, -50), (54, 8), (18, -4)],
                [(-70, 18), (-20, 4), (-8, 72), (-58, 62)],
                [(8, 8), (62, 18), (74, 70), (20, 62)],
            ]:
                parts.append(polygon([(cx + x * s, cy + y * s) for x, y in points], bg, dark, 4 * s))
        return "".join(parts)

    if key == "stretched-chain":
        positions = [(-76, -18), (-38, 18), (0, -18), (38, 18), (76, -18)]
        for (x1, y1), (x2, y2) in zip(positions, positions[1:]):
            parts.append(line(cx + x1 * s, cy + y1 * s, cx + x2 * s, cy + y2 * s, primary, 7 * s))
        for index, (dx, dy) in enumerate(positions):
            parts.append(circle(cx + dx * s, cy + dy * s, 13 * s, accent if index % 2 else primary, dark, 3 * s))
        parts.append(line(cx - 104 * s, cy, cx - 82 * s, cy, dark, 5 * s))
        parts.append(line(cx + 82 * s, cy, cx + 104 * s, cy, dark, 5 * s))
        return "".join(parts)

    if key in {"solid-particles", "liquid-particles", "gas-particles", "ice-lattice", "cold-particles", "warm-particles", "oil-particles", "water-particles"}:
        if key in {"solid-particles", "ice-lattice"}:
            positions = [(x, y) for y in (-42, 0, 42) for x in (-48, 0, 48)]
        elif key in {"gas-particles"}:
            positions = [(-78, -58), (62, -72), (-20, -12), (82, 34), (-65, 65)]
        else:
            positions = [(-65, -48), (-15, -62), (48, -43), (-48, 5), (8, -5), (65, 12), (-28, 50), (38, 58)]
        speed = key in {"warm-particles"}
        oil = key == "oil-particles"
        for i, (dx, dy) in enumerate(positions):
            color = accent if oil else primary if i % 2 else accent
            parts.append(circle(cx + dx * s, cy + dy * s, 12 * s, color, dark, 2 * s))
            if speed:
                parts.append(line(cx + (dx - 24) * s, cy + dy * s, cx + (dx - 7) * s, cy + dy * s, accent, 3 * s))
        if key in {"solid-particles", "ice-lattice"}:
            for y in (-42, 0, 42):
                parts.append(line(cx - 48 * s, cy + y * s, cx + 48 * s, cy + y * s, primary, 2 * s))
            for x in (-48, 0, 48):
                parts.append(line(cx + x * s, cy - 42 * s, cx + x * s, cy + 42 * s, primary, 2 * s))
        return "".join(parts)

    if key == "bubble":
        parts.append(circle(cx, cy, 78 * s, bg, primary, 6 * s))
        parts.append(path(f"M{cx-48*s},{cy-35*s} Q{cx-15*s},{cy-70*s} {cx+12*s},{cy-48*s}", "none", accent, 8 * s))
        parts.append(circle(cx + 42 * s, cy + 35 * s, 13 * s, accent))
        return "".join(parts)

    if key in {"pheromone", "strong-trail"}:
        count = 8 if key == "strong-trail" else 5
        for i in range(count):
            x = cx - 90 * s + i * (180 / max(1, count - 1)) * s
            y = cy + math.sin(i * 1.2) * 28 * s
            parts.append(circle(x, y, (9 if key == "strong-trail" else 7) * s, accent if i % 2 else primary, dark, 2 * s))
        return "".join(parts)

    if key == "gecko-foot":
        parts.append(ellipse(cx, cy, 52 * s, 70 * s, primary, dark, 5 * s))
        for i, dx in enumerate([-65, -32, 0, 32, 65]):
            parts.append(path(f"M{cx+dx*0.45*s},{cy-35*s} Q{cx+dx*s},{cy-105*s} {cx+dx*s},{cy-125*s}", "none", accent, (13 - i % 2 * 2) * s))
        return "".join(parts)

    if key == "duck":
        parts.append(ellipse(cx - 5 * s, cy + 15 * s, 82 * s, 48 * s, primary, dark, 5 * s))
        parts.append(circle(cx + 62 * s, cy - 35 * s, 35 * s, accent, dark, 4 * s))
        parts.append(polygon([(cx + 92 * s, cy - 37 * s), (cx + 130 * s, cy - 20 * s), (cx + 92 * s, cy - 8 * s)], bg, dark, 3 * s))
        parts.append(circle(cx + 70 * s, cy - 45 * s, 5 * s, dark))
        return "".join(parts)

    if key == "deep-object":
        parts.append(ellipse(cx, cy, 90 * s, 48 * s, primary, dark, 5 * s))
        parts.append(rect(cx - 25 * s, cy - 62 * s, 50 * s, 32 * s, primary, dark, 4 * s, 8 * s))
        parts.append(circle(cx - 35 * s, cy, 12 * s, bg, dark, 2 * s))
        parts.append(circle(cx + 12 * s, cy, 12 * s, bg, dark, 2 * s))
        return "".join(parts)

    if key == "weight":
        parts.append(path(f"M{cx-35*s},{cy-55*s} Q{cx},{cy-95*s} {cx+35*s},{cy-55*s}", "none", dark, 10 * s))
        parts.append(path(f"M{cx-68*s},{cy-48*s} H{cx+68*s} L{cx+88*s},{cy+82*s} H{cx-88*s} Z", primary, dark, 5 * s))
        return "".join(parts)

    if key in {"sound-wave", "wave", "vortex", "wind-wave", "wave-particles", "current", "seismic-wave"}:
        if key == "vortex":
            for i in range(3):
                y = cy - 45 * s + i * 45 * s
                parts.append(path(f"M{cx-90*s},{y} C{cx-35*s},{y-35*s} {cx-20*s},{y+35*s} {cx+25*s},{y} C{cx+62*s},{y-28*s} {cx+72*s},{y+25*s} {cx+92*s},{y}", "none", primary if i % 2 else accent, 5 * s))
        elif key == "wave-particles":
            parts.append(path(f"M{cx-100*s},{cy} Q{cx-50*s},{cy-70*s} {cx},{cy} T{cx+100*s},{cy}", "none", primary, 8 * s, True))
            for x in [-60, 0, 60]:
                parts.append(circle(cx + x * s, cy, 13 * s, accent, dark, 2 * s))
                parts.append(path(f"M{cx+(x-20)*s},{cy} A{20*s},{16*s} 0 1 1 {cx+(x+18)*s},{cy-4*s}", "none", dark, 2.5 * s, True))
        else:
            for i in range(3):
                y = cy - 42 * s + i * 42 * s
                parts.append(path(f"M{cx-98*s},{y} Q{cx-52*s},{y-42*s} {cx-5*s},{y} T{cx+90*s},{y}", "none", primary if i % 2 else accent, 6 * s, key in {"current", "seismic-wave"}))
        return "".join(parts)

    if key == "drop-rainbow":
        parts.append(path(f"M{cx},{cy-78*s} C{cx-25*s},{cy-30*s} {cx-62*s},{cy+8*s} {cx-62*s},{cy+40*s} A{62*s},{62*s} 0 0 0 {cx+62*s},{cy+40*s} C{cx+62*s},{cy+8*s} {cx+25*s},{cy-30*s} {cx},{cy-78*s} Z", bg, dark, 5 * s))
        for i, color in enumerate(["#df5b52", "#ed9c42", "#e6ca45", "#4b9c68", "#3e75b9"]):
            parts.append(path(f"M{cx-95*s},{cy+(-34+i*14)*s} Q{cx},{cy+(-75+i*8)*s} {cx+92*s},{cy+(-20+i*14)*s}", "none", color, 4 * s))
        return "".join(parts)

    if key in {"ice", "snow", "snow-rain"}:
        if key == "ice":
            parts.append(polygon([(cx - 70 * s, cy - 48 * s), (cx + 45 * s, cy - 65 * s), (cx + 78 * s, cy + 48 * s), (cx - 48 * s, cy + 72 * s)], bg, primary, 6 * s))
            parts.append(line(cx - 70 * s, cy - 48 * s, cx - 48 * s, cy + 72 * s, dark, 3 * s))
        else:
            for a in [0, math.pi / 3, 2 * math.pi / 3]:
                dx, dy = math.cos(a) * 82 * s, math.sin(a) * 82 * s
                parts.append(line(cx - dx, cy - dy, cx + dx, cy + dy, primary, 5 * s))
            if key == "snow-rain":
                parts.append(path(f"M{cx+70*s},{cy-55*s} C{cx+50*s},{cy-18*s} {cx+35*s},{cy+8*s} {cx+35*s},{cy+30*s} A{35*s},{35*s} 0 0 0 {cx+105*s},{cy+30*s} C{cx+105*s},{cy+8*s} {cx+90*s},{cy-18*s} {cx+70*s},{cy-55*s} Z", accent, dark, 3 * s))
        return "".join(parts)

    if key == "balloon":
        parts.append(ellipse(cx, cy - 15 * s, 70 * s, 88 * s, accent, dark, 5 * s))
        parts.append(polygon([(cx - 10 * s, cy + 72 * s), (cx + 10 * s, cy + 72 * s), (cx, cy + 92 * s)], accent, dark, 3 * s))
        parts.append(path(f"M{cx},{cy+92*s} Q{cx+22*s},{cy+120*s} {cx-5*s},{cy+140*s}", "none", primary, 3 * s))
        return "".join(parts)

    if key == "straw":
        parts.append(path(f"M{cx-72*s},{cy-45*s} H{cx+60*s} L{cx+42*s},{cy+85*s} H{cx-55*s} Z", bg, dark, 5 * s))
        parts.append(path(f"M{cx-50*s},{cy+20*s} H{cx+50*s} L{cx+42*s},{cy+85*s} H{cx-55*s} Z", primary, "none"))
        parts.append(path(f"M{cx+5*s},{cy+55*s} L{cx+20*s},{cy-105*s} H{cx+82*s}", "none", accent, 8 * s))
        return "".join(parts)

    if key == "lightning":
        parts.append(polygon([(cx + 5 * s, cy - 95 * s), (cx - 48 * s, cy + 5 * s), (cx - 5 * s, cy + 2 * s), (cx - 40 * s, cy + 95 * s), (cx + 65 * s, cy - 20 * s), (cx + 18 * s, cy - 15 * s)], accent, dark, 5 * s))
        return "".join(parts)

    if key in {"seed-parachute", "wind"}:
        if key == "wind":
            for i, width in enumerate([95, 70, 105]):
                parts.append(path(f"M{cx-width*s},{cy+(-45+i*45)*s} Q{cx},{cy+(-70+i*45)*s} {cx+width*s},{cy+(-45+i*45)*s}", "none", primary if i % 2 else accent, 6 * s, True))
        else:
            parts.append(path(f"M{cx-75*s},{cy-25*s} Q{cx},{cy-105*s} {cx+75*s},{cy-25*s}", "none", primary, 5 * s))
            for dx in [-60, -30, 0, 30, 60]:
                parts.append(line(cx + dx * s, cy - 25 * s, cx, cy + 62 * s, dark, 3 * s))
            parts.append(ellipse(cx, cy + 78 * s, 13 * s, 24 * s, accent, dark, 3 * s))
        return "".join(parts)

    if key in {"egg", "chrysalis"}:
        if key == "egg":
            parts.append(ellipse(cx, cy, 48 * s, 67 * s, bg, dark, 5 * s))
        else:
            parts.append(path(f"M{cx},{cy-85*s} C{cx-58*s},{cy-52*s} {cx-48*s},{cy+62*s} {cx},{cy+88*s} C{cx+48*s},{cy+62*s} {cx+58*s},{cy-52*s} {cx},{cy-85*s} Z", primary, dark, 5 * s))
            parts.append(line(cx, cy - 86 * s, cx + 32 * s, cy - 112 * s, dark, 4 * s))
        return "".join(parts)

    if key in {"long-worm", "short-worm", "anchor"}:
        if key == "anchor":
            parts.append(line(cx - 100 * s, cy + 58 * s, cx + 100 * s, cy + 58 * s, dark, 6 * s))
            parts.append(path(f"M{cx-70*s},{cy+42*s} Q{cx},{cy-35*s} {cx+70*s},{cy+42*s}", "none", primary, 24 * s))
            for x in [-45, -15, 15, 45]:
                parts.append(line(cx + x * s, cy + 35 * s, cx + (x + 8) * s, cy + 66 * s, accent, 3 * s))
        else:
            rx, ry = (105, 25) if key == "long-worm" else (72, 43)
            parts.append(path(f"M{cx-rx*s},{cy} Q{cx-rx/2*s},{cy-ry*s} {cx},{cy} T{cx+rx*s},{cy}", "none", primary, (18 if key == "long-worm" else 28) * s))
        return "".join(parts)

    if key in {"web", "radial", "spiral"}:
        for a in [i * math.tau / 8 for i in range(8)]:
            parts.append(line(cx, cy, cx + math.cos(a) * 95 * s, cy + math.sin(a) * 95 * s, dark, 2.5 * s))
        for r in [28, 52, 76]:
            parts.append(circle(cx, cy, r * s, "none", primary if r != 52 else accent, 2.5 * s))
        return "".join(parts)

    if key == "skull":
        parts.append(circle(cx, cy - 15 * s, 65 * s, bg, dark, 5 * s))
        parts.append(rect(cx - 38 * s, cy + 30 * s, 76 * s, 52 * s, bg, dark, 5 * s, 8 * s))
        parts.append(circle(cx - 25 * s, cy - 18 * s, 14 * s, dark))
        parts.append(circle(cx + 25 * s, cy - 18 * s, 14 * s, dark))
        return "".join(parts)

    if key in {"giraffe", "camel"}:
        parts.append(ellipse(cx - 15 * s, cy + 30 * s, 75 * s, 45 * s, primary, dark, 5 * s))
        parts.append(rect(cx + 32 * s, cy - 58 * s, 28 * s, 88 * s, primary, dark, 4 * s, 12 * s))
        parts.append(circle(cx + 50 * s, cy - 70 * s, 28 * s, accent, dark, 4 * s))
        if key == "camel":
            parts.append(path(f"M{cx-72*s},{cy+10*s} Q{cx-25*s},{cy-62*s} {cx+20*s},{cy+12*s}", accent, dark, 4 * s))
        return "".join(parts)

    if key == "fat-drop":
        parts.append(circle(cx - 36 * s, cy + 10 * s, 42 * s, accent, dark, 4 * s))
        parts.append(circle(cx + 36 * s, cy + 8 * s, 42 * s, primary, dark, 4 * s))
        parts.append(circle(cx, cy - 38 * s, 38 * s, accent, dark, 4 * s))
        return "".join(parts)

    if key in {"shoulder", "knee"}:
        if key == "shoulder":
            parts.append(path(f"M{cx-78*s},{cy+70*s} Q{cx-70*s},{cy-50*s} {cx+5*s},{cy-62*s}", "none", primary, 32 * s))
            parts.append(circle(cx + 5 * s, cy - 62 * s, 34 * s, bg, dark, 5 * s))
            parts.append(path(f"M{cx+32*s},{cy-45*s} L{cx+78*s},{cy+58*s}", "none", accent, 28 * s))
        else:
            parts.append(line(cx - 12 * s, cy - 95 * s, cx - 5 * s, cy - 12 * s, primary, 30 * s))
            parts.append(circle(cx, cy, 8 * s, bg, dark, 5 * s))
            parts.append(line(cx + 5 * s, cy + 12 * s, cx + 45 * s, cy + 95 * s, accent, 30 * s))
        return "".join(parts)

    if key == "muscle":
        parts.append(path(f"M{cx-80*s},{cy+48*s} Q{cx-20*s},{cy-80*s} {cx+72*s},{cy-22*s} Q{cx+35*s},{cy+72*s} {cx-80*s},{cy+48*s} Z", accent, dark, 5 * s))
        parts.append(line(cx - 92 * s, cy + 58 * s, cx - 55 * s, cy + 35 * s, dark, 8 * s))
        parts.append(line(cx + 72 * s, cy - 22 * s, cx + 105 * s, cy - 45 * s, dark, 8 * s))
        return "".join(parts)

    if key == "motion-arrow":
        parts.append(path(f"M{cx-100*s},{cy+25*s} Q{cx-15*s},{cy-78*s} {cx+85*s},{cy-8*s}", "none", primary, 11 * s, True))
        parts.append(circle(cx - 72 * s, cy + 18 * s, 24 * s, accent, dark, 4 * s))
        return "".join(parts)

    if key == "gravity-center":
        parts.append(circle(cx, cy, 30 * s, accent, dark, 4 * s))
        for a in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            x1, y1 = cx + math.cos(a) * 100 * s, cy + math.sin(a) * 100 * s
            x2, y2 = cx + math.cos(a) * 48 * s, cy + math.sin(a) * 48 * s
            parts.append(line(x1, y1, x2, y2, primary, 7 * s, True))
        return "".join(parts)

    if key == "hot-drop":
        parts.append(path(f"M{cx},{cy-90*s} C{cx-25*s},{cy-42*s} {cx-65*s},{cy+10*s} {cx-65*s},{cy+45*s} A{65*s},{65*s} 0 0 0 {cx+65*s},{cy+45*s} C{cx+65*s},{cy+10*s} {cx+25*s},{cy-42*s} {cx},{cy-90*s} Z", accent, dark, 5 * s))
        parts.append(path(f"M{cx-28*s},{cy+15*s} Q{cx},{cy-35*s} {cx+28*s},{cy+15*s}", "none", bg, 7 * s))
        return "".join(parts)

    if key == "delta":
        parts.append(path(f"M{cx},{cy-95*s} V{cy-20*s} M{cx},{cy-20*s} L{cx-72*s},{cy+92*s} M{cx},{cy-20*s} L{cx},{cy+98*s} M{cx},{cy-20*s} L{cx+72*s},{cy+92*s}", "none", primary, 12 * s))
        parts.append(path(f"M{cx-110*s},{cy+96*s} Q{cx},{cy+60*s} {cx+110*s},{cy+96*s}", "none", accent, 10 * s))
        return "".join(parts)

    if key in {"ocean-light", "current"}:
        parts.append(rect(cx - 105 * s, cy - 75 * s, 210 * s, 155 * s, primary, dark, 5 * s, 5 * s))
        if key == "ocean-light":
            for i, color in enumerate(["#df5b52", "#edbd43", "#4d80c4"]):
                parts.append(line(cx - 70 * s + i * 45 * s, cy - 120 * s, cx - 35 * s + i * 25 * s, cy + (5 + i * 28) * s, color, 8 * s, True))
        else:
            parts.append(path(f"M{cx-90*s},{cy-20*s} Q{cx},{cy-80*s} {cx+85*s},{cy-20*s} M{cx+90*s},{cy+40*s} Q{cx},{cy+95*s} {cx-85*s},{cy+40*s}", "none", bg, 7 * s, True))
        return "".join(parts)

    if key == "filter-mouth":
        parts.append(path(f"M{cx-105*s},{cy-50*s} Q{cx},{cy+5*s} {cx+105*s},{cy-50*s} Q{cx},{cy+95*s} {cx-105*s},{cy-50*s} Z", primary, dark, 5 * s))
        for x in [-35, -10, 15, 40]:
            parts.append(line(cx + x * s, cy - 25 * s, cx + (x - 12) * s, cy + 55 * s, bg, 3 * s))
        return "".join(parts)

    if key == "turtle":
        parts.append(ellipse(cx, cy, 65 * s, 48 * s, primary, dark, 5 * s))
        parts.append(circle(cx + 78 * s, cy - 5 * s, 20 * s, accent, dark, 4 * s))
        for dx, dy in [(-48, -42), (-48, 42), (42, -42), (42, 42)]:
            parts.append(ellipse(cx + dx * s, cy + dy * s, 28 * s, 12 * s, accent, dark, 3 * s))
        return "".join(parts)

    if key == "microwave":
        parts.append(rect(cx - 95 * s, cy - 72 * s, 190 * s, 145 * s, bg, dark, 6 * s, 8 * s))
        parts.append(rect(cx - 70 * s, cy - 45 * s, 112 * s, 88 * s, primary, dark, 4 * s, 5 * s))
        parts.append(circle(cx + 68 * s, cy - 30 * s, 10 * s, accent, dark, 3 * s))
        parts.append(circle(cx + 68 * s, cy + 12 * s, 10 * s, accent, dark, 3 * s))
        return "".join(parts)

    if key == "polar-molecule":
        parts.append(circle(cx - 35 * s, cy, 30 * s, primary, dark, 4 * s))
        parts.append(circle(cx + 35 * s, cy, 22 * s, accent, dark, 4 * s))
        parts.append(line(cx - 5 * s, cy, cx + 13 * s, cy, dark, 5 * s))
        parts.append(path(f"M{cx-88*s},{cy-62*s} A{105*s},{82*s} 0 1 1 {cx+85*s},{cy-50*s}", "none", primary, 5 * s, True))
        return "".join(parts)

    if key in {"bread", "gas-bubble"}:
        parts.append(path(f"M{cx-90*s},{cy+65*s} Q{cx-88*s},{cy-65*s} {cx},{cy-72*s} Q{cx+88*s},{cy-65*s} {cx+90*s},{cy+65*s} Z", accent, dark, 5 * s))
        for dx, dy, r in [(-42, -10, 14), (5, -30, 18), (45, 15, 12), (-10, 35, 10)]:
            parts.append(circle(cx + dx * s, cy + dy * s, r * s, bg, dark, 2 * s))
        return "".join(parts)

    if key in {"sun", "star", "sun-ground", "sun-heat"}:
        for i in range(12):
            a = i * math.tau / 12
            parts.append(line(cx + math.cos(a) * 58 * s, cy + math.sin(a) * 58 * s, cx + math.cos(a) * 82 * s, cy + math.sin(a) * 82 * s, accent, 7 * s))
        parts.append(circle(cx, cy, 48 * s, accent, dark, 5 * s))
        if key == "sun-ground":
            parts.append(line(cx - 85 * s, cy + 82 * s, cx + 85 * s, cy + 82 * s, dark, 8 * s))
        return "".join(parts)

    if key in {"atom", "fusion", "air", "molecule", "collision", "matter", "charge", "salt", "enzyme", "dividing-cells", "cell", "fast-molecule"} or "particles" in key or "molecules" in key:
        colors = [primary, accent, dark]
        offsets = [(-42, -12), (0, -38), (42, -4), (-18, 34), (35, 42)]
        for i, (dx, dy) in enumerate(offsets):
            parts.append(circle(cx + dx * s, cy + dy * s, (13 if key != "fusion" else 17) * s, colors[i % 3], dark, 2.5 * s))
        if key in {"atom", "fusion"}:
            parts.append(ellipse(cx, cy, 86 * s, 35 * s, "none", primary, 3 * s))
            parts.append(ellipse(cx, cy, 35 * s, 86 * s, "none", accent, 3 * s))
        if key == "fusion":
            parts.append(path(f"M{cx-78*s},{cy+58*s} Q{cx},{cy+92*s} {cx+78*s},{cy+58*s}", "none", accent, 8 * s, True))
        return "".join(parts)

    if key in {"light", "blue-light", "radiation", "light-air", "light-water-air", "object-light", "window-light", "sun-wave", "sun-heat", "trapped-heat"}:
        for offset, color in [(-35, accent), (0, primary), (35, accent)]:
            parts.append(path(f"M{cx-82*s},{cy+offset*s} Q{cx-20*s},{cy+(offset-25)*s} {cx+82*s},{cy+offset*s}", "none", color, 7 * s, True))
        return "".join(parts)

    if key in {"eye", "eye-lens"}:
        parts.append(path(f"M{cx-88*s},{cy} Q{cx},{cy-72*s} {cx+88*s},{cy} Q{cx},{cy+72*s} {cx-88*s},{cy} Z", bg, dark, 5 * s))
        parts.append(circle(cx, cy, 38 * s, primary, dark, 4 * s))
        parts.append(circle(cx, cy, 16 * s, dark))
        if key == "eye-lens":
            parts.append(path(f"M{cx-120*s},{cy-48*s} L{cx-18*s},{cy} L{cx-120*s},{cy+48*s}", "none", accent, 5 * s))
        return "".join(parts)

    if key in {"lamp", "energy", "heat", "heat-out", "clock", "oscillator", "counter", "pressure"}:
        if key == "clock" or key == "counter":
            parts.append(circle(cx, cy, 70 * s, bg, dark, 5 * s))
            parts.append(line(cx, cy, cx, cy - 42 * s, primary, 7 * s))
            parts.append(line(cx, cy, cx + 36 * s, cy + 18 * s, accent, 7 * s))
        elif key == "oscillator":
            parts.append(path(f"M{cx-90*s},{cy} Q{cx-45*s},{cy-70*s} {cx},{cy} T{cx+90*s},{cy}", "none", primary, 9 * s))
        elif key in {"heat", "heat-out"}:
            parts.append(rect(cx - 22 * s, cy - 72 * s, 44 * s, 112 * s, bg, dark, 5 * s, 22 * s))
            parts.append(circle(cx, cy + 48 * s, 34 * s, accent, dark, 5 * s))
            parts.append(rect(cx - 10 * s, cy - 50 * s, 20 * s, 92 * s, accent, "none", 0, 9 * s))
            if key == "heat-out":
                parts.append(line(cx + 45 * s, cy - 25 * s, cx + 105 * s, cy - 65 * s, accent, 7 * s, True))
        else:
            parts.append(circle(cx, cy - 22 * s, 52 * s, accent, dark, 5 * s))
            parts.append(rect(cx - 32 * s, cy + 25 * s, 64 * s, 36 * s, primary, dark, 4 * s, 8 * s))
            for a in [-0.7, 0, 0.7]:
                parts.append(line(cx + math.sin(a) * 68 * s, cy - 30 * s - math.cos(a) * 68 * s, cx + math.sin(a) * 95 * s, cy - 30 * s - math.cos(a) * 95 * s, accent, 6 * s))
        return "".join(parts)

    if key in {"drop", "water", "rain", "vapor", "cool-water", "liquid-water", "beading-drop", "hydration", "salt-solution", "clean-water", "water-air"} or "water" in key:
        parts.append(path(f"M{cx},{cy-88*s} C{cx-22*s},{cy-45*s} {cx-65*s},{cy+5*s} {cx-65*s},{cy+42*s} A{65*s},{65*s} 0 0 0 {cx+65*s},{cy+42*s} C{cx+65*s},{cy+5*s} {cx+22*s},{cy-45*s} {cx},{cy-88*s} Z", primary, dark, 5 * s))
        parts.append(path(f"M{cx-34*s},{cy+18*s} Q{cx-15*s},{cy+45*s} {cx+5*s},{cy+20*s}", "none", bg, 7 * s))
        if key in {"vapor", "water-air"}:
            parts.append(path(f"M{cx-45*s},{cy-80*s} Q{cx-70*s},{cy-125*s} {cx-35*s},{cy-145*s} M{cx+15*s},{cy-85*s} Q{cx-5*s},{cy-125*s} {cx+25*s},{cy-150*s}", "none", accent, 6 * s))
        return "".join(parts)

    if key in {"cloud", "charge-cloud", "wavy-air", "air-layers", "warm-air", "cool-air", "tiny-drops", "merge-drops", "ionized-air"}:
        parts.extend([circle(cx - 55 * s, cy + 18 * s, 47 * s, bg, dark, 4 * s), circle(cx, cy - 12 * s, 64 * s, bg, dark, 4 * s), circle(cx + 62 * s, cy + 18 * s, 45 * s, bg, dark, 4 * s), rect(cx - 95 * s, cy + 15 * s, 190 * s, 58 * s, bg, dark, 4 * s, 28 * s)])
        if key == "charge-cloud":
            for dx in [-45, 0, 45]:
                parts.append(text_block(cx + dx * s, cy + 10 * s, "+" if dx < 0 else "−", accent if dx < 0 else primary, int(34 * s)))
        if key in {"warm-air", "cool-air"}:
            direction = -1 if key == "warm-air" else 1
            parts.append(line(cx, cy + 105 * s * direction, cx, cy + 70 * s * direction, accent, 7 * s, True))
        if key in {"tiny-drops", "merge-drops"}:
            for dx, dy in [(-55, -12), (-5, 15), (48, -22)]:
                parts.append(circle(cx + dx * s, cy + dy * s, (12 if key == "tiny-drops" else 24) * s, primary, dark, 2 * s))
        return "".join(parts)

    if key in {"earth", "half-earth", "tilted-earth", "rotation", "globe", "earth-cutaway", "earth-air", "earth-shadow", "earth-person"}:
        parts.append(circle(cx, cy, 72 * s, primary, dark, 5 * s))
        parts.append(path(f"M{cx-52*s},{cy-35*s} Q{cx-15*s},{cy-70*s} {cx+8*s},{cy-32*s} L{cx-5*s},{cy+4*s} L{cx-50*s},{cy+12*s} Z M{cx+18*s},{cy+12*s} Q{cx+65*s},{cy-8*s} {cx+55*s},{cy+45*s} L{cx+12*s},{cy+58*s} Z", accent, dark, 2.5 * s))
        if key == "half-earth":
            parts.append(path(f"M{cx},{cy-72*s} A{72*s},{72*s} 0 0 1 {cx},{cy+72*s} Z", dark, "none"))
        if key == "tilted-earth":
            parts.append(line(cx - 28 * s, cy + 92 * s, cx + 28 * s, cy - 92 * s, dark, 5 * s))
        if key in {"rotation", "earth-person"}:
            parts.append(path(f"M{cx-98*s},{cy+5*s} A{98*s},{82*s} 0 1 1 {cx+92*s},{cy-25*s}", "none", accent, 6 * s, True))
        if key in {"earth-cutaway", "earth-air"}:
            parts.append(circle(cx, cy, 48 * s, accent, dark, 3 * s))
            parts.append(circle(cx, cy, 23 * s, dark))
        return "".join(parts)

    if key in {"moon", "half-moon", "full-moon", "orbit", "orbiting-moon", "red-moon"}:
        color = accent if key == "red-moon" else bg
        parts.append(circle(cx, cy, 66 * s, color, dark, 5 * s))
        parts.append(circle(cx - 22 * s, cy - 18 * s, 11 * s, primary))
        parts.append(circle(cx + 24 * s, cy + 26 * s, 15 * s, primary))
        if key == "moon":
            parts.append(path(f"M{cx},{cy-66*s} A{66*s},{66*s} 0 0 1 {cx},{cy+66*s} Z", dark))
        if key == "half-moon":
            parts.append(path(f"M{cx},{cy-66*s} A{66*s},{66*s} 0 0 0 {cx},{cy+66*s} Z", dark))
        if key in {"orbit", "orbiting-moon"}:
            parts.append(path(f"M{cx-100*s},{cy} A{100*s},{72*s} 0 1 1 {cx+95*s},{cy-15*s}", "none", accent, 5 * s, True))
        return "".join(parts)

    if key in {"block", "shadow", "face", "hand", "hot-hand", "body", "step-one", "step-two", "balance", "no-belt", "seatbelt"}:
        parts.append(circle(cx, cy - 58 * s, 30 * s, accent, dark, 4 * s))
        parts.append(line(cx, cy - 25 * s, cx, cy + 48 * s, dark, 10 * s))
        parts.append(line(cx, cy, cx - 52 * s, cy + 30 * s, primary, 9 * s))
        parts.append(line(cx, cy, cx + 52 * s, cy + 25 * s, primary, 9 * s))
        parts.append(line(cx, cy + 48 * s, cx - 42 * s, cy + 100 * s, dark, 10 * s))
        parts.append(line(cx, cy + 48 * s, cx + 48 * s, cy + 95 * s, dark, 10 * s))
        if key == "shadow":
            parts.append(ellipse(cx + 52 * s, cy + 106 * s, 78 * s, 18 * s, dark))
        if key == "seatbelt":
            parts.append(line(cx - 28 * s, cy - 34 * s, cx + 45 * s, cy + 62 * s, accent, 11 * s))
        return "".join(parts)

    if key in {"mirror", "metal", "glass", "flat-map", "scale-symbol", "map", "solar-cell", "circuit", "closed-circuit", "battery", "microwave"}:
        if key == "battery":
            parts.append(rect(cx - 58 * s, cy - 72 * s, 116 * s, 144 * s, primary, dark, 5 * s, 14 * s))
            parts.append(rect(cx - 22 * s, cy - 91 * s, 44 * s, 20 * s, accent, dark, 3 * s, 4 * s))
        elif key in {"circuit", "closed-circuit"}:
            parts.append(path(f"M{cx-70*s},{cy-55*s} H{cx+70*s} V{cy+55*s} H{cx-70*s} Z", "none", dark, 7 * s))
            parts.append(circle(cx + 70 * s, cy, 20 * s, accent, dark, 3 * s))
        elif key == "solar-cell":
            parts.append(polygon([(cx - 82 * s, cy + 65 * s), (cx - 55 * s, cy - 65 * s), (cx + 82 * s, cy - 65 * s), (cx + 55 * s, cy + 65 * s)], primary, dark, 5 * s))
            for dx in [-30, 20]:
                parts.append(line(cx + dx * s, cy - 58 * s, cx + (dx - 20) * s, cy + 58 * s, bg, 3 * s))
        else:
            parts.append(rect(cx - 72 * s, cy - 92 * s, 144 * s, 184 * s, bg, dark, 5 * s, 5 * s))
            parts.append(line(cx - 55 * s, cy - 55 * s, cx + 45 * s, cy + 45 * s, primary, 6 * s))
        return "".join(parts)

    if key in {"seed", "seed-food", "embryo", "sprout", "root", "stem", "broad-leaf", "leaf", "vein", "tree", "growth-ring", "xylem", "lotus", "micro-bumps", "bent-plant", "stem-growth"}:
        if key in {"seed", "seed-food", "embryo"}:
            parts.append(ellipse(cx, cy, 80 * s, 58 * s, accent, dark, 5 * s))
            parts.append(path(f"M{cx-20*s},{cy+15*s} Q{cx},{cy-25*s} {cx+42*s},{cy-28*s}", "none", bg, 8 * s))
        elif key in {"tree", "growth-ring", "xylem"}:
            parts.append(circle(cx, cy, 80 * s, accent, dark, 5 * s))
            for r in [20, 40, 60]:
                parts.append(circle(cx, cy, r * s, "none", primary, 4 * s))
        else:
            parts.append(line(cx, cy + 78 * s, cx, cy - 35 * s, dark, 9 * s))
            parts.append(path(f"M{cx},{cy} Q{cx-82*s},{cy-78*s} {cx-82*s},{cy+18*s} Q{cx-35*s},{cy+52*s} {cx},{cy} Z", primary, dark, 4 * s))
            parts.append(path(f"M{cx},{cy-30*s} Q{cx+82*s},{cy-90*s} {cx+82*s},{cy+8*s} Q{cx+35*s},{cy+42*s} {cx},{cy-30*s} Z", accent, dark, 4 * s))
            if key == "root":
                for dx in [-40, -15, 15, 40]:
                    parts.append(line(cx, cy + 70 * s, cx + dx * s, cy + 120 * s, primary, 5 * s))
        return "".join(parts)

    if key in {"flower", "fruit", "bee", "insect", "spider", "ant-food", "ant-line", "dragonfly", "grasshopper", "caterpillar", "butterfly", "firefly-light", "varied-insects", "camouflage"}:
        if key == "flower" or key == "fruit":
            for i in range(6):
                a = i * math.tau / 6
                parts.append(circle(cx + math.cos(a) * 45 * s, cy + math.sin(a) * 45 * s, 30 * s, accent, dark, 3 * s))
            parts.append(circle(cx, cy, 30 * s, primary, dark, 4 * s))
        elif key == "spider":
            parts.extend([circle(cx, cy - 25 * s, 35 * s, primary, dark, 4 * s), circle(cx, cy + 30 * s, 47 * s, accent, dark, 4 * s)])
            for side in [-1, 1]:
                for dy in [-45, -15, 20, 55]:
                    parts.append(line(cx + side * 28 * s, cy + dy * s, cx + side * 95 * s, cy + (dy - 18) * s, dark, 5 * s))
        elif key in {"caterpillar", "butterfly"}:
            if key == "caterpillar":
                for i in range(5):
                    parts.append(circle(cx - 60 * s + i * 30 * s, cy, 25 * s, primary if i % 2 else accent, dark, 3 * s))
            else:
                parts.append(ellipse(cx - 45 * s, cy, 50 * s, 72 * s, accent, dark, 4 * s))
                parts.append(ellipse(cx + 45 * s, cy, 50 * s, 72 * s, primary, dark, 4 * s))
                parts.append(rect(cx - 8 * s, cy - 65 * s, 16 * s, 130 * s, dark, dark, 0, 8 * s))
        else:
            parts.append(ellipse(cx, cy, 62 * s, 42 * s, primary, dark, 4 * s))
            parts.append(circle(cx + 62 * s, cy - 10 * s, 28 * s, accent, dark, 4 * s))
            for side in [-1, 1]:
                for dy in [-25, 5, 35]:
                    parts.append(line(cx + side * 20 * s, cy + dy * s, cx + side * 75 * s, cy + (dy + 25) * s, dark, 4 * s))
            if key in {"bee", "dragonfly"}:
                parts.append(ellipse(cx - 20 * s, cy - 48 * s, 48 * s, 24 * s, bg, dark, 3 * s))
                parts.append(ellipse(cx + 20 * s, cy - 48 * s, 48 * s, 24 * s, bg, dark, 3 * s))
        return "".join(parts)

    if key in {"feather", "barbs", "hooklets", "wing", "bird-wing", "penguin-flipper", "oily-feather", "owl-wing", "woodpecker", "bird", "rabbit-ear"}:
        if key == "rabbit-ear":
            parts.append(ellipse(cx - 30 * s, cy, 30 * s, 90 * s, accent, dark, 5 * s))
            parts.append(ellipse(cx + 30 * s, cy, 30 * s, 90 * s, primary, dark, 5 * s))
        else:
            parts.append(path(f"M{cx-75*s},{cy+65*s} Q{cx-35*s},{cy-85*s} {cx+75*s},{cy-70*s} Q{cx+25*s},{cy+45*s} {cx-75*s},{cy+65*s} Z", bg, dark, 5 * s))
            parts.append(line(cx - 62 * s, cy + 53 * s, cx + 60 * s, cy - 58 * s, primary, 6 * s))
            for i in range(5):
                parts.append(line(cx - 35 * s + i * 18 * s, cy + 25 * s - i * 17 * s, cx - 72 * s + i * 12 * s, cy - 5 * s - i * 14 * s, accent, 3 * s))
        return "".join(parts)

    if key in {"fish", "gill", "fish-bladder", "dolphin", "squid", "jellyfish-open", "jellyfish-close", "turtle", "coral", "octopus-skin"}:
        if key.startswith("jellyfish"):
            parts.append(path(f"M{cx-70*s},{cy+5*s} Q{cx-58*s},{cy-85*s} {cx},{cy-90*s} Q{cx+58*s},{cy-85*s} {cx+70*s},{cy+5*s} Z", primary, dark, 5 * s))
            for dx in [-45, -15, 15, 45]:
                parts.append(path(f"M{cx+dx*s},{cy+5*s} Q{cx+(dx-18)*s},{cy+55*s} {cx+dx*s},{cy+95*s}", "none", accent, 5 * s))
        elif key == "coral":
            parts.append(path(f"M{cx},{cy+88*s} V{cy-20*s} M{cx},{cy+15*s} L{cx-58*s},{cy-50*s} M{cx},{cy+35*s} L{cx+60*s},{cy-38*s} M{cx-58*s},{cy-50*s} L{cx-75*s},{cy-90*s} M{cx+60*s},{cy-38*s} L{cx+80*s},{cy-82*s}", "none", accent, 15 * s))
        else:
            parts.append(ellipse(cx, cy, 78 * s, 48 * s, primary, dark, 5 * s))
            parts.append(polygon([(cx - 75 * s, cy), (cx - 125 * s, cy - 52 * s), (cx - 125 * s, cy + 52 * s)], accent, dark, 4 * s))
            parts.append(circle(cx + 45 * s, cy - 12 * s, 8 * s, dark))
            if key == "gill":
                for dx in [5, 20, 35]:
                    parts.append(path(f"M{cx+dx*s},{cy-30*s} Q{cx+(dx+12)*s},{cy} {cx+dx*s},{cy+30*s}", "none", accent, 4 * s))
            if key == "fish-bladder":
                parts.append(ellipse(cx, cy + 5 * s, 33 * s, 16 * s, bg, dark, 3 * s))
        return "".join(parts)

    if key in {"heart", "pacemaker", "heart-signal", "vessel", "blood-oxygen", "lung", "alveoli", "gas-exchange", "brain", "spinal", "cochlea", "ear", "skin", "hair", "follicle", "bone", "compact-bone", "spongy-marrow", "shoulder", "knee", "elbow", "muscle", "inhale", "exhale"}:
        if key in {"heart", "pacemaker", "heart-signal"}:
            parts.append(path(f"M{cx},{cy+82*s} C{cx-90*s},{cy+15*s} {cx-90*s},{cy-70*s} {cx-32*s},{cy-72*s} C{cx},{cy-72*s} {cx},{cy-35*s} {cx},{cy-25*s} C{cx},{cy-35*s} {cx},{cy-72*s} {cx+32*s},{cy-72*s} C{cx+90*s},{cy-70*s} {cx+90*s},{cy+15*s} {cx},{cy+82*s} Z", accent, dark, 5 * s))
            if key != "heart":
                parts.append(path(f"M{cx-65*s},{cy} L{cx-30*s},{cy} L{cx-15*s},{cy-35*s} L{cx+12*s},{cy+30*s} L{cx+30*s},{cy} L{cx+70*s},{cy}", "none", bg, 6 * s))
        elif key in {"lung", "alveoli", "gas-exchange", "inhale", "exhale"}:
            parts.append(line(cx, cy - 80 * s, cx, cy - 10 * s, dark, 10 * s))
            parts.append(path(f"M{cx-8*s},{cy-10*s} C{cx-35*s},{cy-60*s} {cx-105*s},{cy-28*s} {cx-92*s},{cy+65*s} C{cx-82*s},{cy+120*s} {cx-22*s},{cy+95*s} {cx-8*s},{cy+40*s} Z", primary, dark, 4 * s))
            parts.append(path(f"M{cx+8*s},{cy-10*s} C{cx+35*s},{cy-60*s} {cx+105*s},{cy-28*s} {cx+92*s},{cy+65*s} C{cx+82*s},{cy+120*s} {cx+22*s},{cy+95*s} {cx+8*s},{cy+40*s} Z", accent, dark, 4 * s))
            if key in {"alveoli", "gas-exchange"}:
                for dx, dy in [(-62, 10), (-38, 40), (42, 18), (68, 45)]:
                    parts.append(circle(cx + dx * s, cy + dy * s, 14 * s, bg, dark, 2 * s))
            if key == "inhale":
                parts.append(path(f"M{cx-92*s},{cy+112*s} Q{cx},{cy+135*s} {cx+92*s},{cy+112*s}", "none", dark, 6 * s))
                parts.append(line(cx + 115 * s, cy + 30 * s, cx + 115 * s, cy + 105 * s, accent, 6 * s, True))
            elif key == "exhale":
                parts.append(path(f"M{cx-92*s},{cy+112*s} Q{cx},{cy+55*s} {cx+92*s},{cy+112*s}", "none", dark, 6 * s))
                parts.append(line(cx + 115 * s, cy + 105 * s, cx + 115 * s, cy + 30 * s, accent, 6 * s, True))
        elif key == "brain" or key == "spinal":
            parts.extend([circle(cx - 28 * s, cy - 15 * s, 58 * s, primary, dark, 4 * s), circle(cx + 35 * s, cy - 20 * s, 62 * s, accent, dark, 4 * s), circle(cx + 5 * s, cy + 40 * s, 55 * s, primary, dark, 4 * s)])
            if key == "spinal":
                parts.append(line(cx + 8 * s, cy + 85 * s, cx + 8 * s, cy + 140 * s, dark, 12 * s))
        elif key in {"ear", "cochlea"}:
            parts.append(path(f"M{cx+20*s},{cy-85*s} C{cx-75*s},{cy-100*s} {cx-105*s},{cy-15*s} {cx-65*s},{cy+38*s} C{cx-35*s},{cy+78*s} {cx-5*s},{cy+45*s} {cx-8*s},{cy+88*s}", "none", dark, 13 * s))
            parts.append(path(f"M{cx+18*s},{cy-42*s} C{cx-35*s},{cy-48*s} {cx-42*s},{cy+8*s} {cx-12*s},{cy+15*s} C{cx+28*s},{cy+25*s} {cx+45*s},{cy-8*s} {cx+18*s},{cy-18*s}", "none", accent, 7 * s))
        elif key in {"bone", "compact-bone", "spongy-marrow", "shoulder", "knee", "elbow"}:
            parts.append(circle(cx - 55 * s, cy - 55 * s, 30 * s, bg, dark, 4 * s))
            parts.append(circle(cx + 55 * s, cy + 55 * s, 30 * s, bg, dark, 4 * s))
            parts.append(line(cx - 38 * s, cy - 38 * s, cx + 38 * s, cy + 38 * s, primary, 32 * s))
            parts.append(line(cx - 38 * s, cy - 38 * s, cx + 38 * s, cy + 38 * s, dark, 4 * s))
        elif key in {"skin", "hair", "follicle"}:
            parts.append(rect(cx - 90 * s, cy - 65 * s, 180 * s, 42 * s, accent, dark, 3 * s, 3 * s))
            parts.append(rect(cx - 90 * s, cy - 23 * s, 180 * s, 90 * s, primary, dark, 3 * s, 3 * s))
            parts.append(line(cx, cy - 100 * s, cx + 18 * s, cy + 78 * s, dark, 7 * s))
            parts.append(ellipse(cx + 20 * s, cy + 70 * s, 22 * s, 40 * s, bg, dark, 3 * s))
        else:
            parts.append(circle(cx, cy, 66 * s, primary, dark, 5 * s))
            parts.append(line(cx - 55 * s, cy + 55 * s, cx + 55 * s, cy - 55 * s, accent, 14 * s))
        return "".join(parts)

    if key in {"lever", "seesaw", "pulley", "rope-segments", "small-gear", "large-gear", "gear-contact", "incline", "ramp", "steep-lift", "screw", "wedge", "spring", "pressed-spring", "magnet", "field", "iron", "coil", "motor", "generator", "lift"}:
        if key in {"lever", "seesaw"}:
            parts.append(polygon([(cx - 25 * s, cy + 62 * s), (cx + 25 * s, cy + 62 * s), (cx, cy + 12 * s)], accent, dark, 4 * s))
            parts.append(line(cx - 95 * s, cy - 18 * s, cx + 95 * s, cy + 35 * s, primary, 13 * s))
        elif key in {"pulley", "rope-segments", "lift"}:
            parts.append(circle(cx, cy - 25 * s, 48 * s, bg, dark, 9 * s))
            parts.append(path(f"M{cx-50*s},{cy-25*s} V{cy+85*s} M{cx+50*s},{cy-25*s} V{cy+85*s}", "none", primary, 7 * s))
            parts.append(rect(cx + 18 * s, cy + 62 * s, 64 * s, 45 * s, accent, dark, 4 * s, 5 * s))
        elif "gear" in key:
            parts.append(circle(cx, cy, 65 * s, primary if key != "large-gear" else accent, dark, 9 * s))
            parts.append(circle(cx, cy, 21 * s, bg, dark, 5 * s))
            for i in range(8):
                a = i * math.tau / 8
                parts.append(line(cx + math.cos(a) * 64 * s, cy + math.sin(a) * 64 * s, cx + math.cos(a) * 84 * s, cy + math.sin(a) * 84 * s, dark, 14 * s))
        elif key in {"incline", "ramp", "steep-lift"}:
            height = 95 if key == "steep-lift" else 65
            parts.append(polygon([(cx - 90 * s, cy + 70 * s), (cx + 90 * s, cy + 70 * s), (cx + 90 * s, cy - height * s)], bg, dark, 5 * s))
            parts.append(circle(cx + 35 * s, cy - 5 * s, 27 * s, accent, dark, 4 * s))
        elif key == "screw":
            parts.append(line(cx, cy - 95 * s, cx, cy + 95 * s, dark, 18 * s))
            for y in range(-70, 71, 28):
                parts.append(path(f"M{cx-50*s},{cy+y*s} Q{cx},{cy+(y+20)*s} {cx+50*s},{cy+y*s}", "none", primary, 6 * s))
        elif key == "wedge":
            parts.append(polygon([(cx - 20 * s, cy - 90 * s), (cx + 60 * s, cy + 85 * s), (cx - 60 * s, cy + 85 * s)], accent, dark, 5 * s))
        elif key in {"spring", "pressed-spring"}:
            width = 45 if key == "pressed-spring" else 80
            data = f"M{cx},{cy-95*s} " + " ".join(f"L{cx + (width if i % 2 else -width)*s},{cy + (-70+i*28)*s}" for i in range(6)) + f" L{cx},{cy+95*s}"
            parts.append(path(data, "none", primary, 8 * s))
        elif key in {"magnet", "field", "iron"}:
            parts.append(path(f"M{cx-68*s},{cy-75*s} V{cy+15*s} C{cx-68*s},{cy+105*s} {cx+68*s},{cy+105*s} {cx+68*s},{cy+15*s} V{cy-75*s}", "none", primary, 28 * s))
            if key == "field":
                parts.append(path(f"M{cx-100*s},{cy-35*s} Q{cx},{cy-135*s} {cx+100*s},{cy-35*s} M{cx-105*s},{cy+15*s} Q{cx},{cy+150*s} {cx+105*s},{cy+15*s}", "none", accent, 4 * s))
        else:
            parts.append(circle(cx, cy, 70 * s, bg, dark, 5 * s))
            parts.append(circle(cx, cy, 28 * s, accent, dark, 4 * s))
            parts.append(path(f"M{cx-95*s},{cy} A{95*s},{75*s} 0 1 1 {cx+88*s},{cy-20*s}", "none", primary, 6 * s, True))
        return "".join(parts)

    if key in {"mountain-quake", "mountain", "valley", "volcano-conduit", "eruption", "fault", "slip", "seismic-wave", "plates", "plate-motion", "weathering", "glacier", "ice-flow", "delta", "landscape", "gravity-center"}:
        parts.append(polygon([(cx - 105 * s, cy + 82 * s), (cx - 20 * s, cy - 82 * s), (cx + 80 * s, cy + 82 * s)], primary, dark, 5 * s))
        parts.append(polygon([(cx + 10 * s, cy + 82 * s), (cx + 70 * s, cy - 35 * s), (cx + 120 * s, cy + 82 * s)], accent, dark, 4 * s))
        if key in {"volcano-conduit", "eruption"}:
            parts.append(path(f"M{cx-20*s},{cy-82*s} Q{cx+10*s},{cy-125*s} {cx+42*s},{cy-85*s} M{cx-10*s},{cy-72*s} L{cx+25*s},{cy+70*s}", "none", accent, 10 * s))
        if key in {"fault", "slip", "seismic-wave"}:
            parts.append(path(f"M{cx-80*s},{cy+65*s} L{cx-15*s},{cy-5*s} L{cx+18*s},{cy+28*s} L{cx+85*s},{cy-45*s}", "none", accent, 8 * s))
            parts.append(path(f"M{cx-115*s},{cy+100*s} Q{cx},{cy+135*s} {cx+115*s},{cy+100*s}", "none", primary, 5 * s))
        return "".join(parts)

    if key in {"satellite", "receiver", "planet", "comet", "icy-comet", "comet-tail", "mercury-day", "mercury-night", "venus", "saturn", "ring-pieces", "small-orbits", "meteoroid", "meteor", "meteorite", "impactor", "impact", "crater", "rocket", "orbiting-moon"}:
        if key == "rocket":
            parts.append(path(f"M{cx},{cy-100*s} C{cx-62*s},{cy-38*s} {cx-55*s},{cy+45*s} {cx},{cy+85*s} C{cx+55*s},{cy+45*s} {cx+62*s},{cy-38*s} {cx},{cy-100*s} Z", bg, dark, 5 * s))
            parts.append(circle(cx, cy - 15 * s, 24 * s, primary, dark, 3 * s))
            parts.append(polygon([(cx - 24 * s, cy + 78 * s), (cx, cy + 135 * s), (cx + 24 * s, cy + 78 * s)], accent))
        elif key in {"comet", "icy-comet", "comet-tail", "meteor", "meteoroid", "impactor"}:
            parts.append(path(f"M{cx-100*s},{cy+45*s} Q{cx-15*s},{cy-75*s} {cx+60*s},{cy-25*s}", "none", accent, 22 * s))
            parts.append(circle(cx + 65 * s, cy - 28 * s, 38 * s, primary, dark, 5 * s))
        elif key in {"saturn", "ring-pieces", "small-orbits"}:
            parts.append(ellipse(cx, cy, 110 * s, 34 * s, "none", accent, 16 * s))
            parts.append(circle(cx, cy, 57 * s, primary, dark, 5 * s))
        elif key in {"satellite", "receiver"}:
            parts.append(rect(cx - 34 * s, cy - 28 * s, 68 * s, 56 * s, accent, dark, 4 * s, 5 * s))
            parts.append(rect(cx - 105 * s, cy - 42 * s, 68 * s, 84 * s, primary, dark, 4 * s, 4 * s))
            parts.append(rect(cx + 37 * s, cy - 42 * s, 68 * s, 84 * s, primary, dark, 4 * s, 4 * s))
        elif key in {"impact", "crater", "meteorite"}:
            parts.append(ellipse(cx, cy + 20 * s, 90 * s, 42 * s, bg, dark, 6 * s))
            parts.append(circle(cx, cy + 15 * s, 36 * s, primary, dark, 4 * s))
        else:
            color = accent if "day" in key or key == "venus" else primary
            parts.append(circle(cx, cy, 67 * s, color, dark, 5 * s))
            parts.append(circle(cx - 25 * s, cy - 20 * s, 12 * s, bg))
            parts.append(circle(cx + 28 * s, cy + 25 * s, 16 * s, bg))
        return "".join(parts)

    if key in {"oil-molecules", "salt-crystal", "ice-lattice", "folded-protein", "protein-network", "yeast-sugar", "gas-bubble", "bread", "paper-fibers", "capillary", "wet-paper", "thermos", "vacuum-wall", "lid", "evaporator", "compressor", "condenser", "valve", "polar-molecule"}:
        if key in {"thermos", "vacuum-wall", "lid"}:
            parts.append(path(f"M{cx-58*s},{cy-95*s} H{cx+58*s} L{cx+45*s},{cy+95*s} H{cx-45*s} Z", bg, dark, 6 * s))
            parts.append(path(f"M{cx-38*s},{cy-72*s} H{cx+38*s} L{cx+28*s},{cy+72*s} H{cx-28*s} Z", primary, dark, 3 * s))
        elif key in {"paper-fibers", "capillary", "wet-paper"}:
            for i in range(6):
                y = cy - 65 * s + i * 26 * s
                parts.append(path(f"M{cx-90*s},{y} Q{cx-25*s},{y-20*s} {cx+90*s},{y+8*s}", "none", primary if i % 2 else accent, 6 * s))
        elif key in {"evaporator", "compressor", "condenser", "valve"}:
            parts.append(path(f"M{cx-78*s},{cy-55*s} H{cx+55*s} V{cy+55*s} H{cx-55*s} V{cy-25*s} H{cx+28*s}", "none", primary, 9 * s, True))
            parts.append(circle(cx - 35 * s, cy + 45 * s, 18 * s, accent, dark, 3 * s))
        else:
            for i in range(7):
                a = i * math.tau / 7
                parts.append(circle(cx + math.cos(a) * 55 * s, cy + math.sin(a) * 48 * s, 15 * s, primary if i % 2 else accent, dark, 2 * s))
                if key == "protein-network" and i:
                    parts.append(line(cx, cy, cx + math.cos(a) * 55 * s, cy + math.sin(a) * 48 * s, dark, 3 * s))
        return "".join(parts)

    if key in {"boat", "sailboat", "airplane-wing", "rotor", "hot-balloon", "bicycle", "train-wheel", "power-plant", "transformer-up", "substation", "home", "surface-sub", "deep-sub", "ballast-air", "ballast-water"}:
        if key in {"boat", "sailboat"}:
            parts.append(path(f"M{cx-100*s},{cy+20*s} H{cx+100*s} L{cx+62*s},{cy+78*s} H{cx-65*s} Z", primary, dark, 5 * s))
            if key == "sailboat":
                parts.append(line(cx, cy + 18 * s, cx, cy - 100 * s, dark, 6 * s))
                parts.append(polygon([(cx, cy - 90 * s), (cx, cy + 5 * s), (cx + 75 * s, cy)], bg, dark, 4 * s))
        elif key == "airplane-wing" or key == "rotor":
            parts.append(path(f"M{cx-100*s},{cy+20*s} Q{cx-10*s},{cy-65*s} {cx+100*s},{cy} Q{cx},{cy+45*s} {cx-100*s},{cy+20*s} Z", bg, dark, 5 * s))
            if key == "rotor":
                parts.append(line(cx - 115 * s, cy, cx + 115 * s, cy, primary, 8 * s))
        elif key == "hot-balloon":
            parts.append(ellipse(cx, cy - 20 * s, 72 * s, 92 * s, accent, dark, 5 * s))
            parts.append(line(cx - 35 * s, cy + 60 * s, cx - 22 * s, cy + 105 * s, dark, 4 * s))
            parts.append(line(cx + 35 * s, cy + 60 * s, cx + 22 * s, cy + 105 * s, dark, 4 * s))
            parts.append(rect(cx - 30 * s, cy + 100 * s, 60 * s, 38 * s, primary, dark, 4 * s, 5 * s))
        elif key == "bicycle":
            parts.append(circle(cx - 62 * s, cy + 30 * s, 48 * s, "none", dark, 6 * s))
            parts.append(circle(cx + 62 * s, cy + 30 * s, 48 * s, "none", dark, 6 * s))
            parts.append(path(f"M{cx-62*s},{cy+30*s} L{cx},{cy-55*s} L{cx+25*s},{cy+30*s} Z M{cx},{cy-55*s} L{cx+62*s},{cy+30*s}", "none", primary, 7 * s))
        elif key == "train-wheel":
            parts.append(circle(cx, cy + 25 * s, 58 * s, dark))
            parts.append(circle(cx, cy + 25 * s, 26 * s, accent))
            parts.append(line(cx - 105 * s, cy + 88 * s, cx + 105 * s, cy + 88 * s, dark, 12 * s))
        elif key in {"surface-sub", "deep-sub", "ballast-air", "ballast-water"}:
            parts.append(ellipse(cx, cy, 100 * s, 48 * s, primary, dark, 5 * s))
            parts.append(rect(cx - 25 * s, cy - 62 * s, 50 * s, 32 * s, primary, dark, 4 * s, 8 * s))
            parts.append(circle(cx - 42 * s, cy, 12 * s, bg, dark, 2 * s))
            parts.append(circle(cx + 5 * s, cy, 12 * s, bg, dark, 2 * s))
        else:
            parts.append(rect(cx - 85 * s, cy - 55 * s, 170 * s, 110 * s, bg, dark, 5 * s, 6 * s))
            parts.append(path(f"M{cx-100*s},{cy-55*s} L{cx},{cy-115*s} L{cx+100*s},{cy-55*s}", "none", accent, 8 * s))
        return "".join(parts)

    # Abstract fallback: connected observations, used only when the label carries
    # the exact scientific distinction and a pictogram would imply false detail.
    FALLBACK_KEYS.add(key)
    parts.append(circle(cx, cy, 22 * s, accent, dark, 3 * s))
    for a in [0, math.tau / 3, 2 * math.tau / 3]:
        x = cx + math.cos(a) * 72 * s
        y = cy + math.sin(a) * 72 * s
        parts.append(line(cx, cy, x, y, primary, 5 * s))
        parts.append(circle(x, y, 18 * s, bg, dark, 3 * s))
    return "".join(parts)


def node_card(node: tuple[str, str], x: float, y: float, w: float, h: float, palette: tuple[str, str, str, str], scale: float = 1.0) -> str:
    key, label = node
    bg, primary, accent, dark = palette
    parts = [rect(x - w / 2, y - h / 2, w, h, "#ffffff", dark, 2.5, 8)]
    parts.append(icon(key, x, y - 35, scale, primary, accent, dark, bg))
    label_offset = 58 if h <= 220 else 70 if h <= 250 else 105
    parts.append(text_block(x, y + label_offset, label, dark, 26, "middle", 650))
    return "".join(parts)


def render_flow(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    _, primary, accent, dark = palette
    xs = [210, 600, 990]
    parts = [node_card(node, x, 335, 265, 390, palette, 0.78) for node, x in zip(diagram.nodes, xs)]
    parts.append(line(352, 335, 452, 335, primary, 9, True))
    parts.append(line(742, 335, 842, 335, accent, 9, True))
    return "".join(parts)


def render_compare(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    _, primary, accent, dark = palette
    count = len(diagram.nodes)
    xs = [300, 600, 900] if count == 3 else [350, 850]
    width = 270 if count == 3 else 360
    parts = [node_card(node, x, 335, width, 395, palette, 0.9) for node, x in zip(diagram.nodes, xs)]
    for left, right in zip(xs, xs[1:]):
        parts.append(line((left + right) / 2, 170, (left + right) / 2, 500, dark, 3, False, "10 12"))
    parts.append(text_block(600, 570, "并排比较：结构或条件不同，结果也会不同", dark, 23, "middle", 500))
    return "".join(parts)


def render_cycle(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    _, primary, accent, dark = palette
    positions = [(600, 175), (900, 340), (600, 500), (300, 340)] if len(diagram.nodes) == 4 else [(600, 175), (860, 455), (340, 455)]
    parts: list[str] = []
    for i, (node, (x, y)) in enumerate(zip(diagram.nodes, positions)):
        parts.append(node_card(node, x, y, 230, 210, palette, 0.52))
        nx, ny = positions[(i + 1) % len(positions)]
        dx, dy = nx - x, ny - y
        length = math.hypot(dx, dy)
        start_x, start_y = x + dx / length * 125, y + dy / length * 110
        end_x, end_y = nx - dx / length * 135, ny - dy / length * 120
        parts.append(line(start_x, start_y, end_x, end_y, primary if i % 2 == 0 else accent, 7, True))
    return "".join(parts)


def render_forces(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    bg, primary, accent, dark = palette
    center = diagram.nodes[0]
    parts = [node_card(center, 600, 330, 350, 300, palette, 0.9)]
    if len(diagram.nodes) > 1:
        parts.append(line(520, 245, 520, 115, primary, 10, True))
        parts.append(text_block(500, 88, diagram.nodes[1][1], dark, 25, "middle", 650))
    if len(diagram.nodes) > 2:
        parts.append(line(680, 415, 680, 545, accent, 10, True))
        parts.append(text_block(700, 585, diagram.nodes[2][1], dark, 25, "middle", 650))
    parts.append(rect(80, 215, 215, 225, bg, dark, 2.5, 8))
    parts.append(text_block(187, 280, "箭头表示力或运动方向", dark, 24, "middle", 600))
    parts.append(path("M135,385 H240", "none", primary, 9, True))
    return "".join(parts)


def render_orbit(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    bg, primary, accent, dark = palette
    parts = [ellipse(600, 340, 345, 205, "none", primary, 5), ellipse(600, 340, 245, 145, "none", accent, 4)]
    positions = [(600, 340), (355, 215), (835, 230), (700, 500)]
    for i, (node, (x, y)) in enumerate(zip(diagram.nodes, positions)):
        parts.append(circle(x, y, 82 if i == 0 else 66, "#ffffff", dark, 2.5))
        parts.append(icon(node[0], x, y - 7, 0.48 if i else 0.58, primary, accent, dark, bg))
        parts.append(text_block(x, y + (112 if i == 0 else 92), node[1], dark, 24, "middle", 650))
    parts.append(path("M285,390 A345,205 0 0 0 465,520", "none", primary, 7, True))
    return "".join(parts)


def render_network(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    _, primary, accent, dark = palette
    if len(diagram.nodes) == 4:
        positions = [(600, 300), (230, 175), (970, 175), (600, 535)]
        dimensions = [(260, 190), (225, 180), (225, 180), (225, 180)]
        scales = [0.52, 0.44, 0.44, 0.44]
    else:
        positions = [(600, 340), (250, 190), (950, 190)]
        dimensions = [(285, 245), (245, 210), (245, 210)]
        scales = [0.62, 0.5, 0.5]
    parts: list[str] = []
    for i, (node, (x, y), (width, height), scale) in enumerate(zip(diagram.nodes, positions, dimensions, scales)):
        if i:
            parts.append(line(positions[0][0], positions[0][1], x, y, primary if i % 2 else accent, 7, False))
        parts.append(node_card(node, x, y, width, height, palette, scale))
    return "".join(parts)


def render_layers(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    bg, primary, accent, dark = palette
    radii = [(255, 190), (180, 132), (105, 78)]
    colors = [bg, "#ffffff", bg]
    parts: list[str] = []
    for i, ((_, _), (rx, ry), color) in enumerate(zip(diagram.nodes, radii, colors)):
        parts.append(ellipse(480, 335, rx, ry, color, [dark, primary, accent][i], 4))
    parts.append(icon(diagram.nodes[0][0], 480, 335, 0.6, primary, accent, dark, bg))
    for i, ((_, label), (_, _), _) in enumerate(zip(diagram.nodes, radii, colors)):
        label_y = 190 + i * 125
        parts.append(line(650 - i * 60, 250 + i * 48, 795, label_y, [dark, primary, accent][i], 4))
        parts.append(text_block(815, label_y - 8, label, dark, 25, "start", 650))
    parts.append(text_block(480, 575, "由外到内或由整体到细部", dark, 23, "middle", 500))
    return "".join(parts)


def render_moon_phases(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    bg, primary, accent, dark = palette
    parts = [
        ellipse(675, 335, 300, 205, "none", primary, 5),
        icon("sun", 125, 335, 0.72, primary, accent, dark, bg),
        icon("earth", 675, 335, 0.72, primary, accent, dark, bg),
    ]
    for y in [265, 335, 405]:
        parts.append(line(205, y, 1090, y, accent, 3, True))
    moons = [
        (375, 335, "新月附近"),
        (675, 130, "弦月附近"),
        (975, 335, "满月附近"),
        (675, 500, "另一侧弦月"),
    ]
    for x, y, label in moons:
        parts.append(circle(x, y, 34, dark, dark, 3))
        parts.append(path(f"M{x},{y-34} A34,34 0 0 0 {x},{y+34} Z", bg, "none"))
        parts.append(text_block(x, y + (62 if y < 500 else 55), label, dark, 22, "middle", 650))
    parts.append(text_block(675, 625, "月亮受光的一半始终朝向太阳，地球上的观察方向不断改变", dark, 24, "middle", 550))
    return "".join(parts)


def render_seasons(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    bg, primary, accent, dark = palette
    parts = [
        ellipse(600, 335, 390, 205, "none", primary, 5),
        icon("sun", 600, 335, 0.85, primary, accent, dark, bg),
    ]
    for x, label in [(255, "北半球朝向太阳"), (945, "北半球背向太阳")]:
        parts.append(icon("earth", x, 335, 0.68, primary, accent, dark, bg))
        parts.append(line(x - 28, 415, x + 28, 255, dark, 6))
        parts.append(text_block(x, 475, label, dark, 24, "middle", 650))
    for x1, x2 in [(515, 350), (685, 850)]:
        for y in [295, 335, 375]:
            parts.append(line(x1, y, x2, y, accent, 4, True))
    parts.append(path("M250,155 A390,205 0 0 1 950,155", "none", primary, 7, True))
    parts.append(text_block(600, 585, "地轴方向近似不变｜绕行使阳光角度和白昼长短改变", dark, 24, "middle", 550))
    return "".join(parts)


def render_tides(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    bg, primary, accent, dark = palette
    parts = [
        line(225, 335, 500, 335, accent, 4, False, "12 10"),
        line(655, 335, 920, 335, primary, 5, False, "12 10"),
        icon("sun", 125, 335, 0.58, primary, accent, dark, bg),
        ellipse(575, 335, 145, 78, primary, primary, 4),
        icon("earth", 575, 335, 0.64, primary, accent, dark, bg),
        icon("moon", 1010, 335, 0.58, primary, accent, dark, bg),
        text_block(125, 455, "太阳也参与", dark, 23, "middle", 650),
        text_block(575, 475, "海洋沿地月方向形成两个潮汐隆起", dark, 24, "middle", 650),
        text_block(1010, 455, "月球是主要作用者", dark, 23, "middle", 650),
        path("M470,545 Q575,600 680,545", "none", accent, 7, True),
        text_block(575, 610, "地球自转时｜海岸轮流经过较高和较低水位", dark, 23, "middle", 550),
    ]
    return "".join(parts)


def render_orbit_force(diagram: Diagram, palette: tuple[str, str, str, str]) -> str:
    bg, primary, accent, dark = palette
    parts = [
        ellipse(505, 340, 355, 215, "none", primary, 5),
        icon("earth", 505, 340, 0.85, primary, accent, dark, bg),
        icon("moon", 860, 340, 0.63, primary, accent, dark, bg),
        line(850, 275, 850, 125, accent, 10, True),
        text_block(850, 92, "原有速度指向轨道切线", dark, 25, "middle", 650),
        line(790, 340, 620, 340, primary, 10, True),
        text_block(705, 305, "地球引力指向地心", dark, 25, "middle", 650),
        path("M150,340 A355,215 0 0 0 505,555", "none", primary, 7, True),
        text_block(505, 610, "向前运动不断被引力拉弯｜月球便持续绕地球下落", dark, 24, "middle", 550),
    ]
    return "".join(parts)


RENDERERS = {
    "flow": render_flow,
    "compare": render_compare,
    "cycle": render_cycle,
    "forces": render_forces,
    "orbit": render_orbit,
    "network": render_network,
    "layers": render_layers,
}

SPECIAL_RENDERERS = {
    12: render_moon_phases,
    17: render_seasons,
    246: render_tides,
    296: render_orbit_force,
}


def svg_for(diagram: Diagram, title: str) -> str:
    month_index = next(i for i, upper in enumerate([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) if diagram.day <= upper)
    palette = PALETTES[month_index]
    bg, primary, accent, dark = palette
    kind_label = {
        "flow": "过程图",
        "compare": "比较图",
        "cycle": "循环图",
        "forces": "方向与受力图",
        "orbit": "位置关系图",
        "network": "协同关系图",
        "layers": "结构图",
    }[diagram.kind]
    body = SPECIAL_RENDERERS.get(diagram.day, RENDERERS[diagram.kind])(diagram, palette)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680" role="img" aria-labelledby="title desc">
  <title id="title">原理图：{esc(title)}</title>
  <desc id="desc">{esc(diagram.caption)}</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{dark}"/></marker>
  </defs>
  <rect width="1200" height="680" fill="{bg}"/>
  <path d="M0,615 Q260,560 520,615 T1040,615 T1300,615 V680 H0 Z" fill="{primary}" opacity="0.09"/>
  <text x="65" y="64" fill="{dark}" font-family="system-ui, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="30" font-weight="750">{kind_label}</text>
  <text x="1135" y="64" text-anchor="end" fill="{dark}" opacity="0.72" font-family="system-ui, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="21">示意图 · 不按比例</text>
  {body}
  <rect x="12" y="12" width="1176" height="656" rx="8" fill="none" stroke="{dark}" stroke-width="3" opacity="0.22"/>
</svg>
'''


def titles_by_day() -> dict[int, str]:
    titles: dict[int, str] = {}
    heading_re = re.compile(r"^## 第 (\d{3}) 天｜(.+)$", re.MULTILINE)
    for path in MONTHS:
        for day, title in heading_re.findall(path.read_text(encoding="utf-8")):
            titles[int(day)] = title
    return titles


def embed_diagrams() -> None:
    selected = {diagram.day: diagram for diagram in DIAGRAMS}
    old_block_re = re.compile(
        r"\n\n!\[原理图：[^\]]+\]\(images/explainers/day-\d{3}\.svg\)"
        r"\n\n\*图解：.*?\*",
    )
    for path in MONTHS:
        text = old_block_re.sub("", path.read_text(encoding="utf-8"))
        days = [int(day) for day in re.findall(r"^## 第 (\d{3}) 天｜", text, re.MULTILINE)]
        for day in days:
            if day not in selected:
                continue
            diagram = selected[day]
            pattern = re.compile(
                rf"(^## 第 {day:03d} 天｜.+?$.*?^\*\*再想一步：\*\* .+?$)",
                re.MULTILINE | re.DOTALL,
            )
            replacement = (
                r"\1"
                f"\n\n![原理图：{diagram.caption}](images/explainers/day-{day:03d}.svg)"
                f"\n\n*图解：{diagram.caption}*"
            )
            text, count = pattern.subn(replacement, text, count=1)
            if count != 1:
                raise RuntimeError(f"could not embed explainer for day {day:03d} in {path.name}")
        path.write_text(text, encoding="utf-8")


def validate_selection() -> None:
    expected_total = 12 * EXPECTED_DIAGRAMS_PER_MONTH
    if len(DIAGRAMS) != expected_total:
        raise RuntimeError(f"expected {expected_total} selected diagrams, found {len(DIAGRAMS)}")
    days = [diagram.day for diagram in DIAGRAMS]
    if len(days) != len(set(days)):
        raise RuntimeError("duplicate explainer day in selection")
    month_index = lambda day: next(i for i, upper in enumerate([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) if day <= upper)
    counts = Counter(month_index(day) for day in days)
    if counts != Counter({index: EXPECTED_DIAGRAMS_PER_MONTH for index in range(12)}):
        raise RuntimeError(f"selection is not {EXPECTED_DIAGRAMS_PER_MONTH} diagrams per month: {counts}")
    for diagram in DIAGRAMS:
        if diagram.kind not in RENDERERS:
            raise RuntimeError(f"unknown renderer {diagram.kind} on day {diagram.day}")
        expected = 2 if diagram.kind == "compare" else 3
        if len(diagram.nodes) < expected:
            raise RuntimeError(f"too few nodes on day {diagram.day}")


def main() -> None:
    validate_selection()
    titles = titles_by_day()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_names = {f"day-{diagram.day:03d}.svg" for diagram in DIAGRAMS}
    for stale in OUTPUT.glob("day-*.svg"):
        if stale.name not in selected_names:
            stale.unlink()
    for diagram in DIAGRAMS:
        title = titles.get(diagram.day)
        if title is None:
            raise RuntimeError(f"missing source title for day {diagram.day:03d}")
        (OUTPUT / f"day-{diagram.day:03d}.svg").write_text(svg_for(diagram, title), encoding="utf-8")
    embed_diagrams()
    print(f"generated={len(DIAGRAMS)} embedded={len(DIAGRAMS)} output={OUTPUT.relative_to(ROOT)}")
    if FALLBACK_KEYS:
        raise RuntimeError("abstract fallback icons: " + ", ".join(sorted(FALLBACK_KEYS)))


if __name__ == "__main__":
    main()
