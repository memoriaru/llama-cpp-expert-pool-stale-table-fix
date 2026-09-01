# Flash-Next × moe-expert-pool Windows 测试反馈(矩阵 3b7a32d,含修复验证 + 分歧根因定位)

测试机:RTX 4090 24GB / 7800X3D / 128GB DDR5-3800(2DPC)/ Windows 11 / Docker(WSL2,内存=120GB)
后端:**CUDA0 = RTX 4090,compute capability 8.9,VMM yes**(非 Vulkan)
构建:full-cuda 容器内 Ubuntu CUDA 12.0 + g++-12(CUDAHOSTCXX 指定),`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`

## 0. 结论速览

- **矩阵 40/40 全过**(5 量化 × 双池 × 双 ubatch × 4 轮,CUDA)
- ~~真实模型贪心分歧已被 3b7a32d 修复~~ **更正(9/2 复验):3b7a32d 未修复服务器级分歧;根因已定位并修复(见 §8),补丁 `expert-pool-stale-table-fix.patch`**
- **decode 收益:基线 11.55 → mec64 21.21 t/s(+84%)**,单调上升,96 槽为 24GB 卡显存上限
- **prefill 零回归验证通过**(两条腿 104-119 t/s,池不参与 prefill 的设计成立)

## 1. 矩阵结果:40/40 全过(CUDA)

q2_K / q3_K / q4_K / q6_K / q8_0 × nt=1 / nt=3 × (cold / full eviction / hit+reload / after graph rebuild) = 每类型 8 轮全部 `ok`。
含 Q3_K(qwen4exp 实际量化类型)与双池形态(日志:`expert pool for 'w_gu'` + `'w_dn'` 两条注册)。

按判定树:后端 kernel 层排除。**但见 §8:单后端/少 split 场景在结构上无法触发该时序 bug——矩阵全过与服务器分歧并存不矛盾。**
修复后该测试在 CUDA 上复跑仍 PASSED(无回归)。

## 2. 真实模型复现与修复验证(Flash-Next UD-Q3_K_XL,85GB,3 卷)

```bash
./build-bin/llama-server -m Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
  -ngl 99 --cpu-moe --no-mmap -mec {0|16|32|48|64} -c 8192 --port 8999
curl localhost:8999/completion -d @req-completion.json   # 834 tok prompt,n_predict 128,temperature 0
```

- 未使用 `--no-warmup`;`-ngl 99`;`--cpu-moe`(= 全部 48 层专家 CPU)

### 修复前(升级版矩阵对应的上一提交)

mec16 内容第 **21** 字节即分叉并产出 `<tool_call>`(幻觉);mec32 第 **30** 字节分叉(均在 `<think>\n` 头部)→ 首 token 级数值分歧。

### ~~修复后(3b7a32d)~~ ——该验证已作废,见 §3.5

| 配置 | content 提取字节数 | 判定 |
|---|---|---|
| mec0(独立重跑) | 425 | 基线确定(mec0 vs mec0 逐字一致) |
| mec32 | **425** | **与 mec0 逐字一致 ✅,且与历史 mec0 基线吻合** |

## 3. ⚠️ 方法论警示:一次假"分歧"数据已作废

第一轮修复后复测曾报"仍分歧",作废原因:**生产 27B 服务(占 ~22GB 显存)未停,A/B 的 llama-server 分不到显存,实际在溢出/失败状态下跑完了流程**。
教训:**池路径 A/B 必须独占显存**——开跑前 `docker compose down` 生产服务并确认 `nvidia-smi` 无大占用;这是本次测试唯一的方法论坑。

## 3.5 ⚠️ 二次更正(9/1-9/2 根因定位期间的考古发现)

§2 的"修复后"表**无效**:fix32 的 timings 显示 `prompt_n=4, cache_n=830`(830 token 来自缓存),且 fix0/fix32 时间戳仅差 13 秒——两次请求打在**同一个 mec0 实例**上;当时 mec32 server 启动命令本身失败(`f32.err`:`No such file or directory`)。"mec0/mec32 一致"实为 mec0 vs mec0。
**方法论教训(与 §3 对偶):A/B 必须逐配置起全新 server 实例,并核验响应 timings 的 `prompt_n/cache_n` 字段。**

## 8. 根因定位与修复(9/1-9/2,定性实验 + 插桩)

### 8.1 干净环境定性实验(推翻"显存伪影"假说,也推翻"3b7a32d 已修复")

独占显存(起跑 2.4GB)、每配置全新实例、贪心解码:

| 运行 | 代码 | content | 判定 |
|---|---|---|---|
| mec0(det0/fix0/当日三次) | 新旧皆可 | 402B | 逐字节一致,基线牢固 |
| mec32(历史 det32) | 修复前 | 662B | 分歧,byte 17 起("wants"→"needs") |
| mec32(当日) | **修复前(revert)** | 662B | **与 det32 逐字节一致** → 非显存伪影,真 bug |
| mec32(当日) | **3b7a32d** | 662B | **仍分歧,逐字节同上** → 3b7a32d 对该 bug 无效 |

模型形状:qwen4exp,n_expert=512,top-k=10,48 层,**分离 gate/up + down 各自成池(288 池,非文档先前所说"双池")**;mec32 每步每层 miss 2-3 专家,decode step 3-4 首次逐出 —— 与 token 5 分叉时序吻合。

### 8.2 三步插桩收敛(bypass 探针 → 槽位/表校验 → split 结构 dump)

1. **bypass 探针**(env 强制 host 路径,池照常分配 VRAM):mec32 → **402B 与 mec0 逐字节一致** → 排除显存布局/分配效应,**bug 在池数据路径**。
2. **零行为校验探针**(每步同步读回):槽位字节 vs host 权重 **0 mismatch**;设备端 map table vs host 表 **0 mismatch** → 写侧全对。
3. **ids-stale 探针 + split dump**(决定性):`GET_ROWS(table→slots)` 重映射落在**前一个 split**,池更新却在其后 mm_id split 的 prologue —— **kernel 消费的是上一个 ubatch 的映射表**。每次 miss(冷加载/逐出重载)读到的是该槽位**上一个专家的权重**;thrashing 下每步 ~20-30% 专家权重错误,扰动恰好让贪心解码撑过 4 token、在 token 5 近似平局处确定性翻转 → "流畅但分歧"。
   - warmup 首步全零表消费(9413 条 stale 记录)与各步逐出数完全对上时序。

### 8.3 修复(expert-pool-stale-table-fix.patch,+47/-7,主体为注释)

1. map table 直接建成 **2D [1, n_expert]**(图内不再有 reshape view——view 无 buffer,边界检查看不见);
2. `table_buf` 注册进 `expert_pool_by_buf` → 重映射 GET_ROWS **自成 split**;
3. 池更新扫描扩展到 GET_ROWS 节点(按表的设备拷贝匹配):**更新 → 上传新表 → 同 split 重映射消费**,顺序不变式成立(MMID 分支保留兜底单 split 后端)。

### 8.4 修复验证(最终干净补丁二进制)

| 对照 | 结果 |
|---|---|
| 单元矩阵 test-expert-pool(CUDA) | **PASSED**(无回归) |
| mec32 修复后 | **402B 中前 354B 与 mec0 逐字节一致**,fork 从 byte 17 → 354;跨运行逐字节确定 |
| mec0 / bypass32(池分配+host kernel) | **互为逐字节一致**(再证布局无效应) |
| mec64 | **与修复后 mec32 逐字节一致**(同在 byte 354 vs mec0 分叉)→ 残留与池大小/逐出无关,系池 kernel 路径系统性单点翻转 |

**残留**:token ~88 处单点确定性翻转("Length matching"→"Length match",词边界近似平局)。逐专家 GEMM 数学已由单元矩阵证明位精确,bypass 对照位精确 → 残留定性为池 kernel 路径数值差异(容器 ne[2]=槽位数/槽位对齐可触发不同 kernel 配置),**非数据正确性问题**;完整 logit 边际留档待查(`/completion` 端点 return_probs 未生效,建议 PR 前用 llama-cli --n-probs 补测)。

### 8.5 对 RFC/PR 的叙事修正

- 不能写"预算 cap 后从未有过真 bug":有真 bug(跨 split 陈旧映射),且 3b7a32d 未修复它;现补丁修复。
- 矩阵测试结构性盲区:单后端/少 split 图无法暴露跨 split 时序 bug——建议 PR 中给 test-expert-pool 增加"host 权重 + GPU 池"双后端用例。
- §2 修复后验证表与 §3.5 作废说明必须随 PR 呈现,数据文件齐全(det*/fix*/exp12-14 系列留档于 repo-moe/)。

## 4. decode 收益扫描(-ncmoe 99,q4→ 实测用 q3_K,tg512,r=2)

| -mec | tg512 (t/s) | vs 基线 |
|---|---|---|
| 0 | 11.55 ± 0.41 | — |
| 16 | 12.89 ± 0.33 | +12% |
| 32 | 17.54 ± 0.15 | +52% |
| 48 | 19.09 ± 0.62 | +65% |
| **64** | **21.21 ± 0.46** | **+84%** |

- 单调上升未见平台;mec=64(6.4× top-k)为 24GB 卡实用上限(96 槽池 ~27GB 超预算)
- 注:`-p 0` 纯生成路由偏斜中性;代码类真实工作流(模板化输出)按文档预期应更高,待复测
- 注:上表测于修复前;时序 bug 不影响拷贝量与命中/逐出行为,吞吐数字应不受影响,PR 前建议抽一档复测

## 5. prefill 零回归:通过

| 配置 | pp512 | pp2048 |
|---|---|---|
| -mec 0 | 104.5 ± 9.9 | 113.4 ± 5.7 |
| -mec 32 | 119.1 ± 10.9 | 112.9 ± 4.8 |

两配置噪声范围内一致,"prefill 走回退路径"的设计声明被实机验证。

## 6. 剩余待办

- [x] ~~长会话稳定性~~(exp12/13 系列 128-token 多轮独占显存跑通;2000+ token 过夜批跑仍待)
- [ ] 代码类工作流(高路由偏斜)下的真实收益复测——本扫描为路由中性载荷
- [ ] 模板化输出场景的 tg 上限复测(t5 类载荷在 27B/MTP 侧曾实测 2-3× 加速)
- [ ] 残留单点翻转的 logit 边际定量(llama-cli --n-probs)
- [ ] test-expert-pool 增加 host 权重 + GPU 池双后端用例(暴露跨 split 时序)
- [ ] 修复后 decode 收益抽档复测

## 7. RFC 论据小结

确定性基线(mec0×3 逐字一致)、根因定位三步证据链(bypass 探针/写侧零 mismatch/split dump)、最小修复与双端验证(矩阵 PASSED + 服务器 fork 17→354 + bypass 对照位精确)、decode +84%@mec64(6.4×top-k)、prefill 零回归、消费级 24GB 单卡 WDDM/Windows 实机数据、复现命令与数据文件齐备。修复前分歧样本(首 token 分叉)与根因机制说明是 PR 叙事的核心资产。
