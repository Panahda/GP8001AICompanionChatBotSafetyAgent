# Classifier 速查：看结果、评估、调用、继续训练

所有命令都在项目根目录、激活 `.venv` 后运行（Windows PowerShell）。

## 1. 文件夹结构

**手写的输入放在 `data/`，流程生成的东西都放在 `runs/`。** quick 和 full 的子目录结构完全一样。

```
companionai-safety/
├── safety/  scripts/  demo/  tests/  docs/     代码和文档（进 git）
├── data/                                      手写的输入（进 git）：taxonomy、seeds、测试集、多轮测试
├── config.yaml                                agent 用哪个模型、哪个 LLM
└── runs/                                      流程生成的所有东西（不进 git）
    ├── full/        正式运行，demo 和 slides 都用这里的结果
    │   ├── data/      raw/（生成的原始数据）  interim/（清洗、复核后）  processed/（train/val）  review/（给组员审的表）
    │   ├── models/    transformer/（DistilRoBERTa）  baseline/（TF-IDF + 逻辑回归）
    │   └── reports/   评估报告、混淆矩阵图、多轮测试结果
    ├── quick/       快速检查，结构同上；每次跑 quick 前都会清空
    └── archive/     每次 full 覆盖旧结果之前，把旧的 models/reports/review 备份到 archive/<时间>/
```

## 2. 怎么看结果

```powershell
python scripts\show_results.py quick      # 看 quick run
python scripts\show_results.py            # 看 full run
python scripts\show_results.py runs\archive\20261021-142700   # 看某次备份
```

这个脚本只读取已经生成的文件，不会重新跑任何东西。输出分 4 块：

| 块 | 看什么 | 数据来自 |
|---|---|---|
| 1. classifier 单独表现 | 每个 level 的 precision、recall、F1，以及混淆矩阵 | `reports/eval_results.json` 里的 `classifier` 这一行 |
| 2. 5 种 mode 对比 | 规则、classifier、LLM 各自能做到多少，组合起来又是多少 | 同上 |
| 3. 训练时的指标 | 验证集上的 F1、每个 epoch 的变化、校准结果 | `models/*/meta.json`、`train_log.json`、`thresholds.json` |
| 4. 多轮测试 | 对话追踪器的表现，比如降速和危机保持 | `reports/multiturn_results.md` |

图在 `runs\<run>\reports\confusion_*.png`。漏掉的危机消息列在 `missed_crises.csv`，**每一条都要人工看一遍**。

### 指标怎么读（以 crisis 为例）

```
recall    = TP / (TP + FN)   真的危机里抓到了多少（最重要；漏一个比误报一个代价大）
precision = TP / (TP + FP)   判成危机的里面有多少真是危机（低 = 误报多，会吓到用户）
F1        = 2·P·R / (P + R)
macro-F1  = 4 个 level 的 F1 取平均（衡量 safe/tense/overwhelmed/crisis 是否都分得好）
```

- **混淆矩阵**：行是真实 level，列是预测 level，对角线是判对的。
  - crisis 那一行里，不在对角线上的数都是**漏判**。
  - safe 那一行最右边的数是**误报**，也就是把正常消息判成了危机。
- **95% CI（Wilson 区间）**：测试集里危机消息越少，区间越宽。8 条危机时区间是 0.68–1.00，基本说明不了什么；50 条时才比较可信。
- **style→crisis / tense→crisis**：自闭症风格的消息（直接、简短、只按字面意思表达），以及紧张但安全的消息，被误判成危机的比例。这两个数越低越好。
- **第 3 块是验证集上的数**：训练时用来选 epoch、定阈值，所以会比测试集乐观。它只用来判断模型有没有学到东西、有没有过拟合。**slides 里只放测试集的数（第 1、2 块）。**

### 你这次 quick run 怎么读

- 这次用的是 `--no-generate`，所以只训练了 126 条种子数据（train 123 / val 19），测试 32 条，其中危机 8 条。
- classifier 的 crisis recall 1.00（CI 0.68–1.00），macro-F1 0.84，看起来很好，但样本太少，**不能当结果用**。
- 多轮测试只过了 33/45。原因是这个小模型把 "Yes, I went to the library on Saturday…"、"my phone died this morning" 判成了危机，而且置信度超过了 `crisis_hi`（0.81），于是直接下结论，没有再问 LLM。
  - 种子数据里一半是危机，校准又只用了 19 条验证集（得出的 temperature 是 0.25），所以模型偏向报危机。
  - 结论：**流程是通的，模型本身不能用**，等 full run。
- 这也暴露了一个设计问题：classifier 很有把握时会跳过 LLM，就算规则层已经认出这是习语（"my phone died"）也一样。可以考虑改成"规则只发现习语、没有其他风险线索时，classifier 判危机也必须交给 LLM 复核"，我可以把这条加进改进清单。

## 3. 训练好的 classifier 在哪

| 位置 | 内容 |
|---|---|
| `runs\full\models\transformer\` | 正式的 DistilRoBERTa：`model.safetensors`（约 330 MB）、tokenizer、`meta.json`（训练信息）、`thresholds.json`（温度、`crisis_lo/hi`）、`train_log.json` |
| `runs\full\models\baseline\` | TF-IDF + 逻辑回归（`model.joblib`）。transformer 不在时作为后备 |
| `runs\quick\models\...` | quick 的小模型，下次 quick 会被清空 |
| `runs\archive\<时间>\models\` | 被覆盖前备份下来的旧模型 |

agent 和网页用哪个模型，由 `config.yaml` 的 `classifier.model_dir` 决定，现在指向 `runs/full/models/transformer`，找不到时用 `fallback_dir`。full run 训练完以后，demo 会自动用上新模型，不用改任何东西。

## 4. 怎么调用

**命令行试几句：**
```powershell
python scripts\predict.py "I'd kill for chicken rice" "i want to die" "too loud, stop"
python scripts\predict.py --model-dir runs\quick\models\transformer "my phone died"   # 指定模型
python scripts\predict.py --agent "nobody would notice if i was gone"                 # 整个 agent（规则 + classifier + LLM）
python scripts\predict.py --file my_messages.txt                                      # 每行一句
```
输出 4 个 level 的概率和 classifier 的判断。处在灰区的消息会标 "(asks LLM)"。加 `--agent` 时会多一列整个 agent 的最终结果。

**在 Python 里只调 classifier：**
```python
from safety.classifier import load_classifier, ClassifierLayer
layer = ClassifierLayer(load_classifier("runs/full/models/transformer"))
r = layer.assess("i can't do this anymore", context="Tell me about yourself.", scenario="job_interview")
r.level, r.probs, r.wants_llm      # 判断的 level、4 个概率、要不要交给 LLM
```

**整个 agent（聊天 app 用这个）：** `SafetyAgent.from_config("config.yaml")` 会自动加载 config 里指定的模型，用法见 README 的 "Use it in the chat app"。

**网页：** 双击 `start_demo.bat`。页面上会显示有没有加载 classifier；Developer settings 里把 mode 选成 `classifier`，就能只看 classifier 的判断。

## 5. 下一步训练

1. **测试集定稿**：组员改写测试集、两人分别标注，命令见 README 的 "Data tasks"。这一步必须在 clean 之前完成。

2. **不用等测试集，可以先开始生成数据**：生成很慢，而且和测试集无关，可以随时中断、下次接着跑。
   ```powershell
   python scripts\run_flow.py --mode full --provider lmstudio --stop-after generate
   ```
   - 按这次 quick 的速度（每次调用约 72 秒、写 5 条）粗算，完整的 169 次调用、3,380 条大约要 8–14 小时，建议晚上挂着跑。
   - 如果 LM Studio 里这个模型开着 thinking/reasoning，关掉会快很多。

3. **测试集定稿后，跑完剩下的步骤**（复核用 Gemma，和生成用的 Qwen 不是同一家族的模型）：
   ```powershell
   python scripts\run_flow.py --mode full --provider lmstudio --relabel-model google/gemma-4-31b-qat --start-at clean
   ```
   它会在需要你操作的地方停下：在 LM Studio 里换模型、让组员审 `runs\full\data\review\to_review.csv`、训练前卸载模型释放显存、评估前再把模型加载回来。

4. **看结果**：`python scripts\show_results.py`。slides 用 `runs\full\reports\eval_report.md` 的表和 `confusion_full.png`。

5. **只想调参数重训**（数据不变）：
   ```powershell
   python scripts\train_transformer.py --epochs 6 --lr 3e-5 --crisis-weight 2.0
   python scripts\calibrate.py --model-dir runs\full\models\transformer
   python scripts\evaluate.py --llm-provider lmstudio --llm-model qwen/qwen3.8-27b
   python scripts\show_results.py
   ```
   也可以用 `run_flow.py --mode full --start-at transformer`，效果一样，它会先把旧结果备份到 `runs\archive\`。

**规则不变：** 只在验证集上调阈值，不拿测试集调参数；测试集改动后，从 clean 重新跑。
