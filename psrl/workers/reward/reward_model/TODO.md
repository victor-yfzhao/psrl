## 目标架构设计

```
n AgentLoopWorker -- n RewardManager -- n RewardLoopManager -- 1 RewardModelManager
                                                                    |
                                                                    +-- 1 RewardModelRouter
                                                                    +-- m RewardModelWorker
```

**设计说明**：
- 每个 `AgentLoopWorker` 拥有独立的 `RewardManager` 和 `RewardLoopManager`，实现并行处理
- 所有 `RewardLoopManager` 共享一个 `RewardModelManager`，统一管理 reward model 资源
- `RewardModelManager` 内部包含一个 `RewardModelRouter` 和多个 `RewardModelWorker`，实现负载均衡和资源复用

## 待完成任务

1. [DONE] Reward Model Rollout 管理：实现多个 rollout 的管理机制（参照现有的 Rollout 实现）
   - 已完成 `RewardModelManager` 和 `RewardModelReplica` 实现
   - 支持多 replica 并行
   - 已实现 Ray-based router 用于负载均衡

2. [WIP] GenRewardLoopManager 实现：完成 `GenRewardLoopManager.run_single()` 的逻辑实现
   - 已实现完整的 `run_single()` 方法
   - 支持构造 RM prompt、调用 RM inference、计算 reward score
   - 支持自定义 `compute_score` 函数和 prompt 模板

3. [TODO] RewardManager 重构：重构 `RewardManager` 的创建逻辑，支持每个 `AgentLoopWorker` 拥有独立的 `RewardManager` 实例
   - 当前状态：所有 `AgentLoopWorker` 仍共享同一个 `RewardManager` 实例
   - 需要修改 `ray_trainer.py` 中的 RewardManager 创建逻辑
   - 每个 AgentLoopWorker 应该拥有独立的 RewardManager 实例
   - 所有 RewardManager 共享同一个 RewardModelManager
   - 预期收益：消除串行瓶颈，提升并行处理能力

4. [TODO] 验证功能扩展：扩展验证功能以支持 gen-rm
   - 需要在验证阶段支持使用 gen-rm 计算奖励
   - 需要集成 RewardModelManager 到验证流程中

