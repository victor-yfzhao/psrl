In `third_party/verl/verl/utils/dataset/rl_dataset.py`, these code need to be added.

```python
class RLHFDataset(Dataset):
    def __getitem__(self, item):
        # ...
        # original code above

        # --------------- new code ----------------
        # add reward loop type and reward fn
        reward_loop_type = self.config.reward_loop_type
        reward_fn = self.config.reward_fn
        reward_model_name = self.config.reward_model_name

        reward_model_dict = {
            "reward_loop_type": reward_loop_type,
            "reward_fn": reward_fn,
            "reward_model_name": reward_model_name,
        }
        row_dict["reward_model_dict"] = reward_model_dict
        # --------------- new code ----------------

        # original code below
        return row_dict
```