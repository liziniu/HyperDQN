# HyperDQN: A Randomized Exploration Method for Deep Reinforcement Learning

HyperDQN is a randomized exploration based on Deep Q-Network (DQN). The paper can be found [here](https://openreview.net/forum?id=X0nrKAXu7g-).


## Requirement

Experiments are based on ``Python 3.6``. Packages can be installed by the following cmd:

```
pip install -r requirement.txt
```

Note that our implementation highly relies on ``tianshou==0.4.1``.


## Usage


```
bash scripts/hyper_dqn/run_atari.sh
```


## Bibtex
```
@inproceedings{
    li2022hyperdqn,
    title={Hyper{DQN}: A Randomized Exploration Method for Deep Reinforcement Learning},
    author={Ziniu Li and Yingru Li and Yushun Zhang and Tong Zhang and Zhi-Quan Luo},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=X0nrKAXu7g-}
}
```

## Acknowledgment


Our codebase is based on the [Tianshou](https://github.com/thu-ml/tianshou) framework.