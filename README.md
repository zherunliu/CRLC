# Contrastive Representation Learning for Conservative Offline RL

## quick start

### environment setup

```bash
conda create -n crlc python=3.9
conda activate crlc

pip install -r requirements.txt

export MUJOCO_DIR=/root/.mujoco/mujoco210
export MUJOCO_INCLUDE=$MUJOCO_DIR/include
export MUJOCO_LIBRARY=$MUJOCO_DIR/bin
export MUJOCO_PLUGIN_PATH=$MUJOCO_DIR/bin
export LD_LIBRARY_PATH=$MUJOCO_DIR/bin:$LD_LIBRARY_PATH

pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

### mini-test

```bash
# -m module
python -m envs.point_maze
```
