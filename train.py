from args import parse_train_opt
from EDGE import EDGE
import os

# 设置NCCL环境变量
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_TIMEOUT"] = "3600"


def train(opt):
    model = EDGE(opt.feature_type)
    model.train_loop(opt)



if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
