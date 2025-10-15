from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_3_with_7
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_2_with_7
from federated_learning.utils import replace_6_with_0
from federated_learning.utils import replace_7_with_1
from federated_learning.utils import replace_8_with_6
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
import numpy as np

def multi_label_flipper(Y, classes):
    Y_new = []
    for label in Y:
        # Apply multiple flipping rules
        if label == 1:
            Y_new.append(9)
        elif label == 0:
            Y_new.append(2)
        elif label == 5:
            Y_new.append(3)
        elif label == 4:
            Y_new.append(6)
        elif label == 6:
            Y_new.append(0)
        elif label == 3:
            Y_new.append(7)
        elif label == 2:
            Y_new.append(7)
        elif label == 7:
            Y_new.append(1)
        elif label == 8:
            Y_new.append(6)
        elif label == 9:
            Y_new.append(3)
        else:
            # Keep unchanged
            Y_new.append(label)
    return np.array(Y_new)


if __name__ == '__main__':
    START_EXP_IDX = 3011
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 3
    REPLACEMENT_METHOD = multi_label_flipper
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 10,
        "aggregator" : "trimmed_mean"
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)

