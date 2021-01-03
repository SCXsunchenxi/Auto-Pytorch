__license__ = "BSD"

import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from autoPyTorch import AutoNetImageClassification

# Note: You can write your own datamanager! Call fit with respective train, valid data (numpy matrices) 
csv_dir = os.path.abspath("../../datasets/example.csv")

def main():
    X_train = np.array([csv_dir])
    Y_train = np.array([0])

    # Note: every parameter has a default value, you do not have to specify anything. The given parameter allow a fast test.
    autonet = AutoNetImageClassification(config_preset="full_cs", result_logger_dir="logs/")

    res = autonet.fit(X_train=X_train,
                      Y_train=Y_train,
                      images_shape=[3, 32, 32],
                      min_budget=600,
                      max_budget=900,
                      max_runtime=1800,
                      save_checkpoints=True,
                      images_root_folders=[os.path.abspath("../../datasets/example_images")])

    print(res)
    print("Score:", autonet.score(X_test=X_train, Y_test=Y_train))


if __name__ == '__main__':
    main()