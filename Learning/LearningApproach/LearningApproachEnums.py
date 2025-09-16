from enum import Enum

class LearningApproachTypeEnum(Enum):
    SK_LEARN = 1
    DO_WHY_CAUSALML = 2
    KERAS_LEARN = 3

class LearningApproachMethodName(str, Enum):
    LINEAR_REGRESSION = "Linear Regression"
    RANDOM_FOREST = "Random Forest"
    POLYNOMIAL_REGRESSION = "Polynomial Regression"
    MULTI_LAYER_PERCEPTRON = "Multi-Layer Perceptron"
    CAUSAL_STRUCTURE_MODEL = "Causal Structure Model"
    DNN_MODEL = "DNN Model"