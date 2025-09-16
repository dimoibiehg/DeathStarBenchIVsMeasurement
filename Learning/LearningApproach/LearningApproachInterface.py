from Learning.LearningApproach import LearningApproachEnums
from Learning.LearningApproach.LearningApproachEnums import LearningApproachMethodName

class   LearningApproachInterface:
    def train(self, X,y):
        pass

    def validate(self, X,y) -> float:
        pass

    def get_learning_type(self) -> LearningApproachEnums:
        return self.learning_type    
    def get_learning_name(self) -> LearningApproachMethodName:
        return self.learning_name