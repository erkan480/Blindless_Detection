from src.ai.retina import RetinaPredictor

class LoadModel:
    def __init__(self, model=None):
        self.model = model if model is not None else self.get_model()
    def get_model(self):
        return RetinaPredictor()
    def get_disease(self, path):
        return self.model.predict(path)['class_id']
