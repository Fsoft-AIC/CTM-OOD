class BaseDetector():
    def __init__(self, name) -> None:
        self.name = name

    def adapt(self, forward_fn, ref_id_dataloader, **kwargs):
        '''
        Use this method for adaptation to reference Id data
        '''
        pass

    @property
    def require_adapt(self):
        return False

    def score(self, loader):
        raise NotImplemented
    
    def score_batch(self, forward_fn, data):
        raise NotImplemented
    
    def __str__(self) -> str:
        return f"{self.name}"