from .base_tta import TTABaseClass
import torch
import torch.nn as nn


class NoTTA(TTABaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def adapt(self):
        print("NoTTA adapt")


class SimpleTTA(TTABaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.after_classifier = nn.Linear(2, 2, device=self.device)
        # self.opt = torch.optim.Adam(self.after_classifier.parameters(), lr=0.01)

    def adapt(self):
        ...
        # self.model.train()
        # for _ in range(100):
        #     self.opt.zero_grad()
        #     logits = self.after_classifier(self.model(self.test_time_graph))
        #     loss = nn.functional.cross_entropy(logits, self.test_time_graph.ndata['label'])  # not a seriously TTA
        #     loss.backward()
        #     self.opt.step()