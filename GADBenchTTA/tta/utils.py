import torch
import torch.nn as nn
from typing import Tuple
import dgl

def hack_model_for_embedding(model: nn.Module, expect_num_classess: int = 2, args: tuple = (), kwargs: dict = {}) -> Tuple[torch.Tensor, torch.Tensor | None] | None:
    """
    skip the last classifier(which out_features == expect_num_classess)
    """
    if "get_emb" in dir(model):
        return model.get_emb(*args, **kwargs)
    
    # FIXME it works in several cases, but not in all cases
    catch_result = None
    def hack_wrapper(og_func):
        def catch_func(*args): 
            nonlocal catch_result
            print("[DEBUG] hack_wrapper args:", args)
            if len(args) == 1:
                catch_result = args[0]
            else:
                catch_result = args
            return og_func(*args)
        return catch_func

    found_layer = None
    og_forward = None
    for layer in model.modules():
        if isinstance(layer, nn.Linear) and layer.out_features == expect_num_classess:
            found_layer = layer
            og_forward = layer.forward
            layer.forward = hack_wrapper(layer.forward)
            break
    if found_layer is None or og_forward is None:
        return None

    og_res = model(*args, **kwargs)
    found_layer.forward = og_forward
    return og_res, catch_result

def get_classifier_layer(model: nn.Module, expect_num_classess: int = 2) -> nn.Linear | None:
    for layer in model.modules():
        if isinstance(layer, nn.Linear) and layer.out_features == expect_num_classess:
            return layer
    return None

def make_dgl_graph(feat: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None, self_loop: bool = True, num_nodes: int | None = None):
    graph = dgl.graph((edge_index[0], edge_index[1]), device=feat.device, num_nodes=num_nodes)
    graph.ndata['feature'] = feat
    if edge_weight is not None:
        graph.edata['edge_weight'] = edge_weight
    if self_loop:
        graph = dgl.add_self_loop(graph)
    return graph
