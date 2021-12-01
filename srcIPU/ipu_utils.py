import logging
import sys
import poptorch
import torch

def pipeline_model(model, pipeline_splits):
    """
    Split the model into stages.
    """
    for name, modules in model.named_modules():
        name = name.replace('.', '/')
        if name in pipeline_splits:
            logging.debug('--------')
        logging.debug(name)

    for split_idx, split in enumerate(pipeline_splits):
        split_tokens = split.split('/')
        logging.info(f'Processing pipeline split {split_tokens}')
        parent, node, field_or_idx_str = get_module_and_parent_by_name(model, split_tokens)
        if parent is None:
            logging.error(f'Split {split} not found')
            sys.exit()
        else:
            replace_layer(parent, field_or_idx_str, poptorch.BeginBlock(ipu_id=split_idx + 1, layer_to_call=node))

def replace_layer(parent, field_name, new_layer):
    if isinstance(parent, torch.nn.Sequential):
        parent[int(field_name)] = new_layer
    else:
        setattr(parent, field_name, new_layer)

def get_module_and_parent_by_name(node, split_tokens):
    child_to_find = split_tokens[0]
    for name, child in node.named_children():
        if name == child_to_find:
            if len(split_tokens) == 1:
                return node, child, name
            else:
                return get_module_and_parent_by_name(child, split_tokens[1:])
                
    return None, None, None