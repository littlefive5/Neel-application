from typing import List, Callable, Tuple, Union
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import (
    HookPoint,
) 
from jaxtyping import Float
from fancy_einsum import einsum
import transformer_lens.utils as utils
import plotly.graph_objects as go
from transformer_lens.ActivationCache import ActivationCache


def get_source_representation(cache, layer_id: int, pos_id: Union[int, List[int]]=None) -> torch.Tensor:
    """Get source hidden representation represented by (S, i, M, l)
    
    Args:
        - prompts (List[str]): a list of source prompts
        - layer_id (int): the layer id of the model
        - model (HookedTransformer): the source model
        - pos_id (Union[int, List[int]]): the position id(s) of the model, if None, return all positions

    Returns:
        - source_rep (torch.Tensor): the source hidden representation
    """
    layer_name = "blocks.{id}.hook_resid_post"
    layer_name = layer_name.format(id=layer_id)
    if pos_id is None:
        return cache[layer_name][:, :, :]
    else:
        return cache[layer_name][:, pos_id, :]

# recall the target representation (T,i*,f,M*,l*), and we also need the hidden representation from our source model (S, i, M, l)
def feed_source_representation(source_rep: torch.Tensor, prompt: List[str], f: Callable, model: HookedTransformer, layer_id: int, pos_id: Union[int, List[int]]=None) -> ActivationCache:
    """Feed the source hidden representation to the target model
    
    Args:
        - source_rep (torch.Tensor): the source hidden representation
        - prompt (List[str]): the target prompt
        - f (Callable): the mapping function
        - model (HookedTransformer): the target model
        - layer_id (int): the layer id of the target model
        - pos_id (Union[int, List[int]]): the position id(s) of the target model, if None, return all positions
    """
    mapped_rep = f(source_rep)
    # similar to what we did for activation patching, we need to define a function to patch the hidden representation
    def resid_ablation_hook(
        value: Float[torch.Tensor, "batch pos d_resid"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos d_resid"]:
        # print(f"Shape of the value tensor: {value.shape}")
        # print(f"Shape of the hidden representation at the target position: {value[:, pos_id, :].shape}")
        value[:, pos_id, :] = mapped_rep
        return value

    input_tokens = model.to_tokens(prompt)
    logits = model.run_with_hooks(
        input_tokens,
        return_type="logits",
        fwd_hooks=[(
            utils.get_act_name("resid_post", layer_id),
            resid_ablation_hook
            )]
        )
    
    return logits

def generate_with_patching(model: HookedTransformer, prompts: List[str], target_f: Callable, max_new_tokens: int = 50):
    temp_prompts = prompts
    input_tokens = model.to_tokens(temp_prompts)
    for _ in range(max_new_tokens):
        logits = target_f(
            prompt=temp_prompts,
        )
        next_tok = torch.argmax(logits[:, -1, :])
        input_tokens = torch.cat((input_tokens, next_tok.view(input_tokens.size(0), 1)), dim=1)
        temp_prompts = model.to_string(input_tokens)
    return model.to_string(input_tokens)[0]

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=orig_tokens,
        y=['Layer {}'.format(i) for i in range(num_layers)][::-1],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='Value')
    ))
    # Add text annotations
    annotations = []
    for i in range(num_layers):
        for j in range(seq_len):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=tokens[i, j],
                    showarrow=False,
                    font=dict(color='black')
                )
            )

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed'),
        margin=dict(l=50, r=50, t=100, b=50),
        width=1300,
        height=600,
        plot_bgcolor='white'
    )
    # Show the plot
    fig.show()