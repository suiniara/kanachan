"""Non-exported Q-only views; parameters remain owned by the original actor."""

from torch import nn
from tensordict.nn import TensorDictSequential

from kanachan.nn.decode_converter import DecodeConverter
from kanachan.nn.decision_decoder import DenseDecisionDecoder, GlobalActionHead
from kanachan.nn.decision_state import DecisionStateHead


OBSERVATION_KEYS = ["sparse", "numeric", "progression", "candidates"]


def transfer_bc_features(backbone, head, bc_model):
    """Copy learned features, including the hidden head, never BC logits as Q."""
    bodies = [m for m in bc_model.modules() if type(m) is DenseDecisionDecoder]
    heads = [m for m in bc_model.modules() if type(m) is GlobalActionHead]
    if type(backbone) is not DenseDecisionDecoder or type(head) not in (GlobalActionHead, DecisionStateHead) or len(bodies) != 1 or len(heads) != 1:
        raise ValueError("BC transfer needs one matching dense decoder and global action head.")
    teacher, teacher_head = bodies[0], heads[0]

    def signature(body):
        return (body.dimension, body.numeric_dimension, body.history_attention,
                tuple((b.num_heads, b.head_dimension, b.rope_base, b.gate_up.out_features) for b in body.blocks))

    if signature(backbone) != signature(teacher):
        raise ValueError("BC and Q feature architectures differ.")
    if head.layers[0].weight.shape != teacher_head.layers[0].weight.shape:
        raise ValueError("BC and Q hidden feature dimensions differ.")
    backbone.load_state_dict(teacher.state_dict(), strict=True)
    head.layers[0].load_state_dict(teacher_head.layers[0].state_dict(), strict=True)
    if getattr(head, "candidate_layers", None) is not None and teacher_head.candidate_layers is not None:
        head.candidate_embedding.load_state_dict(teacher_head.candidate_embedding.state_dict(), strict=True)
        head.candidate_layers[0].load_state_dict(teacher_head.candidate_layers[0].state_dict(), strict=True)
    head.zero_output()


class DecisionQNetwork(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, data):
        representation, valid = self.backbone.forward_with_validity(
            *(data[key] for key in OBSERVATION_KEYS)
        )
        data["action_value"] = self.head._from_validated(
            data["candidates"], representation, valid, data["sparse"], data["progression"]
        )
        return data


def q_only(model):
    """Strip action selection, fusing validation only for our exact topology.

    Never bypass a DDP/compiled/custom wrapper: its forward may carry required
    synchronization or additional value transforms. Portable export/state keys
    continue to belong to the original TensorDict actor.
    """
    if not isinstance(model, TensorDictSequential):
        return model
    layers = list(model.module)
    if len(layers) not in (2, 3):
        return model
    if len(layers) == 3 and (
        type(getattr(layers[-1], "module", None)) is not DecodeConverter
        or layers[-1].in_keys != ["candidates", "action_value"]
        or layers[-1].out_keys != ["action"]
    ):
        return model
    first, second = layers[:2]
    if (
        first.in_keys == OBSERVATION_KEYS and first.out_keys == ["encode"]
        and second.in_keys in (["candidates", "encode"], ["candidates", "encode", "sparse", "progression"])
        and second.out_keys == ["action_value"]
        and type(getattr(first, "module", None)) is DenseDecisionDecoder
        and type(getattr(second, "module", None)) is GlobalActionHead
    ):
        return DecisionQNetwork(first.module, second.module)
    if len(layers) == 3 and layers[1].out_keys == ["action_value"]:
        return model.select_subsequence(out_keys=["action_value"])
    return model
