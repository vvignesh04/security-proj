# encryptionhe_utils.py
from tenseal import TenSEALContext, CKKSVector, scheme_type
import torch

def get_ckks_context():
    ctx = TenSEALContext(
        scheme_type.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 21, 21, 40]
    )
    ctx.global_scale = 2**21
    ctx.generate_galois_keys()
    return ctx

def encrypt_model_weights(model_state_dict, ctx):
    return [CKKSVector(ctx, tensor.flatten().tolist()) for tensor in model_state_dict.values()]

def decrypt_model_weights(encrypted_weights, template_state_dict):
    decrypted = {}
    for (key, param), enc_tensor in zip(template_state_dict.items(), encrypted_weights):
        decrypted[key] = torch.tensor(enc_tensor.decrypt()).reshape(param.shape)
    return decrypted
