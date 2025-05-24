# encryptionfhe_wrapper.py
from config import USE_TENSEAL

if USE_TENSEAL:
    from encryptionhe_utils import get_ckks_context, encrypt_model_weights, decrypt_model_weights

class FHEHandler:
    def __init__(self):
        if USE_TENSEAL:
            self.ctx = get_ckks_context()
        else:
            self.ctx = None

    def encrypt(self, model_state_dict):
        if USE_TENSEAL:
            return encrypt_model_weights(model_state_dict, self.ctx)
        else:
            return list(model_state_dict.values())

    def decrypt(self, encrypted_weights, template_state_dict):
        if USE_TENSEAL:
            return decrypt_model_weights(encrypted_weights, template_state_dict)
        else:
            return {k: v for k, v in zip(template_state_dict.keys(), encrypted_weights)}
