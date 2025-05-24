def secure_average(encrypted_weights_list):
    # Assume weights are encrypted tensors
    avg_encrypted = encrypted_weights_list[0]
    for enc_weights in encrypted_weights_list[1:]:
        avg_encrypted = [a + b for a, b in zip(avg_encrypted, enc_weights)]
    avg_encrypted = [w / len(encrypted_weights_list) for w in avg_encrypted]
    return avg_encrypted
