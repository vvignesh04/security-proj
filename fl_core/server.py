class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_weights):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.mean(torch.stack([weights[k] for weights in client_weights]), dim=0)
        self.global_model.load_state_dict(global_dict)
        return self.global_model