ONE_PARAM = "1p"
TWO_K = "2k"
BETA_MODELS = [ONE_PARAM, TWO_K]


def check_valid_model(beta_model):
    """
        Checking whether models fall into either single beta, 2k
    """
    if beta_model not in BETA_MODELS:
        raise ValueError("Invalid beta model, please choose one of: {}".format(BETA_MODELS))
