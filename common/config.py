class TrainConfig:
    """
    Holds all paths, hyperparameters, and settings
    for training and testing the transformer model.
    """

    # ------------------------------------------------------
    # Paths
    # ------------------------------------------------------
    # Your Excel dataset (already placed in data/)
    excel_path = "data/CA3D_constrained_vel_0to1_10k.xlsx"

    # Where the transformer weights will be saved/loaded
    model_path = "models/best_transformer.pth"

    # ------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------
    # Control point outputs (7 per axis → 21 values)
    output_cols = [
        "x2","x3","x4","x5","x6","x7","x8",
        "y2","y3","y4","y5","y6","y7","y8",
        "z2","z3","z4","z5","z6","z7","z8"
    ]

    # Input tokens are 3D vectors
    input_dim = 3

    # Token sequence:
    #   0: start
    #   1: end
    #   2: obstacle
    #   3: initial velocity / heading vector
    seq_len = 4

    # Predicting 3D control points
    output_dim = 3

    # ------------------------------------------------------
    # Transformer hyperparameters
    # ------------------------------------------------------
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 256
    dropout = 0.1

    # MLP head hyperparameters
    head_hidden = 256

    # ------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 200

    # Learning rate schedule + early stopping
    lr_patience = 5
    early_stop_patience = 12

    # ------------------------------------------------------
    # General settings
    # ------------------------------------------------------
    obstacle_radius = 0.1
    seed = 42
