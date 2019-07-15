from nn import infer

params = {
    "batch_size": 128,
    "model_path": "models/model_final.pth",
    "headless": False,
    "predictions_save_path": "predictions/"
}

infer("test_data/valid", params)
