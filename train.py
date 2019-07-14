from nn import train

params = {
    "batch_size" : 5,
    "epochs" : 1000,
    "learning_rate" : 0.0005,
    "beta1" : 0.5,
    "print_interval": 10,
    "save_interval": 100,
    "save_path": "models/"
    }

train("test_data/train", "test_data/valid", params)
