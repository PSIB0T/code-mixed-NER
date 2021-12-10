from dataloader import filterDataset

with open("config.json", "r") as f:
    configJSON = json.load(f)

config = configJSON["config"]["POS"]

filterDataset(config["filepath_train_og"], config["filepath_train"])
filterDataset(config["filepath_valid_og"], config["filepath_valid"])
filterDataset(config["filepath_test_og"], config["filepath_test"])