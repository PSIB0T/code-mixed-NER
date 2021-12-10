import json
import os
from dataloader import getDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loadDataLoaders():
    configJSON = json.load("./config.json")
    config = configJSON["config"]
    datasets = {
        "train": {},
        "valid": {},
        "test": {}
    }
    for task in ["NER", "POS"]:
        taskConfig = config[task]
        for _type in ["train", "valid", "test"]:
            filePathVar = "filepath_" + _type
            datasets[_type][task] = getDataset(taskConfig[filePathVar], taskConfig["uniqueLabels"], taskConfig["indices"])

    dataloaders = {
        "train": {},
        "valid": {},
        "test": {}
    }

    for task in ["NER", "POS"]:
        for _type in ["train", "valid", "test"]:
            dataloaders[_type][task] = DataLoader(datasets[_type][task], **params[_type], collate_fn=collate_fn)

    
    return dataloaders

def train(model, optimizer, lossFn, dataloaders, tag_vals, epoch=1):
  model.train()


    for _ in tqdm(range(epoch)):
        i = 0
        tasks = ["NER", "POS"]
        batches = {}
        for task in tasks:
            batches[task] = iter(dataloaders["train"][task])

        while len(tasks) > 0:
            task = np.random.choice(tasks)
            try:
                data = next(batches[task])
            except StopIteration:
                tasks.remove(task)
                continue
    
            optimizer.zero_grad()

            tag, label = data

            tag = tag.to(device)
            label = label.to(device)
            logits = model(tag, task)

            logits = torch.transpose(logits, 2, 1)

            loss = lossFn(logits, label)

            if i%100==0:
                print(f'Epoch: {_}, Loss:  {loss.item()}')

            loss.backward()

            optimizer.step()

            i += 1

        torch.save(model.state_dict(), "./bert_multitask_{}.pt".format(_))
        test(model, dataloaders["valid"]["NER"], tag_vals)


def test(model, loader, tags_vals):
    tag_list = []
    pred_list = []
    label_list = []


    with torch.no_grad():
    
        for _, data in enumerate(loader):
      
            tags, labels = data
            tags = tags.to(device)
            labels = labels.to(device)

            output = model(tags, "NER")
            logits = output
            logits = logits.detach().cpu().numpy()
            predictions = np.array([list(p) for p in np.argmax(logits, axis=2)])
            labels = labels.detach().cpu().numpy()

            tags = [auto_tokenizer.convert_ids_to_tokens(t) for t in tags]


            for pred, label, tag in zip(predictions, labels, tags):
                pred_sent, label_sent, tag_sent = [], [], []
                tagString = ""
                for p, l, t in zip(pred, label, tag):
                if l != pad_token_label_id:
                    pred_sent.append(p)
                    label_sent.append(l)
                    if len(tagString) > 0:
                    tag_sent.append(tagString)
                    tagString = t
                elif t not in ['[CLS]', '[SEP]', '[PAD]']:
                    if t.startswith("##"):
                    t = t.lstrip("#")
                    tagString += t
                
                if len(tagString) > 0:
                    tag_sent.append(tagString)
                if len(pred_sent) > 0:
                    pred_list.append(pred_sent)
                    label_list.append(label_sent)
                    tag_list.append(tag_sent)

    pred_list = [[tags_vals[l] for l in p] for p in pred_list]
    label_list = [[tags_vals[l] for l in p] for p in label_list]

    print(classification_report(label_list, pred_list))

    return tag_list, pred_list, label_list

if __name__ == "__main__":
    configJSON = json.load("./config.json")

    model  = BERTClass()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    lossFn = nn.NLLLoss()

    model.to(device)

    dataloaders = loadDataLoaders()

    train(model, optimizer, lossFn, dataloaders["valid"]["NER"], config["NER"]["uniqueLabels"], 1)
