# Performance

Сначала я получал подгружал данные каждый раз, когда получал элемент из DataLoader. 


``` python
def __getitem__(self, idx):
    file_name = os.path.join(self.data_dir, self.file_list[idx])
    num_data = pd.read_csv(file_name, sep=" ", header=None).to_numpy()
    num_data = self.transform(num_data)
    return num_data
```

Потом я поменял стратегию, и даже на CPU результаты стали намного лучше

``` python

def __init__(...)
...

    for file_name in self.file_list:
        file_name = os.path.join(self.data_dir, file_name)
        num_data = pd.read_csv(file_name, sep=" ", header=None).to_numpy()
        num_data = self.transform(num_data)
        self.data.append(num_data)

    self.data = torch.stack(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```


# Processing 

## Images (input)

Изначально для обработки изображений я ипользовал следующий подоход:

``` python 
...
    self.transform = (
        transform
        if transform
        else transforms.Compose(
            [
                transforms.ToTensor(),  # Конвертация в тензор [0, 1]
                # Нормализация [-1, 1]
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    )
...
```



## Targets (output)