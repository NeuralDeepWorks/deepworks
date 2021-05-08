# IDataset
An abstract class representing a Dataset.
Custom datasets defines at [samples](./samples/README.md)

## Methods
`size` - get size of dataset.
`shape` - get shape of image and label from dataset.
`pull` - fill tensors with data batch and label from dataset.


# DataLoader
The class represents a utility for loading a dataset.

## Methods
`pull` - get batch (samples) from dataset.