import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os

#issue: when tensor declared outside main pytorch-lighting module or functionally, does not auto-migrate.

def main():
    train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()),num_workers=16)
    trainer = pl.Trainer(fast_dev_run=True,gpus=1)
    model = LitModel()
    trainer.fit(model, train_loader)


class LitModel(pl.LightningModule):

     def __init__(self):
         super().__init__()
         self.l1 = torch.nn.Linear(28 * 28, 28 * 28)
         self.masker = TrivialMasking()
         self.l2 = torch.nn.Linear(28 * 28, 10)


     def forward(self, x):
         return torch.relu(self.l1(x.view(x.size(0), -1)))

     def training_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self(x)
         loss = F.cross_entropy(y_hat, y)
         return loss

     def configure_optimizers(self):
         return torch.optim.Adam(self.parameters(), lr=0.02)

    

class TrivialMasking(pl.LightningModule):
    def __init__(self):
        super(TrivialMasking, self).__init__()

    def forward(self, input_tensor):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, T_max, key_size) Key Projection from Encoder per time step
        :param value: (N, T_max, value_size) Value Projection from Encoder per time step
        :param lens: (N, T) Length of key and value, used for binary masking
        :return output: Attended Context
        :return attention: Attention mask that can be plotted
        '''

        mask = torch.arange(len(input)).unsqueeze(0) #>= lens.unsqueeze(1) # (1, T) >= (B, 1) -> (N, T_max)
        return input_tensor*mask


def generate_mask(length):
    lens = torch.tensor(lens).to(DEVICE)
    max_len = torch.max(lens)
    mask = (torch.arange(0, max_len).repeat(lens.size(0), 1).to(DEVICE) < \
                lens.unsqueeze(1).expand(lens.size(0), max_len)).int()
    return mask

class Attention(pl.LightningModule):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, T_max, key_size) Key Projection from Encoder per time step
        :param value: (N, T_max, value_size) Value Projection from Encoder per time step
        :param lens: (N, T) Length of key and value, used for binary masking
        :return output: Attended Context
        :return attention: Attention mask that can be plotted
        '''

        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2) # (N, T_max, key_size) * (N, context_size, 1) = (N, T_max, 1) -> (N, T_max)

        # binary masking for padded positions
        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1) # (1, T) >= (B, 1) -> (N, T_max)
        # mask = mask.to(DEVICE)
        energy.masked_fill_(mask, -1e9) # (N, T_max)
        attention = nn.functional.softmax(energy, dim=1) # (N, T_max)
        output = torch.bmm(attention.unsqueeze(1), value).squeeze(1) # (N, T_max)

        return output, attention


if __name__=='__main__':
    main()