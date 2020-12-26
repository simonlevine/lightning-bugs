# lightning-bugs

A repository noting various Pytorch-Lightning bugs I've found during usage.
Please see device_bugs.ipynb for a detailed look at a tensor on-device error within a LightningModule.
This results when torch returns a CPU tensor within a LightningModule, but Pytorch-Lighting does not migrate the new tensor.
