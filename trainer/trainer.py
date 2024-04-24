import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.plot_utils import *

from trainer.base_trainer import BaseTrainer
plt.switch_backend('agg')

class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0
        
        for i, S_in in enumerate(self.train_data_loader):
            S_in = S_in.to(self.device)
            S_in = S_in.unsqueeze(1)
            S_out,_ = self.model(S_in)
            S_out = S_out.unsqueeze(1)
            #S_in_re_im = torch.cat((torch.real(S_in), torch.imag(S_in)), dim=1)
            #S_out_re_im = torch.cat((torch.real(S_out), torch.imag(S_out)), dim=1)

            loss = self.loss_function(S_out, S_in) #S_in_re_im, S_out_re_im)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_limit = self.validation_custom_config["visualize_limit"]
        for i, S_in in enumerate(self.validation_data_loader):
            S_in = S_in.to(self.device)
            S_in = S_in.unsqueeze(1)

            S_out, latent_x = self.model(S_in)
            latent_I = torch.abs(latent_x[0]).unsqueeze(0).detach().cpu().numpy()
            latent_I /= latent_I.max()
            latent_I = np.tile(latent_I, (3, 1))
            R_field = get_field()
            ## Generated tesselation for Robinson projection
            if i <= visualize_limit:
                arg_lonticks = np.linspace(-180, 180, 5)
                fig, ax, triangulation = draw_map(latent_I, R_field,
                        lon_ticks=arg_lonticks,
                        catalog=None,
                        show_labels=True,
                        show_axis=True)
            
                self.writer.add_figure(f"Acoustic Map - latent {i}", fig, epoch)

            if i <= visualize_limit:
                fig, ax = plt.subplots(1, 4)
                mat_out = S_out.detach().cpu().numpy()
                mat_in = S_in.squeeze(1).detach().cpu().numpy()
                min_val = min(np.abs(mat_out.min().item()), np.abs(mat_in.min().item()))
                max_val = max(np.abs(mat_out.max().item()), np.abs(mat_in.max().item()))
                for j, y in enumerate([mat_in, mat_out, mat_in, mat_out]):
                    vis_matrix = y
                    if j < 2:
                        im = ax[j].imshow(np.angle(vis_matrix[0]), cmap='viridis', interpolation='nearest') #, vmin=min_val, vmax=max_val)
                        ax[j].set_title("Visibility matrix (phase) - label & prediction")
                    else:
                        im = ax[j].imshow(np.abs(vis_matrix[0]), cmap='viridis', interpolation='nearest') #, vmin=min_val, vmax=max_val)
                        ax[j].set_title("Visibility matrix (mag) - label & prediction")
                    ax[j].set_xlabel('Column Index')
                    ax[j].set_ylabel('Row Index')
                #fig.colorbar(im, ax=ax[1], label='Intensity')
                plt.tight_layout()
                self.writer.add_figure(f"Visibility matrix - sample {i}", fig, epoch)
        return 0
