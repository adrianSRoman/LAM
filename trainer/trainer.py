import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim

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
            upsample = False,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.upsample = upsample

    def _train_epoch(self, epoch):
        loss_total = 0.0
        
        for i, (S_em32, S_mic, _, _) in enumerate(self.train_data_loader):
            if self.upsample:
                # call 4 channel model (CDBPN->Bproj)
                S_mic = S_mic.to(self.device)
                S_mic = S_mic.unsqueeze(1)
                S_out,_ = self.model(S_mic)
            else:
                # call 32 channel model (Bproj)
                S_em32 = S_em32.to(self.device)
                S_em32 = S_em32.unsqueeze(1)
                S_out,_ = self.model(S_em32)
            
            S_out = S_out.unsqueeze(1)

            loss = self.loss_function(S_out, S_em32) # compare prediction with 32 channel visibility matrix
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_limit = self.validation_custom_config["visualize_limit"]
        loss_total = 0.0
        avg_psnr = 0.0
        avg_diff = 0.0
        psnr_vs_dur = {}

        is_dur_active = False

        for i, (S_em32, S_mic, apgd_label, dur_list) in enumerate(self.validation_data_loader):
            if self.upsample:
                # call 4 channel model (CDBPN->Bproj)
                S_mic = S_mic.to(self.device)
                S_mic = S_mic.unsqueeze(1)
                S_out, latent_x = self.model(S_mic)
            else:
                # call 32 channel model (Bproj)
                S_em32 = S_em32.to(self.device)
                S_em32 = S_em32.unsqueeze(1)
                S_out, latent_x = self.model(S_em32)
            loss = self.loss_function(S_out.unsqueeze(1), S_em32)
            loss_total += loss.item()
            latent_I = torch.abs(latent_x[0]).unsqueeze(0).detach().cpu().numpy()
            latent_I /= latent_I.max()
            latent_I = np.tile(latent_I, (3, 1))
            R_field = get_field()
            
            ##############################################################################
            ############# Generated tesselation for Robinson projection ##################
            ##############################################################################
            if i <= visualize_limit:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
                ax = axes[0] # subplot 1
                arg_lonticks = np.linspace(-180, 180, 5)
                fig1, ax1, triangulation1 = draw_map(latent_I, R_field,
                        lon_ticks=arg_lonticks,
                        catalog=None,
                        show_labels=True,
                        show_axis=True,
                        fig=fig,
                        ax=ax)
                
            if i <= visualize_limit:
                ax = axes[1] # subplot 2
                arg_lonticks = np.linspace(-180, 180, 5)
                apgd_map = torch.abs(apgd_label[0])
                apgd_map /= apgd_map.max()
                apgd_map = np.tile(apgd_map, (3, 1))
                fig2, ax2, triangulation2 = draw_map(apgd_map, R_field,
                        lon_ticks=arg_lonticks,
                        catalog=None,
                        show_labels=True,
                        show_axis=True,
                        fig=fig,
                        ax=ax)
            
                
                gt = apgd_map[0]
                pred = latent_I[0]
                ssim_gt = ssim(gt, gt, data_range=gt.max() - gt.min())
                ssim_pred = ssim(gt, pred, data_range=pred.max() - pred.min())
                ax1.set_title(f"LAM (pred): SSIM {round(ssim_pred, 2)}") 
                ax2.set_title(f"APGD (gt): SSIM {round(ssim_gt, 2)}")
                self.writer.add_figure("Acoustic Map {}".format(i), fig, epoch)

            ##############################################################################
            ##############################################################################

            ####################################################
            # Compute peak SNR between acustic maps (latent_x) #
            ####################################################
            psnr_val = psnr(apgd_map[0], latent_I[0])            
            avg_psnr += psnr_val
            ####################################################
            ######## Comput mean absolute difference ###########
            ####################################################
            diff = abs_diff(apgd_map[0], latent_I[0])
            avg_diff = diff
            ####################################################

            # Compute peak SNR as a function of duration
            if dur_list is not None:
                for dur in dur_list:
                    dur_key = dur.item()
                    if dur_key not in psnr_vs_dur:
                        psnr_vs_dur[dur_key] = {}
                        psnr_vs_dur[dur_key]["psnr"] = psnr_val
                        psnr_vs_dur[dur_key]["count"] = 1
                    else:
                        psnr_vs_dur[dur_key]["psnr"] += psnr_val
                        psnr_vs_dur[dur_key]["count"] += 1

                is_dur_active = True

            ##############################################################################
            ##################### Visualize visibility matrices ##########################
            ##############################################################################
            if i <= visualize_limit:
                fig, ax = plt.subplots(1, 4)
                mat_out = S_out.detach().cpu().numpy()
                mat_in = S_em32.squeeze(1).detach().cpu().numpy()
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

                plt.tight_layout()
                self.writer.add_figure(f"Visibility matrix - sample {i}", fig, epoch)
        
            ##############################################################################
            ##############################################################################

        dl_len = len(self.validation_data_loader)
        self.writer.add_scalar(f"Validation/Loss", loss_total / dl_len, epoch)
        self.writer.add_scalar(f"Peak SNR", avg_psnr / dl_len, epoch)
        self.writer.add_scalar(f"Abs Diff", avg_diff / dl_len, epoch)
        
        ###############################################
        ########### Make batplot for PSNR #############
        ###############################################
        if is_dur_active:
            for dur in psnr_vs_dur:
                # compute average PSNR per duration                  
                psnr_vs_dur[dur]["psnr"] /= psnr_vs_dur[dur]["count"]
            psnr_bins = sorted(list(psnr_vs_dur.keys())) 
            psnr_vals = [psnr_vs_dur[dur]["psnr"] for dur in psnr_bins]
            psnr_bins_sec = [round(time_bin * 10, 2) for time_bin in psnr_bins] # convert to seconds scale
            x = np.arange(len(psnr_bins_sec))
            fig, ax = plt.subplots(1, 1)       
            ax.set_ylabel('PSNR')
            ax.set_xlabel('Audio frame duration (s)')
            ax.set_title('PSRN vs. frame duration')

            ax.bar(x, psnr_vals, width = 0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(psnr_bins_sec)
            
            self.writer.add_figure(f"Dur vs PSNR/Barplot", fig, epoch)
        return 0
