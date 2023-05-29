# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import pdb
import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from contrastive_loss import SupConLoss 



#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()



#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self,  device, G_mapping, G_synthesis, D, C, Diffusion_D=None, Diffusion_C=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2
                 , cls_lambda=0.05): 
        super().__init__()
        self.device = device
        
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.C = C # modifed
        self.diffusion_D = Diffusion_D
        self.diffusion_C = Diffusion_C # modified
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma 
        
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.cls_lambda = cls_lambda
        self.pl_mean = torch.zeros([], device=device)

        

    def classification_loss(self, logit, target):
        return torch.nn.functional.cross_entropy(logit, target) # added
    
    def TwoCropTransform(self, image): #added
        
        return [image, image]
    
    def run_G(self, z, c, sync): 
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c) 
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws 

    def run_D(self, img, prior, sync):  #modified
        if self.diffusion_D is not None:
            img, t = self.diffusion_D(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, prior, t) 
        return logits 
    

    def run_C(self, double_img, labels, sync): #added
        if self.diffusion_C is not None:
            double_img, t = self.diffusion_C(double_img) 
        with misc.ddp_sync(self.C, sync):
            features = self.C(double_img, t) 
            if features.shape[0] > 16: 
                f1,f2 = torch.split(features, [16,16], dim=0)
                features = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)],dim=1)
                criterion = SupConLoss(temperature=0.07)
                contrastive_loss = criterion(features, labels)
                return contrastive_loss
            else:
                return features 


    def accumulate_gradients(self,  phase, real_img, real_c, gen_z, gen_c, sync, gain): #modified
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth','Cmain', 'Cboth'] 
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Cmain = (phase in ['Cmain', 'Cboth']) #변경
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        

        
        # Gmain: Maximize logits for generated images.
        fake_contrastive_loss = 0
        if do_Gmain: 
            with torch.autograd.profiler.record_function('Gmain_forward'): # modified
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))  
                gen_img_og = gen_img
                double_gen_img = self.TwoCropTransform(gen_img) 
                gen_img_2 = torch.cat([double_gen_img[0],double_gen_img[1]], dim=0) 
                fake_labels = gen_c.max(dim=1).indices
                fake_contrastive_loss= self.run_C(gen_img_2, fake_labels,sync=False) 
                gen_logits = self.run_D(gen_img_og, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/C/fake_contra', fake_contrastive_loss)
                training_stats.report('Loss/signs/fake', gen_logits.sign())                
                loss_Gmain = torch.nn.functional.softplus((-gen_logits))  
                loss_Gmain_tot = 0.95 * loss_Gmain + 0.05 * fake_contrastive_loss          
                training_stats.report('Loss/G/loss', loss_Gmain_tot)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain_tot.mean().mul(gain).backward()
           

        # Gpl: Apply path length regularization.
        if do_Gpl: 
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = \
                    torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                        only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
 
        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain: 
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _ = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) 
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync) 
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                        torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                            only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


        contrastive_loss = 0
        if do_Cmain: # added
            with torch.autograd.profiler.record_function('Cmain_forward'):
                real_img_for_C = real_img.detach().requires_grad_(do_Cmain)
                real_img_og = real_img_for_C
                double_real_img = self.TwoCropTransform(real_img_for_C) 
                real_img_for_C = torch.cat([double_real_img[0],double_real_img[1]], dim=0)
                real_c_for_C = real_c.detach().requires_grad_(do_Cmain)
                labels = real_c_for_C.max(dim=1).indices 
                contrastive_loss= self.run_C(real_img_for_C, labels, sync=sync) 
                class_inf = self.run_C(real_img_og, labels, sync=sync)
                label_recon_loss = self.classification_loss(class_inf, labels)
                training_stats.report('Loss/C/real_contra', contrastive_loss) # output of contrastive loss
                training_stats.report('Loss/C/recon',label_recon_loss)
                C_loss_tot = 0.95 * contrastive_loss + 0.05 * label_recon_loss

            with torch.autograd.profiler.record_function('Cmain_backward'):
                C_loss_tot.mean().mul(gain).backward()