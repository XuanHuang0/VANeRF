import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gc
import copy
import math
import subprocess
import time
import cv2
import yaml
import tqdm
import torch
import kornia as K
import numpy as np
import pytorch_lightning
import imageio
from kornia.utils import tensor_to_image
from pytorch_lightning.utilities.apply_func import move_data_to_device

from .utils import *
from . import evaluator
from .spatial import SpatialEncoder
from .dataset import Dataset
from .networks import GeoVisFusion, GeoVisFusion_spconv, TexVisFusion, TexVisFusion_spconv, Discriminator_vis, d_logistic_loss, g_nonsaturating_loss, d_r1_loss
from .lib.dataset.mesh_util import cal_vert_sdf_batch,cal_vis_sdf_batch
from .lib.dataset.mesh_util import seal as seal_torch
from .render_vis import render_vis
import mesh_to_sdf
import trimesh
import skimage
import smplx

# mano layer
smplx_path = 'smplx/models/'
mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1


class VANeRFLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, cfg: dict, cfg_model: dict):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.cfg_model = cfg_model
        self.idx = 0
        self.expname = cfg['expname']
        self.save_dir = f'{cfg["out_dir"]}/{cfg["expname"]}'
        self.save_hyperparameters()
        self.dataset = Dataset
        self.video_dirname = 'video_interhand'
        self.images_dirname = 'images'
        self.test_dst_name = cfg['test_dst_name']
        self.nkpt_r,self.nkpt_l=21,21
        self.model = VANeRF(self.cfg)
        self.discriminator=Discriminator_vis()
        self.znear, self.zfar = 0.71, 1.42
        self.evaluator = evaluator.Evaluator()

    def configure_optimizers(self):
        opt_g=torch.optim.Adam(self.model.parameters(), lr=self.cfg['training'].get('lr', 1e-5))
        opt_d=torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg['training'].get('lr', 1e-5))
        StepLR_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[2,5,10,20,35], gamma=0.5)
        StepLR_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=[2,5,10,20,35], gamma=0.5)
        optim_dict_g = {'optimizer': opt_g, 'lr_scheduler': StepLR_g}
        optim_dict_d = {'optimizer': opt_d, 'lr_scheduler': StepLR_d}
        return [optim_dict_g,optim_dict_d]

    @classmethod
    def from_config(cls, cfg, cfg_model):
        return cls(cfg, cfg_model)

    def train_dataloader(self, batch_size=None):
        train_dataset = self.dataset.from_config(self.cfg['dataset'], 'train', self.cfg)
        return torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            num_workers=self.cfg['training'].get('train_num_workers', 0),
            batch_size=self.cfg['training'].get('train_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self, batch_size=None):
        val_dataset = self.dataset.from_config(self.cfg['dataset'], 'val', self.cfg)
        return torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            num_workers=self.cfg['training'].get('val_num_workers', 0),
            batch_size=self.cfg['training'].get('val_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self, batch_size=None):
        test_dataset = self.dataset.from_config(self.cfg['dataset'], 'test', self.cfg)
        return torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            num_workers=self.cfg['training'].get('val_num_workers', 0),
            batch_size=self.cfg['training'].get('val_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def save_ckpt(self, **kwargs):
        pass

    def test_epoch_end(self, outputs):
        results = {key: torch.stack([x[key] for x in outputs]).mean() for key in outputs[0].keys()}
        results = {key: float(val.item()) if torch.is_tensor(val) else float(val) for key, val in results.items()}

        path = os.path.join(
            self.save_dir,  f'test_{self.test_dst_name}_{self.current_epoch}_{self.global_step}.yml')

        with open(path, 'w') as f:
            yaml.dump(results, f)

        print('Results saved in', path)
        print(results)

    @staticmethod
    def collate_fn(items):
        """ Modified form of :func:`torch.utils.data.dataloader.default_collate` that will strip samples from
        the batch if they are ``None``.
        """
        try:
            items = [item for item in items if item is not None]
            return torch.utils.data.dataloader.default_collate(items) if len(items) > 0 else None
        except Exception as e:
            return None
    
    def load_ckpt(self, ckpt_path):
        assert os.path.exists(ckpt_path), f'Checkpoint ({ckpt_path}) does not exists!'
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["state_dict"])
        return ckpt['epoch'], ckpt['global_step']

    @torch.no_grad()
    def render_video(self, data_loader, label='', **kwargs):
        sc_factor = data_loader.dataset.sc_factor
        trans = 10#5.0
        n_frames= 20
        znear = (trans - 5.0)*sc_factor
        zfar = (trans + 5.0)*sc_factor
        im_w, im_h = 256, 256
        fstart = im_w * 30
        fend = im_w * .05
        focal = fstart + 0.01 * (fend - fstart)

        dst_dir = os.path.join(self.save_dir, f'{self.video_dirname}{label}')
        self.eval()
        cameras = dict()
        sub_dir_path_set = set()
        for batch in tqdm.tqdm(data_loader):
            batch = move_data_to_device(batch, self.device)
            # update batch
            if 'near_fars' in batch:
                batch['near_fars'][..., 0] = znear
                batch['near_fars'][..., 1] = zfar

            # create dst directory
            session = str(batch['index']['segment'][0])
            identity = str(int(batch['human'][0]))
            frame_index = int(batch['frame_index'][0])

            sub_dir_path = os.path.join(dst_dir, session, identity)
            sub_dir_path_set.add(sub_dir_path)
            cond_mkdir(sub_dir_path)

            # create cameras
            if identity not in cameras:
                print('Processing,', identity)
                cameras[identity] = get_360cameras(batch['headpose'][0], focal,
                    trans, sc_factor, im_w, im_h, znear, zfar, n_frames)

            camera = cameras[identity]

            rnd_images = self.render_novel_views(camera, batch)

            for fi,rnd_img in enumerate(rnd_images):
                img_name = os.path.join(sub_dir_path, f'{fi:06d}.png')
                cv2.imwrite(img_name, rnd_img[:, :, ::-1])

            # clean cache
            torch.cuda.empty_cache()
            gc.collect()

        # create videos via ffmpeg
        for sub_dir_path in sub_dir_path_set:
            video_path = f'{sub_dir_path}_nvs.mp4'
            gif_path = f'{sub_dir_path}_nvs.gif'
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
            cap_fps = 15
            size = (512,256)
            video = cv2.VideoWriter(video_path, fourcc, cap_fps, size)
            path = sub_dir_path+"/"
            file_lst = os.listdir(path)
            frames = []
            for fi in range(n_frames):
                img = cv2.imread(os.path.join(sub_dir_path, f'{fi:06d}.png'))
                frames.append(img[...,[2,1,0]])
                video.write(img)
            video.release()
            imageio.mimsave(gif_path, frames, 'GIF', duration=0.1)
            print('Saved:', video_path)

    @staticmethod
    def compute_test_metric(rendered_img, gt_img, mask=None, max_val=1.0):
        """
        Args:
            rendered_img (torch.tensor): (3, H, W) or (B, 3, H, W) [0, 1.0]
            gt_img (torch.tensor): (3, H, W) or (B, 3, H, W) [0, 1.0]
            mask (torch.tensor: torch.bool): (3, H, W) or (B, 3, H, W) [0, 1.0]
        """
        assert rendered_img.shape == gt_img.shape
        if len(rendered_img.shape) == 3:
            rendered_img = rendered_img.unsqueeze(0)
            gt_img = gt_img.unsqueeze(0)
        mask = mask.view(1, *mask.shape[-2:]) if mask is not None else mask

        # B,3,H,W
        ssim = K.metrics.ssim(rendered_img, gt_img, window_size=7, max_val=max_val)
        ssim = ssim.permute(0, 2, 3, 1)[mask] if mask is not None else ssim
        ssim = ssim.mean()

        if mask is not None:
            rendered_img = rendered_img.permute(0, 2, 3, 1)[mask]
            gt_img = gt_img.permute(0, 2, 3, 1)[mask]

        return {
            f'psnr': K.metrics.psnr(rendered_img, gt_img, max_val=max_val),
            f'ssim': ssim,
        }

    def save_test_image(self, batch, rendered_img, gt_img, mask=None, face_mask=None):
        """
        Args:
            rendered_img (torch.tensor): (3, H, W) [0, 1.0]
        """
        index = batch['index']
        sub_id = index['frame'][0]
        tar_cam_id = index['tar_cam_id'][0]
        # prepare directory
        dst_dir = os.path.join(
            self.save_dir,
            f'{self.images_dirname}_{self.test_dst_name}',  # _{self.current_epoch}_{self.global_step}
            sub_id)
        cond_mkdir(dst_dir)

        # save images
        if rendered_img is not None:
            rendered_img = tensor_to_image(rendered_img)  # H,W,3
            rendered_img = (rendered_img*255.).astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.pred.png')
            cv2.imwrite(path, rendered_img[:, :, ::-1])

        if gt_img is not None:
            gt_img = tensor_to_image(gt_img)
            gt_img = (gt_img*255.).astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.gt.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])
        
        if mask is not None:
            mask = (mask*255.).squeeze().unsqueeze(-1).repeat(1, 1,  3)
            gt_img = mask.detach().cpu().numpy().astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.mask.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])

        if face_mask is not None:
            face_mask = (face_mask*255.).squeeze().unsqueeze(-1).repeat(1, 1,  3)
            gt_img = face_mask.detach().cpu().numpy().astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.face_mask.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])

    def decode_batch(self, batch, use_dr=False):
        if self.cfg.get('provide_znear_zfar',False):
            self.znear, self.zfar = batch['znear'].float(), batch['zfar'].float()
        img_mask = batch['images_masks'].float()
        img = batch["images"].float()
        hand_type=batch["hand_type"]
        targets=batch["targets"]
        tar_img_mask = batch['images_masks'][:, 0]
        src_img_mask = batch['images_masks'][:, 1:]
        use_dr = True

        Rt = batch['Rt']
        K = batch['K']

        n_batch = Rt.shape[0]
        n_views = 1
        if len(Rt.shape) != 3:
            if use_dr:
                dr_Rt = Rt[:, 0]
                Rt = Rt[:, 1:].contiguous()
                dr_K =  K[:, 0]
                K = K[:, 1:].contiguous()
                dr_img = img[:, 0]
                img = img[:, 1:].contiguous()
            n_views = Rt.shape[1]
            Rt = Rt.view(-1, *Rt.shape[2:])
            K = K.view(-1, *K.shape[2:])
            img = img.view(-1, *img.shape[2:])
            img_mask = img_mask.view(-1, *img_mask.shape[2:])
        extrin = torch.eye(4, device=self.device)[None].repeat(n_batch * n_views, 1, 1)
        extrin[:, :3, :4] = Rt

        height, width = img.shape[-2:]
        intrin = torch.eye(4, device=self.device)[None].repeat(n_batch * n_views, 1, 1)
        intrin[:, :3, :3] = K[..., :3, :3]
        KRT = torch.bmm(intrin, extrin)
        cam = {
            "KRT": KRT, 'K': intrin, 'Rt': Rt, 'extrin': extrin,
            "znear": self.znear, "zfar": self.zfar,
            "width": width, "height": height, "nml_scale": 100.0
        }
        hT = None
        invhT = None

        sp_data = {"extrin": extrin}

        kpt3d = batch["kpt3d"]
        if kpt3d is not None:
            sp_data["kpt3d"] = kpt3d

        # uniform sampling within frustum
        # in case of multi-view, it picks the first view frustum
        bbox_min = batch["kpt3d"].min(1)[0]
        bbox_max = batch["kpt3d"].max(1)[0]
        center = 0.5 * (bbox_min + bbox_max)
        length = 0.7 * (bbox_max - bbox_min)
        bbox_min = center - length
        bbox_max = center + length

        dr_data = None
        if use_dr:
            dr_extrin = torch.eye(4, device=self.device)[None].repeat(n_batch, 1, 1)
            dr_extrin[:, :3, :4] = dr_Rt.clone()

            dr_intrin = torch.eye(4, device=self.device)[None].repeat(n_batch, 1, 1)
            dr_intrin[:, :3, :3] = dr_K.clone()

            dr_data = {
                'cam_tar': {
                    "K": dr_intrin, "RT": dr_extrin, "KRT": torch.bmm(dr_intrin, dr_extrin),
                    "width": width, "height": height, "nml_scale": 100.,
                    "znear": self.znear, "zfar": self.zfar,
                },
                "tar": dr_img,
                'hT': hT,
                "img": img,
                "cam": cam,
                "mask_at_box": batch['mask_at_box'],
                "bounds": batch["bounds"],
            }
            dr_data["sp_data"] = sp_data
            dr_data["objcenter"] = kpt3d[:, 0, :]  # select pelvis
            dr_data["bg"] = bkg if "bkg" in batch else None
            dr_data["camcenter"] = batch.get("camcenter", None)
            dr_data['msk'] = tar_img_mask

        # apply mask over the source images
        params = {
            'im': img,
            'cam': cam,
            'hand_type':hand_type,
            'targets':targets,
            'data': batch,
            'bbox': (bbox_max, bbox_min),
            'n_views': n_views,
            'sp_data': sp_data,
            'invhT': invhT,
            'dr_data': dr_data,
            'tar_img_mask': tar_img_mask,
            'src_foreground_mask': src_img_mask,
            "bounds": batch["bounds"],
        }
        return params

    def training_step(self, batch, batch_idx, optimizer_idx):
        tr_batch = self.decode_batch(batch)
        lambda_dis=self.cfg['models']['Discriminator']['lambdas']
        # Train Generator
        if optimizer_idx == 0:
            loss_dict = self.model(**tr_batch)
            out=loss_dict['out']
            out_nerf=out['nerf']
            vis_img_gt=out_nerf['vis_img']
            rendered_image = out_nerf["tex_fg_fine"].clamp(min=0.0, max=1.0)  # 3, H, W
            gt_img = out_nerf["tar_img"]
            msk = out_nerf["tar_alpha"]
            vis_img_gt=out_nerf['vis_img']
            input_img=out_nerf['img_in']
            input_densepose=out_nerf['input_densepose']
            tar_densepose=out_nerf['tar_densepose']

            fake_pred,fake_vis_pred=self.discriminator(input_img, input_densepose, tar_densepose,rendered_image)
            bce_loss = nn.BCELoss(reduce=False)
            vis_pix_loss=bce_loss(fake_vis_pred, torch.ones_like(fake_vis_pred).to(fake_vis_pred.device) )
            vis_pix_loss[msk==0]=0
            vis_pix_loss=vis_pix_loss.mean()

            g_gan_loss = g_nonsaturating_loss(fake_pred)
            loss_dict['loss'] = loss_dict['loss']+lambda_dis['lambda_dis1']*g_gan_loss+lambda_dis['lambda_dis2']*vis_pix_loss
            loss_dict['err_dict']['gan_loss']=g_gan_loss*lambda_dis['lambda_dis1']
            loss_dict['err_dict']['vis_pix_loss']=vis_pix_loss*lambda_dis['lambda_dis2']
            
            for k, val in loss_dict['err_dict'].items():
                self.log(f'train/{k}', val, prog_bar=True)
            return loss_dict['loss']
        
        # Train discriminator
        if optimizer_idx == 1:
            gan_loss_dict = {}
            
            loss_dict = self.model(**tr_batch)
            out=loss_dict['out']
            out_nerf=out['nerf']
            vis_img_gt=out_nerf['vis_img']
            rendered_image = out_nerf["tex_fg_fine"].clamp(min=0.0, max=1.0)  # 3, H, W
            gt_img = out_nerf["tar_img"]
            input_img=out_nerf['img_in'] 
            msk = out_nerf["tar_alpha"]
            input_densepose=out_nerf['input_densepose']
            tar_densepose=out_nerf['tar_densepose']
            
            gt_img.requires_grad = True
            real_pred,real_vis_pred=self.discriminator(input_img, input_densepose, tar_densepose, gt_img)
            fake_pred,fake_vis_pred=self.discriminator(input_img, input_densepose, tar_densepose, rendered_image.detach())

            bce_loss=nn.BCELoss(reduce=False)
            real_vis_pix_loss=bce_loss(real_vis_pred, torch.ones_like(real_vis_pred).to(real_vis_pred.device) )
            fake_vis_pix_loss=bce_loss(fake_vis_pred, vis_img_gt)
            
            real_vis_pix_loss[msk==0]=0
            fake_vis_pix_loss[msk==0]=0
            fake_vis_pix_loss[vis_img_gt==0]=fake_vis_pix_loss[vis_img_gt==0]*5

            real_vis_pix_loss=real_vis_pix_loss.mean()
            fake_vis_pix_loss=fake_vis_pix_loss.mean()

            d_gan_loss = d_logistic_loss(real_pred, fake_pred)
            grad_penalty = d_r1_loss(real_pred, gt_img)
            r1_loss = 300 * 0.5 * grad_penalty
            
            gan_loss_dict["d"] = d_gan_loss
            gan_loss_dict["r1"] = r1_loss
            gan_loss_dict["real_score"] = real_pred.mean()
            gan_loss_dict["fake_score"] = fake_pred.mean()
            gan_loss_dict["real_vis_pix_loss"] = real_vis_pix_loss
            gan_loss_dict["fake_vis_pix_loss"] = fake_vis_pix_loss
            d_loss = d_gan_loss + r1_loss + real_vis_pix_loss + fake_vis_pix_loss
            prefix = 'train/'
            
            for k, val in gan_loss_dict.items():
                self.log(f'train/{k}', val, prog_bar=True)

            return {'loss':d_loss}

    @staticmethod
    def _arrange_nerf_images(out_nerf, znear, zfar):
        tex_map = out_nerf["tex_fg_fine"].clamp(min=0.0, max=1.0)
        tex_map = tex_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return tex_map

    @staticmethod
    def _arrange_src_images(im, render_size=None):
        """
        Args:
            im (torch.tensor): B, 3, H, W
            render_size (int):
        Returns:
            stacked images (np.ndarray) [H, B*W, 3]
        """
        gt_img_stack = []
        im_h, im_w = im.shape[-2:]
        if render_size is not None:
           sc = render_size/max(im_h, im_w)
        for b_ind in range(im.shape[0]):
            gt_img = im[b_ind].permute(1, 2, 0).cpu().numpy()
            if render_size is not None:
                gt_img = cv2.resize(gt_img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            gt_img_stack.append(gt_img)
        gt_img_stack = np.concatenate(gt_img_stack, axis=1)
        return gt_img_stack

    def render_full_nerf_image(self, tr_batch, nerf_level):
        out_nerf = self.model.render_pifu_nerf(
            self=self,
            net=self.model,
            img_in=tr_batch['im'],
            cam_in=tr_batch['cam'],
            hand_type=tr_batch['hand_type'],
            targets=tr_batch['targets'],
            cam_tar=tr_batch['dr_data']['cam_tar'],
            tar_img=tr_batch['dr_data']['tar'],
            sp_data=tr_batch['sp_data'],
            objcenter=tr_batch['dr_data']['objcenter'].squeeze(-1),
            fine=self.model.kwargs["dr_kwargs"]["fine"],
            uniform=True,
            objrad=250.,
            blur=3,
            level=nerf_level,
            sample_per_ray_c=self.model.kwargs["dr_kwargs"]["sample_per_ray_c"],
            sample_per_ray_f=self.model.kwargs["dr_kwargs"]["sample_per_ray_f"],
            src_foreground_mask=tr_batch['src_foreground_mask'],
            bounds=tr_batch['dr_data']['bounds'],
            mask_at_box=tr_batch['dr_data']['mask_at_box'],
        )
        return out_nerf

    @torch.no_grad()
    def render_novel_views(self, cameras, batch, **kwargs):
        # gen voxel grid
        tr_batch = self.decode_batch(batch)
        self.model.attach_im_feat(tr_batch['im'])
        tr_batch['dr_data']['tar'] = None

        rgb_imgs = []
        for camera in tqdm.tqdm(cameras):
            nerf_level = max(0, int(math.log(camera['im_h'], 2)) - 5)
            cam_tar = {
                "K": camera['intrinsics'],
                "RT": camera['w2cs'].unsqueeze(0),
                "KRT": camera['intrinsics'] @ camera['w2cs'].unsqueeze(0),
                "width": camera['im_w'], "height": camera['im_h'], "nml_scale": 100.,
                "znear": camera['znear'], "zfar": camera['zfar'],
            }
            tr_batch['dr_data']['cam_tar'] = cam_tar
            out_nerf = self.render_full_nerf_image(tr_batch, nerf_level)
            rgb_imgs.append(self._arrange_nerf_images(out_nerf, cam_tar['znear'], cam_tar['zfar']))

        rgb_imgs = np.stack(rgb_imgs)  # N, H, Wt, 3
        if kwargs.get('only_renderings', False):
            rgb_imgs = (rgb_imgs*255.).astype(np.uint8)
            src_imgs = (tensor_to_image(tr_batch['im'])*255.).astype(np.uint8)
            return rgb_imgs, src_imgs

        # align images
        src_imgs = self._arrange_src_images(tr_batch['im'], cameras[0]['im_h']) # H,Ws,3
        src_imgs = src_imgs[None].repeat(rgb_imgs.shape[0], axis=0)
        out_img = np.concatenate((src_imgs, rgb_imgs), axis=-2) # N,H,W,3
        out_img = (out_img*255.).astype(np.uint8)
        return out_img

    def validation_step(self, batch, batch_nb):
        prefix = 'val/'
        tr_batch = self.decode_batch(batch)
        out_dict = self.model(**tr_batch)
        
        nerf_level = max(0, int(math.log(tr_batch['im'].shape[-2], 2)) - 5)
        out_nerf = self.render_full_nerf_image(tr_batch, nerf_level)
        renderings = self._arrange_nerf_images(out_nerf, self.znear, self.zfar)  # H, Wt, 3
        src_imgs = self._arrange_src_images(tr_batch['im'], renderings.shape[0])  # H,Ws,3
        gt_img = out_nerf['tar_img'].permute(1, 2, 0).cpu().numpy()
        input_densepose=out_nerf['input_densepose'].permute(1, 2, 0).cpu().numpy()
        tar_densepose=out_nerf['tar_densepose'].permute(1, 2, 0).cpu().numpy()
        vis_img_gt = out_nerf['vis_img'].squeeze(0).expand(3,-1,-1).permute(1, 2, 0).cpu().numpy()
        fake_vis_pred = out_nerf['fake_vis_pred'].squeeze(0).expand(3,-1,-1).permute(1, 2, 0).cpu().numpy()
        real_vis_pred = out_nerf['real_vis_pred'].squeeze(0).expand(3,-1,-1).permute(1, 2, 0).cpu().numpy()
        msk = tr_batch['tar_img_mask'].squeeze(0).expand(3,-1,-1).permute(1, 2, 0).cpu().numpy()
        fake_vis_pred[~msk]=1.
        real_vis_pred[~msk]=1.

        log_img = torch.from_numpy(np.concatenate((src_imgs, gt_img, renderings, input_densepose, tar_densepose, msk, vis_img_gt, real_vis_pred, fake_vis_pred), axis=-2))
        self.logger.experiment.add_image(f'{prefix}renderings', log_img.permute(2, 0, 1), self.global_step)

        out_losses = out_dict['err_dict']
        log = {f'{prefix}{key}': val for key, val in out_losses.items() if torch.is_tensor(val)}
        log['val_total_loss'] = out_dict['loss']

        return log

    def test_step(self, batch, batch_nb):
        self.evaluator.result_dir = os.path.join(
            self.save_dir,
            f'{self.images_dirname}_{self.test_dst_name}')

        tr_batch = self.decode_batch(batch)
        nerf_level = max(0, int(math.log(tr_batch['im'].shape[-2], 2)) - 5)
        out_nerf = self.render_full_nerf_image(tr_batch, nerf_level)
        rendered_image = out_nerf["tex_fg_fine"].clamp(min=0.0, max=1.0)  # 3, H, W
        human_idx = str(batch['human_idx'][0])
        frame_index = str(batch['frame_index'][0])
        view_index = str(batch['cam_ind'][0])
        scores = self.evaluator.compute_score(
            rendered_image,
            tr_batch['dr_data']['tar'],
            input_imgs=tr_batch['im'],
            mask_at_box=tr_batch['dr_data']['mask_at_box'],
            human_idx=human_idx,
            frame_index=frame_index,
            view_index=view_index,
        )
        scores = {key: torch.tensor(val) for key, val in scores.items()}
        return scores

    def validation_epoch_end(self, outputs):
        for key in outputs[0].keys():
            self.log(key, torch.stack([x[key] for x in outputs]).mean(), prog_bar=False)
        return

class VANeRF(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        model_cfg = cfg['models']['VANeRF']
        self.train_out_h = model_cfg.get('train_out_h', 64)
        self.train_out_w = model_cfg.get('train_out_w', 64)
        mlp_geo, mlp_tex, sp_encoder_postfusion = None, None, None
        self.disable_fg_mask = model_cfg.get('disable_fg_mask', False)
        self.nkpt_r,self.nkpt_l=21,21
        self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))
        
        if model_cfg.get('sp_conv'):
            self.geo_vis_fusion_spconv = GeoVisFusion_spconv()
            self.tex_vis_fusion_spconv = TexVisFusion_spconv()
        else:
            self.geo_vis_fusion = GeoVisFusion()
            self.tex_vis_fusion = TexVisFusion()

        sp_encoder = SpatialEncoder(**model_cfg["sp_args"])
        mlp_geo_args = copy.deepcopy(model_cfg['mlp_geo_args'])
        mlp_geo_args["n_dims1"][0] = sp_encoder.get_dim() 


        mlp_geo = MLPUNetFusion(**mlp_geo_args)

        geo_encoder = HGFilterV2(**model_cfg['geo_args'])

        mlp_tex = IBRRenderingHead(**model_cfg['mlp_tex_args']["args"])
        self.ibr_compress_gfeat = torch.nn.Linear(
            model_cfg['mlp_tex_args']['gcompress']['in_ch'],
            model_cfg['mlp_tex_args']['gcompress']['out_ch'], 
        )
        geo_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(geo_encoder)

        # set modules
        self.mlp_geo = mlp_geo
        self.mlp_tex = mlp_tex
        self.geo_encoder = geo_encoder
        self.sp_encoder = sp_encoder
        self.sp_encoder_r = SpatialEncoder(**model_cfg["sp_args"])
        self.sp_encoder_l = SpatialEncoder(**model_cfg["sp_args"])
        self.sp_encoder_postfusion = sp_encoder_postfusion

        # downsampling factor of image image
        self.ds_geo = model_cfg.get('ds_geo', 0)
        self.ds_tex = model_cfg.get('ds_tex', 0)
        self.v_level = model_cfg.get('v_level', 0)

        self.dr_level = model_cfg.get('dr_level', 5)
        self.tex_encoder = ResBlkEncoder(**model_cfg['tex_args'])

        self.feat_geo = None
        self.feat_tex = None

        self.kwargs = model_cfg

        self.init_weights(self)
        self.init_weights(self.mlp_geo, "kaiming", nl="relu")
        self.init_weights(self.mlp_tex, "kaiming", nl="leaky_relu")

        # parse kwargs
        self.vgg_loss = VGGLoss()
        self.disable_bg = True

    @staticmethod
    def init_weights(net, init_type="normal", gain=0.02, nl="relu"):
        def init_func(m):
            torch.manual_seed(125)
            torch.cuda.manual_seed_all(125)
            torch.backends.cudnn.deterministic = True

            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in", nonlinearity=nl)
                elif init_type == "orthogonal":
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

        net.apply(init_func)

    def attach_im_feat(self, im, return_val=False):
        if return_val:
            out = {}
            out["feat_geo"] = self.attach_geo_feat(im, return_val)
            feat_tex = self.attach_tex_feat(im, return_val)
            if feat_tex is not None:
                out["feat_tex"] = feat_tex
        else:
            self.attach_geo_feat(im, return_val)
            self.attach_tex_feat(im, return_val)

    def attach_geo_feat(self, im, return_val=False):
        if not return_val:
            self.im = im.clone()
        if len(im.shape) == 5:
            im = im.view(-1, *im.shape[2:])
        im_ds = im
        for _ in range(self.ds_geo):
            im_ds = thf.avg_pool2d(im_ds, 2, stride=2)
        # normalize to -1 to 1
        feat_geo = self.geo_encoder(2.0 * im_ds - 1.0)

        self.feat_geo = feat_geo
        if return_val:
            return feat_geo

    def attach_tex_feat(self, im, return_val=False):
        if self.tex_encoder is None:
            return None
        if len(im.shape) == 5:
            im = im.view(-1, *im.shape[2:])
        for _ in range(self.ds_tex):
            im = thf.avg_pool2d(im, 2, stride=2)
        
        # normalize to -1 to 1
        feat_tex = self.tex_encoder(2.0 * im - 1.0)
        self.feat_tex = feat_tex
        if return_val:
            return feat_tex
        
    def detach_im_feat(self):
        if self.feat_geo is not None:
            del self.feat_geo
            self.feat_geo = None
        if self.feat_tex is not None:
            del self.feat_tex
            self.feat_tex = None

    def query(
        self,
        pts,  # (B, N, 3)
        cam,
        hand_type,
        targets,
        feat_geo=None,
        feat_tex=None,
        vert=None,
        vert_vis=None,
        query_vis=None,
        query_sdf=None,
        closest_face=None,
        n_views=1,
        sp_data={},
        tx_data={},
        view=None,
        n_pts_samples=-1,
        **kwargs):
        # B, T, _ = pts.shape
        B = hand_type.shape[0]
        T = pts.shape[1]
        sample_func = feat_sample

        if n_views != 1 and pts.shape[0] == B:
            v = pts[:, None].expand(-1, n_views, -1, -1)
            v = v.reshape(-1, *v.shape[2:])
        else: 
            v = pts
        if feat_geo is None:
            feat_geo = self.feat_geo

        vh = v @ cam["KRT"][:, :3, :3].transpose(1, 2) + cam["KRT"][:, :3, 3][:, None]
        z = vh[..., 2:3]
        xy = vh[..., :2] / z
        if "transf" in cam:
            transf = cam["transf"]
            xy = xy @ transf[:, :2, :2].transpose(1, 2) + transf[:, :, 2][:, None]
        xy[..., 0] = 2.0 * (xy[..., 0] / (cam["width"] - 1.0)) - 1.0
        xy[..., 1] = 2.0 * (xy[..., 1] / (cam["height"] - 1.0)) - 1.0
        z = 2.0 * (z - cam["znear"]) / (cam["zfar"] - cam["znear"]) - 1.0
        epsilon = 1e-2
        mask_xy = (xy >= -1.0 - epsilon) & (xy <= 1.0 + epsilon)
        mask_z = (z >= -1.0)
        out_mask = (mask_xy[..., 0] & mask_xy[..., 1] & mask_z[..., 0])[..., None].float()
        out_mask = out_mask.view(-1, n_views, *out_mask.shape[1:])
        fg_mask = kwargs['src_foreground_mask'] #V,1,H,W
        fg_mask = fg_mask.view(-1, 1, *fg_mask.shape[-2:])
        if True:
            if self.disable_fg_mask:
                out_mask = out_mask*out_mask.bool().all(1, keepdim=True)
            else:
                fg_mask_xy = sample_func(fg_mask.float(), xy)
                fg_mask_xy = fg_mask_xy.view(-1, n_views, *fg_mask_xy.shape[1:])
                out_mask = out_mask*(fg_mask_xy > 0.1).all(1, keepdim=True)*out_mask.bool().all(1, keepdim=True)
        # view dropout
        if self.training and n_views > 1:
            dropout = torch.zeros_like(out_mask[:, :, :1])
            dropout[:, :1] = 1.0
            dropout[:, 1:] = (torch.rand_like(dropout[:, 1:]) > 0.5).float()
            rand_perm = torch.rand_like(dropout).argsort(dim=1)
            dropout = torch.gather(dropout, 1, rand_perm)
            out_mask *= dropout

        # smooth mask
        std = 0.1
        with torch.no_grad():
            xyz = 0.5 * torch.cat([xy, z], -1) + 0.5  # [0, 1] # (BV, N, 3)
            dist_boundary = torch.min(xyz, 1.0 - xyz)
            pix_weight = torch.sigmoid(5.0 * (dist_boundary / std - 1.0))  # sigmoid
            pix_weight = pix_weight[..., 0] * pix_weight[..., 1] * pix_weight[..., 2]
            pix_weight = pix_weight.view(-1, n_views, pix_weight.shape[1], 1)
            pix_weight = pix_weight * out_mask
            pix_weight = pix_weight / (pix_weight.sum(1, keepdim=True) + 1e-6)  # normalize

        if isinstance(feat_geo, list):
            feat_sampled = []
            for f in feat_geo:
                feat = sample_func(f, xy)
                feat_sampled.append(feat.view(-1, n_views, *feat.shape[-2:]))
        else:
            feat = sample_func(feat_geo, xy)
            feat_sampled = [feat.view(-1, n_views, *feat.shape[-2:])]

        kpt3do = sp_data['kpt3d']
        sp_data.update({"n_view": n_views, "pts": pts, "v": v, "z": z, "xy": xy})
        sp_data.update(cam)

        y = self.sp_encoder(**sp_data)

        if y is not None:
            y = y.view(-1, n_views, *y.shape[1:])
        y_all = None

        vert3d=targets['vert_world']
        face=targets['face_world'].long()

        vert3d_img = vert3d @ cam["KRT"][:, :3, :3].transpose(1, 2) + cam["KRT"][:, :3, 3][:, None]
        z = vert3d_img[..., 2:3]
        vert_xy = vert3d_img[..., :2] / (z + 1e-8)
        if "transf" in cam:
            transf = cam["transf"]
            vert_xy = vert_xy @ transf[:, :2, :2].transpose(1, 2) + transf[:, :, 2][:, None]
        # normalize it to [-1, 1]
        vert_xy[..., 0] = 2.0 * (vert_xy[..., 0] / (cam["width"] - 1.0)) - 1.0
        vert_xy[..., 1] = 2.0 * (vert_xy[..., 1] / (cam["height"] - 1.0)) - 1.0

        if self.kwargs.get('sp_conv'):
            coord, out_sh, bounds=targets['coord'], targets['out_sh'],targets['bounds']
            out_sh, _ = torch.max(out_sh, dim=0)
            out_sh = out_sh.tolist()
            sh = coord.shape
            idx = [torch.full([sh[1]], i) for i in range(sh[0])]
            idx = torch.cat(idx).to(coord)
            coord = coord.view(-1, sh[-1])
            coord= torch.cat([idx[:, None], coord], dim=1)
            query_sdf = self.sdf_activation(-query_sdf)
            feat_sampled=self.geo_vis_fusion_spconv(vert_xy, feat_geo, feat_sampled, vert3d, v, vert_vis.type(torch.int), query_vis, closest_face, query_sdf.unsqueeze(-1), coord, out_sh, bounds)
        else:
            if n_views != 1:
                vert3d = vert3d[:, None].expand(-1, n_views, -1, -1)
                vert3d = vert3d.reshape(-1, *vert3d.shape[2:])
            feat_sampled=self.geo_vis_fusion(vert_xy, feat_geo, feat_sampled, vert3d, v, vert_vis.type(torch.int), query_vis, closest_face, query_sdf.unsqueeze(-1))

        out, valid, latent_view, latent_fused = self.mlp_geo(y, feat_sampled, out_mask, pix_weight, y_all)
        img = tx_data['img']
        out_mask = out_mask.view(B*n_views, T, 1)
        rgb = self.query_color(vert3d,vert_xy,vert_vis,query_vis,closest_face,v, xy, view, n_views, feat_tex, latent_fused, cam, img, out_mask, n_pts_samples, targets)
        out = torch.cat([out, rgb], -1)  # [_, alpha, rgb]
        return out, valid

    def sdf_activation(self, input):
        self.sigmoid_beta.data.copy_(max(torch.zeros_like(self.sigmoid_beta.data) + 2e-3, self.sigmoid_beta.data))
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta
        return sigma

    def query_color(self,vert3d,vert_xy,vert_vis,query_vis,closest_face, v, xy, view, n_views, feat_tex, latent_fused, cam, img, out_mask, n_samples, targets):
        '''
        Args:
            v (torch.tensor): Replicated query points (B*V, N, 3)
            xy (torch.tensor): Corresponding pixel locations of the query points (B*V, N, 2)
            view (torch.tensor): ray directions (B, N, 3)
            n_views (int): the number of input images

            feat_tex (torch.tensor): Feature planes (B*V, C, H, W)
            latent_fused (torch.tensor): Fused features at the query points `v`. (B, N, f_len)

            cam (dict): parameters of the input cameras
            img (torch.tensor): input source images  (B*V, 3, H, W)
            out_mask (torch.tensor): Validity mask (B*V, N, 1)

            n_samples(int): the number of samples per ray. 
                (N is composed of path H*W*n_samples)
        Returns:
            rgb (torch.tensor): (B,N,3)
        '''
        BV, N, _ = v.shape
        B = BV//n_views
        img_xy = feat_sample(img, xy).view(B, n_views, N, 3)  # (B, V, N, 3)
        # img_xy = img_xy*query_vis

        pHW = N//n_samples
        if n_views > 1:  # pad tensors
            assert (B, N) == latent_fused.shape[:2]
            latent_fused = latent_fused.unsqueeze(1).expand(-1, n_views, -1, -1).reshape(BV, N, -1)
            view = view.view(B, 1, N, 3).expand(-1, n_views, -1, -1).reshape(BV, N, 3)  # BV,N,3

        # query texture
        if feat_tex is None:
            feat_tex = self.tex_feat
            assert v.shape[0] == feat_tex.shape[0]
        feat_xy = feat_sample(feat_tex, xy)  # (BV, N, feat_leat)

        latent_fused = self.ibr_compress_gfeat(latent_fused)
        
        if self.kwargs.get('sp_conv'):
            coord, out_sh, bounds=targets['coord'], targets['out_sh'],targets['bounds']
            out_sh, _ = torch.max(out_sh, dim=0)
            out_sh = out_sh.tolist()
            sh = coord.shape
            idx = [torch.full([sh[1]], i) for i in range(sh[0])]
            idx = torch.cat(idx).to(coord)
            coord = coord.view(-1, sh[-1])
            coord= torch.cat([idx[:, None], coord], dim=1)
            rgb_feat =self.tex_vis_fusion_spconv(vert_xy,feat_tex,feat_xy,vert3d,v, vert_vis.type(torch.int), query_vis, img_xy.view(-1, *img_xy.shape[-2:]), img, latent_fused, coord, out_sh, bounds)
        else:
            rgb_feat =self.tex_vis_fusion(vert_xy,feat_tex,feat_xy,vert3d,v, vert_vis.type(torch.int), query_vis, img_xy.view(-1, *img_xy.shape[-2:]), img, latent_fused)
        
        calib = cam['KRT']
        inv_calib = torch.inverse(calib.float())  # (BV, 4, 4)
        cam_pos = inv_calib[:, :3, 3:4]  # (BV, 3, 1)
        cam_rays = thf.normalize(v - cam_pos.view(-1, 1, 3), p=2, dim=-1)  # (BV, N, 3)

        # encode cam rays of input images
        ray_diff = (view - cam_rays).view(B, n_views, N, 3)  # B,V,N,3
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = (cam_rays * view).sum(-1).view(B,n_views,N,1)  # B,V,N,1
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1) # BVN4

        # prep ibr net params
        rgb_feat = rgb_feat.view(B, n_views, pHW, n_samples, -1).permute(0, 2, 3, 1, 4).reshape(B*pHW, n_samples, n_views, -1)
        ray_diff = ray_diff.view(B, n_views, pHW, n_samples, -1).permute(0, 2, 3, 1, 4).reshape(B*pHW, n_samples, n_views, -1)
        out_mask = out_mask.view(B, n_views, pHW, n_samples, -1).permute(0, 2, 3, 1, 4).reshape(B*pHW, n_samples, n_views, -1)

        out = self.mlp_tex(
            rgb_feat, ray_diff, out_mask
        )  # B*pHW,n_samples,3
        out = out.reshape(B, N, 3)
        return out

    def forward(self, im, cam,hand_type, targets, data, bbox, n_views=1, sp_data={}, dr_data=None, **kwargs):
        '''
        args:
            im: (B*V, C, H, W)
            v: (B, N, 3)
        '''
        assert (len(im.shape) == 4 and len(cam["KRT"].shape) == 3)
        n_batch = im.shape[0] // n_views

        lambdas = self.kwargs.get("lambdas", {})
        dr_kwargs = self.kwargs.get("dr_kwargs", {})

        feat_geo = self.attach_geo_feat(im, return_val=True) #([1, 64, 32, 32])  ([1, 8, 128, 128])
        feat_tex = self.attach_tex_feat(im, return_val=True) #([1, 8, 64, 64]) 
        
        kpt3d=sp_data["kpt3d"]
        
        kapt3d_img = kpt3d @ cam["KRT"][:, :3, :3].transpose(1, 2) + cam["KRT"][:, :3, 3][:, None]
        z = kapt3d_img[..., 2:3]
        ka_xy = kapt3d_img[..., :2] / (z + 1e-8)
        if "transf" in cam:
            transf = cam["transf"]
            ka_xy = ka_xy @ transf[:, :2, :2].transpose(1, 2) + transf[:, :, 2][:, None]
        # normalize it to [-1, 1]
        ka_xy[..., 0] = 2.0 * (ka_xy[..., 0] / (cam["width"] - 1.0)) - 1.0
        ka_xy[..., 1] = 2.0 * (ka_xy[..., 1] / (cam["height"] - 1.0)) - 1.0

        # random stride
        if self.dr_level == 1:
            stride = 0
        else:
            stride = torch.randint(high=(2 ** (self.dr_level - 1) - 1), size=(n_batch, 2))

        out_nerf = self.batch_render_pifu_nerf(
            net=self,
            img_in=dr_data['img'],
            cam_in=dr_data['cam'],
            hand_type=hand_type,
            targets=targets,
            n_views=n_views,
            cam_tar=dr_data['cam_tar'],
            level=self.dr_level,
            stride=stride,
            tar_img=dr_data['tar'],
            bg_img=dr_data['bg'],
            feat_geo=feat_geo,
            feat_tex=feat_tex,
            # mano_vert_world=mano_vert_world,
            sp_data=sp_data,
            camcenter=dr_data.get('camcenter', None),
            objcenter=dr_data.get('objcenter', None),
            msk=dr_data['msk'],
            src_foreground_mask=kwargs['src_foreground_mask'],
            bounds=kwargs['bounds'],
            **dr_kwargs)
        if self.disable_bg:
            out_nerf["tex_bg"] = 0.0

        out_nerf["tex"] = out_nerf["tex_fg"]
        out_nerf["tex_cal"] = out_nerf["tex_fg"]
        if "tex_fg_fine" in out_nerf:
            out_nerf["tex_fine"] = out_nerf["tex_fg_fine"]
            out_nerf["tex_cal_fine"] = out_nerf["tex_fg_fine"]

        loss, err_dict = compute_error(inter_loss=None, out_nerf=out_nerf, vggloss=self.vgg_loss, lambdas=lambdas)
        return dict(loss=loss, err_dict=err_dict, out={"nerf": out_nerf})

    @staticmethod
    def render_pifu_nerf(
        self,
        net,
        img_in,  # for pifu (V, C, H, W)
        cam_in,
        hand_type,
        targets,
        cam_tar,
        level=5,
        sp_data={},
        bkg_emb=None,
        camcenter=None,
        objcenter=None,
        tar_img=None,
        **config
    ):
        # n_views = img_in.shape[0]
        n_views = 1


        feat_geo = net.attach_geo_feat(img_in, return_val=True)
        feat_tex = net.attach_tex_feat(img_in, return_val=True)
        
        stride = 2 ** (level - 1)
        ret = {}
        for i in range(stride):
            for j in range(stride):
                strd = th.Tensor([[j, i]]).cuda()
                # NOTE: better to take args with **kwargs to avoid bugs
                out = net.batch_render_pifu_nerf(net, img_in, cam_in,hand_type,targets, n_views, cam_tar, level, strd,
                    tar_img, feat_geo, feat_tex, None, sp_data, objcenter, **config)
                
                if 'input_densepose' in out:
                    gt_img = out["tar_img"]
                    input_img=out["img_in"]
                    msk = out["input_mask"]
                    rendered_image = out["tex_fg_fine"].clamp(min=0.0, max=1.0)  # 3, H, W
                    input_densepose=out['input_densepose']
                    tar_densepose=out['tar_densepose']
                    fake_pred,fake_vis_pred=self.discriminator(input_img, input_densepose, tar_densepose, rendered_image.detach())
                    out["fake_vis_pred"]=fake_vis_pred
                    real_pred,real_vis_pred=self.discriminator(input_img, input_densepose, tar_densepose, gt_img)
                    out["real_vis_pred"]=real_vis_pred

                
                for k, v in out.items():
                    if v is None or len(v.shape) < 3:
                        continue
                    if len(v.shape) == 3:
                        v = v[:, None]
                    if k in ret:
                        ret[k].append(v)
                    else:
                        ret[k] = [v]

        for k in ret.keys():
            ret[k] = th.stack(ret[k], 2)
            ret[k] = ret[k].reshape(ret[k].shape[0], -1, *ret[k].shape[3:])
            ret[k] = thf.pixel_shuffle(ret[k], stride)[0]

        cam=cam_tar
        vert3d=targets['vert_world']
        face=targets['face_world'].long()
        mano_vert_world_img = vert3d @ cam["KRT"][:, :3, :3].transpose(1, 2) + cam["KRT"][:, :3, 3][:, None]
        z = mano_vert_world_img[..., 2:3]
        vert_xy = mano_vert_world_img[..., :2] / (z + 1e-8)
        if "transf" in cam:
            transf = cam["transf"]
            vert_xy = vert_xy @ transf[:, :2, :2].transpose(1, 2) + transf[:, :, 2][:, None]
        ret['vert_xy']=vert_xy

        ret['vert_vis']=out['vert_vis']

        return ret

    @staticmethod
    def batch_render_pifu_nerf(
        net,
        img_in,  # for pifu (B*V, C, H, W)
        cam_in,
        hand_type,
        targets,
        n_views,
        cam_tar,
        level=2,  # 1 is the finest
        stride=0,  # this is valid only if min_level != 1
        tar_img=None,
        feat_geo=None,
        feat_tex=None,
        mano_vert_world=None,
        sp_data={},
        objcenter=None,
        **config
    ):
        batch_size = cam_tar["K"].shape[0]
        # get config values
        sample_per_ray_c = config.get("sample_per_ray_c", 64)
        sample_per_ray_f = config.get("sample_per_ray_f", 64)
        fine = config.get("fine", False)
        uniform = config.get("uniform", False)
        separate_cf = config.get("separate_cf", False)
        rand_noise_std = config.get("rand_noise_std", 0.0)

        if feat_geo is None:
            feat_geo = net.attach_geo_feat(img_in, return_val=True)
        if feat_tex is None:
            feat_tex = net.attach_tex_feat(img_in, return_val=True)
        
        width = cam_tar.get("width", cam_in["width"])
        height = cam_tar.get("height", cam_in["height"])
        znear = cam_tar.get("znear", cam_in["znear"])
        zfar = cam_tar.get("zfar", cam_in["zfar"])

        def eval_func(v, view, n_pts_samples,vert_vis=None,query_vis=None,query_sdf=None,closest_face=None, fine=False):  # (B, N, 3), (B, N, 3)
            rgba, mask = net.query(v, cam_in,hand_type, targets, feat_geo, feat_tex,vert_vis=vert_vis,query_sdf=query_sdf,query_vis=query_vis,closest_face=closest_face, n_views=n_views, view=view,
                nerf=True, sp_data=sp_data, tx_data={"img": img_in}, bbox_center=objcenter, n_pts_samples=n_pts_samples, src_foreground_mask=config['src_foreground_mask'])
            mask = mask.float()
            sdf = mask * rgba[..., :1] + (1.0 - mask) * (0.1 / cam_in["nml_scale"])
            rad_sign = 1
            if separate_cf:
                assert rgba.shape[-1] == 6 or rgba.shape[-1] == 9
                rad = rad_sign*rgba[..., 2:3] if fine else rad_sign*rgba[..., 1:2]
                rgb = rgba[..., 3:]
            else:
                assert rgba.shape[-1] == 5 or rgba.shape[-1] == 8
                rad = rad_sign*rgba[..., 1:2]
                rgb = rgba[..., 2:]
            
            if rand_noise_std > 0.0:
                rad += th.randn_like(rad) * rand_noise_std

            alpha = mask * thf.relu(rad)

            return th.cat([alpha, sdf, rgb], -1) #, offset

        assert width % (2 ** (level - 1)) == 0 and height % (2 ** (level - 1)) == 0
        if isinstance(stride, int):
            assert stride < 2 ** (level - 1)
        elif isinstance(stride, th.Tensor):
            assert stride.max().item() < 2 ** (level - 1)
            stride = stride[:, None].cuda()
        else:
            raise NotImplementedError("unsupported stride type")

        grids_b=[]
        if net.training and 'msk' in config:
            out_h, out_w = net.train_out_h, net.train_out_w

            for b in range(batch_size):
                msk = config['msk'][b,:,:].squeeze() # B,H,W

                msk_coords = th.stack(th.where(msk)[::-1], -1)
                if msk_coords.shape[0]<=0:
                    grid_center=torch.zeros((1,2),device=msk.device)
                else:
                    grid_center = msk_coords[np.random.randint(0, msk_coords.shape[0], 1)]
                y_grid, x_grid = th.meshgrid(th.arange(0, out_h).cuda(), th.arange(0, out_w).cuda())
                grids = th.stack([x_grid, y_grid], -1).view(-1, 2).contiguous()
                grids = grids + (grid_center-out_h//2)
                grids = grids.clamp(0, min(width-1, height-1))
                grids = grids[None].expand(1, -1, -1)  # position grid at center
                grids_b.append(grids)
            grids=torch.cat(grids_b,0)
        else:
            out_w, out_h = width // (2 ** (level - 1)), height // (2 ** (level - 1))

            for b in range(batch_size):
                y_grid, x_grid = th.meshgrid(th.arange(0, height, 2 ** (level - 1)).cuda(), th.arange(0, width, 2 ** (level - 1)).cuda())
                grids = th.stack([x_grid, y_grid], -1).view(-1, 2).contiguous()
                grids = grids[None].expand(1, -1, -1) + stride
                grids_b.append(grids)
            grids=torch.cat(grids_b,0)

        index = grids[..., 0] + grids[..., 1] * width
        grids = grids.float() # (B, h*w, xy)

        grids_h = th.cat([grids, th.ones_like(grids[..., :1])], -1)
        grids_znear_h = znear * th.cat([grids, th.ones_like(grids[..., :1])], -1)
        grids_zfar_h = zfar * th.cat([grids, th.ones_like(grids[..., :1])], -1)

        # transform ray into world space
        inv_K = th.inverse(cam_tar["K"][:, :3, :3]).transpose(1, 2)
        cam_rays = th.bmm(grids_h, inv_K)
        znear_rays = th.norm(th.bmm(grids_znear_h, inv_K), p=2, dim=-1, keepdim=True)
        zfar_rays = th.norm(th.bmm(grids_zfar_h, inv_K), p=2, dim=-1, keepdim=True)
        cam_rays = thf.normalize(th.bmm(cam_rays, cam_tar["RT"][:, :3, :3]), p=2, dim=-1)  # (B, h*w, 3)
        cam_pos = -th.bmm(cam_tar["RT"][:, :3, 3][:, None], cam_tar["RT"][:, :3, :3])  # (B, 1, 3)

        with torch.no_grad():
            z1, z2, hit = net.ray_bbox_intersection(config['bounds'], cam_pos, cam_rays)
        mask_z1 = (hit & (z1 > znear_rays)).float()
        znear_rays = mask_z1 * z1 + (1.0 - mask_z1) * znear_rays
        mask_z2 = (hit & (z2 < zfar_rays)).float()
        zfar_rays = mask_z2 * z2 + (1.0 - mask_z2) * zfar_rays  # (B,h*w,1)
        
        z = th.linspace(0.0, 1.0, steps=sample_per_ray_c).to(cam_pos.device)
        z = z[None, None, :].expand(*znear_rays.shape[:2], -1)  # (B,h*w,sample_c)
        z_mid = 0.5 * (z[..., 1:] + z[..., :-1])  # (B,h*w,sample_c)

        if not uniform:
            z_lower = th.cat([z[..., :1], z_mid], -1)
            z_upper = th.cat([z_mid, z[..., -1:]], -1)
            z = z_lower + th.rand_like(z) * (z_upper - z_lower)
            z = znear_rays + (zfar_rays - znear_rays) * z
        else:
            z = znear_rays + (zfar_rays - znear_rays) * z

        eval_pts = cam_pos[:, :, None] + cam_rays[:, :, None] * z[..., None]  # (B,h*w,sample_c,3)
        eval_pts = eval_pts.view(batch_size, -1, 3)  # (B,h*w*sample_c,3)

        view = cam_rays[:, :, None, :].expand(-1, -1, sample_per_ray_c, -1)  # (B,h*w,sample_c,3)
        view = view.reshape(batch_size, -1, 3)  # (B,h*w*sample_c,3)
        
        
        face=targets['face_world'].long()
        vert3d=targets['vert_world']

        cam=cam_in
        vert3d_img = vert3d @ cam["KRT"][:, :3, :3].transpose(1, 2) + cam["KRT"][:, :3, 3][:, None]
        vert_z = vert3d_img[..., 2:3]
    
        vert_xy = vert3d_img[..., :2] / (vert_z + 1e-8)
        if "transf" in cam:
            transf = cam["transf"]
            vert_xy = vert_xy @ transf[:, :2, :2].transpose(1, 2) + transf[:, :, 2][:, None]
        
        vert_xy[..., 0] =  (vert_xy[..., 0] / (cam["width"] - 1.0)) 
        vert_xy[..., 1] =  (vert_xy[..., 1] / (cam["height"] - 1.0)) 
        vert_z =  (vert_z - cam["znear"]) / (cam["zfar"] - cam["znear"])

        q3d_img = eval_pts @ cam["KRT"][:, :3, :3].transpose(1, 2) + cam["KRT"][:, :3, 3][:, None]
        q_z = q3d_img[..., 2:3]
    
        q_xy = q3d_img[..., :2] / (q_z + 1e-8)
        if "transf" in cam:
            transf = cam["transf"]
            q_xy = q_xy @ transf[:, :2, :2].transpose(1, 2) + transf[:, :, 2][:, None]
        
        q_xy[..., 0] =  (q_xy[..., 0] / (cam["width"] - 1.0)) 
        q_xy[..., 1] =  (q_xy[..., 1] / (cam["height"] - 1.0)) 
        q_z =  (q_z - cam["znear"]) / (cam["zfar"] - cam["znear"])

        # print("vis")
        # print(config['src_foreground_mask'].shape)
        sample_func = feat_sample
        vert_vis0 = sample_func(config['src_foreground_mask'].squeeze(1).squeeze(2).float(), vert_xy.squeeze(2))
        # print(vert_vis0.shape)

        q_vis0 = sample_func(config['src_foreground_mask'].squeeze(1).squeeze(2).float(), q_xy.squeeze(2))
        # print(q_vis0.shape)

        query_sdfs, query_viss, vert_viss, closest_faces=[],[],[],[]
        for b in range(batch_size):
            query_sdf, query_vis, vert_vis, closest_face=cal_vis_sdf_batch(vert3d[b:b+1,...], face, eval_pts[b:b+1,...],vert_xy[b:b+1,...],vert_z[b:b+1,...])
            query_sdfs.append(query_sdf)
            query_viss.append(query_vis)
            vert_viss.append(vert_vis)
            closest_faces.append(closest_face)
        query_sdf=torch.cat(query_sdfs,0)
        query_vis=torch.cat(query_viss,0)
        vert_vis=torch.cat(vert_viss,0)
        closest_face=torch.cat(closest_faces,0)
        rgba_coarse = eval_func(eval_pts, view, sample_per_ray_c, vert_vis=vert_vis,query_vis=query_vis,query_sdf=query_sdf,closest_face=closest_face)  # (B,h*w*sample_c,5)
        rgba = rgba_coarse.view(batch_size, -1, sample_per_ray_c, rgba_coarse.shape[-1])  # (B,h*w,sample_c,5)
        
        query_sdf=query_sdf.view(batch_size, -1, sample_per_ray_c, 1)

        color, depth, alpha, contrib, sdf = net.rgba2out(net, rgba, z, query_sdf)  # (B,h*w,3)
        color = color.view(batch_size, out_h, out_w, 3).permute(0, 3, 1, 2)  # (B,h*w,3) -> (B,h,w,3)
        depth = depth.view(batch_size, out_h, out_w)
        alpha = alpha.view(batch_size, out_h, out_w)
        contrib = contrib.view(batch_size, out_h, out_w, sample_per_ray_c) # (B,h,w,samp_c)
        out = {"tex_fg": color, "depth": depth, "alpha": alpha}

        if fine:
            contrib = contrib.view(batch_size, -1, sample_per_ray_c)
            z_mid = 0.5 * (z[..., 1:] + z[..., :-1])
            z_fine = net.importance_sample(contrib[..., 1:-1], z_mid, sample_per_ray_f, uniform=uniform)
            z_fine = th.sort(th.cat([z, z_fine], -1), -1)[0]
            eval_pts = cam_pos[:, :, None] + cam_rays[:, :, None] * z_fine[..., None]
            eval_pts = eval_pts.view(batch_size, -1, 3)  # (B,h*w*sample_f,3)

            q3d_img = eval_pts @ cam["KRT"][:, :3, :3].transpose(1, 2) + cam["KRT"][:, :3, 3][:, None]
            q_z = q3d_img[..., 2:3]
        
            q_xy = q3d_img[..., :2] / (q_z + 1e-8)
            if "transf" in cam:
                transf = cam["transf"]
                q_xy = q_xy @ transf[:, :2, :2].transpose(1, 2) + transf[:, :, 2][:, None]
            
            q_xy[..., 0] =  (q_xy[..., 0] / (cam["width"] - 1.0)) 
            q_xy[..., 1] =  (q_xy[..., 1] / (cam["height"] - 1.0)) 
            q_z =  (q_z - cam["znear"]) / (cam["zfar"] - cam["znear"])

            # print("vis")
            # print(config['src_foreground_mask'].shape)
            sample_func = feat_sample
            q_vis0 = sample_func(config['src_foreground_mask'].squeeze(1).squeeze(2).float(), q_xy.squeeze(2))
            # print(q_vis0.shape)

            query_sdfs, query_viss, vert_viss, closest_faces=[],[],[],[]
            for b in range(batch_size):
                query_sdf, query_vis, vert_vis, closest_face=cal_vis_sdf_batch(vert3d[b:b+1,...], face, eval_pts[b:b+1,...],vert_xy[b:b+1,...],vert_z[b:b+1,...])
                query_sdfs.append(query_sdf)
                query_viss.append(query_vis)
                vert_viss.append(vert_vis)
                closest_faces.append(closest_face)
            query_sdf=torch.cat(query_sdfs,0)
            query_vis=torch.cat(query_viss,0)
            vert_vis=torch.cat(vert_viss,0)
            closest_face=torch.cat(closest_faces,0)

            query_sdf_fine=query_sdf.view(batch_size, -1, sample_per_ray_c*2, 1)


            view = cam_rays[:, :, None, :].expand(-1, -1, z_fine.shape[-1], -1)
            view = view.reshape(batch_size, -1, 3)

            rgba_fine = eval_func(eval_pts, view, sample_per_ray_f,vert_vis=vert_vis,query_vis=query_vis,query_sdf=query_sdf,closest_face=closest_face,fine=separate_cf)
            rgba_fine = rgba_fine.view(*z_fine.shape, rgba_fine.shape[-1])

            
            color_fine, depth_fine, alpha_fine, contrib, sdf = net.rgba2out(net, rgba_fine, z_fine, query_sdf_fine)  # (B,h*w,3)
            color_fine = color_fine.view(batch_size, out_h, out_w, 3).permute(0, 3, 1, 2)
            depth_fine = depth_fine.view(batch_size, out_h, out_w)
            alpha_fine = alpha_fine.view(batch_size, out_h, out_w)
            sdf = sdf.view(batch_size, out_h, out_w)

            out.update({
                "tex_fg_fine": color_fine,
                "depth_fine": depth_fine,
                "alpha_fine": alpha_fine,
                "sdf": sdf,
            })
        if tar_img is not None:
            index = index.long()
            with th.no_grad():
                assert tar_img.shape[0] == index.shape[0]
                tar_img = tar_img.reshape(*tar_img.shape[:2], -1)
                tar_img = th.gather(tar_img, 2, index[:, None].expand(-1, 3, -1))
                out["tar_img"] = tar_img.view(*tar_img.shape[:2], out_h, out_w)
                if 'msk' in config:
                    alpha_img = config['msk'].reshape(batch_size, 1, -1)
                    alpha_img = th.gather(alpha_img, 2, index[:, None].expand(-1, 1, -1))
                    out["tar_alpha"] = alpha_img.view(*alpha_img.shape[:2], out_h, out_w).float()

        tar_cam=targets['tar_cam']

        vis_img_rbgs,vis_imgs=[],[]
        for b in range(batch_size):
            vis_img_rbg,vis_img=render_vis(vert3d[b:b+1,...].float(), face, vert_vis[b:b+1,...].float(), tar_cam['tar_R'][b:b+1,...].float(), tar_cam['tar_T'][b:b+1,...].float(), tar_cam['tar_focal'][b:b+1,0].float(), tar_cam['tar_focal'][b:b+1,1].float(), tar_cam['tar_princpt'][b:b+1,0].float(), tar_cam['tar_princpt'][b:b+1,1].float(), device=vert3d.device)
            vis_img_rbgs.append(vis_img_rbg)
            vis_imgs.append(vis_img)
        vis_img_rbg=torch.cat(vis_img_rbgs,0)
        vis_img=torch.cat(vis_imgs,0)

        out["vis_img_all"]=vis_img
        index=index.long()
        with th.no_grad():
            assert vis_img.shape[0] == index.shape[0]
            vis_img = vis_img.reshape(*vis_img.shape[:2], -1)
            vis_img = th.gather(vis_img, 2, index[:, None].expand(-1, 1, -1))
            out["vis_img"] = vis_img.view(*vis_img.shape[:2], out_h, out_w)
        
        input_mask=config['src_foreground_mask']
        with th.no_grad():
            assert input_mask.shape[0] == index.shape[0]
            input_mask = input_mask.reshape(*input_mask.shape[:2], -1)
            input_mask = th.gather(input_mask, 2, index[:, None].expand(-1, 1, -1))
            out["input_mask"] = input_mask.view(*input_mask.shape[:2], out_h, out_w) 
        
        if 'input_densepose' in targets:
            input_densepose=targets['input_densepose']
            with th.no_grad():
                assert input_densepose.shape[0] == index.shape[0]
                input_densepose = input_densepose.reshape(*input_densepose.shape[:2], -1)
                input_densepose = th.gather(input_densepose, 2, index[:, None].expand(-1, 3, -1))
                out["input_densepose"] = input_densepose.view(*input_densepose.shape[:2], out_h, out_w) 
        
        if 'tar_densepose' in targets:
            tar_densepose=targets['tar_densepose']
            with th.no_grad():
                assert tar_densepose.shape[0] == index.shape[0]
                tar_densepose = tar_densepose.reshape(*tar_densepose.shape[:2], -1)
                tar_densepose = th.gather(tar_densepose, 2, index[:, None].expand(-1, 3, -1))
                out["tar_densepose"] = tar_densepose.view(*tar_densepose.shape[:2], out_h, out_w) 
        
        with th.no_grad():
            assert img_in.shape[0] == index.shape[0]
            img_in = img_in.reshape(*img_in.shape[:2], -1)
            img_in = th.gather(img_in, 2, index[:, None].expand(-1, 3, -1))
            out["img_in"] = img_in.view(*img_in.shape[:2], out_h, out_w)

        out['vert_vis']=vert_vis
        
        return out

    @staticmethod
    def importance_sample(contrib, z, sample_per_ray, uniform=False):
        '''
        args:
            contrib: (B, N, D-2)
            z: (B, N, D-1)

        '''
        with th.no_grad():
            assert contrib.shape[-1] == z.shape[-1] - 1
            contrib = contrib + 1e-5
            pdf = contrib / contrib.sum(-1, keepdim=True)
            cdf = th.cumsum(pdf, -1)
            cdf = th.cat([th.zeros_like(cdf[:, :, :1]), cdf], 2)

            if uniform:
                sample = th.linspace(0.0, 1.0, steps=sample_per_ray)
                sample = sample[None, None, :].expand(*cdf.shape[:-1], -1).to(contrib.device)
            else:
                sample = th.rand(*cdf.shape[:-1], sample_per_ray).to(contrib.device)

            idx = th.searchsorted(cdf, sample, right=True)
            idx_prev = (idx - 1).clamp(min=0)
            idx = idx.clamp(max=cdf.shape[-1] - 1)
            idx = th.cat([idx_prev, idx], -1)

            cdf_idx = th.gather(cdf, -1, idx)
            cdf_prev = cdf_idx[:, :, :sample_per_ray]
            cdf_next = cdf_idx[:, :, sample_per_ray:]

            z_idx = th.gather(z, -1, idx)
            z_prev = z_idx[:, :, :sample_per_ray]
            z_next = z_idx[:, :, sample_per_ray:]

            num = sample - cdf_prev
            den = cdf_next - cdf_prev
            den = th.where(den < 1e-5, th.ones_like(den), den)
            sample = z_prev + (num / den) * (z_next - z_prev)
            return sample

    @staticmethod
    def rgba2out(self, rgba, z, vert_sdf):
        '''
        args:
            rgba: (B, N, D, 5), sorted in D
            z: (B, N, D), sorted in D
        return:
            color: (B, N, 3)
            depth: (B, N)
            alpha: (B, N),
            contrib: (B, N, D)
        '''
        alpha = rgba[..., 0]
        if self.kwargs.get('sp_conv'):
            alpha=alpha
        else:
            alpha=alpha+vert_sdf.squeeze(-1)
            alpha = self.sdf_activation(-alpha)
        sdf = rgba[..., 1]
        rgb = rgba[..., 2:]

        dist = th.cat([(z[..., 1:] - z[..., :-1]), 1e10 * th.ones_like(z[..., :1])], -1)
        contrib = 1.0 - th.exp(-alpha * dist)
        contrib = contrib * \
            th.cumprod(th.cat([th.ones_like(contrib[..., :1]), 1 - contrib[..., :-1]], -1), -1)  # (B, N, D)

        color = (rgb * contrib[..., None]).sum(-2)  # (B, N, 3)
        alpha = contrib.sum(-1)  # (B, N)
        sdf = (sdf * contrib).sum(-1) / (alpha + 1e-8)  # (B, N)
        depth = (z * contrib).sum(-1) / (alpha + 1e-8)  # (B, N)
        return color, depth, alpha, contrib, sdf

    @staticmethod
    def ray_bbox_intersection(bounds, orig, direct, boffset=(-0.01, 0.01)):
        '''
        args:
            bounds: (B, 2, 3)
            orig: (B, 1, 3)
            direct: (B, N, 3)
            cent: (B, N, 3)
            rad: radian
        return:
            intersection point: (B, N, 3)
        # '''

        near_finals=[]
        far_finals=[]
        mask_at_boxs=[]
        batch_size=bounds.shape[0]
        bounds0, orig0, direct0=bounds, orig, direct
        for b in range(batch_size):
            bounds, orig, direct = bounds0[b,...], orig0[b,...], direct0[b,...]
            
            orig = orig.expand(direct.shape[0], -1)
            bounds = bounds + th.tensor([boffset[0], boffset[1]])[:, None].to(device=orig.device)
            nominator = bounds[None] - orig[:, None]

            # calculate the step of intersections at six planes of the 3d bounding box
            direct = direct.detach().clone()
            direct[direct.abs() < 1e-5] = 1e-5
            d_intersect = (nominator / direct[:, None]).reshape(-1, 6)

            # calculate the six interections
            p_intersect = d_intersect[..., None] * direct[:, None] + orig[:, None]

            # calculate the intersections located at the 3d bounding box
            bounds = bounds.reshape(-1)
            min_x, min_y, min_z, max_x, max_y, max_z =\
                bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
            eps = 1e-6
            p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                            (p_intersect[..., 0] <= (max_x + eps)) * \
                            (p_intersect[..., 1] >= (min_y - eps)) * \
                            (p_intersect[..., 1] <= (max_y + eps)) * \
                            (p_intersect[..., 2] >= (min_z - eps)) * \
                            (p_intersect[..., 2] <= (max_z + eps))

            # obtain the intersections of rays which intersect exactly twice
            mask_at_box = p_mask_at_box.sum(-1) == 2
            p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(-1, 2, 3)

            # calculate the step of intersections
            norm_ray = th.linalg.norm(direct[mask_at_box], dim=1)
            d0 = th.linalg.norm(p_intervals[:, 0] - orig[mask_at_box], dim=1) / norm_ray
            d1 = th.linalg.norm(p_intervals[:, 1] - orig[mask_at_box], dim=1) / norm_ray
            d01 = th.stack((d0, d1), -1)
            near = d01.min(-1).values
            far = d01.max(-1).values

            # fix dimensions
            near_final = th.ones_like(mask_at_box.float())
            far_final = th.ones_like(mask_at_box.float())
            near_final[mask_at_box] = near.float()
            far_final[mask_at_box] = far.float()

            near_final = near_final.unsqueeze(0).unsqueeze(-1)
            far_final = far_final.unsqueeze(0).unsqueeze(-1)
            mask_at_box = mask_at_box.unsqueeze(0).unsqueeze(-1)
        
            near_finals.append(near_final)
            far_finals.append(far_final)
            mask_at_boxs.append(mask_at_box)

        near_final=torch.cat(near_finals,0)
        far_final=torch.cat(far_finals,0)
        mask_at_box=torch.cat(mask_at_boxs,0)
        return near_final, far_final, mask_at_box

class IBRRenderingHead(nn.Module):
    """ This rendering head is insipred by IBRNet (CVPR'21). """

    def __init__(self, in_channels=32+5, **kwargs):
        super().__init__()

        self.ani_al = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.ray_encoder = nn.Sequential(nn.Linear(4, 16), nn.ELU(inplace=True), nn.Linear(16, in_channels + 3), nn.ELU(inplace=True))

        self.base_layer = nn.Sequential(nn.Linear((in_channels + 3) * 3, 64), nn.ELU(inplace=True), nn.Linear(64, 32), nn.ELU(inplace=True))
        self.base_layer.apply(self._init_block)

        self.vis_layer1 = nn.Sequential(nn.Linear(32, 32), nn.ELU(inplace=True), nn.Linear(32, 33), nn.ELU(inplace=True))
        self.vis_layer1.apply(self._init_block)

        self.vis_layer2 = nn.Sequential(nn.Linear(32, 32), nn.ELU(inplace=True), nn.Linear(32, 1), nn.Sigmoid())
        self.vis_layer2.apply(self._init_block)

        self.out_layer = nn.Sequential(nn.Linear(32 + 1 + 4, 16), nn.ELU(inplace=True), nn.Linear(16, 8), nn.ELU(inplace=True), nn.Linear(8, 1))
        self.out_layer.apply(self._init_block)

    @staticmethod
    def _init_block(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, rgb_feats, ray_diffs, proj_mask):
        ''' Forward pass.

        Args:
            rgb_feats (torch.tensor): RGB colors and image features [rays, query samples, number of views, feat lean]
            ray_diffs (torch.tensor): difference of ray direction and their dot product [rays, query samples, number of views, 4]
            proj_mask (torch.tensor): whether projection is valid or not. [rays, query samples, number of views, 1]

        Returns:
            predicted color [rays, query samples, 3]
        '''
        V = rgb_feats.shape[2]
        dir_feat = self.ray_encoder(ray_diffs)
        src_rgb = rgb_feats[..., :3]

        rgb_feats = torch.cat((
            rgb_feats[..., :dir_feat.shape[-1]] + dir_feat,
            rgb_feats[..., dir_feat.shape[-1]:]
        ), dim=-1)

        _, dot_prod = torch.split(ray_diffs, [3, 1], dim=-1)
        exp_dot_prod = torch.exp(torch.abs(self.ani_al) * (dot_prod - 1))
        weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * proj_mask
        weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)

        fused_feat = torch.cat(fused_mean_variance(rgb_feats, weight), dim=-1)
        x = self.base_layer(torch.cat([fused_feat.expand(-1, -1, V, -1), rgb_feats], dim=-1))

        pred_vis = self.vis_layer1(x * weight)
        res, _vis = torch.split(pred_vis, [pred_vis.shape[-1] - 1, 1], dim=-1)
        x = x + res
        _vis = self.vis_layer2(x * torch.sigmoid(_vis) * proj_mask) * proj_mask

        # color prediction
        x = self.out_layer(torch.cat([x, _vis, ray_diffs], dim=-1)).masked_fill(proj_mask == 0, -1e4)
        pred_color = torch.sum(src_rgb * torch.softmax(x, dim=2), dim=2)
        return pred_color
