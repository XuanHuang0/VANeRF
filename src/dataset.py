import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import imageio.v2 as imageio
import copy
import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from src.transforms import world2cam, cam2pixel, pixel2cam, cam2world
from src.mis_utils import edge_subdivide
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio
import smplx
from torchvision import transforms
import trimesh
import torchvision.transforms as transforms
import pickle

# mano layer
smplx_path = 'smplx/models/'
mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1


def seal(mesh_to_seal,face_to_seal,hand_type):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    if hand_type=='left':
        circle_v_id=circle_v_id[::-1]
    center = (mesh_to_seal[circle_v_id, :]).mean(0)

    mesh_to_seal = np.vstack([mesh_to_seal, center])
    center_v_id = mesh_to_seal.shape[0] - 1

    # # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        face_to_seal = np.vstack([face_to_seal, new_faces])
    return mesh_to_seal, face_to_seal

def concat_meshes(mesh_list):
    '''manually concat meshes'''
    cur_vert_number = 0
    cur_face_number = 0
    verts_list = []
    faces_list = []
    for idx, m in enumerate(mesh_list):
        verts_list.append(m.vertices)
        faces_list.append(m.faces + cur_vert_number)
        cur_vert_number += len(m.vertices)

    combined_mesh = trimesh.Trimesh(np.concatenate(verts_list),
        np.concatenate(faces_list), process=False
    )
    return combined_mesh

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, **kwargs):
        self.split = split # train, test, val
        self.mode = split # train, test
        self.sc_factor = 1
        if split == 'val':
            self.mode = 'train'
        self.nearmin = 2
        self.farmax = 0
        self.input_per_frame = kwargs.get('input_per_frame_test', 1)
        self.num_input_view = kwargs.get('num_input_view', 1)
        self.if_color_jitter=kwargs.get('color_jitter', False)
        self.if_mask_sa=kwargs.get('mask_sa', False)
        self.if_edge_subdivide=kwargs.get('edge_subdivide', False)
        self.big_view_variation = kwargs.get('big_view_variation', False)
        if self.mode == 'train' and self.if_color_jitter:
            self.jitter = self.color_jitter()
        self.annot_path = 'InterHand2.6M/annotations'
        self.processed_data_path = './processed_dataset/'
        joint_regressor = np.load('smplx/models/mano/J_regressor_mano_ih26m.npy')
        self.joint_regressor=torch.tensor(joint_regressor)
        self.image2tensor = transforms.Compose([transforms.ToTensor(), ])
        self.sequence_names = []
        self.cam_list=torch.load(os.path.join(self.processed_data_path,self.mode,"cam_list.pth"))
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            self.joints = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f:
            self.manos = json.load(f)

        self.use_intag_preds = kwargs.get('use_intag_preds', False)
        if self.use_intag_preds:
            print('using intaghand prediction')

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def color_jitter(self):
        ops = []
        ops.extend(
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )
        return transforms.Compose(ops)

    @staticmethod
    def get_mask_at_box(bounds, K, R, T, H, W):
        ray_o, ray_d = Dataset.get_rays(H, W, K, R, T)

        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = Dataset.get_near_far(bounds, ray_o, ray_d)
        return mask_at_box.reshape((H, W)),near.min(),far.max()
    
    def load_human_bounds_pred(self, vert_world_pred): 
        xyz=vert_world_pred
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds

    def load_human_bounds(self, capture_id,frame_idx, hand_type):
        mano_valid=np.zeros((2,))
        if hand_type == 'right' or hand_type == 'left':
            
            mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['pose']).view(-1,3)
            root_pose = mano_pose[0].view(1,3)
            hand_pose = mano_pose[1:,:].view(1,-1)
            shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['shape']).view(1,-1)
            trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['trans']).view(1,3)
            output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
            mesh = output.vertices[0].detach().numpy()
            if hand_type == 'left':
                mano_valid[1]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh0,mesh,axis=0) 
            else:
                mano_valid[0]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh,mesh0,axis=0) 
            
        else:
            for hand in ('right', 'left'):
                try:
                    mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['pose']).view(-1,3)
                    root_pose = mano_pose[0].view(1,3)
                    hand_pose = mano_pose[1:,:].view(1,-1)
                    shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['shape']).view(1,-1)
                    trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['trans']).view(1,3)
                    output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                    mesh = output.vertices[0].detach().numpy()
                    if hand == 'left':
                        mano_valid[1]=1
                    else:
                        mano_valid[0]=1
                except:
                    mesh=np.zeros((778,3))
                    mano_pose=np.zeros((16,3))
                    shape=np.zeros((1,10))
                    trans=np.zeros((1,3))
                if hand == 'left':
                    mesh_left=mesh
                else:
                    mesh_right=mesh
            mesh=np.append(mesh_right,mesh_left,axis=0) 

        xyz=mesh
        if hand_type == 'right':
            xyz=mesh[:778,:]
        elif hand_type == 'left':
            xyz=mesh[778:,:]

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
 
    def load_mano(self, capture_id,frame_idx, hand_type):
        mano_valid=np.zeros((2,))
        if hand_type == 'right' or hand_type == 'left':
            
            mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['pose']).view(-1,3)
            root_pose = mano_pose[0].view(1,3)
            
            Rh = root_pose
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)

            hand_pose = mano_pose[1:,:].view(1,-1)
            shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['shape']).view(1,-1)
            trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['trans']).view(1,3)
            output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
            mesh = output.vertices[0].detach().numpy()
            face = output.faces[0].detach().numpy()
            Th = trans.astype(np.float32)
            mano_pose=mano_pose.view(1,-1)
            mano_pose=np.squeeze(mano_pose)
            shape=np.squeeze(shape)
            trans=np.squeeze(trans)
            
            mano_pose0=np.zeros(mano_pose.shape)
            shape0=np.zeros(shape.shape)
            trans0=np.zeros(trans.shape)

            if hand_type == 'left':
                mano_valid[1]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh0,mesh,axis=0)

                mano_pose=np.append(mano_pose0,mano_pose,axis=0)
                shape=np.append(shape0,shape,axis=0)
                trans=np.append(trans0,trans,axis=0)
            else:
                mano_valid[0]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh,mesh0,axis=0)

                mano_pose=np.append(mano_pose,mano_pose0,axis=0)
                shape=np.append(shape,shape0,axis=0)
                trans=np.append(trans,trans0,axis=0)
            
        else:
            for hand in ('right', 'left'):
                mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['pose']).view(-1,3)
                root_pose = mano_pose[0].view(1,3)

                Rh = root_pose.numpy()
                R = cv2.Rodrigues(Rh)[0].astype(np.float32)

                hand_pose = mano_pose[1:,:].view(1,-1)
                shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['shape']).view(1,-1)
                trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['trans']).view(1,3)
                
                Th = trans.numpy().astype(np.float32)

                output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                mesh = output.vertices[0].detach().numpy()
                face = mano_layer[hand].faces

                joint= torch.matmul(self.joint_regressor, torch.tensor(mesh).float())
                mesh,face=seal(mesh,face,hand)
                if self.if_edge_subdivide:
                    mesh,face,_=edge_subdivide(mesh, face)
              
                nxyz = np.zeros_like(mesh).astype(np.float32)

                if hand == 'left':
                    mano_valid[1]=1
                else:
                    mano_valid[0]=1
           
                if hand == 'left':
                    mesh_left=mesh
                    face_left=face
                    joint_left=joint
                    mano_pose=mano_pose.reshape(1,-1)
                    mano_pose_left=np.squeeze(mano_pose)
                    shape_left=np.squeeze(shape)
                    trans_left=np.squeeze(trans)

                    xyz = np.dot(mesh - Th_r, R_r)
                    xyz_l=xyz
                    Rh_l=Rh
                    Th_l=Th
                    R_l=R
                    cxyz_l = xyz.astype(np.float32)
                    nxyz_l = nxyz.astype(np.float32)

                else:
                    mesh_right=mesh
                    face_right=face
                    joint_right=joint

                    mano_pose=mano_pose.reshape(1,-1)
                    mano_pose_right=np.squeeze(mano_pose)
                    shape_right=np.squeeze(shape)
                    trans_right=np.squeeze(trans)
                    
                    Rh_r=Rh
                    Th_r=Th
                    R_r=R
                    xyz = np.dot(mesh - Th, R)
                    xyz_r=xyz
                    
                    cxyz_r = xyz.astype(np.float32)
                    nxyz_r = nxyz.astype(np.float32)

            xyz = np.append(xyz_r,xyz_l,axis=-2)
            cxyz = np.append(cxyz_r,cxyz_l,axis=-2)
            nxyz = np.append(nxyz_r,nxyz_l,axis=-2)
            Rh = np.append(Rh_r,Rh_l,axis=-2)
            Th = np.append(Th_r,Th_l,axis=-2)
            feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= 0.05
            max_xyz += 0.05

            bounds = np.stack([min_xyz, max_xyz], axis=0)

            xyz=torch.from_numpy(xyz)
            xyz=xyz.numpy()

            dhw = xyz[:, [2, 1, 0]]
            min_dhw = min_xyz[[2, 1, 0]]
            max_dhw = max_xyz[[2, 1, 0]]
            voxel_size = np.array([0.005, 0.005, 0.005])
            coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)
            # construct the output shape
            out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)

            x = 32
            out_sh = (out_sh | (x - 1)) + 1
            joint_world=np.append(joint_right,joint_left,axis=0)
            mesh_right=trimesh.Trimesh(mesh_right,face_right)
            mesh_left=trimesh.Trimesh(mesh_left,face_left)
            mesh=concat_meshes([mesh_right,mesh_left])
            face=mesh.faces
            mesh=mesh.vertices

            mano_pose=np.append(mano_pose_right.unsqueeze(0),mano_pose_left.unsqueeze(0),axis=0)
            shape=np.append(shape_right.unsqueeze(0),shape_left.unsqueeze(0),axis=0)
            trans=np.append(trans_right.unsqueeze(0),trans_left.unsqueeze(0),axis=0)
            
            # obtain the original bounds for point sampling
            min_mesh = np.min(mesh, axis=0)
            max_mesh = np.max(mesh, axis=0)
            min_mesh -= 0.05
            max_mesh += 0.05
            can_bounds = np.stack([min_mesh, max_mesh], axis=0)
        return joint_world, mesh, face, mano_pose, shape, trans, Rh_r, Th_r, R_r, coord, out_sh, can_bounds, bounds, feature


    def __len__(self):
        if self.split=='train':
            return 5423
        elif self.split=='val':
            return 8
        else:
            return 1895*self.input_per_frame
    
    def __getitem__(self, index):

        prob = np.random.randint(9000000)

        if self.mode == 'test':
            index_res = int(index % self.input_per_frame)
            index = int((index-index_res) / self.input_per_frame)

        with open(osp.join(self.processed_data_path,self.mode,'index', '{}.pkl'.format(index)), 'rb') as file:
            data = pickle.load(file)
        idx=data['idx']
        frame_idx = data['frame']
        capture_id = data['capture']
        cam = data['cam']
        hand_type ='interacting'

        if not self.use_intag_preds:
            kpt3d = np.array(self.joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)/1000
        all_input_view=self.cam_list[frame_idx][capture_id]
        # randomply sample input views for training
        if self.mode == 'train':  
            input_view = copy.deepcopy(all_input_view)
            random.shuffle(input_view)
            input_view = input_view[:self.num_input_view]
        # select a target view
        if self.mode == 'train':
            tar_pool = list(set(all_input_view) - set(input_view))
            random.shuffle(tar_pool)
            tar_view_ind = tar_pool[0]
            (tar_cam_id,_)=tar_view_ind
            input_view = [tar_view_ind] + input_view
        else:
            input_view = copy.deepcopy(all_input_view)

            if not self.big_view_variation:
                input_list_01={'0':[37,44],'1':[8,16],'2':[23,25],'3':[41,43],'4':[55,56]}
                input_list_27={'0':[0,3],'1':[1,2],'2':[4,5],'3':[8,9],'4':[16,17]}
            else:
                input_list_01={'0':[0,1],'1':[1,2],'2':[2,3],'3':[5,6],'4':[11,12]}
                input_list_27={'0':[0,3],'1':[0,4],'2':[0,6],'3':[4,8],'4':[0,13]}

            if '0' in str(capture_id) or '1' in str(capture_id):
                input_list=input_list_01[str(index_res)]
            else:
                input_list=input_list_27[str(index_res)]
            
            input_view = [input_view[i] for i in input_list]
            (tar_cam_id,_)=input_view[0]
            tar_view_ind = input_view[0]

        tar_cam={}  
        input_imgs, input_msks, input_K, input_Rt = [], [], [], []
        for idx,(cam,aid) in enumerate(input_view):
            with open(osp.join(self.processed_data_path,self.mode,'annotation', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.pkl'), 'rb') as file:
                anno = pickle.load(file)

            if self.use_intag_preds:
                with open(osp.join(self.processed_data_path,self.mode,'verts_preds', str(int(aid))+'.pkl'), 'rb') as file:
                    vert_cam_pred = pickle.load(file)

            in_T, in_R=anno['camera']['t'].reshape(3), anno['camera']['R']
            in_Rt = np.concatenate((in_R.reshape(3,3), in_T.reshape(3, 1)), axis=1)
            in_K = anno['camera']['in_K']
            princpt = in_K[0:2, 2].astype(np.float32)
            focal = np.array( [in_K[0, 0], in_K[1, 1]], dtype=np.float32)
            campos = anno['camera']['campos']
            camrot = anno['camera']['camrot']
            img_info=anno['image_info']

            img_path=osp.join(self.processed_data_path,self.mode,'image', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')
            if self.if_mask_sa:
                mask_path=osp.join(self.processed_data_path,self.mode,'mask_sa', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
            else:
                mask_path=osp.join(self.processed_data_path,self.mode,'mask', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
            
            densepose_path=osp.join(self.processed_data_path,self.mode,'densepose', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')
            try:
                input_msk = (imageio.imread(mask_path) >=100).astype(np.uint8) #0为黑色
            except:
                mask_path=osp.join(self.processed_data_path,self.mode,'mask', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
                input_msk = (imageio.imread(mask_path) >=100).astype(np.uint8) #0为黑色
       
            input_img = imageio.imread(img_path)
            if self.mode == 'train' and self.if_color_jitter:
                input_img = Image.fromarray(input_img)
                torch.manual_seed(prob)
                input_img = self.jitter(input_img)
                input_img = np.array(input_img)
            input_img = input_img.astype(np.float32) / 255.
            if self.mode == 'train':
                input_densepose = imageio.imread(densepose_path).astype(np.float32) / 255.
                input_densepose[input_msk == 0] = 0
            self.ratio=1
            H, W = int(input_img.shape[0] * self.ratio), int(input_img.shape[1] * self.ratio)
            input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)
            H, W = int(input_img.shape[0] ), int(input_img.shape[1])
            input_img[input_msk == 0] = 0
            input_msk = (input_msk != 0)  # bool mask : foreground (True) background (False)
            if idx == 0 and not self.if_color_jitter:
                input_msk[input_img[:,:,1]<= 0.1] = 0 
                input_img[input_msk == 0] = 0
            if idx == 0 and self.if_color_jitter:
                input_msk[input_img[:,:,1]<= 0.03] = 0 
                input_img[input_msk == 0] = 0
            if not 1. in input_msk:
                print('input_msk black!!!!!'+img_path)

            input_img0=input_img
            input_msk = input_msk.astype(np.uint8) * 255
            input_img = self.image2tensor(input_img)
            input_msk = self.image2tensor(input_msk).bool()
            in_K[:2] = in_K[:2] * self.ratio
            if idx==0:

                if self.use_intag_preds:
                    v3d_left=torch.from_numpy(vert_cam_pred[778:,:])
                    v3d_right=torch.from_numpy(vert_cam_pred[:778,:])
                    joint_left = torch.matmul(self.joint_regressor, v3d_left)
                    joint_right = torch.matmul(self.joint_regressor, v3d_right)
                    joint_cam_pred=torch.cat((joint_right, joint_left), dim=0).cpu().numpy()
                    joint_world_pred=cam2world(joint_cam_pred.transpose(1,0),camrot, campos.reshape(3,1)/1000).transpose(1,0)
                    kpt3d = joint_world_pred
                    vert_world_pred=cam2world(vert_cam_pred.transpose(1,0),camrot, campos.reshape(3,1)/1000).transpose(1,0)
                    joint_world = joint_world_pred
                    mesh = vert_world_pred

                else:
                    joint_world, mesh, face, mano_pose, shape, trans, Rh_r, Th_r, R_r, coord, out_sh, can_bounds, bounds, feature=self.load_mano(capture_id,frame_idx, hand_type)

                torch3d_T_colmap = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                tar_R = (torch3d_T_colmap @ in_R).T
                tar_T = torch3d_T_colmap @ in_T
                tar_cam['tar_R']=torch.from_numpy(tar_R).float()
                tar_cam['tar_T']=torch.from_numpy(tar_T).float()
                tar_cam['tar_focal']=torch.from_numpy(focal).float()
                tar_cam['tar_princpt']=torch.from_numpy(princpt).float()

                targets={
                        "joint_world":torch.from_numpy(joint_world).float(),
                        'vert_world':torch.from_numpy(mesh).float(),
                        'face_world':torch.from_numpy(face).float(),
                        'tar_cam':tar_cam,
                }

                if not self.use_intag_preds:
                    targets['coord'] = coord
                    targets['out_sh'] = out_sh
                    targets['bounds'] = bounds

            if self.mode == 'train':
                input_densepose = self.image2tensor(input_densepose)
                if self.num_input_view==1:
                    if idx == 0:
                        targets['tar_densepose']=input_densepose
                    else:
                        targets['input_densepose']=input_densepose
                       
            # append data
            input_imgs.append(input_img)
            input_msks.append(input_msk)
            input_K.append(torch.from_numpy(in_K))
            input_Rt.append(torch.from_numpy(in_Rt))
        
        hand_type_array=self.handtype_str2array(hand_type)
        ret = {
            'images': torch.stack(input_imgs),
            'images_masks': torch.stack(input_msks),
            'K': torch.stack(input_K),
            'Rt': torch.stack(input_Rt),
            'kpt3d': torch.from_numpy(kpt3d),
            'hand_type':torch.from_numpy(hand_type_array),
            'i': frame_idx,
            'human_idx': capture_id,
            'sessision': capture_id,
            'frame_index': frame_idx,
            'human': capture_id,
            'cam_ind': tar_cam_id,
            "index": {"camera": "cam", "segment": 'VANeRF', "tar_cam_id": tar_view_ind,
                "frame": f"{capture_id}_{frame_idx}", "ds_idx": cam},
        }

        ret['targets']=targets
        if self.use_intag_preds:
            bounds = self.load_human_bounds_pred(vert_world_pred)
        else:
            bounds = self.load_human_bounds(capture_id,frame_idx, hand_type)
        ret['mask_at_box'],near,far = self.get_mask_at_box(
            bounds,
            input_K[0].numpy(),
            input_Rt[0][:3, :3].numpy(),
            input_Rt[0][:3, -1].numpy(),
            H, W)
        ret['znear'], ret['zfar']=near,far
        if near<self.nearmin:
            self.nearmin=near
        if far>self.farmax:
            self.farmax=far  
        ret['bounds'] = bounds
        ret['mask_at_box'] = ret['mask_at_box'].reshape((H, W))
        x, y, w, h = cv2.boundingRect(ret['mask_at_box'].astype(np.uint8))

        mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)]['right']['pose']).view(-1,3)
        root_pose = mano_pose[0].reshape(-1)
        
        Rh = root_pose.numpy()
        R,_ = cv2.Rodrigues(Rh)
        R = torch.from_numpy(R)

        headpose = torch.eye(4)
        headpose[:3, :3] = input_Rt[1][:3, :3].t()
        headpose[:3, 3] = torch.from_numpy(kpt3d[0])
        ret['headpose'] = headpose

        return ret

    @classmethod
    def from_config(cls, dataset_cfg, data_split, cfg):
        ''' Creates an instance of the dataset.

        Args:
            dataset_cfg (dict): input configuration.
            data_split (str): data split (`train` or `val`).
        '''
        assert data_split in ['train', 'val', 'test', 'test_visualize']

        dataset_cfg = copy.deepcopy(dataset_cfg)
        dataset_cfg['is_train'] = data_split == 'train'
        if f'{data_split}_cfg' in dataset_cfg:
            dataset_cfg.update(dataset_cfg[f'{data_split}_cfg'])
        if dataset_cfg['is_train']:
            dataset = cls(split=data_split, **dataset_cfg)
        elif data_split == 'test_visualize':
            dataset = TestDataset(split='test', sample_frame=1, sample_camera=6, **dataset_cfg)
        else:
            dataset = TestDataset(split=data_split, **dataset_cfg)
        return dataset

    @staticmethod
    def get_rays(H, W, K, R, T):
        rays_o = -np.dot(R.T, T).ravel()

        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32), indexing='xy')

        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_world = np.dot(pixel_camera - T.ravel(), R)
        rays_d = pixel_world - rays_o[None, None]
        rays_o = np.broadcast_to(rays_o, rays_d.shape)

        return rays_o, rays_d

    @staticmethod
    def get_near_far(bounds, ray_o, ray_d, boffset=(-0.01, 0.01)):
        """calculate intersections with 3d bounding box"""
        bounds = bounds + np.array([boffset[0], boffset[1]])[:, None]
        nominator = bounds[None] - ray_o[:, None]
        # calculate the step of intersections at six planes of the 3d bounding box
        ray_d[np.abs(ray_d) < 1e-5] = 1e-5
        d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
        # calculate the six interections
        p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
        # calculate the intersections located at the 3d bounding box
        min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
        eps = 1e-6
        p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                        (p_intersect[..., 0] <= (max_x + eps)) * \
                        (p_intersect[..., 1] >= (min_y - eps)) * \
                        (p_intersect[..., 1] <= (max_y + eps)) * \
                        (p_intersect[..., 2] >= (min_z - eps)) * \
                        (p_intersect[..., 2] <= (max_z + eps))
        # obtain the intersections of rays which intersect exactly twice
        mask_at_box = p_mask_at_box.sum(-1) == 2
        p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
            -1, 2, 3)

        # calculate the step of intersections
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        norm_ray = np.linalg.norm(ray_d, axis=1)
        d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
        d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
        near = np.minimum(d0, d1)
        far = np.maximum(d0, d1)

        return near, far, mask_at_box

def draw_keypoints(img, kpts, color=(255, 0, 0), size=3):
    for i in range(kpts.shape[0]):
        kp2 = kpts[i].tolist()
        kp2 = [int(kp2[0]), int(kp2[1])]
        img = cv2.circle(img, kp2, 0, color, size)
    return img

class TestDataset(Dataset):
    def __init__(self, split, sample_frame=30, sample_camera=1, **kwargs):
        super().__init__( split, **kwargs)

    
       



