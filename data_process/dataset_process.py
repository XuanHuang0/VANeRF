import os
import sys
import pickle
import numpy as np
import json
from glob import glob
import os.path as osp
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import trimesh
import smplx
import torch
from pycocotools.coco import COCO
import cv2
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam, cam2world
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.renderer import (
    BlendParams,
    look_at_view_transform,
    PerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex, 
    SoftSilhouetteShader,
    HardPhongShader
)
import torch
from tqdm import tqdm

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


dense_path='./processed_dataset/v_color.pkl'
with open(dense_path, 'rb') as file:
    dense_coor = pickle.load(file)
dense_coor = torch.from_numpy(dense_coor)
dense_coor = torch.cat((dense_coor,dense_coor[-1,:].unsqueeze(0),dense_coor,dense_coor[-1,:].unsqueeze(0)), dim=0)

def render_img(verts, faces, mesh_i_xy, mesh_i_z, R,T, fx, fy, px, py, mask_path='vis.jpg',image_size=(512,334),device='cuda:3'):

    device = torch.device(device)
    verts=torch.FloatTensor(verts).to(device)
    faces = torch.tensor(faces).to(device)

    mesh_i_xy=torch.FloatTensor(mesh_i_xy).to(device)
    mesh_i_z=torch.FloatTensor(mesh_i_z).to(device)

    znear, zfar = 0.71, 1.42
    h,w=image_size
    mesh_i_xy[..., 0] =  (mesh_i_xy[..., 0] / (w - 1.0)) 
    mesh_i_xy[..., 1] =  (mesh_i_xy[..., 1] / (h - 1.0)) 
    mesh_i_z =  (mesh_i_z - znear) / (zfar - znear)

    if verts.shape[0] > 800:
        v_color=(dense_coor.expand(*verts.shape)).to(device).unsqueeze(0)
    else:
        v_color=(dense_coor[:779].expand(*verts.shape)).to(device).unsqueeze(0)
    tex = TexturesVertex(verts_features=v_color)
    R=torch.FloatTensor(R).view(1, 3, 3).to(device)
    T=torch.FloatTensor(T).view(1, 3).to(device)
    mesh = Meshes(verts=[verts], faces=[faces], textures=tex)

    cameras =  PerspectiveCameras(device=device,R=R,T=T,
        focal_length=((fx, fy),),
        principal_point=((px, py),),
        in_ndc=False,
        image_size=(image_size,),)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )

    # Make an arbitrary light source
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    im_vis = renderer(mesh.to(device))
    return im_vis.detach().cpu().numpy()[0, ..., :3]*255, im_vis.detach().cpu().numpy()[0, ..., -1]*255


def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

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

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

if __name__ == "__main__": 
    # mano layer
    smplx_path = './smplx/models/'
    mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
    # vert_face=torch.load(os.path.join("/data4/huangx/KeypointNeRF/face.pth"))
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:,0,:] *= -1
                
    root_path = './InterHand2.6M/'
    processed_data_path = './processed_dataset/'
    img_root_path = osp.join(root_path, 'images')
    annot_root_path = osp.join(root_path, 'annotations')

    for split in ['train','test']:
        db = COCO(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_data.json'))
        with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)
        with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_camera.json')) as f:
            cam_params = json.load(f)
        with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_joint_3d.json')) as f:
            joints = json.load(f)

        occlusion_cam=[400006,400008,400015,400035,400049]

        idx = 0
        index = 0
        cam_list = {}
        os.makedirs(osp.join(processed_data_path, split, 'index'), exist_ok=True)
        
        for acount, aid in tqdm(enumerate(db.anns.keys())):
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            capture_id = str(img['capture'])
            seq_name = str(img['seq_name'])
            cam_idx = str(img['camera'])
            cam=img['camera']
            hand_type = ann['hand_type']

            if int(cam) in occlusion_cam :
                continue  
            if str(cam)[:2]=='41':
                continue

            if hand_type == 'left':
                continue
            if hand_type == 'right':
                continue

            img_width, img_height = img['width'], img['height']

            bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
            bbox = process_bbox(bbox, (img_height, img_width))

            frame_idx = str(img['frame_idx'])
            img_path = osp.join(img_root_path, split, img['file_name'])
            save_path = osp.join(img_root_path, split+'mask', img['file_name']+'.npy')
            meshdir=os.path.split(save_path)[0]

            if split == 'test':
                if 'ROM01_No_Interaction_2_Hand' in img['file_name']:
                    continue

            save_path = meshdir
            frame_idx = img_path.split('/')[-1][5:-4]
            image = load_img(img_path)

            img_height, img_width, _ = image.shape

            prev_depth = None
            mesh={}
            face={}
            data_info = {}
            data_info['mano'] = {'pose':{},'shape':{},'trans':{}}
            if_continue = 0
            if hand_type == 'interacting':
                for hand in ('right', 'left'):
                    try:
                        mano_param = mano_params[capture_id][frame_idx][hand]
                        if mano_param is None:
                            if_continue = 1
                            continue
                    except KeyError:
                        if_continue = 1
                        continue

                    # get MANO 3D mesh coordinates (world coordinate)
                    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
                    root_pose = mano_pose[0].view(1,3)
                    hand_pose = mano_pose[1:,:].view(1,-1)
                    shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
                    trans = torch.FloatTensor(mano_param['trans']).view(1,3)

                    data_info['mano']['pose'][hand]=mano_pose
                    data_info['mano']['shape'][hand]=shape
                    data_info['mano']['trans'][hand]=trans

                    output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                    mesh[hand] = output.vertices[0].numpy() # meter to milimeter
                    face[hand] = mano_layer[hand].faces
                    # face = np.array(face, dtype=np.int32)
                    mesh[hand], face[hand]=seal(mesh[hand],face[hand],hand)
                try:
                    mesh_i=np.concatenate([mesh['right'],mesh['left']],0)
                except KeyError:
                    if_continue = 1
                    continue

                mesh_right=trimesh.Trimesh(mesh['right'],face['right'])
                mesh_left=trimesh.Trimesh(mesh['left'],face['left'])
                mesh=concat_meshes([mesh_right,mesh_left])
                face=mesh.faces
                mesh_i=mesh.vertices
            else:  
                try:
                    mano_param = mano_params[capture_id][frame_idx][hand_type]
                    if mano_param is None:
                        if_continue = 1
                        continue
                except KeyError:
                    if_continue = 1
                    continue
                # get MANO 3D mesh coordinates (world coordinate)
                mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
                root_pose = mano_pose[0].view(1,3)
                hand_pose = mano_pose[1:,:].view(1,-1)
                shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
                trans = torch.FloatTensor(mano_param['trans']).view(1,3)

                data_info['mano']['pose'][hand_type]=mano_pose
                data_info['mano']['shape'][hand_type]=shape
                data_info['mano']['trans'][hand_type]=trans

                output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                mesh[hand_type] = output.vertices[0].numpy() # meter to milimeter
                face = mano_layer[hand_type].faces
                face = np.array(face, dtype=np.int32)
                mesh[hand_type], face=seal(mesh[hand_type],face,hand_type)
                mesh_i=mesh[hand_type]
            
            if if_continue == 1:
                continue

            data = {'idx':idx,'capture': capture_id, 'cam': cam, 'frame':frame_idx}
            data_info['aid'] = aid
            data_info['idx'] = idx
            data_info['image_info'] = img

            cam_param = cam_params[capture_id]
            focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
            princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
            t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3)/1000, np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3,3)
            t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
            
            mesh_i_cam = np.dot(R, mesh_i.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
            mesh_i_img = cam2pixel(mesh_i_cam, focal, princpt)
            mesh_i_xy = mesh_i_img[:, :2]
            mesh_i_z = mesh_i_img[:, 2:]

            torch3d_T_colmap = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            R = (torch3d_T_colmap @ R).T
            t = torch3d_T_colmap @ t
            
            densepose, rendered_mask=render_img(mesh_i, face, mesh_i_xy, mesh_i_z, R=R,T=t, fx=focal[0], fy=focal[1], px=princpt[0], py=princpt[1],image_size=(img_height, img_width))

            os.makedirs(osp.join(processed_data_path, split, 'densepose', 'capture'+str(capture_id),'cam'+str(cam)), exist_ok=True)
            os.makedirs(osp.join(processed_data_path, split, 'mask', 'capture'+str(capture_id),'cam'+str(cam)), exist_ok=True)
            os.makedirs(osp.join(processed_data_path, split, 'image', 'capture'+str(capture_id),'cam'+str(cam)), exist_ok=True)

            image,trans = augmentation(image,bbox)
            densepose,trans = augmentation(densepose,bbox)
            rendered_mask,trans = augmentation(rendered_mask,bbox)

            campos, camrot = np.array(cam_params[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cam_params[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cam_params[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cam_params[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            in_T, in_R = np.array(campos, dtype=np.float32).reshape(3)/1000., np.array(camrot, dtype=np.float32).reshape(3,3)
            in_T = -np.dot(in_R,in_T.reshape(3,1)).reshape(3) # -Rt -> t
            in_Rt = np.concatenate((in_R.reshape(3,3), in_T.reshape(3, 1)), axis=1)

            focal = trans_point2d(focal, trans)
            princpt = trans_point2d(princpt, trans)
            in_K=np.array([[focal[0],0,princpt[0]],[0,focal[1],princpt[1]],[0,0,1]])

            data_info['camera'] = {'R': in_R.reshape(3,3), 't': in_T.reshape(3, 1), 'in_K': in_K, 'campos':campos, 'camrot':camrot}

            os.makedirs(osp.join(processed_data_path, split, 'annotation', 'capture'+str(capture_id)+'/cam'+str(cam)), exist_ok=True)
            with open(osp.join(processed_data_path, split, 'annotation', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.pkl'), 'wb') as file:
                pickle.dump(data_info, file)

            if not frame_idx in cam_list.keys():
                cam_list[frame_idx]={}
            if capture_id not in cam_list[frame_idx].keys():
                cam_list[frame_idx][capture_id]=[]
            cam_list[frame_idx][capture_id].append((cam,aid))

            if frame_idx in cam_list.keys() and capture_id in cam_list[frame_idx].keys():
                if len(cam_list[frame_idx][capture_id])==4:
                    with open(osp.join(processed_data_path ,split,'index', '{}.pkl'.format(index)), 'wb') as file:
                        pickle.dump(data, file)
                    index=index+1
                    
            image_save_path=processed_data_path+split+'/image/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg'
            densepose_save_path=processed_data_path+split+'/densepose/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg'
            mask_save_path=processed_data_path+split+'/mask/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg'

            cv2.imwrite(densepose_save_path, densepose)
            cv2.imwrite(mask_save_path, rendered_mask)
            cv2.imwrite(image_save_path, image[...,[2,1,0]])

            idx=idx+1
            # print(mask_save_path)
        torch.save(cam_list, os.path.join(processed_data_path, split, "cam_list.pth"))
