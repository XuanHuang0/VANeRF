import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from .transforms import world2cam, cam2pixel, pixel2cam, cam2world
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

def render_img(verts, faces,R,T, fx, fy, px, py, mask_path='mask.jpg',image_size=(512,334),device='cuda:2'):

    device = torch.device(device)
    verts=torch.FloatTensor(verts)
    faces = torch.FloatTensor(faces.astype(np.float32))
    R=torch.FloatTensor(R).view(1, 3, 3).to(device)
    T=torch.FloatTensor(T).view(1, 3).to(device)
    tex = torch.zeros_like(verts) 
    tex[:, 2] = 1.0 # white

    tex = TexturesVertex(verts_features=[tex])

    mesh = Meshes(verts=[verts], faces=[faces], textures=tex)

    cameras =  PerspectiveCameras(device=device,R=R,T=T,
        focal_length=((fx, fy),),
        principal_point=((px, py),),
        in_ndc=False,
        image_size=(image_size,),)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings = RasterizationSettings(
        image_size=(512,334), 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
        bin_size=None, 
        max_faces_per_bin=None
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
        shader=SoftPhongShader(
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

    # phong shader
    image = renderer(mesh.to(device)) 
    # softsilhouetteshader
    im_silhouette = silhouette_renderer(mesh.to(device))
    im_silhouette[im_silhouette > 0] = 255  # Binarize segmentation mask

    return im_silhouette.detach().cpu().numpy()[0, ..., 3]

dense_path='processed_dataset/v_color.pkl'
with open(dense_path, 'rb') as file:
    dense_coor = pickle.load(file)
dense_coor = torch.from_numpy(dense_coor)
dense_coor = torch.cat((dense_coor,dense_coor[-1,:].unsqueeze(0),dense_coor,dense_coor[-1,:].unsqueeze(0)), dim=0)

def render_img_vis(verts, faces, mesh_i_xy, mesh_i_z, R,T, fx, fy, px, py, mask_path='vis.jpg',image_size=(512,334),device='cuda:5'):

    device = torch.device(device)
    verts=torch.FloatTensor(verts).to(device)
    faces = torch.tensor(vert_face,dtype=torch.long).to(device)

    mesh_i_xy=torch.FloatTensor(mesh_i_xy).to(device)
    mesh_i_z=torch.FloatTensor(mesh_i_z).to(device)

    znear, zfar = 0.71, 1.42
    h,w=image_size
    mesh_i_xy[..., 0] =  (mesh_i_xy[..., 0] / (w - 1.0)) 
    mesh_i_xy[..., 1] =  (mesh_i_xy[..., 1] / (h - 1.0)) 
    mesh_i_z =  (mesh_i_z - znear) / (zfar - znear)

    v_color=(dense_coor.expand(*verts.shape)*0+1).to(device).unsqueeze(0)
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

    # phong shader
    # image = renderer(mesh.to(device)) 
    # softsilhouetteshader
    im_vis = renderer(mesh.to(device))
    # im_silhouette[im_silhouette > 0] = 255  # Binarize segmentation mask
    # cv2.imwrite(mask_path, im_vis.detach().cpu().numpy()[0, ..., :3]*255)
    # print(mask_path)
    return im_vis.detach().cpu().numpy()[0, ..., :3]*255


def render_vis(verts, faces, vert_vis, R,T, fx, fy, px, py, mask_path='vis2.jpg',image_size=(256,256),device='cuda:1'):

    v_color=torch.tile(vert_vis.squeeze(0), (1, 3)).unsqueeze(0).to(device)
    tex = TexturesVertex(verts_features=v_color.float())
    mesh = Meshes(verts=verts.float(), faces=faces, textures=tex)

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

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    # softsilhouetteshader
    im_vis = renderer(mesh.to(device))
    im_vis_rbg=im_vis[..., :3]*255
    im_vis_rbg_mean=im_vis_rbg.mean(-1).unsqueeze(-1)
    im_vis_rbg_mean[im_vis_rbg_mean>=50]=255
    im_vis_rbg_mean[im_vis_rbg_mean<50]=0
   
    return im_vis_rbg.permute(0,3,1,2)/255, im_vis_rbg_mean.permute(0,3,1,2)/255

def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def seal(mesh_to_seal,hand_type):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    if hand_type=='left':
        circle_v_id=circle_v_id[::-1]
    center = (mesh_to_seal[circle_v_id, :]).mean(0)

    mesh_to_seal = np.vstack([mesh_to_seal, center])
  
    return mesh_to_seal

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

if __name__ == "__main__": 
    # mano layer
    # smplx_path = '/home/3DHandReconstruction2/smplx/models/'
    smplx_path = '/data1/huangxuan/interhand/smplx/models/'
    mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
    vert_face=torch.load(os.path.join("/data4/huangx/KeypointNeRF/face.pth"))

    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:,0,:] *= -1
                
    root_path = '/home/huangxuan_m2023/InterHand2.6M-my/data/InterHand2.6M/'
    img_root_path = osp.join(root_path, 'images')
    annot_root_path = osp.join(root_path, 'annotations')
    split='train'
    # for split in ['train','val','test']:
    db = COCO(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_data.json'))
    with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_MANO_NeuralAnnot.json')) as f:
        mano_params = json.load(f)
    with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_camera.json')) as f:
        cam_params = json.load(f)
    with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_joint_3d.json')) as f:
        joints = json.load(f)

    occlusion_cam=[400006,400008,400015,400035,400049]
    grey_cam=[410001, 410003, 410004, 410014, 410015,410018, 410019, 410027, 410028, 410029, 410033, 410049, 410050,410053, 410057, 410061, 410062, 410063, 410066, 410067, 410068, 410069, 410208, 410210, 410211, 410213, 410214, 410215, 410216, 410218, 410219, 410220, 410226, 410229, 410231,410232, 410234, 410236, 410237, 410238]

    for acount, aid in enumerate(db.anns.keys()):
        if acount>10000:
            break 
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]
        capture_idx = str(img['capture'])
        capture_id=capture_idx
        seq_name = str(img['seq_name'])
        cam_idx = str(img['camera'])
        cam=img['camera']

        if str(cam)[:2]=='41':
            continue

        hand_type = ann['hand_type']
        if hand_type != 'interacting':
            continue

        img_width, img_height = img['width'], img['height']
        bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
        bbox = process_bbox(bbox, (img_height, img_width))

        frame_idx = str(img['frame_idx'])
        img_path = osp.join(img_root_path, split, img['file_name'])
        save_path = osp.join(img_root_path, split+'mask', img['file_name']+'.npy')
        meshdir=os.path.split(save_path)[0]
        os.makedirs(meshdir, exist_ok=True)
        save_path = meshdir
        frame_idx = img_path.split('/')[-1][5:-4]
        img = load_img(img_path)
        img_height, img_width, _ = img.shape
        
        prev_depth = None
        mesh={}
        for hand_type in ('right', 'left'):
            try:
                mano_param = mano_params[capture_idx][frame_idx][hand_type]
                if mano_param is None:
                    continue
            except KeyError:
                continue
            
            # get MANO 3D mesh coordinates (world coordinate)
            mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
            root_pose = mano_pose[0].view(1,3)
            hand_pose = mano_pose[1:,:].view(1,-1)
            shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
            trans = torch.FloatTensor(mano_param['trans']).view(1,3)
            output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
            mesh[hand_type] = output.vertices[0].numpy() # meter to milimeter
            mesh[hand_type]=seal(mesh[hand_type],hand_type)
        try:
            mesh_i=np.concatenate([mesh['right'],mesh['left']],0)
        except KeyError:
            continue
        # apply camera extrinsics
        cam_param = cam_params[capture_idx]
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
        
        vis_img=render_img_vis(mesh_i, vert_face, mesh_i_xy, mesh_i_z, R=R,T=t, fx=focal[0], fy=focal[1], px=princpt[0], py=princpt[1],image_size=(img_height, img_width))
        os.makedirs('/home/huangxuan_m2023/KeypointNeRF/'+split+'/densepose/capture'+str(capture_id)+'/cam'+str(cam)+'/', exist_ok=True)
        
        img,vis_img,trans = augmentation(img,vis_img,bbox)
        mask_save_path='/home/huangxuan_m2023/KeypointNeRF/'+split+'/densepose/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg'
        cv2.imwrite(mask_save_path, vis_img)
        print(mask_save_path)
