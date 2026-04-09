import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import json
from collections import defaultdict

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

def process_video(video_path, output_path, annot_path, model, model_cfg, detector, cpm, renderer, device, args):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0 # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} ({frame_count} frames)")
    
    video_annotations = []
    
    # Pre-populate empty annotations
    for i in range(frame_count):
        video_annotations.append({
            'frame_idx': i,
            'boxes': [],
            'right': [],
            'vertices': [],
            'cam_t': [],
            'keypoints_3d': []
        })

    buffered_frames = []
    buffered_indices = []

    tbar = tqdm(total=frame_count, desc=video_path.name)

    def process_buffered_frames():
        if not buffered_frames:
            return
            
        datasets = []
        dataset_frame_indices = []
        
        all_bboxes_per_frame = defaultdict(list)
        all_right_per_frame = defaultdict(list)

        # 1. Sequential Detectron2 and ViTPose per frame
        for buf_idx, img_cv2 in enumerate(buffered_frames):
            
            det_out = detector(img_cv2)
            img = img_cv2.copy()[:, :, ::-1]

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores=det_instances.scores[valid_idx].cpu().numpy()

            if len(pred_bboxes) > 0:
                vitposes_out = cpm.predict_pose(
                    img,
                    [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
                )

                bboxes_list = []
                is_right_list = []

                for vitposes in vitposes_out:
                    left_hand_keyp = vitposes['keypoints'][-42:-21]
                    right_hand_keyp = vitposes['keypoints'][-21:]

                    # Left hand
                    keyp = left_hand_keyp
                    valid = keyp[:,2] > 0.5
                    if sum(valid) > 3:
                        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                        bboxes_list.append(bbox)
                        is_right_list.append(0)
                        
                    # Right hand
                    keyp = right_hand_keyp
                    valid = keyp[:,2] > 0.5
                    if sum(valid) > 3:
                        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                        bboxes_list.append(bbox)
                        is_right_list.append(1)

                if len(bboxes_list) > 0:
                    boxes = np.stack(bboxes_list)
                    right = np.stack(is_right_list)
                    
                    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
                    datasets.append(dataset)
                    dataset_frame_indices.extend([buf_idx] * len(boxes))
                    all_bboxes_per_frame[buf_idx] = boxes
                    all_right_per_frame[buf_idx] = right

        # 2. Batched HaMeR Inference
        all_verts = defaultdict(list)
        all_cam_t = defaultdict(list)
        all_right_labels = defaultdict(list)
        all_kpts_3d = defaultdict(list)
        
        scaled_focal_length = None

        if len(datasets) > 0:
            concat_dataset = torch.utils.data.ConcatDataset(datasets)
            # Dataloader handles up to batch_size hands at once
            dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            out_idx = 0
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out_net = model(batch)

                multiplier = (2*batch['right']-1)
                pred_cam = out_net['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                
                # Note: scaled_focal_length is consistent for all frames in video
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                batch_size_n = batch['img'].shape[0]
                for n in range(batch_size_n):
                    buf_idx = dataset_frame_indices[out_idx]
                    
                    verts = out_net['pred_vertices'][n].detach().cpu().numpy()
                    is_right_n = batch['right'][n].cpu().numpy()
                    verts[:,0] = (2*is_right_n-1)*verts[:,0]
                    cam_t = pred_cam_t_full[n]
                    
                    if 'pred_keypoints_3d' in out_net:
                        kpts_3d = out_net['pred_keypoints_3d'][n].detach().cpu().numpy()
                        kpts_3d[:,0] = (2*is_right_n-1)*kpts_3d[:,0]
                        all_kpts_3d[buf_idx].append(kpts_3d)

                    all_verts[buf_idx].append(verts)
                    all_cam_t[buf_idx].append(cam_t)
                    all_right_labels[buf_idx].append(is_right_n)
                    
                    out_idx += 1

        # 3. Render and Write
        if scaled_focal_length is None:
            # Fallback if no hands were found in the batch to get focal length
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * width

        for buf_idx, img_cv2 in enumerate(buffered_frames):
            frame_idx = buffered_indices[buf_idx]
            verts_list = all_verts[buf_idx]
            
            # Save annotations
            if len(verts_list) > 0:
                video_annotations[frame_idx]['boxes'] = list(all_bboxes_per_frame[buf_idx])
                video_annotations[frame_idx]['right'] = list(all_right_labels[buf_idx])
                video_annotations[frame_idx]['vertices'] = all_verts[buf_idx]
                video_annotations[frame_idx]['cam_t'] = all_cam_t[buf_idx]
                video_annotations[frame_idx]['keypoints_3d'] = all_kpts_3d[buf_idx]
                
                # Render front view
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length.item() if isinstance(scaled_focal_length, torch.Tensor) else scaled_focal_length,
                )
                render_res = [width, height] 
                cam_view = renderer.render_rgba_multiple(verts_list, cam_t=all_cam_t[buf_idx], render_res=render_res, is_right=all_right_labels[buf_idx], **misc_args)
                
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                
                output_frame = (255*input_img_overlay[:, :, ::-1]).astype(np.uint8)
                out.write(output_frame)
            else:
                out.write(img_cv2)
                
        buffered_frames.clear()
        buffered_indices.clear()

    # Read Loop
    frame_idx = 0
    while True:
        ret, img_cv2 = cap.read()
        if not ret:
            break
            
        buffered_frames.append(img_cv2)
        buffered_indices.append(frame_idx)
        frame_idx += 1
        
        if len(buffered_frames) >= args.batch_size:
            process_buffered_frames()
            tbar.update(args.batch_size)

    if len(buffered_frames) > 0:
        n_rem = len(buffered_frames)
        process_buffered_frames()
        tbar.update(n_rem)

    tbar.close()
    cap.release()
    out.release()
    
    # Save the annotations using pickle
    annot_dir = os.path.dirname(annot_path)
    os.makedirs(annot_dir, exist_ok=True)
    
    with open(annot_path, 'wb') as f:
        pickle.dump(video_annotations, f)
    print(f"Saved annotations (pickle) to {annot_path}")

    # Save the annotations using json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super(NumpyEncoder, self).default(obj)
            
    json_path = str(annot_path).replace('.pkl', '.json')
    with open(json_path, 'w') as f:
        json.dump(video_annotations, f, cls=NumpyEncoder)
    print(f"Saved annotations (JSON) to {json_path}")


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code for a single video')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Output folder to save rendered videos')
    parser.add_argument('--annot_dir', type=str, default='outputs/annotations_new', help='Output folder to save framewise annotations')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of frames to buffer and HaMeR batch size')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directories if they do not exist
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.annot_dir, exist_ok=True)

    vid_path = Path(args.video_path)
    if not vid_path.exists():
        print(f"Video file not found: {vid_path}")
        return

    vid_fn = vid_path.name
    out_path = Path(args.out_dir) / vid_fn
    # Save annotations with .pkl extension in annot_dir
    annot_fn = vid_path.stem + '.pkl'
    annot_path = Path(args.annot_dir) / annot_fn
    process_video(vid_path, out_path, annot_path, model, model_cfg, detector, cpm, renderer, device, args)

if __name__ == '__main__':
    main()


# docker compose -f ./docker/docker-compose.yml exec hamer-dev python demo_single_video_with_annotations.py \
#     --video_path videos/DJI_20260219190950_0066_D.MP4 \
#     --out_dir outputs/annotations_new/ \
#     --annot_dir outputs/annotations_new/ \
#     --batch_size 32
