import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
from torchvision import transforms
from collections import defaultdict
from scipy.spatial import distance
import matplotlib.pyplot as plt
import json
import random
import os
import base64
from backend.models.analyzer import TrackAnalyzer
from typing import Dict, Tuple, List

class VideoProcessor:
    def __init__(self):
        # Initialize models
        self.yolo_model = YOLO("yolov8m.pt")
        self.reid_model = self._initialize_reid_model()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def _initialize_reid_model(self):
        model = torchreid.models.build_model(name='osnet_x1_0', num_classes=751)
        torchreid.utils.load_pretrained_weights(model, 'osnet_x1_0_imagenet.pth')
        model.eval()
        return model
    
    def get_features(self, roi):
        if roi.size == 0: 
            return None
        roi = cv2.resize(roi, (128, 256), interpolation=cv2.INTER_LANCZOS4)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = torch.from_numpy(roi).float().permute(2, 0, 1) / 255.0
        roi = self.normalize(roi).unsqueeze(0)
        with torch.no_grad():
            return self.reid_model(roi).squeeze().cpu().numpy()
    def video_to_base64(self, video_path: str) -> str:
        """Convert video file to base64 encoded string"""
        with open(video_path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
        return encoded_string
    def _draw_curved_line(self, frame, pt1, pt2, color, thickness=2):
        """Draw a curved connection line between two points"""
    # Calculate control point for the curve
        offset = 30  # Curve amplitude
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        angle = np.arctan2(dy, dx)
    
    # Calculate perpendicular offset direction
        ctrl_pt = (
        (pt1[0] + pt2[0]) // 2 + int(offset * np.cos(angle + np.pi/2)),
        (pt1[1] + pt2[1]) // 2 + int(offset * np.sin(angle + np.pi/2))
        )
    
    # Draw Bezier curve
        pts = np.array([pt1, ctrl_pt, pt2], np.int32)
        cv2.polylines(frame, [pts], False, color, thickness, lineType=cv2.LINE_AA)

    def _draw_dashed_line(self, frame, pt1, pt2, color, thickness=2, dash_length=10):
        """Draw dashed line between two points"""
        dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start = (int(pt1[0] + (pt2[0]-pt1[0])*i/dashes),
                    int(pt1[1] + (pt2[1]-pt1[1])*i/dashes))
            end = (int(pt1[0] + (pt2[0]-pt1[0])*(i+0.5)/dashes),
                    int(pt1[1] + (pt2[1]-pt1[1])*(i+0.5)/dashes))
            cv2.line(frame, start, end, color, thickness, lineType=cv2.LINE_AA) 

    def _draw_interaction_label(self, frame, text, position, angle, color):
        """Draw rotated text label"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_img = np.zeros_like(frame)
        # Get text size and create rotation matrix
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        M = cv2.getRotationMatrix2D(position, angle, 1)
    
        # Create background rectangle
        rect_w = text_size[0] + 10
        rect_h = text_size[1] + 10
        rect_pts = np.array([
            [-rect_w//2, -rect_h//2],
            [rect_w//2, -rect_h//2],
            [rect_w//2, rect_h//2],
            [-rect_w//2, rect_h//2]
        ])
        rect_pts = (rect_pts @ M[:2,:2].T + position).astype(int)
        cv2.fillPoly(frame, [rect_pts], (40, 40, 40))
    
        text_origin = np.array([position[0] - text_size[0]//2, position[1] + text_size[1]//2])
        cv2.putText(
        text_img, text, 
        tuple(text_origin.astype(int)),
        font, font_scale, color, thickness, lineType=cv2.LINE_AA
        )

        rotated = cv2.warpAffine(text_img, M, (frame.shape[1], frame.shape[0]))
        frame[:] = cv2.addWeighted(frame, 1.0, rotated, 1.0, 0)
    
    def _draw_style_line(self, frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                    color: Tuple[int, int, int], interaction_type: str, confidence: float) -> None:
        """Draw stylish connection line with appropriate style for each interaction type"""
        line_thickness = max(1, min(4, int(4 * confidence)))
        
        if interaction_type == "Handshake":
            # Elegant curved double line with gradient
            for offset in [-2, 2]:
                pts = np.array([
                    pt1,
                    (pt1[0] + (pt2[0]-pt1[0])//3, pt1[1] + (pt2[1]-pt1[1])//3 + offset*3),
                    (pt1[0] + 2*(pt2[0]-pt1[0])//3, pt1[1] + 2*(pt2[1]-pt1[1])//3 - offset*3),
                    pt2
                ], np.int32)
                cv2.polylines(frame, [pts], False, color, line_thickness, lineType=cv2.LINE_AA)
                
        elif interaction_type == "Pushing":
            # Angled arrow-like dashed line
            arrow_length = min(15, int(np.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])/3))
            cv2.arrowedLine(frame, pt1, pt2, color, line_thickness, 
                          line_type=cv2.LINE_AA, tipLength=0.2)
            
        else:
            # Smooth bezier curve for other interactions
            control_pt = (
                (pt1[0] + pt2[0]) // 2 + random.randint(-20, 20),
                (pt1[1] + pt2[1]) // 2 + random.randint(-20, 20))
            pts = np.array([pt1, control_pt, pt2], np.int32)
            cv2.polylines(frame, [pts], False, color, line_thickness, lineType=cv2.LINE_AA)


    def _draw_elegant_label(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                          color: Tuple[int, int, int]) -> None:
        """Draw professional-looking label with rounded background"""
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1
        padding = 7
        radius = 10
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Create rounded rectangle background
        pt1 = (position[0] - padding, position[1] - text_height - padding)
        pt2 = (position[0] + text_width + padding, position[1] + padding)
        
        # Draw rounded rectangle
        cv2.rectangle(frame, 
                     (pt1[0], pt1[1] + radius),
                     (pt2[0], pt2[1] - radius), 
                     (40, 40, 40), -1)
        cv2.rectangle(frame, 
                     (pt1[0] + radius, pt1[1]),
                     (pt2[0] - radius, pt2[1]), 
                     (40, 40, 40), -1)
        cv2.circle(frame, (pt1[0] + radius, pt1[1] + radius), radius, (40, 40, 40), -1)
        cv2.circle(frame, (pt2[0] - radius, pt1[1] + radius), radius, (40, 40, 40), -1)
        cv2.circle(frame, (pt1[0] + radius, pt2[1] - radius), radius, (40, 40, 40), -1)
        cv2.circle(frame, (pt2[0] - radius, pt2[1] - radius), radius, (40, 40, 40), -1)
        
        # Draw text with subtle shadow
        cv2.putText(frame, text, 
                   (position[0] + 1, position[1] + 1), 
                   font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
        cv2.putText(frame, text, 
                   position, 
                   font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    def process_video(self, video_path: str, output_folder: str, 
                     desired_fps: int = 10, confidence_threshold: float = 0.3, 
                     iou_threshold: float = 0.4):
        """Process video and return analysis results"""
        os.makedirs(output_folder, exist_ok=True)

        # Initialize tracker and analyzer
        tracker = DeepSort(max_age=15, n_init=3, max_cosine_distance=0.4, nn_budget=50)
        analyzer = TrackAnalyzer()

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video metadata
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps

        if duration < 1:
            original_fps = 125
            duration = total_frames / original_fps

        frame_skip = max(1, int(round(original_fps / desired_fps)))
        target_frame_count = int(desired_fps * duration)

        # Set up output video
        output_video_path = os.path.join(output_folder, "output-behavior-analysis.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                            desired_fps, (512, 384))

        frame_counter = 0
        processed_frames = 0

        interaction_colors = {
            "Handshake": (0, 255, 0),
            "Pushing": (0, 0, 255),
            "Wrestling": (255, 0, 0),
            "Close Proximity": (255, 255, 0),
            "Interaction": (255, 0, 255)
        }

        while cap.isOpened() and processed_frames < target_frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % frame_skip != 0:
                frame_counter += 1
                continue

            # Process frame
            frame = cv2.resize(frame, (512, 384))
            # Person detection
            results = self.yolo_model(frame, classes=[0], 
                                    conf=confidence_threshold, 
                                   iou=iou_threshold)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]
                    features = self.get_features(roi)
                    if features is not None:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf[0], features))

            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Update analyzer
            analyzer.update_tracks(tracks, processed_frames)

            # Draw tracks and interactions
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                track_id = track.track_id

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                info_text = f"ID: {track_id} ({x2 - x1}x{y2 - y1})"
                cv2.putText(frame, info_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            
            interactions = analyzer.get_interactions()
            drawn_pairs = set()

            for interaction in interactions:
                id1, id2, interaction_type, confidence = interaction
                if (id1, id2) in drawn_pairs or (id2, id1) in drawn_pairs:
                    continue
        
                # Get current positions
                try:
                    box1 = analyzer.track_history[id1]['boxes'][-1]
                    box2 = analyzer.track_history[id2]['boxes'][-1]
                    center1 = analyzer._box_center(box1)
                    center2 = analyzer._box_center(box2)
                    color = interaction_colors.get(interaction_type, (255, 255, 255))

                    self._draw_style_line(frame, center1, center2, color, interaction_type, confidence)
                    label_pos = (
                        int(center1[0] + (center2[0]-center1[0])*0.4 + random.randint(-15, 15)),
                        int(center1[1] + (center2[1]-center1[1])*0.4 + random.randint(-15, 15))
                    )
                    self._draw_elegant_label(
                        frame, 
                        f"{interaction_type.upper()} {confidence*100:.0f}%",
                        label_pos,
                        color
                    )
                    drawn_pairs.add((id1, id2))
                except (IndexError, KeyError):
                    continue
        
                
    

            out.write(frame)
            processed_frames += 1
            frame_counter += 1

        cap.release()
        out.release()

        # Generate behavior report
        behavior_report = analyzer.get_behavior_analysis()
        report_path = os.path.join(output_folder, "behavior_report.json")
        with open(report_path, 'w') as f:
            json.dump(behavior_report, f, indent=2)

        # Generate interaction plots
        plot_folder = os.path.join(output_folder, 'interaction_plots')
        os.makedirs(plot_folder, exist_ok=True)
        analyzer.generate_interaction_plots(output_folder)

        return {
            "output_video_path": output_video_path,
            "report_path": report_path,
            "plots_path": plot_folder,
            "processed_frames": processed_frames
        }