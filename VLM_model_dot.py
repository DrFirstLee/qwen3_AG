#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen 2.5-VL 바운딩 박스 탐지 및 시각화 스크립트
Qwen 모델로 객체 탐지 후 바운딩 박스를 이미지에 그려주는 코드
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.utils import logging
# ✨ Qwen3 전용 클래스 임포트 (사용자 제공 코드 반영)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

logging.set_verbosity_error()
import timm
from qwen_vl_utils import process_vision_info
import torch
import re
import json
import math
import os
from datetime import datetime
from config import model_name, model_size
from collections import defaultdict
import numpy as np
from torchvision.transforms import GaussianBlur
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as T
import cv2


class MetricsTracker:
    def __init__(self, name):
        self.metrics = defaultdict(list)
        self.total_samples = 0  
        self.name = name
    def update(self, new_metrics):
        """Update metrics with new values"""
        for key, value in new_metrics.items():
            # nan 또는 None은 저장하지 않음
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                self.metrics[key].append(value)
        self.total_samples += 1
    
    def get_averages(self):
        """Calculate average metrics"""
        return {
            key: (sum(values) / len(values)) if values else float('nan')
            for key, values in self.metrics.items()
        }
    
    def print_metrics(self, current_metrics, image_name):
        """Print current and cumulative metrics"""
        print(f"\n{'='*50}")
        print(f"Metrics for {self.name} {image_name}:")
        print(f" {self.name} Current - KLD: {current_metrics['KLD']:.4f} | SIM: {current_metrics['SIM']:.4f} | NSS: {current_metrics['NSS']:.4f}")
        
        if self.total_samples > 0:
            averages = self.get_averages()
            print(f"\nCumulative {self.name}  Averages over {self.total_samples} samples:")
            print(f"Average - KLD: {averages['KLD']:.4f} | SIM: {averages['SIM']:.4f} | NSS: {averages['NSS']:.4f}")
        print(f"{'='*50}\n")


class QwenVLModel:
    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct"):
        """
        Qwen3-VL 모델 및 DINO 초기화
        """
        self.model_name = model_name
        
        print(f"🤖 {self.model_name} 모델 로딩중...")
        # 1. Processor 로드
        
        
        # 2. Qwen3 Model 로드
        # 사용자가 제공한 코드에 맞춰 Qwen3VLForConditionalGeneration 사용
        # dtype="auto", device_map="auto" 적용
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        # 모델이 로드된 주 디바이스 확인 (DINO를 같은 곳에 올리기 위해)
        self.device = self.model.device
        print(f"✅ Qwen3 모델 로딩 완료! (Main Device: {self.device})")


    def create_heatmap_from_dots_v2(self, image_size, dots):
        """
        Create a heatmap from dot coordinates using Gaussian kernels with dynamic sigma.
        Args:
            image_size (tuple): Size of the image (height, width)
            dots (list): List of dot coordinates [x, y]
        Returns:
            torch.Tensor: Heatmap tensor
        """
        height, width = image_size

        # Dynamic sigma based on image dimensions (simple linear scaling)
        base_size = 640  # Reference size
        base_sigma = 60
        scale_factor = ((height + width) / 2) / base_size
        sigma = int( base_sigma * scale_factor)
        heatmap = torch.zeros((height, width))
        for dot in dots:
            # Convert coordinates to integers
            x, y = map(int, dot)
            # Ensure coordinates are within image bounds
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            # Create coordinate grids for the entire image
            y_grid, x_grid = torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing='ij'
            )
            # Calculate Gaussian values centered at the dot
            gaussian = torch.exp(
                -((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2)
            )
            # Add to heatmap
            heatmap += gaussian
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        return heatmap

    def draw_dots_on_single_image(self, image, dots, color='red', radius=15):
        """
        Draw dots on an image
        Args:
            image (PIL.Image): Image to draw on
            dots (list): List of dot coordinates [x, y]
            color (str): Color of the dots
            radius (int): Radius of the dots
        Returns:
            PIL.Image: Image with dots drawn
        """
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        for dot in dots:
            x, y = map(int, dot)
            # Draw circle
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=color, outline=color)
        
        return img_copy

    def load_ground_truth(self, gt_path):
        """
        Load and process ground truth image
        Args:
            gt_path (str): Path to the ground truth image
        Returns:
            torch.Tensor: Processed ground truth tensor normalized to [0, 1]
        """
        try:
            # Load the ground truth image
            gt_img = Image.open(gt_path)
            
            # Convert to grayscale if image is RGB
            if gt_img.mode == 'RGB':
                gt_img = gt_img.convert('L')
            
            # Convert to tensor
            gt_tensor = transforms.ToTensor()(gt_img).squeeze(0)
            
            # Normalize to [0, 1]
            if gt_tensor.max() > 0:
                gt_tensor = (gt_tensor - gt_tensor.min()) / (gt_tensor.max() - gt_tensor.min())
            
            return gt_tensor
            
        except Exception as e:
            print(f"⚠️ Failed to load ground truth image: {str(e)}")
            return None

    def draw_dots_on_image(self, image_path, dots, gt_path, action, exo_path=None, exo_type=None, output_path=None):
        """
        Draw dots and create heatmap, save results side by side with GT
        Args:
            image_path (str): Path to the ego image
            dots (list): List of dot coordinates [x, y]
            gt_path (str): Path to the ground truth image
            action (str): Action name for the filename
            exo_path (str, optional): Path to the exo image (if provided, creates 3x2 layout)
            exo_type (str, optional): Type of exo image ('random' or 'selected')
            output_path (str, optional): Path to save the result image
        Returns:
            str: Path to the saved image
            torch.Tensor: Generated heatmap for metric calculation
        """
        # Load the ego image
        ego_img = Image.open(image_path)
        if exo_path is not None:
            exo_file_name = os.path.basename(exo_path)
        width, height = ego_img.size
        
        # Load exo image if provided
        exo_img = None
        if exo_path:
            exo_img = Image.open(exo_path)
        
        # Create heatmap from dots
        heatmap_tensor = self.create_heatmap_from_dots_v2((height, width), dots)
        
        # Convert heatmap to RGB image
        heatmap_img = transforms.ToPILImage()(heatmap_tensor.unsqueeze(0).repeat(3, 1, 1))
        
        # Create a copy for dot drawing
        dot_img = self.draw_dots_on_single_image(ego_img, dots, color='red', radius=15)
        
        # Determine layout based on image aspect ratio
        aspect_ratio = width / height
        
        # For very wide images (aspect ratio > 2), adjust font size based on width
        if aspect_ratio > 2:
            font_size = min(50, width // 12)  # Larger font for wide images
            header_height = 110  # Increased header height
            spacing = 30  # Normal spacing
        elif aspect_ratio > 1.5:  # For moderately wide images
            font_size = min(55, width // 10)  # Larger font for moderately wide images
            header_height = 120  # Increased header height
            spacing = 35  # Slightly increased spacing
        else:
            font_size = max(60, width // 8)  # Largest font size for normal images
            header_height = 130  # Normal header height
            spacing = 40  # Normal spacing
        
        # Create a new image with 3x2 layout
        combined_width = width * 3
        combined_height = height * 2 + header_height * 2 + spacing * 3 + 40  # Dynamic height
        combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
        
        # Try to load fonts (size proportional to image width and aspect ratio)
        try:
            # Try to load a font that supports Korean
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
        except:
            try:
                # Fallback to DejaVu font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                # Last resort: default font
                font = ImageFont.load_default()

        # Get file names
        ego_filename = os.path.basename(image_path)
        gt_filename = os.path.basename(gt_path) if gt_path else "No GT"
        
        # Draw file names and titles for 3x2 layout
        draw = ImageDraw.Draw(combined_img)
        
        # Configure headers based on whether exo image is provided
        if exo_img:
            exo_filename = os.path.basename(exo_path)
            # Top row headers for exo version
            top_headers = [
                ("Ego", ego_filename),
                ("Exo", exo_filename),
                ("", "")  # Empty space
            ]
        else:
            # Top row headers for ego only version
            top_headers = [
                ("Original", ego_filename),
                ("", ""),  # Empty space
                ("", "")   # Empty space
            ]
        
        # Bottom row headers (same for both versions)
        bottom_headers = [
            ("Dots", action+"_"+ego_filename),
            ("Heatmap", action+"_"+ ego_filename),
            ("GT", action+"_"+gt_filename)
        ]
        
        # Draw top row headers with background
        for idx, (title, filename) in enumerate(top_headers):
            if title:  # Only draw if not empty
                section_width = width
                section_x = idx * section_width
                
                # Draw white background for text area
                draw.rectangle([section_x, 0, section_x + section_width, header_height], fill='white', outline='lightgray')
                
                # Draw title
                title_width = draw.textlength(title, font=font)
                title_x = section_x + (section_width - title_width) // 2
                draw.text((title_x, 5), title, fill='black', font=font)
                
                # Draw filename (truncate if too long)
                max_filename_width = section_width - 20
                filename_truncated = filename
                while draw.textlength(filename_truncated + "...", font=font) > max_filename_width and len(filename_truncated) > 0:
                    filename_truncated = filename_truncated[:-1]
                if filename_truncated != filename:
                    filename_truncated += "..."
                
                filename_width = draw.textlength(filename_truncated, font=font)
                filename_x = section_x + (section_width - filename_width) // 2
                draw.text((filename_x, header_height // 2 + 5), filename_truncated, fill='black', font=font)
        
        # Draw bottom row headers with background
        for idx, (title, filename) in enumerate(bottom_headers):
            section_width = width
            section_x = idx * section_width
            section_y = height + header_height + spacing  # Position below first row
            
            # Draw white background for text area
            draw.rectangle([section_x, section_y - 10, section_x + section_width, section_y + header_height], fill='white', outline='lightgray')
            
            # Draw title
            title_width = draw.textlength(title, font=font)
            title_x = section_x + (section_width - title_width) // 2
            draw.text((title_x, section_y), title, fill='black', font=font)
            
            # Draw filename (truncate if too long)
            max_filename_width = section_width - 20
            filename_truncated = filename
            while draw.textlength(filename_truncated + "...", font=font) > max_filename_width and len(filename_truncated) > 0:
                filename_truncated = filename_truncated[:-1]
            if filename_truncated != filename:
                filename_truncated += "..."
            
            filename_width = draw.textlength(filename_truncated, font=font)
            filename_x = section_x + (section_width - filename_width) // 2
            draw.text((filename_x, section_y + header_height // 2), filename_truncated, fill='black', font=font)
        
        # Paste images in 3x2 layout
        # Top row: Ego image and optionally Exo image
        top_image_y = header_height + spacing
        combined_img.paste(ego_img, (0, top_image_y))  # Ego
        if exo_img:
            combined_img.paste(exo_img, (width, top_image_y))  # Exo
        
        # Bottom row: Dots, Heatmap, GT
        bottom_image_y = height + header_height * 2 + spacing * 2
        combined_img.paste(dot_img, (0, bottom_image_y))  # Image with dots
        combined_img.paste(heatmap_img, (width, bottom_image_y))  # Heatmap
        
        # Add GT image and calculate metrics
        gt_map = self.load_ground_truth(gt_path)
        metrics_text = "No GT provided"
        
        if gt_map is not None:
            if isinstance(gt_map, torch.Tensor):
                gt_img = transforms.ToPILImage()(gt_map.unsqueeze(0).repeat(3, 1, 1))
            else:
                gt_map_tensor = torch.tensor(gt_map)
                gt_img = transforms.ToPILImage()(gt_map_tensor.unsqueeze(0).repeat(3, 1, 1))
            combined_img.paste(gt_img, (width * 2, bottom_image_y))  # GT heatmap
            
            # Calculate metrics
            metrics = self.calculate_metrics(heatmap_tensor, gt_map)
            metrics_text = f"KLD: {metrics['KLD']:.4f} | SIM: {metrics['SIM']:.4f} | NSS: {metrics['NSS']:.4f}"
        else:
            # If no GT provided, create blank white image
            blank_img = Image.new('RGB', (width, height), 'white')
            combined_img.paste(blank_img, (width * 2, bottom_image_y))
            metrics_text = "ERRRR"
        
        # Draw metrics text at the bottom with background
        text_width = draw.textlength(metrics_text, font=font)
        text_x = (combined_width - text_width) // 2
        text_y = bottom_image_y + height + spacing
        
        # Draw white background for metrics text
        padding = 10
        draw.rectangle([text_x - padding, text_y - padding, 
                       text_x + text_width + padding, text_y + font_size + padding], 
                      fill='white', outline='gray')
        
        draw.text((text_x, text_y), metrics_text, fill='black', font=font)
        
        # Create res_images directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if exo_type is None:
            res_dir = os.path.join(script_dir, f'dot_images')
            os.makedirs(res_dir, exist_ok=True)
            os.makedirs(os.path.join(res_dir, "with_exo"), exist_ok=True)
            os.makedirs(os.path.join(res_dir, "only_ego"), exist_ok=True)
        else:
            res_dir = os.path.join(script_dir, f'dot_images_{exo_type}')
            os.makedirs(res_dir, exist_ok=True)
            os.makedirs(os.path.join(res_dir, "with_exo"), exist_ok=True)
            os.makedirs(os.path.join(res_dir, f"{exo_type}"), exist_ok=True)
            os.makedirs(os.path.join(res_dir, "only_ego"), exist_ok=True)            
            
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(ego_filename)[0]
            ext = os.path.splitext(ego_filename)[1]
            if exo_img and exo_type:
                # Format: skis_002829_jump_exo_random.jpg or skis_002829_jump_exo_selected.jpg
                output_filename = f"{base_name}_{action}_exo_{exo_type}{ext}"
                output_path = os.path.join(res_dir, f"with_exo/{output_filename}")
            elif exo_img:
                # Fallback if exo_type not specified
                output_filename = f"{base_name}_{action}_exo_{exo_file_name}"
                output_path = os.path.join(res_dir, f"with_exo/{output_filename}")
            elif exo_type is not None:
                output_filename = f"{base_name}_{action}_exo_{exo_type}{ext}"
                output_path = os.path.join(res_dir, f"{exo_type}/{output_filename}")
            else:
                # Format: skis_002829_jump.jpg
                output_filename = f"{base_name}_{action}{ext}"
                output_path = os.path.join(res_dir, f"only_ego/{output_filename}")
        
        # Save the combined image
        combined_img.save(output_path)
        # print(f"✅ Saved comparison image with heatmap and GT: {output_path}")
        
        return output_path, heatmap_tensor

    def ask(self, question: str) -> str:
        """
        주어진 텍스트 질문에 대해 model.generate()를 사용하여 답변을 반환합니다.
        """
        # 1) 모델이 이해할 수 있는 대화 형식(chat format)으로 메시지 구성
        messages = [{'role': 'user', 'content': question}]

        # 2) processor를 사용해 메시지를 모델 입력 형식으로 변환
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # MODIFIED: processor() 대신 tokenizer를 직접 호출하여 텍스트만 처리
        inputs = self.processor.tokenizer(
            [text], return_tensors="pt"
        ).to(self.device)

        # 3) 모델 추론 (generate 호출)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0
            )

        # 4) 결과 디코딩
        input_ids_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0, input_ids_len:]
        
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return response

    def ask_with_image(self, question: str, image_path:str) -> str:
        """
        주어진 텍스트 질문에 대해 model.generate()를 사용하여 답변을 반환합니다.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]
            }
        ]
                
        # 채팅 템플릿 적용
        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    def process_image_ego(self, image_path, prompt, gt_path, action):
        """
        Process an image with the given prompt
        Args:
            image_path (str): Path to the image
            prompt (str): Prompt for the model
            gt_path (str): Path to the ground truth image
        Returns:
            dict: Model's response and processed information
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        result = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"qwen ego Results!! : {result}")
        # dot 좌표 파싱
        dots = self.parse_dot_coordinates(result)
        print(f"parsed dots!!! : {dots}")
        
        # Draw dots on the image and get metrics
        dot_image_path, heatmap_tensor = self.draw_dots_on_image(image_path, dots, gt_path, action)
        
        # Save heatmap image
        script_dir = os.path.dirname(os.path.abspath(__file__))
        res_dir = os.path.join(script_dir, f'dot_images', 'heatmaps')
        os.makedirs(res_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        heatmap_filename = f"{base_name}_{action}_heatmap{ext}"
        heatmap_path = os.path.join(res_dir, heatmap_filename)
        
        # Convert heatmap tensor to image and save
        heatmap_img = transforms.ToPILImage()(heatmap_tensor.unsqueeze(0).repeat(3, 1, 1))
        heatmap_img.save(heatmap_path)
        
        # Save dot image separately for validation
        dot_res_dir = os.path.join(script_dir, f'dot_images', 'dots_only')
        os.makedirs(dot_res_dir, exist_ok=True)
        dot_only_filename = f"{base_name}_{action}_dots{ext}"
        dot_only_path = os.path.join(dot_res_dir, dot_only_filename)
        
        # Create dot image (ego image with dots)
        ego_img = Image.open(image_path)
        dot_only_img = self.draw_dots_on_single_image(ego_img, dots, color='red', radius=15)
        dot_only_img.save(dot_only_path)
        
        # Calculate metrics if GT is available
        metrics = None
        gt_map = self.load_ground_truth(gt_path)
        if gt_map is not None and len(dots) > 0:
            metrics = self.calculate_metrics(heatmap_tensor, gt_map)
        
        return {
            'text_result': result.strip(),
            'dots': dots,
            'dot_image_path': dot_image_path,
            'dot_only_image_path': dot_only_path,
            'heatmap_image_path': heatmap_path,
            'heatmap_tensor': heatmap_tensor,
            'metrics': metrics
        }
    
    def parse_dot_coordinates(self, text):
        """
        Parse list of dot coordinates from a model-generated text response.

        Supported formats:
        - JSON list of lists: [[x1, y1], [x2, y2], ...]
        - Individual points: 
          point1: [x, y]
          point2: [x, y]
          point1: (x, y)
        - Raw coordinates like: [x, y] or (x, y)
        - Bounding boxes (converted to center points): [x1, y1, x2, y2]

        Args:
            text (str): Text containing keypoint coordinates

        Returns:
            list: List of [x, y] integer coordinates
        """
        points = []

        # First try to parse as JSON array
        try:
            # Look for JSON array pattern
            json_pattern = r"\[\s*\[[\d\s,]+\](?:\s*,\s*\[[\d\s,]+\])*\s*\]"
            match = re.search(json_pattern, text)
            if match:
                json_like = match.group(0)
                parsed = json.loads(json_like)
                # print(f"Found JSON array: {parsed}")
                
                for pt in parsed:
                    if isinstance(pt, list):
                        if len(pt) == 2:
                            # Standard dot format [x, y]
                            x, y = map(int, pt)
                            points.append([x, y])
                        elif len(pt) == 4:
                            # Bounding box format [x1, y1, x2, y2] - convert to center point
                            x1, y1, x2, y2 = map(int, pt)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            points.append([center_x, center_y])
                            # print(f"Converted bbox {pt} to center point [{center_x}, {center_y}]")
                
                if points:
                    return points
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            pass

        # Try to parse individual coordinate patterns
        patterns = [
            # 2-coordinate patterns (dots)
            r'point\d*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            r'point\d*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
            r'\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            r'\(\s*(\d+)\s*,\s*(\d+)\s*\)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    x, y = map(int, match)
                    points.append([x, y])
                except Exception as e:
                    continue

        # If still no points found, try to parse as bounding boxes and convert to center points
        if not points:
            print("No dot coordinates found, trying to parse as bounding boxes...")
            print(f"text : {text}")

        print(f"final points :{points}")
        return points

    def calculate_metrics(self, pred_heatmap, gt_map):
        """
        Calculate comparison metrics between predicted heatmap and GT (following original metric.py)
        Args:
            pred_heatmap (torch.Tensor): Predicted heatmap
            gt_map (torch.Tensor): Ground truth map
        Returns:
            dict: Dictionary containing KLD, SIM, and NSS metrics
        """
        # Ensure inputs are proper tensors
        if not isinstance(pred_heatmap, torch.Tensor):
            pred_heatmap = torch.tensor(pred_heatmap)
        if not isinstance(gt_map, torch.Tensor):
            gt_map = torch.tensor(gt_map)
        
        # Flatten tensors and add batch dimension for compatibility
        pred = pred_heatmap.flatten().float().unsqueeze(0)  # [1, H*W]
        gt = gt_map.flatten().float().unsqueeze(0)          # [1, H*W]
        
        eps = 1e-10
        
        # Calculate KLD following original implementation
        # Normalize to probability distributions
        pred_norm = pred / pred.sum(dim=1, keepdim=True)
        gt_norm = gt / gt.sum(dim=1, keepdim=True)
        pred_norm += eps
        kld = F.kl_div(pred_norm.log(), gt_norm, reduction="batchmean").item()
        
        # Calculate SIM following original implementation
        pred_sim = pred / pred.sum(dim=1, keepdim=True)
        gt_sim = gt / gt.sum(dim=1, keepdim=True)
        sim = torch.minimum(pred_sim, gt_sim).sum().item() / len(pred_sim)
        
        # Calculate NSS following original implementation
        # First normalize by max values
        pred_nss = pred / pred.max(dim=1, keepdim=True).values
        gt_nss = gt / gt.max(dim=1, keepdim=True).values
        
        # Calculate z-score for prediction
        std = pred_nss.std(dim=1, keepdim=True)
        u = pred_nss.mean(dim=1, keepdim=True)
        smap = (pred_nss - u) / (std + eps)
        
        # Create fixation map from GT
        fixation_map = (gt_nss - torch.min(gt_nss, dim=1, keepdim=True).values) / (
            torch.max(gt_nss, dim=1, keepdim=True).values - torch.min(gt_nss, dim=1, keepdim=True).values + eps)
        fixation_map = (fixation_map >= 0.1).float()
        
        # Calculate NSS
        nss_values = smap * fixation_map
        nss = nss_values.sum(dim=1) / (fixation_map.sum(dim=1) + eps)
        nss = nss.mean().item()
        
        return {
            'KLD': kld,
            'SIM': sim,
            'NSS': nss
        }

    def process_image_exo(self, image_path, prompt, gt_path, exo_path, action, exo_type=None):
        """
        Process an image with the given prompt (exo view version)
        Args:
            image_path (str): Path to the ego image
            prompt (str): Prompt for the model
            gt_path (str): Path to the ground truth image
            exo_path (str): Path to the exo view image
            action (str): Action name for the filename
        Returns:
            dict: Model's response and processed information
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "image", "image": exo_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        exo_filename =  os.path.basename(exo_path)
        print(f"exo file name : {exo_filename} / exo_path")

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        result = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # print(f"qwen with exo Results!! : {result}")
        # dot 좌표 파싱
        dots = self.parse_dot_coordinates(result)
        # print(f"parsed dots!!! : {dots}")
        # Draw dots on the image and get metrics
        dot_image_path, heatmap_tensor = self.draw_dots_on_image(image_path, dots, gt_path, action, exo_path, exo_type)
        
        # Save heatmap image
        script_dir = os.path.dirname(os.path.abspath(__file__))
        res_dir = os.path.join(script_dir, f'dot_images', 'heatmaps')
        os.makedirs(res_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        heatmap_filename = f"{base_name}_{action}_heatmap_exo_reference_{exo_filename}"
        heatmap_path = os.path.join(res_dir, heatmap_filename)

        # Convert heatmap tensor to image and save
        heatmap_img = transforms.ToPILImage()(heatmap_tensor.unsqueeze(0).repeat(3, 1, 1))
        heatmap_img.save(heatmap_path)
        
        # Save dot image separately for validation
        dot_res_dir = os.path.join(script_dir, f'dot_images', 'dots_only')
        os.makedirs(dot_res_dir, exist_ok=True)
        dot_only_filename = f"{base_name}_{action}_dots_exo{ext}"
        dot_only_path = os.path.join(dot_res_dir, dot_only_filename)
        
        # Create dot image (ego image with dots)
        ego_img = Image.open(image_path)
        dot_only_img = self.draw_dots_on_single_image(ego_img, dots, color='red', radius=15)
        dot_only_img.save(dot_only_path)

        # Calculate metrics if GT is available
        metrics = None
        gt_map = self.load_ground_truth(gt_path)
        if gt_map is not None and len(dots) > 0:
            metrics = self.calculate_metrics(heatmap_tensor, gt_map)
        
        return {
            'text_result': result.strip(),
            'dots': dots,
            'dot_image_path': dot_image_path,
            'dot_only_image_path': dot_only_path,
            'heatmap_image_path': heatmap_path,
            'heatmap_tensor': heatmap_tensor,
            'metrics': metrics
        }


if __name__ == "__main__":
    # Test code
    model = QwenVLModel()
    # Add test code here if needed 