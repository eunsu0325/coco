# model_ops.py - 기본 CCNet용 정리된 버전
"""
BioSecure Palm - 기본 CCNet 모델 연동 모듈

정리된 기능:
- 기본 CCNet 모델 로드
- ROI 추출 및 전처리
- 기본 임베딩 추출
- 시각화 기능
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import math
from PIL import Image
import torchvision.transforms as transforms
from ccnet import ccnet  # 기본 CCNet 모델

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe 초기화
def init_mediapipe_for_web():
    """웹 실시간 처리에 최적화된 Mediapipe 설정"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3
    )
    return mp_hands, hands

mp_hands, hands = init_mediapipe_for_web()
mp_drawing = mp.solutions.drawing_utils

# ROI 통계 추적용
class ROIStats:
    def __init__(self):
        self.total_frames = 0
        self.roi_detected_frames = 0
        self.avg_quality = 0
        self.quality_history = []
        
    def update(self, roi_detected, quality=None):
        self.total_frames += 1
        if roi_detected:
            self.roi_detected_frames += 1
            if quality is not None:
                self.quality_history.append(quality)
                if len(self.quality_history) > 100:
                    self.quality_history.pop(0)
                self.avg_quality = sum(self.quality_history) / len(self.quality_history)
    
    def get_detection_rate(self):
        if self.total_frames == 0:
            return 0
        return (self.roi_detected_frames / self.total_frames) * 100
    
    def reset(self):
        self.total_frames = 0
        self.roi_detected_frames = 0
        self.avg_quality = 0
        self.quality_history = []

web_roi_stats = ROIStats()

#########################
# 기본 CCNet 모델 로드
#########################
def load_ccnet_model(model_path="/Users/kimeunsu/Desktop/공부/졸작 논문/CCNet-main-2/access_system/models/checkpointsnet_params_best.pth",
                     num_classes=600,
                     weight=0.8):
    """
    기본 CCNet 모델 로드 함수
    """
    print(f"🚀 기본 CCNet 모델 초기화 중...")
    model = ccnet(num_classes=num_classes, weight=weight)

    try:
        print(f"📁 모델 로드 시도: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 기본 CCNet 모델 로드 완료!")
        
        # 모델 상세 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 모델 상세 정보:")
        print(f"   - 총 파라미터: {total_params:,}개")
        print(f"   - 학습 가능 파라미터: {trainable_params:,}개")
        print(f"   - 아키텍처: 기본 CCNet")
        
    except FileNotFoundError as e:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        raise RuntimeError("모델 로드 실패")
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        raise e
    
    model.eval()
    model.to(torch.device('cpu'))
    print("🚀 기본 CCNet이 evaluation 모드로 설정되었습니다.")
    return model

#########################
# ROI 추출 및 전처리 함수들
#########################

def normalize_for_ccnet(image):
    """CCNet NormSingleROI와 동일한 정규화"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    normalized = gray.astype(np.float32)
    
    mask = normalized > 0
    if mask.sum() > 0:
        mean_val = normalized[mask].mean()
        std_val = normalized[mask].std()
        normalized[mask] = (normalized[mask] - mean_val) / (std_val + 1e-6)
    
    return normalized

def assess_roi_quality(roi, threshold=0.4):
    """ROI 품질을 평가하여 불량 이미지 필터링"""
    if roi is None or roi.size == 0:
        return False, 0.0
    
    # 선명도 (Laplacian variance)
    laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 100, 1.0)
    
    # 대비도
    contrast = roi.std()
    contrast_score = min(contrast / 50, 1.0)
    
    # 콘텐츠 비율
    if len(roi.shape) == 3:
        content_pixels = np.count_nonzero(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    else:
        content_pixels = np.count_nonzero(roi)
    content_ratio = content_pixels / (128 * 128)
    
    # 종합 점수
    overall_score = (sharpness_score * 0.4 + contrast_score * 0.3 + content_ratio * 0.3)
    
    is_good = overall_score > threshold
    return is_good, overall_score

def classify_hand(mp_hands, hand_landmarks, image_width):
    """손 방향 분류 (좌/우)"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_tip_x_px = thumb_tip.x * image_width
    index_base_x_px = index_base.x * image_width
    return 'Right' if thumb_tip_x_px > index_base_x_px else 'Left'

def align_hand_rotation(image, mp_hands, hand_landmarks):
    """손 회전 보정"""
    image_width, image_height = image.shape[1], image.shape[0]
    hand_orientation = classify_hand(mp_hands, hand_landmarks, image_width)

    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    x1, y1 = index_mcp.x, index_mcp.y
    x2, y2 = pinky_mcp.x, pinky_mcp.y
    distance = math.hypot(x2 - x1, y2 - y1)
    
    if distance == 0:
        return image, 0
    
    unit_vector = ((x2 - x1) / distance, (y2 - y1) / distance)
    angle_with_horizontal = math.degrees(math.atan2(unit_vector[1], unit_vector[0]))

    if hand_orientation == "Right":
        if -180 <= angle_with_horizontal <= -90:
            rotation_angle = angle_with_horizontal + 180
        elif 0 <= angle_with_horizontal <= 180:
            rotation_angle = angle_with_horizontal - 180
        else:
            rotation_angle = 180 - angle_with_horizontal
    else:
        if -180 <= angle_with_horizontal <= -90:
            rotation_angle = angle_with_horizontal + 180
        else:
            rotation_angle = angle_with_horizontal

    center = (image_width // 2, image_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image, rotation_angle

def enhance_for_ccnet(image):
    """CCNet에 최적화된 이미지 품질 개선"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    unsharp = cv2.addWeighted(enhanced, 1.2, gaussian, -0.2, 0)
    
    return unsharp

def ccnet_optimized_square_transform(image, target_size=128):
    """CCNet에 최적화된 정사각형 변환"""
    h, w = image.shape[:2]
    ratio = max(h, w) / min(h, w)
    
    if ratio < 1.15:
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    elif ratio < 1.4:
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        
    else:
        if h > w:
            target_h = int(w * 1.25)
            start_y = max(0, int((h - target_h) * 0.4))
            cropped = image[start_y:start_y + target_h, :]
        else:
            target_w = int(h * 1.25)
            start_x = (w - target_w) // 2
            cropped = image[:, start_x:start_x + target_w]
        
        ch, cw = cropped.shape[:2]
        max_dim = max(ch, cw)
        pad_h = max_dim - ch
        pad_w = max_dim - cw
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    
    result = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return result

#########################
# 시각화 함수들
#########################

def get_hand_roi_with_visualization(img_bgr):
    """시각화 정보가 포함된 ROI 추출"""
    try:
        vis_image = img_bgr.copy()
        
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            cv2.putText(vis_image, "No Hand Detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_image

        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            vis_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        
        h, w, _ = img_bgr.shape
        x_list = [lm.x for lm in hand_landmarks.landmark]
        y_list = [lm.y for lm in hand_landmarks.landmark]

        xmin, xmax = int(min(x_list) * w), int(max(x_list) * w)
        ymin, ymax = int(min(y_list) * h), int(max(y_list) * h)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
            cv2.putText(vis_image, "Invalid ROI", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_image
        
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
        
        if len(hand_landmarks.landmark) >= 9:
            middle_tip = hand_landmarks.landmark[8]
            wrist = hand_landmarks.landmark[0]
            
            middle_tip_px = (int(middle_tip.x * w), int(middle_tip.y * h))
            wrist_px = (int(wrist.x * w), int(wrist.y * h))
            
            cv2.arrowedLine(vis_image, wrist_px, middle_tip_px, (255, 0, 255), 3)
        
        cv2.putText(vis_image, "Basic CCNet", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image
        
    except Exception as e:
        cv2.putText(img_bgr, f"Error: {str(e)[:30]}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img_bgr

def extract_palm_roi_like_roi1(img_bgr):
    """roi1.py 방식과 동일한 손바닥 ROI 추출"""
    try:
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None, "No Hand Detected"

        hand_landmarks = results.multi_hand_landmarks[0]
        image_width, image_height = img_bgr.shape[1], img_bgr.shape[0]
        
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

        points_dict = {
            'index_mcp': (int(index_mcp.x * image_width), int(index_mcp.y * image_height)),
            'middle_mcp': (int(middle_mcp.x * image_width), int(middle_mcp.y * image_height)),
            'ring_mcp': (int(ring_mcp.x * image_width), int(ring_mcp.y * image_height)),
            'pinky_mcp': (int(pinky_mcp.x * image_width), int(pinky_mcp.y * image_height)),
            'wrist': (int(wrist.x * image_width), int(wrist.y * image_height)),
            'thumb_mcp': (int(thumb_mcp.x * image_width), int(thumb_mcp.y * image_height))
        }

        wrist_pinky_horiz = (points_dict['wrist'][0], points_dict['pinky_mcp'][1])

        points = np.array([
            points_dict['index_mcp'],
            points_dict['middle_mcp'],
            points_dict['ring_mcp'],
            points_dict['pinky_mcp'],
            wrist_pinky_horiz,
            points_dict['wrist'],
            points_dict['thumb_mcp']
        ], dtype=np.int32)

        rotated_image, rotation_angle = align_hand_rotation(img_bgr, mp_hands, hand_landmarks)
        
        center = (image_width // 2, image_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones]).astype(np.float32)
        
        new_points = cv2.transform(np.array([points_homogeneous]), rotation_matrix).squeeze().astype(np.int32)

        x, y, w, h = cv2.boundingRect(new_points)
        
        if w == 0 or h == 0:
            return None, "Invalid bounding box"

        roi = rotated_image[y:y+h, x:x+w]
        
        is_good, quality_score = assess_roi_quality(roi, threshold=0.4)
        if not is_good:
            return None, f"Low quality (score: {quality_score:.3f})"
        
        enhanced_roi = enhance_for_ccnet(roi)
        final_roi = ccnet_optimized_square_transform(enhanced_roi, target_size=128)
        
        if len(final_roi.shape) == 2:
            final_roi_bgr = cv2.cvtColor(final_roi, cv2.COLOR_GRAY2BGR)
        else:
            final_roi_bgr = final_roi
            
        return final_roi_bgr, f"Success (quality: {quality_score:.3f})"
        
    except Exception as e:
        print(f"ROI 추출 중 오류: {e}")
        return None, f"Error: {str(e)}"

def get_hand_roi(img_bgr):
    """기존 호환성을 위한 래퍼 함수"""
    roi, message = extract_palm_roi_like_roi1(img_bgr)
    return roi

#########################
# 전처리 설정
#########################
transform_ops = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

#########################
# 기본 CCNet 임베딩 추출 함수
#########################
def extract_embedding(img_bgr, model):
    """
    기본 CCNet 임베딩 추출
    1) CCNet 최적화된 손바닥 ROI 추출
    2) CCNet 정규화 적용
    3) 기본 CCNet -> 임베딩
    """
    roi = get_hand_roi(img_bgr)
    if roi is None:
        print("❌ 기본 CCNet용 손바닥 ROI 추출 실패!")
        return None

    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi

    normalized_roi = normalize_for_ccnet(roi_gray)
    display_roi = ((normalized_roi + 3) * 40).clip(0, 255).astype(np.uint8)
    roi_for_transform = display_roi
    
    roi_tensor = transform_ops(roi_for_transform).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.getFeatureCode(roi_tensor)
    
    emb_np = emb.squeeze().cpu().numpy()
    
    print("✅ 기본 CCNet으로 임베딩 추출 성공!")
    
    return emb_np

#########################
# 테스트 함수
#########################
if __name__ == "__main__":
    print("🚀 기본 CCNet 테스트 시작...")
    print("="*60)
    
    # 모델 로드
    model = load_ccnet_model()
    
    # 기본 기능 테스트
    print("\n📱 기본 기능 테스트:")
    dummy_input = torch.randn(1, 1, 128, 128)
    
    with torch.no_grad():
        feature_code = model.getFeatureCode(dummy_input)
        print(f"  특징 코드 shape: {feature_code.shape}")
        print(f"  정규화 확인: {torch.norm(feature_code[0]).item():.6f}")
    
    print(f"\n✅ 기본 CCNet이 정상적으로 로드되었습니다.")
    print(f"🎯 실시간 시각화 기능이 포함된 웹 시스템이 준비되었습니다.")
    print("="*60)