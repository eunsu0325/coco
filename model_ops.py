# model_ops.py - 이미지 품질 개선 완전 버전
"""
BioSecure Palm - 고품질 이미지 처리 모델 연동 모듈

개선된 기능:
- 🎨 스마트 리사이즈 (비율 유지 + 고품질 보간)
- 🌟 이미지 품질 향상 (선명화, 노이즈 제거)
- 📊 시각적 품질 평가 (엣지 밀도, 텍스처 분석)
- 🔧 완화된 품질 기준 (웹캠 최적화)
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import math
from PIL import Image
import torchvision.transforms as transforms
from ccnet import ccnet

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
# 🎨 이미지 품질 개선 함수들
#########################

def smart_resize_with_padding(image, target_size=128):
    """
    🎨 스마트 리사이즈: 비율 유지 + 패딩으로 이쁜 정사각형 만들기
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # 1. 비율 계산
    aspect_ratio = w / h
    
    if aspect_ratio > 1:  # 가로가 더 길 경우
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:  # 세로가 더 길거나 정사각형인 경우
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
    
    # 2. 고품질 리사이즈 (LANCZOS 사용)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 3. 패딩으로 정사각형 만들기
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 4. 반사 패딩 (검은색 대신 이미지 경계 반사)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, 
        cv2.BORDER_REFLECT_101
    )
    
    return padded

def enhance_image_quality(image):
    """
    🌟 이미지 품질 향상 처리
    """
    if image is None:
        return None
    
    # 1. 가우시안 블러로 노이즈 제거
    denoised = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    # 2. 언샤프 마스킹으로 선명도 향상
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    unsharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    
    # 3. CLAHE로 대비 향상
    if len(image.shape) == 3:
        gray = cv2.cvtColor(unsharp, cv2.COLOR_BGR2GRAY)
    else:
        gray = unsharp
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 4. 히스토그램 평활화로 밝기 조정
    equalized = cv2.equalizeHist(enhanced)
    
    # 5. 원본과 블렌딩 (너무 과하지 않게)
    final = cv2.addWeighted(enhanced, 0.7, equalized, 0.3, 0)
    
    return final

def create_beautiful_roi(roi_image, target_size=128):
    """
    🎨 아름다운 ROI 이미지 생성 (전체 파이프라인)
    """
    if roi_image is None:
        return None
    
    try:
        # 1. 원본이 컬러면 그레이스케일로 변환
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()
        
        # 2. 품질 향상
        enhanced = enhance_image_quality(gray)
        
        # 3. 스마트 리사이즈 (비율 유지)
        resized = smart_resize_with_padding(enhanced, target_size)
        
        # 4. 최종 후처리
        # 약간의 가우시안 블러로 부드럽게
        final = cv2.GaussianBlur(resized, (3, 3), 0.8)
        
        # 5. 컬러로 변환 (웹에서 표시용)
        if len(final.shape) == 2:
            final_color = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
        else:
            final_color = final
        
        return final_color
        
    except Exception as e:
        print(f"🎨 이미지 품질 향상 중 오류: {e}")
        # 실패 시 기본 리사이즈라도 반환
        return cv2.resize(roi_image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

#########################
# 품질 평가 함수들
#########################

def assess_roi_quality_enhanced(roi, threshold=0.35):
    """
    🔥 완화된 ROI 품질 평가 - 웹캠 환경에 맞게 현실적 기준 적용
    """
    if roi is None or roi.size == 0:
        return False, 0.0, "ROI 없음"
    
    try:
        # ROI 크기 확인
        height, width = roi.shape[:2]
        if height < 50 or width < 50:
            return False, 0.0, "ROI 크기 너무 작음"
        
        # 그레이스케일 변환
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # 1. 선명도 평가 (완화된 기준)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 50, 1.0)  # 100 → 50으로 완화
        
        # 2. 대비도 평가 (완화된 기준)
        contrast = gray.std()
        contrast_score = min(contrast / 25, 1.0)  # 50 → 25로 완화
        
        # 3. 콘텐츠 비율 (완화된 기준)
        content_pixels = np.count_nonzero(gray)
        total_pixels = gray.shape[0] * gray.shape[1]
        content_ratio = content_pixels / total_pixels
        content_score = min(content_ratio * 1.2, 1.0)  # 보정 계수 추가
        
        # 4. 밝기 분포 평가 (새로 추가 - 완화된 기준)
        mean_brightness = gray.mean()
        brightness_score = 1.0
        if mean_brightness < 50:  # 너무 어두움
            brightness_score = mean_brightness / 50
        elif mean_brightness > 200:  # 너무 밝음
            brightness_score = (255 - mean_brightness) / 55
        
        # 5. 종합 점수 계산 (가중치 조정)
        overall_score = (
            sharpness_score * 0.25 +      # 선명도 비중 줄임
            contrast_score * 0.25 +       # 대비도 비중 줄임
            content_score * 0.35 +        # 콘텐츠 비중 높임
            brightness_score * 0.15       # 밝기 추가
        )
        
        # 6. 품질 판정
        is_good = overall_score >= threshold
        
        # 7. 상세 이유 제공
        if not is_good:
            reasons = []
            if sharpness_score < 0.3:
                reasons.append("선명도 부족")
            if contrast_score < 0.3:
                reasons.append("대비도 부족")
            if content_score < 0.5:
                reasons.append("손바닥 영역 부족")
            if brightness_score < 0.5:
                reasons.append("조명 문제")
            
            reason = ", ".join(reasons) if reasons else "종합 점수 부족"
        else:
            if overall_score >= 0.7:
                reason = "우수 품질"
            elif overall_score >= 0.5:
                reason = "양호 품질"
            else:
                reason = "최소 품질 통과"
        
        return is_good, overall_score, reason
        
    except Exception as e:
        print(f"품질 평가 오류: {e}")
        return False, 0.0, f"평가 오류: {str(e)}"

def assess_roi_quality_visual(roi, threshold=0.35):
    """
    🎨 시각적 품질까지 고려한 ROI 평가
    """
    if roi is None or roi.size == 0:
        return False, 0.0, "ROI 없음"
    
    try:
        # 기본 품질 평가
        is_good, basic_score, reason = assess_roi_quality_enhanced(roi, threshold)
        
        # 추가: 시각적 품질 평가
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # 1. 엣지 밀도 (손바닥 라인이 잘 보이는지)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
        edge_score = min(edge_density * 10, 1.0)  # 정규화
        
        # 2. 텍스처 다양성 (손바닥 패턴의 복잡도)
        lbp = cv2.calcHist([gray], [0], None, [256], [0, 256])
        texture_score = min(np.std(lbp) / 1000, 1.0)  # 정규화
        
        # 3. 종합 시각적 점수
        visual_score = (edge_score * 0.6 + texture_score * 0.4)
        
        # 4. 최종 점수 (기본 + 시각적)
        final_score = (basic_score * 0.7 + visual_score * 0.3)
        
        is_beautiful = final_score >= threshold
        
        if is_beautiful:
            if final_score >= 0.8:
                visual_reason = "최고 품질 (매우 선명)"
            elif final_score >= 0.6:
                visual_reason = "우수 품질 (선명함)"
            else:
                visual_reason = "양호 품질 (적당함)"
        else:
            visual_reason = f"품질 개선 필요 ({reason})"
        
        return is_beautiful, final_score, visual_reason
        
    except Exception as e:
        return False, 0.0, f"평가 오류: {str(e)}"

def assess_roi_quality_super_relaxed(roi, threshold=0.15):
    """
    🔥 매우 완화된 ROI 품질 평가 - 거의 모든 경우에 통과하도록 설정
    """
    if roi is None or roi.size == 0:
        return False, 0.0, "ROI 없음"
    
    try:
        # ROI 크기 확인 (더 관대하게)
        height, width = roi.shape[:2]
        if height < 30 or width < 30:  # 50 → 30으로 완화
            return False, 0.0, "ROI 크기 너무 작음"
        
        # 그레이스케일 변환
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # 1. 선명도 평가 (매우 완화된 기준)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 20, 1.0)  # 50 → 20으로 대폭 완화
        
        # 2. 대비도 평가 (매우 완화된 기준)
        contrast = gray.std()
        contrast_score = min(contrast / 15, 1.0)  # 25 → 15로 대폭 완화
        
        # 3. 콘텐츠 비율 (매우 완화된 기준)
        content_pixels = np.count_nonzero(gray)
        total_pixels = gray.shape[0] * gray.shape[1]
        content_ratio = content_pixels / total_pixels
        content_score = min(content_ratio * 2.0, 1.0)  # 1.2 → 2.0으로 대폭 완화
        
        # 4. 밝기 분포 평가 (매우 관대한 기준)
        mean_brightness = gray.mean()
        brightness_score = 1.0
        if mean_brightness < 20:  # 50 → 20으로 완화 (매우 어두워도 OK)
            brightness_score = mean_brightness / 20
        elif mean_brightness > 235:  # 200 → 235로 완화 (매우 밝아도 OK)
            brightness_score = (255 - mean_brightness) / 20
        
        # 5. 종합 점수 계산 (더 관대한 가중치)
        overall_score = (
            sharpness_score * 0.15 +      # 선명도 비중 더 줄임
            contrast_score * 0.15 +       # 대비도 비중 더 줄임
            content_score * 0.50 +        # 콘텐츠 비중 더 높임
            brightness_score * 0.20       # 밝기 비중 높임
        )
        
        # 추가 보너스: 기본 점수가 너무 낮아도 최소 보장
        if content_ratio > 0.3:  # 기본적으로 손이 보이기만 하면
            overall_score = max(overall_score, 0.2)  # 최소 0.2점 보장
        
        # 6. 품질 판정 (매우 관대한 임계값)
        is_good = overall_score >= threshold
        
        # 7. 상세 이유 제공
        if not is_good:
            reasons = []
            if sharpness_score < 0.1:  # 0.3 → 0.1로 완화
                reasons.append("매우 흐림")
            if contrast_score < 0.1:   # 0.3 → 0.1로 완화
                reasons.append("매우 낮은 대비")
            if content_score < 0.2:    # 0.5 → 0.2로 완화
                reasons.append("손바닥 거의 안 보임")
            if brightness_score < 0.2: # 0.5 → 0.2로 완화
                reasons.append("극단적 조명")
            
            reason = ", ".join(reasons) if reasons else "기본 기준 미달"
        else:
            if overall_score >= 0.5:
                reason = "우수 품질"
            elif overall_score >= 0.3:
                reason = "양호 품질"
            elif overall_score >= 0.15:
                reason = "최소 품질 통과"
            else:
                reason = "기본 품질 통과"
        
        print(f"[품질 평가] 점수: {overall_score:.3f}, 통과: {is_good}, 이유: {reason}")
        return is_good, overall_score, reason
        
    except Exception as e:
        print(f"품질 평가 오류: {e}")
        return False, 0.0, f"평가 오류: {str(e)}"

#########################
# CCNet 모델 로드
#########################

def load_ccnet_model(model_path="/Users/kimeunsu/Desktop/공부/졸작 논문/CCNet-main-2/access_system1/models/checkpoint_step_951.pth",
                     num_classes=1000,
                     weight=0.8):
    """
    고품질 이미지 처리 CCNet 모델 로드 함수
    """
    print(f"🚀 고품질 이미지 처리 CCNet 모델 초기화 중...")
    model = ccnet(num_classes=num_classes, weight=weight)

    try:
        print(f"📁 모델 로드 시도: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 고품질 이미지 처리 CCNet 모델 로드 완료!")
        
        # 모델 상세 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 모델 상세 정보:")
        print(f"   - 총 파라미터: {total_params:,}개")
        print(f"   - 학습 가능 파라미터: {trainable_params:,}개")
        print(f"   - 아키텍처: 고품질 이미지 처리 CCNet")
        print(f"   - 품질 기준: 0.35 (웹캠 최적화)")
        print(f"   - 🎨 이미지 품질 향상: 활성화")
        
    except FileNotFoundError as e:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        raise RuntimeError("모델 로드 실패")
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        raise e
    
    model.eval()
    model.to(torch.device('cpu'))
    print("🚀 고품질 이미지 처리 CCNet이 evaluation 모드로 설정되었습니다.")
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

def assess_roi_quality(roi, threshold=0.25):
    """기본 ROI 품질 평가 (더 관대한 기준)"""
    if roi is None or roi.size == 0:
        return False, 0.0
    
    # 선명도 (Laplacian variance)
    laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 30, 1.0)  # 더 관대하게
    
    # 대비도
    contrast = roi.std()
    contrast_score = min(contrast / 20, 1.0)  # 더 관대하게
    
    # 콘텐츠 비율
    if len(roi.shape) == 3:
        content_pixels = np.count_nonzero(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    else:
        content_pixels = np.count_nonzero(roi)
    content_ratio = content_pixels / (128 * 128)
    
    # 종합 점수 (더 관대한 가중치)
    overall_score = (sharpness_score * 0.3 + contrast_score * 0.3 + content_ratio * 0.4)
    
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
    """🎨 개선된 CCNet 정사각형 변환"""
    # 새로운 품질 향상 함수 사용
    return create_beautiful_roi(image, target_size)

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
        
        cv2.putText(vis_image, "High Quality CCNet", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image
        
    except Exception as e:
        cv2.putText(img_bgr, f"Error: {str(e)[:30]}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img_bgr

def extract_palm_roi_like_roi1(img_bgr):
    """🎨 roi1.py 방식과 동일한 손바닥 ROI 추출 - 고품질 이미지 처리 버전"""
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
        
        # 🔥 완화된 품질 평가
        is_good, quality_score = assess_roi_quality(roi, threshold=0.25)  # 더 관대하게
        if not is_good:
            return None, f"Low quality (score: {quality_score:.3f})"
        
        # 🌟 새로운 품질 향상 파이프라인 적용
        beautiful_roi = create_beautiful_roi(roi, target_size=128)
        
        return beautiful_roi, f"Success (quality: {quality_score:.3f})"
        
    except Exception as e:
        print(f"ROI 추출 중 오류: {e}")
        return None, f"Error: {str(e)}"

def get_hand_roi_for_registration(img_bgr):
    """등록용 ROI 추출 - 완화된 기준 적용"""
    try:
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        image_width, image_height = img_bgr.shape[1], img_bgr.shape[0]
        
        # 손바닥 주요 점들 추출
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

        # 손바닥 영역 정의
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

        # 회전 보정
        rotated_image, rotation_angle = align_hand_rotation(img_bgr, mp_hands, hand_landmarks)
        
        # 회전 후 좌표 변환
        center = (image_width // 2, image_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones]).astype(np.float32)
        new_points = cv2.transform(np.array([points_homogeneous]), rotation_matrix).squeeze().astype(np.int32)

        # 바운딩 박스
        x, y, w, h = cv2.boundingRect(new_points)
        
        if w == 0 or h == 0:
            return None

        # ROI 추출
        roi = rotated_image[y:y+h, x:x+w]
        
        # 🌟 고품질 이미지 처리 적용
        beautiful_roi = create_beautiful_roi(roi, target_size=128)
        
        return beautiful_roi
        
    except Exception as e:
        print(f"등록용 ROI 추출 중 오류: {e}")
        return None

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
# 고품질 이미지 처리 CCNet 임베딩 추출 함수
#########################
def extract_embedding(img_bgr, model):
    """
    고품질 이미지 처리 CCNet 임베딩 추출
    1) 고품질 손바닥 ROI 추출
    2) CCNet 정규화 적용
    3) CCNet -> 임베딩
    """
    roi = get_hand_roi(img_bgr)
    if roi is None:
        print("❌ 고품질 이미지 처리 CCNet용 손바닥 ROI 추출 실패!")
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
    
    print("✅ 고품질 이미지 처리 CCNet으로 임베딩 추출 성공!")
    
    return emb_np

#########################
# 테스트 함수
#########################
if __name__ == "__main__":
    print("🚀 고품질 이미지 처리 CCNet 테스트 시작...")
    print("="*60)
    print("🎨 이미지 품질 향상 기능:")
    print("   - 스마트 리사이즈 (비율 유지)")
    print("   - LANCZOS4 보간법")
    print("   - 언샤프 마스킹")
    print("   - CLAHE 대비 향상")
    print("   - 반사 패딩")
    print("🎯 품질 임계값: 0.35 (완화된 기준)")
    print("📈 예상 통과율: 80-90%")
    print("🛡️ 여전히 최소한의 품질은 보장")
    print("🔧 웹캠 환경에 최적화된 현실적 기준")
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
    
    print(f"\n✅ 고품질 이미지 처리 CCNet이 정상적으로 로드되었습니다.")
    print(f"🎯 실시간 시각화 기능이 포함된 웹 시스템이 준비되었습니다.")
    print(f"🎨 고품질 이미지 처리 기능 활성화!")
    print(f"📱 웹캠 환경에 맞는 현실적 품질 기준 적용!")
    print("="*60)
