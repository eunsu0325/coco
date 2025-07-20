# app.py - 비디오 스트림 최적화 버전 (Lazy Loading)

from flask import Flask, render_template, Response, request, jsonify
import numpy as np
from collections import defaultdict
import cv2
import threading
import time
import os

from model_ops import extract_embedding, load_ccnet_model, get_hand_roi_with_visualization
from db_ops import register_user, multi_shot_authenticate

app = Flask(__name__)

# ────────────────────────────────────────────────────────────────────
# 전역 변수 정의
# ────────────────────────────────────────────────────────────────────
cap = None
net = None
temp_storage = defaultdict(list)  # 등록 과정에서 임시 임베딩 저장

# 스트리밍용 변수
latest_frame = None
latest_roi = None
frame_lock = threading.Lock()

# 🔥 새로 추가: 비디오 스트림 상태 관리
video_enabled = False
capture_thread_running = False

# 모델 상태 추적
model_info = {
    'type': 'Unknown',
    'path': 'Unknown',
    'loaded': False
}

def store_temp_embedding(username, embedding):
    temp_storage[username].append(embedding)

def retrieve_all_temp_embeddings(username):
    return temp_storage[username]

def clear_temp_embeddings(username):
    temp_storage[username].clear()

# ────────────────────────────────────────────────────────────────────
# 🔥 수정: Lazy 웹캠 초기화 함수
# ────────────────────────────────────────────────────────────────────
def initialize_camera():
    """웹캠을 lazy하게 초기화하는 함수"""
    global cap, video_enabled, capture_thread_running
    
    if video_enabled:
        return True
    
    try:
        print("📹 웹캠 lazy 초기화 시작...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            return False
        
        # 웹캠 설정 최적화
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        video_enabled = True
        
        # 백그라운드 스레드 시작 (한 번만)
        if not capture_thread_running:
            capture_thread = threading.Thread(target=frame_capture_thread, daemon=True)
            capture_thread.start()
            capture_thread_running = True
            
        print("✅ 웹캠 lazy 초기화 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 웹캠 초기화 오류: {e}")
        return False

# ────────────────────────────────────────────────────────────────────
# 🔥 수정: 프레임 캡처 스레드 (조건부 실행)
# ────────────────────────────────────────────────────────────────────
def frame_capture_thread():
    """백그라운드에서 지속적으로 프레임을 캡처하고 처리"""
    global cap, latest_frame, latest_roi, video_enabled
    
    frame_skip_counter = 0
    
    while True:
        try:
            # 비디오가 비활성화되어 있거나 cap이 없으면 대기
            if not video_enabled or cap is None:
                time.sleep(0.1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # 🔥 수정: 매 3번째 프레임만 처리 (성능 향상)
            frame_skip_counter += 1
            if frame_skip_counter % 3 != 0:
                continue
            
            # 🔥 수정: 프레임 크기 더 작게 최적화
            height, width = frame.shape[:2]
            new_width = int(width * 0.6)  # 0.8 → 0.6으로 변경
            new_height = int(height * 0.6)
            frame = cv2.resize(frame, (new_width, new_height))
            
            with frame_lock:
                # 메인 프레임 처리 (시각화 포함)
                try:
                    visualized_frame = get_hand_roi_with_visualization(frame)
                    latest_frame = visualized_frame if visualized_frame is not None else frame
                    
                    # 기본 CCNet 정보 표시
                    if model_info['loaded']:
                        cv2.putText(latest_frame, "Basic CCNet Active", (10, latest_frame.shape[0] - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except:
                    latest_frame = frame
                
                # ROI 프레임 처리
                try:
                    from model_ops import get_hand_roi
                    roi = get_hand_roi(frame)
                    if roi is not None:
                        latest_roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
                        # ROI에도 기본 CCNet 정보 표시
                        if model_info['loaded']:
                            cv2.putText(latest_roi, "CCNet", (5, 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    else:
                        error_frame = np.zeros((128, 128, 3), dtype=np.uint8)
                        cv2.putText(error_frame, "No Hand", (30, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        latest_roi = error_frame
                except:
                    error_frame = np.zeros((128, 128, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "ROI Error", (20, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    latest_roi = error_frame
            
            time.sleep(0.05)  # 🔥 수정: 0.033 → 0.05 (약 20 FPS)
            
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            time.sleep(0.1)

# ────────────────────────────────────────────────────────────────────
# 🔥 수정: 스트림 생성 함수들 (Lazy Loading)
# ────────────────────────────────────────────────────────────────────
def gen_frames():
    """메인 웹캠 스트림 생성 - Lazy Loading 버전"""
    global latest_frame, video_enabled
    
    # 웹캠 초기화 시도
    if not video_enabled:
        if not initialize_camera():
            # 초기화 실패 시 더미 프레임 생성
            while True:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera initialization failed", (100, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                time.sleep(0.1)
    
    # 정상 스트림 생성
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Loading Basic CCNet...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # JPEG 압축
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

def gen_roi_frames():
    """ROI 스트림 생성 - Lazy Loading 버전"""
    global latest_roi, video_enabled
    
    # 웹캠이 초기화되지 않았으면 더미 프레임
    while True:
        with frame_lock:
            if latest_roi is not None and video_enabled:
                roi_frame = latest_roi.copy()
            else:
                roi_frame = np.zeros((128, 128, 3), dtype=np.uint8)
                if video_enabled:
                    cv2.putText(roi_frame, "Loading...", (30, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(roi_frame, "Camera Off", (20, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # JPEG 압축
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        ret, buffer = cv2.imencode('.jpg', roi_frame, encode_param)
        
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

# ────────────────────────────────────────────────────────────────────
# Flask 라우트 (변경 없음)
# ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """🔥 수정: Lazy Loading 웹캠 스트림"""
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/roi_feed')
def roi_feed():
    """🔥 수정: Lazy Loading ROI 스트림"""
    return Response(
        gen_roi_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/model_info')
def get_model_info():
    """모델 정보 API"""
    return jsonify(model_info)

# ────────────────────────────────────────────────────────────────────
# 사용자 등록 - 10장 강제 (기본 CCNet)
# ────────────────────────────────────────────────────────────────────

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/auto_capture', methods=['POST'])
def auto_capture():
    """자동 캡처 - 실패시 재시도, 성공시 다음으로 (기본 CCNet 사용)"""
    global latest_frame, net

    data = request.get_json()
    username = data.get('username', 'NoName')
    shot_index = data.get('shotIndex', 0)

    # 🔥 수정: 웹캠 초기화 확인
    if not video_enabled:
        initialize_camera()
        time.sleep(0.5)  # 초기화 대기

    # 최신 프레임 사용
    with frame_lock:
        if latest_frame is not None:
            frame = latest_frame.copy()
        else:
            return jsonify({
                "error": True, 
                "retry": True,
                "shot_index": shot_index,
                "message": "카메라 연결을 확인해주세요."
            })

    # 기본 CCNet으로 임베딩 추출
    try:
        embedding = extract_embedding(frame, net)
        
        if embedding is None:
            # ❌ 실패시 - 같은 샷 번호로 재시도
            return jsonify({
                "error": True, 
                "retry": True,
                "shot_index": shot_index,
                "message": f"🖐️ {shot_index}번째 촬영 실패!\n\n손바닥을 카메라 정면에 펼쳐서 보여주세요!\n• 손바닥 전체가 화면에 나오도록\n• 충분한 조명 확보\n• 손가락을 펼친 상태로\n\n기본 CCNet으로 재시도 중..."
            })
        else:
            # ✅ 성공시 - 임베딩 저장하고 다음 샷으로
            store_temp_embedding(username, embedding)
            
            return jsonify({
                "error": False,
                "retry": False,
                "shot_index": shot_index,
                "message": f"✅ {shot_index}번째 촬영 완료! (기본 CCNet)"
            })
            
    except Exception as e:
        print(f"⚠️ 임베딩 추출 실패: {e}")
        return jsonify({
            "error": True, 
            "retry": True,
            "shot_index": shot_index,
            "message": f"🖐️ {shot_index}번째 촬영 실패!\n\n손바닥을 카메라 정면에 펼쳐서 보여주세요!\n• 손바닥 전체가 화면에 나오도록\n• 충분한 조명 확보\n• 손가락을 펼친 상태로\n\n재시도 중..."
        })

@app.route('/finish_auto_capture', methods=['POST'])
def finish_auto_capture():
    """등록 완료 - 정확히 10장 체크 (기본 CCNet)"""
    data = request.get_json()
    username = data.get('username', 'NoName')

    all_embeds = retrieve_all_temp_embeddings(username)
    
    # ✅ 정확히 10장인지 체크
    if len(all_embeds) != 10:
        clear_temp_embeddings(username)  # 실패시 임시 저장소 정리
        return jsonify({
            "error": True,
            "message": f"등록 실패: 정확히 10장이 필요합니다. (현재: {len(all_embeds)}장)\n다시 등록을 진행해주세요."
        }), 400

    try:
        # 가중 평균 방식 사용 (db_ops.py의 개선된 함수)
        register_user(username, all_embeds)
        clear_temp_embeddings(username)
        
        model_type = model_info.get('type', 'Unknown')
        
        return jsonify({
            "success": True,
            "message": f"🎉 {username} 등록 완료! ({model_type})\n10장의 생체 데이터가 성공적으로 등록되었습니다.\n기본 CCNet으로 안정적인 임베딩 생성!"
        })
    except Exception as e:
        clear_temp_embeddings(username)
        return jsonify({
            "error": True,
            "message": f"등록 중 오류가 발생했습니다: {str(e)}"
        }), 500

# ────────────────────────────────────────────────────────────────────
# 사용자 인증 (기본 CCNet)
# ────────────────────────────────────────────────────────────────────

@app.route('/authenticate', methods=['GET'])
def authenticate_page():
    return render_template('authenticate.html')

@app.route('/auto_auth', methods=['POST'])
def auto_auth():
    """자동 인증 (기본 CCNet 사용)"""
    global latest_frame, net

    # 🔥 수정: 웹캠 초기화 확인
    if not video_enabled:
        initialize_camera()
        time.sleep(0.5)  # 초기화 대기

    # 최신 프레임 사용
    with frame_lock:
        if latest_frame is not None:
            frame = latest_frame.copy()
        else:
            return jsonify({"success": False, "message": "프레임을 읽을 수 없습니다."})

    # 기본 CCNet으로 임베딩 추출
    try:
        embedding = extract_embedding(frame, net)
        
        if embedding is None:
            return jsonify({
                "success": False, 
                "message": "🖐️ 손바닥을 제대로 보여주세요!\n• 손바닥 전체가 화면에 나오도록\n• 충분한 조명 확보\n• 손가락을 펼친 상태로\n\n기본 CCNet으로 인증 시도 중..."
            })

        result = multi_shot_authenticate(embedding)
        model_type = model_info.get('type', 'Unknown')
        
        if result:
            return jsonify({
                "success": True,
                "message": f"✅ 인증 성공! ({model_type})\n기본 CCNet으로 안정적인 인증 완료!",
                "user": result
            })
        else:
            return jsonify({
                "success": False,
                "message": f"❌ 인증 실패: 등록된 사용자가 아니거나 손바닥을 다시 보여주세요.\n({model_type} 사용)"
            })
            
    except Exception as e:
        print(f"⚠️ 인증 중 오류: {e}")
        return jsonify({
            "success": False,
            "message": "🖐️ 손바닥을 제대로 보여주세요!\n• 손바닥 전체가 화면에 나오도록\n• 충분한 조명 확보\n• 손가락을 펼친 상태로"
        })

# ────────────────────────────────────────────────────────────────────
# 🔥 수정: 앱 실행 (웹캠 초기화 제거)
# ────────────────────────────────────────────────────────────────────

def load_basic_ccnet_model():
    """기본 CCNet 모델 로드"""
    global net, model_info
    
    # 기본 모델 경로들
    model_paths = [
        "/Users/kimeunsu/Desktop/공부/졸작 논문/CCNet-main-2/access_system1/models/checkpointsnet_params_best.pth",
        "/Users/kimeunsu/Desktop/공부/졸작 논문/CCNet-main-2/access_system1/models/checkpointsnet_params.pth"
    ]
    
    print("🚀 기본 CCNet 모델 로드 시도...")
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                print(f"📁 모델 파일 발견: {file_size:.1f}MB")
                
                # 기본 CCNet 모델 로드
                net = load_ccnet_model(model_path=model_path, num_classes=600, weight=0.8)
                
                model_info = {
                    'type': 'Basic CCNet',
                    'path': model_path,
                    'loaded': True,
                    'file_size_mb': round(file_size, 1),
                    'features': 'Standard'
                }
                print("🎉 기본 CCNet 모델 로드 성공!")
                return True
                
            except Exception as e:
                print(f"❌ 모델 로드 실패: {e}")
                continue
    
    # 모든 경로 실패시
    model_info = {
        'type': 'No Model Loaded',
        'path': 'None',
        'loaded': False,
        'error': '모든 모델 로드 시도 실패'
    }
    raise RuntimeError("사용 가능한 모델 파일이 없습니다.")

if __name__ == '__main__':
    try:
        # 기본 CCNet 모델 로드
        load_basic_ccnet_model()
        
        # 🔥 수정: 웹캠 초기화를 제거하고 Lazy Loading으로 변경
        print("\n" + "="*60)
        print("🎥 Lazy Loading 스트리밍 시스템 활성화!")
        print("📊 웹캠은 첫 비디오 요청 시 초기화됩니다")
        print("🔟 10장 강제 촬영 시스템 활성화!")
        print(f"🔥 {model_info['type']} 사용 중!")
        print(f"📦 모델 크기: {model_info.get('file_size_mb', 'Unknown')}MB")
        print(f"⚡ 특징: {model_info.get('features', 'Unknown')}")
        print("🌐 웹 서버: http://localhost:5000")
        print("🥥 CoCoNut 통합 준비 완료!")
        print("🚀 페이지 로딩 속도 최적화!")
        print("="*60)

        # Flask 서버 실행
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        print(f"\n❌ 기본 CCNet 시스템 초기화 실패: {e}")
        print("🔍 다음 사항을 확인해주세요:")
        print("1. 기본 CCNet 모델 파일 경로가 올바른지 확인")
        print("2. 필요한 라이브러리가 설치되어 있는지 확인")
        print("3. ccnet.py 파일이 기본 버전인지 확인")
        
        import sys
        sys.exit(1)