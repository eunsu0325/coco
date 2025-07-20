# app.py - 인터랙티브 등록 시스템 (사용자 확인 후 저장) + 관리자 페이지 추가

from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import base64
from collections import defaultdict
import cv2
import threading
import time
import os
import uuid
import mariadb

from model_ops import extract_embedding, load_ccnet_model, get_hand_roi_with_visualization
from db_ops import register_user, multi_shot_authenticate, connect_to_db

app = Flask(__name__)

# ────────────────────────────────────────────────────────────────────
# 전역 변수 정의
# ────────────────────────────────────────────────────────────────────
cap = None
net = None

# 🔥 NEW: 인터랙티브 등록용 스토리지
user_sessions = defaultdict(lambda: {
    'images': [],        # 촬영된 이미지들 (임시)
    'embeddings': [],    # 대응하는 임베딩들 (임시)
    'confirmed': [],     # 사용자가 확인한 이미지들
    'confirmed_embeddings': [],  # 확인된 임베딩들
    'session_id': str(uuid.uuid4())
})

# 스트리밍용 변수
latest_frame = None
latest_roi = None
original_frame = None
frame_lock = threading.Lock()

# 비디오 스트림 상태 관리
video_enabled = False
capture_thread_running = False

# 모델 상태 추적
model_info = {
    'type': 'Unknown',
    'path': 'Unknown',
    'loaded': False
}

def clear_user_session(username):
    """사용자 세션 정리"""
    if username in user_sessions:
        del user_sessions[username]

# ────────────────────────────────────────────────────────────────────
# 웹캠 초기화 및 스트림 (기존과 동일)
# ────────────────────────────────────────────────────────────────────

def initialize_camera():
    """웹캠을 lazy하게 초기화하는 함수"""
    global cap, video_enabled, capture_thread_running
    
    if video_enabled:
        return True
    
    try:
        print("📹 고해상도 웹캠 lazy 초기화 시작...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        video_enabled = True
        
        if not capture_thread_running:
            capture_thread = threading.Thread(target=frame_capture_thread, daemon=True)
            capture_thread.start()
            capture_thread_running = True
            
        print("✅ 고해상도 웹캠 lazy 초기화 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 웹캠 초기화 오류: {e}")
        return False

def frame_capture_thread():
    """백그라운드에서 지속적으로 프레임을 캡처하고 처리"""
    global cap, latest_frame, latest_roi, original_frame, video_enabled
    
    frame_skip_counter = 0
    
    while True:
        try:
            if not video_enabled or cap is None:
                time.sleep(0.1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_skip_counter += 1
            if frame_skip_counter % 2 != 0:
                continue
            
            height, width = frame.shape[:2]
            new_width = int(width * 0.9)
            new_height = int(height * 0.9)
            frame = cv2.resize(frame, (new_width, new_height))
            
            with frame_lock:
                original_frame = frame.copy()
                
                try:
                    visualized_frame = get_hand_roi_with_visualization(frame)
                    latest_frame = visualized_frame if visualized_frame is not None else frame
                    
                    if model_info['loaded']:
                        cv2.putText(latest_frame, "Interactive Registration Active", (10, latest_frame.shape[0] - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except:
                    latest_frame = frame
                
                try:
                    from model_ops import get_hand_roi
                    roi = get_hand_roi(original_frame)
                    if roi is not None:
                        latest_roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
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
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            time.sleep(0.1)

def gen_frames():
    """메인 웹캠 스트림 생성"""
    global latest_frame, video_enabled
    
    if not video_enabled:
        if not initialize_camera():
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
    
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Loading Interactive System...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
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
    """ROI 스트림 생성"""
    global latest_roi, video_enabled
    
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
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', roi_frame, encode_param)
        
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

# ────────────────────────────────────────────────────────────────────
# Flask 라우트 (기본)
# ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/roi_feed')
def roi_feed():
    return Response(
        gen_roi_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/model_info')
def get_model_info():
    """모델 정보 API"""
    return jsonify(model_info)

# ────────────────────────────────────────────────────────────────────
# 🔥 NEW: 인터랙티브 등록 시스템
# ────────────────────────────────────────────────────────────────────

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    """새로운 등록 세션 시작"""
    data = request.get_json()
    username = data.get('username', 'NoName')
    
    # 기존 세션 정리
    clear_user_session(username)
    
    # 새 세션 생성
    user_sessions[username] = {
        'images': [],
        'embeddings': [],
        'confirmed': [],
        'confirmed_embeddings': [],
        'session_id': str(uuid.uuid4()),
        'created_at': time.time()
    }
    
    return jsonify({
        "success": True,
        "session_id": user_sessions[username]['session_id'],
        "message": f"🚀 {username}의 인터랙티브 등록 세션 시작!"
    })

@app.route('/capture_single', methods=['POST'])
def capture_single():
    """한 장씩 촬영 (🎨 고품질 이미지 처리 버전)"""
    global latest_roi, original_frame, net

    data = request.get_json()
    username = data.get('username', 'NoName')

    if not video_enabled:
        initialize_camera()
        time.sleep(0.5)

    # ROI 스트림에서 고품질 이미지 가져오기
    with frame_lock:
        if latest_roi is not None and original_frame is not None:
            roi_clean = latest_roi.copy()
            current_original = original_frame.copy()
        else:
            return jsonify({
                "error": True,
                "message": "카메라에서 이미지를 가져올 수 없습니다."
            })

    try:
        # 🎨 개선된 품질 평가 사용
        from model_ops import assess_roi_quality_visual, create_beautiful_roi
        is_good, quality_score, reason = assess_roi_quality_visual(roi_clean, threshold=0.15)
        
        if not is_good:
            return jsonify({
                "error": True,
                "message": f"❌ 품질이 낮습니다!\n\n문제: {reason}\n점수: {quality_score:.3f}/1.0\n\n손바닥을 더 선명하게 보여주세요."
            })
        
        # 🌟 아름다운 ROI 생성
        beautiful_roi = create_beautiful_roi(roi_clean, target_size=128)
        if beautiful_roi is None:
            beautiful_roi = roi_clean  # 실패 시 원본 사용
        
        # 임베딩 추출
        embedding = extract_embedding(current_original, net)
        if embedding is None:
            return jsonify({
                "error": True,
                "message": "❌ 임베딩 추출 실패!\n\nCCNet 처리 중 오류가 발생했습니다."
            })
        
        # 세션에 임시 저장
        session = user_sessions[username]
        image_id = len(session['images']) + 1
        
        session['images'].append({
            'id': image_id,
            'image': beautiful_roi.copy(),  # 🎨 개선된 이미지 저장
            'quality_score': quality_score,
            'reason': reason,
            'timestamp': time.time()
        })
        session['embeddings'].append(embedding)
        
        # Base64 인코딩해서 응답 (JPEG 품질 95로 향상)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # 80 → 95로 향상
        _, buffer = cv2.imencode('.jpg', beautiful_roi, encode_param)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image_id": image_id,
            "image_data": f"data:image/jpeg;base64,{img_base64}",
            "quality_score": quality_score,
            "reason": reason,
            "message": f"✅ {image_id}번째 이미지 촬영 완료!\n\n품질: {quality_score:.3f} ({reason})\n\n🎨 고품질 처리 완료!"
        })
        
    except Exception as e:
        print(f"❌ 촬영 실패: {e}")
        return jsonify({
            "error": True,
            "message": f"촬영 중 오류: {str(e)}"
        })

@app.route('/get_session_images/<username>')
def get_session_images(username):
    """현재 세션의 모든 이미지 가져오기"""
    try:
        session = user_sessions[username]
        
        images_data = []
        for img_data in session['images']:
            _, buffer = cv2.imencode('.jpg', img_data['image'])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            images_data.append({
                'id': img_data['id'],
                'image_data': f"data:image/jpeg;base64,{img_base64}",
                'quality_score': img_data['quality_score'],
                'reason': img_data['reason'],
                'timestamp': img_data['timestamp'],
                'confirmed': img_data['id'] in [c['id'] for c in session['confirmed']]
            })
        
        confirmed_count = len(session['confirmed'])
        total_count = len(session['images'])
        
        return jsonify({
            'success': True,
            'images': images_data,
            'stats': {
                'total_captured': total_count,
                'confirmed': confirmed_count,
                'remaining_needed': max(0, 10 - confirmed_count),
                'can_register': confirmed_count >= 10
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/confirm_image', methods=['POST'])
def confirm_image():
    """이미지 확인 (최종 등록용으로 선택)"""
    data = request.get_json()
    username = data.get('username')
    image_id = data.get('image_id')
    
    try:
        session = user_sessions[username]
        
        # 이미지 찾기
        target_img = None
        target_embedding = None
        for i, img_data in enumerate(session['images']):
            if img_data['id'] == image_id:
                target_img = img_data
                target_embedding = session['embeddings'][i]
                break
        
        if target_img is None:
            return jsonify({
                "error": True,
                "message": "이미지를 찾을 수 없습니다."
            })
        
        # 이미 확인된 이미지인지 체크
        if any(c['id'] == image_id for c in session['confirmed']):
            return jsonify({
                "error": True,
                "message": "이미 선택된 이미지입니다."
            })
        
        # 10개 초과 방지
        if len(session['confirmed']) >= 10:
            return jsonify({
                "error": True,
                "message": "최대 10개까지만 선택할 수 있습니다."
            })
        
        # 확인 목록에 추가
        session['confirmed'].append(target_img)
        session['confirmed_embeddings'].append(target_embedding)
        
        confirmed_count = len(session['confirmed'])
        
        return jsonify({
            "success": True,
            "confirmed_count": confirmed_count,
            "remaining_needed": max(0, 10 - confirmed_count),
            "can_register": confirmed_count >= 10,
            "message": f"✅ 이미지 #{image_id} 선택 완료!\n\n현재 선택: {confirmed_count}/10개"
        })
        
    except Exception as e:
        return jsonify({
            "error": True,
            "message": f"오류: {str(e)}"
        })

@app.route('/remove_confirmed', methods=['POST'])
def remove_confirmed():
    """확인된 이미지 제거"""
    data = request.get_json()
    username = data.get('username')
    image_id = data.get('image_id')
    
    try:
        session = user_sessions[username]
        
        # 확인 목록에서 제거
        session['confirmed'] = [img for img in session['confirmed'] if img['id'] != image_id]
        
        # 대응하는 임베딩도 제거 (ID 매칭으로)
        original_ids = [img['id'] for img in session['images']]
        confirmed_indices = []
        for conf_img in session['confirmed']:
            try:
                idx = original_ids.index(conf_img['id'])
                confirmed_indices.append(idx)
            except ValueError:
                continue
        
        session['confirmed_embeddings'] = [session['embeddings'][i] for i in confirmed_indices]
        
        confirmed_count = len(session['confirmed'])
        
        return jsonify({
            "success": True,
            "confirmed_count": confirmed_count,
            "remaining_needed": max(0, 10 - confirmed_count),
            "can_register": confirmed_count >= 10,
            "message": f"🗑️ 이미지 #{image_id} 선택 해제!\n\n현재 선택: {confirmed_count}/10개"
        })
        
    except Exception as e:
        return jsonify({
            "error": True,
            "message": f"오류: {str(e)}"
        })

@app.route('/finalize_registration', methods=['POST'])
def finalize_registration():
    """최종 등록 실행"""
    data = request.get_json()
    username = data.get('username')
    
    try:
        session = user_sessions[username]
        
        if len(session['confirmed']) < 10:
            return jsonify({
                "error": True,
                "message": f"❌ 10개의 이미지가 필요합니다!\n\n현재 선택: {len(session['confirmed'])}개\n부족: {10 - len(session['confirmed'])}개"
            })
        
        # 정확히 10개만 사용
        final_embeddings = session['confirmed_embeddings'][:10]
        
        if len(final_embeddings) != 10:
            return jsonify({
                "error": True,
                "message": "임베딩 개수가 맞지 않습니다."
            })
        
        # 임베딩 품질 검증
        quality_passed = 0
        for i, emb in enumerate(final_embeddings):
            emb_norm = np.linalg.norm(emb)
            emb_std = np.std(emb)
            
            if 0.1 <= emb_norm <= 5.0 and emb_std >= 0.01:
                quality_passed += 1
        
        if quality_passed < 8:  # 10개 중 8개는 통과해야 함
            return jsonify({
                "error": True,
                "message": f"❌ 임베딩 품질 검증 실패!\n\n통과: {quality_passed}/10개\n최소 필요: 8개\n\n더 선명한 이미지들을 선택해주세요."
            })
        
        # DB에 등록
        register_user(username, final_embeddings)
        
        # 세션 정리
        clear_user_session(username)
        
        return jsonify({
            "success": True,
            "message": f"🎉 {username} 등록 완료!\n\n📊 등록 통계:\n• 선택한 이미지: 10장\n• 품질 통과: {quality_passed}장\n• 등록 방식: 사용자 직접 선택\n\n✨ 인터랙티브 등록 시스템으로\n최고 품질의 생체 데이터가 등록되었습니다!"
        })
        
    except Exception as e:
        clear_user_session(username)
        return jsonify({
            "error": True,
            "message": f"등록 중 오류: {str(e)}"
        })

# ────────────────────────────────────────────────────────────────────
# 기존 인증 시스템 (변경 없음)
# ────────────────────────────────────────────────────────────────────

@app.route('/authenticate', methods=['GET'])
def authenticate_page():
    return render_template('authenticate.html')

@app.route('/auto_auth', methods=['POST'])
def auto_auth():
    """자동 인증"""
    global original_frame, net

    if not video_enabled:
        initialize_camera()
        time.sleep(0.5)

    with frame_lock:
        if original_frame is not None:
            frame = original_frame.copy()
        else:
            return jsonify({
                "success": False, 
                "message": "프레임을 읽을 수 없습니다."
            })

    try:
        embedding = extract_embedding(frame, net)
        
        if embedding is None:
            return jsonify({
                "success": False, 
                "message": "🖐️ 손바닥을 보여주세요!"
            })

        result = multi_shot_authenticate(embedding)
        model_type = model_info.get('type', 'Unknown')
        
        if result:
            return jsonify({
                "success": True,
                "message": f"✅ 인증 성공!\n\n👤 사용자: {result}\n🤖 모델: {model_type}\n🔒 인터랙티브 등록 데이터로 인증!",
                "user": result
            })
        else:
            return jsonify({
                "success": False,
                "message": f"❌ 인증 실패\n\n등록된 사용자가 아니거나\n손바닥을 다시 보여주세요."
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"인증 중 오류: {str(e)}"
        })

# ────────────────────────────────────────────────────────────────────
# 🔥 NEW: 관리자 페이지 라우트 추가
# ────────────────────────────────────────────────────────────────────

@app.route('/admin')
def admin_page():
    """관리자 페이지"""
    return render_template('admin.html')

@app.route('/admin/stats')
def admin_stats():
    """시스템 통계 API"""
    try:
        conn = connect_to_db()
        if conn is None:
            return jsonify({"success": False, "message": "DB 연결 실패"})
        
        cursor = conn.cursor()
        
        # 총 사용자 수
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        # 총 인증 시도
        cursor.execute("SELECT COUNT(*) FROM access_logs")
        total_auth_attempts = cursor.fetchone()[0]
        
        # 성공한 인증
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE status = 'Success'")
        successful_auths = cursor.fetchone()[0]
        
        # 성공률 계산
        success_rate = round((successful_auths / total_auth_attempts * 100), 1) if total_auth_attempts > 0 else 0
        
        conn.close()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_users": total_users,
                "total_auth_attempts": total_auth_attempts,
                "successful_auths": successful_auths,
                "success_rate": success_rate
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/admin/users')
def admin_users():
    """사용자 목록 API"""
    try:
        conn = connect_to_db()
        if conn is None:
            return jsonify({"success": False, "message": "DB 연결 실패"})
        
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, created_at, LENGTH(palm_embedding) as embedding_size FROM users ORDER BY created_at DESC")
        users = cursor.fetchall()
        
        user_list = []
        for user in users:
            user_list.append({
                "id": user[0],
                "name": user[1],
                "registration_date": user[2].strftime("%Y-%m-%d %H:%M:%S") if user[2] else "Unknown",
                "embedding_size": user[3] if user[3] else 0
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "users": user_list
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    """사용자 삭제 API"""
    try:
        conn = connect_to_db()
        if conn is None:
            return jsonify({"success": False, "message": "DB 연결 실패"})
        
        cursor = conn.cursor()
        
        # 사용자 이름 가져오기
        cursor.execute("SELECT name FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({"success": False, "message": "사용자를 찾을 수 없습니다."})
        
        username = user[0]
        
        # 사용자 삭제
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        
        # 관련 로그도 삭제
        cursor.execute("DELETE FROM access_logs WHERE user_name = %s", (username,))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": f"사용자 '{username}' 삭제 완료"
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/admin/similarity/<int:user_id>')
def admin_similarity(user_id):
    """사용자별 유사도 분석 API (개발 중)"""
    return jsonify({
        "success": True,
        "similarities": [
            {"user_name": "Sample User", "euclidean_distance": 0.5432, "cosine_similarity": 0.8765}
        ]
    })

# ────────────────────────────────────────────────────────────────────
# 앱 실행
# ────────────────────────────────────────────────────────────────────

def load_basic_ccnet_model():
    """CCNet 모델 로드"""
    global net, model_info
    
    model_paths = [
        "/Users/kimeunsu/Desktop/공부/졸작 논문/CCNet-main-2/access_system1/models/checkpoint_step_951.pth"
    ]
    
    print("🚀 인터랙티브 등록용 CCNet 모델 로드 시도...")
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                print(f"📁 모델 파일 발견: {file_size:.1f}MB")
                
                net = load_ccnet_model(model_path=model_path, num_classes=1000, weight=0.8)
                
                model_info = {
                    'type': 'Interactive CCNet',
                    'path': model_path,
                    'loaded': True,
                    'file_size_mb': round(file_size, 1),
                    'features': 'User-Controlled Interactive Registration',
                    'registration_type': 'Interactive',
                    'user_control': True
                }
                print("🎉 인터랙티브 등록용 CCNet 모델 로드 성공!")
                return True
                
            except Exception as e:
                print(f"❌ 모델 로드 실패: {e}")
                continue
    
    raise RuntimeError("사용 가능한 모델 파일이 없습니다.")

if __name__ == '__main__':
    try:
        load_basic_ccnet_model()
        
        print("\n" + "="*80)
        print("🎮 인터랙티브 등록 시스템 활성화!")
        print("👤 사용자가 직접 품질을 확인하고 선택!")
        print("🖼️ 10장 촬영 → 미리보기 → 선택 → 삭제/재촬영 → 최종 등록")
        print("🎯 사용자 중심의 품질 관리!")
        print("📱 실시간 미리보기로 최고 품질 보장!")
        print(f"🔥 {model_info['type']} 사용 중!")
        print("🛡️ 관리자 패널 활성화!")
        print("🌐 웹 서버: http://localhost:5000")
        print("✨ 완전히 새로운 등록 경험!")
        print("="*80)

        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        print(f"\n❌ 인터랙티브 등록 시스템 초기화 실패: {e}")
        import sys
        sys.exit(1)
