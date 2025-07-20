import mariadb
import numpy as np
import cv2

def connect_to_db():
    try:
        conn = mariadb.connect(
            user="root",
            password="rladmstn12@",  # 실제 비밀번호
            host="localhost",
            port=3306,
            database="access_system"
        )
        print("DB 연결 성공")
        return conn
    except mariadb.Error as e:
        print(f"DB 연결 실패: {e}")
        return None

def log_access(user_name, status, device_info="Unknown", reason="N/A"):
    """인증 로그를 access_logs 테이블에 기록"""
    conn = connect_to_db()
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        sql = "INSERT INTO access_logs (user_name, status, device_info, reason) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (user_name, status, device_info, reason))
        conn.commit()
        print(f"[로그 기록] 사용자: {user_name}, 상태: {status}, 장치: {device_info}, 이유: {reason}")
    except mariadb.Error as e:
        print(f"로그 기록 실패: {e}")
        conn.rollback()
    finally:
        conn.close()

def calculate_embedding_weights(embeddings):
    """임베딩들의 가중치 계산 (일관성 기반)"""
    if len(embeddings) <= 1:
        return [1.0] * len(embeddings)
    
    weights = []
    
    for i, emb1 in enumerate(embeddings):
        # 다른 모든 임베딩과의 평균 유사도 계산
        similarities = []
        for j, emb2 in enumerate(embeddings):
            if i != j:
                # 코사인 유사도 계산 (거리가 아닌 유사도)
                norm1 = emb1 / np.linalg.norm(emb1)
                norm2 = emb2 / np.linalg.norm(emb2)
                similarity = np.dot(norm1, norm2)  # 코사인 유사도
                similarities.append(similarity)
        
        # 평균 유사도를 가중치로 사용
        avg_similarity = np.mean(similarities)
        weight = max(0.1, avg_similarity)  # 최소 가중치 0.1
        weights.append(weight)
    
    # 가중치 정규화
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    return weights

def weighted_average_template(embeddings, weights):
    """가중 평균으로 템플릿 생성"""
    if len(embeddings) == 0:
        return None
    
    if len(embeddings) == 1:
        return embeddings[0]
    
    # 가중 평균 계산
    template = np.zeros_like(embeddings[0])
    for emb, w in zip(embeddings, weights):
        template += emb * w
    
    return template

def register_user(username, embeddings):
    """가중 평균 기반 사용자 등록"""
    conn = connect_to_db()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        if len(embeddings) == 0:
            print("등록 실패: 유효한 임베딩이 없습니다.")
            return

        # 🎯 가중 평균 적용
        if len(embeddings) > 1:
            # 1단계: 가중치 계산 (일관성 기반)
            weights = calculate_embedding_weights(embeddings)
            print(f"[등록 로그] 가중치: {[f'{w:.3f}' for w in weights]}")
            
            # 2단계: 극단적 아웃라이어 제거 (가중치가 너무 낮은 것)
            if len(embeddings) >= 5:
                avg_weight = np.mean(weights)
                threshold_weight = avg_weight * 0.3  # 평균의 30% 이하는 제거
                
                keep_indices = [i for i, w in enumerate(weights) if w > threshold_weight]
                if len(keep_indices) >= 3:  # 최소 3개는 유지
                    filtered_embeddings = [embeddings[i] for i in keep_indices]
                    filtered_weights = [weights[i] for i in keep_indices]
                    # 가중치 재정규화
                    filtered_weights = np.array(filtered_weights)
                    filtered_weights = filtered_weights / filtered_weights.sum()
                    
                    embeddings = filtered_embeddings
                    weights = filtered_weights
                    print(f"[등록 로그] 아웃라이어 제거 후: {len(embeddings)}개 유지")
            
            # 3단계: 가중 평균 계산
            center_embedding = weighted_average_template(embeddings, weights)
        else:
            center_embedding = embeddings[0]

        center_embedding = center_embedding.astype(np.float32).tobytes()
        print(f"[등록 로그] 사용자: {username}, 가중 평균 템플릿 크기: {len(center_embedding)}")

        # users 테이블에 사용자 정보와 중심 벡터 추가
        sql_user = "INSERT INTO users (name, palm_embedding) VALUES (%s, %s)"
        cursor.execute(sql_user, (username, center_embedding))
        conn.commit()
        log_access(username, "Registered", "Web")
        print(f"사용자 {username} 등록 완료! (가중 평균 임베딩 {len(embeddings)}개)")
        
    except mariadb.Error as e:
        print(f"등록 실패: {e}")
        conn.rollback()
    finally:
        conn.close()

def multi_shot_authenticate(embedding):
    """DB에 저장된 중심 벡터들과 비교해 가장 가까운 사용자 찾기"""
    conn = connect_to_db()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()

        # 데이터베이스에서 사용자 이름과 중심 벡터 가져오기
        sql = "SELECT name, palm_embedding FROM users"
        cursor.execute(sql)
        rows = cursor.fetchall()

        min_dist = float('inf')
        best_user = None
        threshold = 0.75  # 🎯 임계값 조정 (더 엄격하게)

        for (username, db_embed) in rows:
            if db_embed is not None:
                db_embedding = np.frombuffer(db_embed, dtype=np.float32)

                # 벡터 정규화
                embedding = embedding / np.linalg.norm(embedding)
                db_embedding = db_embedding / np.linalg.norm(db_embedding)

                # 거리 계산
                dist = np.linalg.norm(embedding - db_embedding)
                print(f"[인증 로그] 사용자: {username}, 거리: {dist:.4f}")

                if dist < min_dist:
                    min_dist = dist
                    best_user = username

        print(f"[인증 로그] 최소 거리: {min_dist:.4f}, 임계값: {threshold}")
        
        if min_dist < threshold:
            confidence = (1 - min_dist / threshold) * 100  # 신뢰도 계산
            log_access(best_user, "Success", "Web", f"신뢰도: {confidence:.1f}%")
            print(f"[인증 성공] 사용자: {best_user}, 신뢰도: {confidence:.1f}%")
            return best_user
        else:
            log_access("Unknown", "Failure", "Web", f"임계값 초과 (거리: {min_dist:.4f})")
            print(f"[인증 실패] 임계값 초과 (거리: {min_dist:.4f})")
            return None
            
    except mariadb.Error as e:
        print(f"인증 오류: {e}")
        return None
    finally:
        conn.close()