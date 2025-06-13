###############################################################################
# load_recommendations.py (F1 스코어 추가 예시 코드)
###############################################################################
import numpy as np
import pandas as pd
import os
import random
import torch
from datetime import datetime

# evaluator.py / DataManager 등
from evaluator import Evaluator
from src.data_manager.data_manager import DataManager


###############################################################################
# 1) 롱테일 지표 계산을 위한 유틸 함수들
###############################################################################

def safe_clamp_recos(recos, n_tracks):
    """
    만약 추천 결과 recos가 곡 ID 범위를 벗어나는 경우를 방지하기 위해,
    0 <= track_id < n_tracks 로 clamp 처리
    """
    return np.clip(recos, 0, n_tracks - 1)

def compute_longtail_metrics(recos, evaluator, long_tail_tracks):
    ground_truths = evaluator.gt  # shape [num_playlists]
    num_playlists = len(ground_truths)
    
    lt_recall_numer = 0  # 롱테일 정답 중 맞춘 수
    lt_recall_denom = 0  # 롱테일 정답 총 개수

    recommended_longtail_tracks = set()  # 롱테일 커버리지 계산용

    for i in range(num_playlists):
        gt_tracks = ground_truths[i]  # 플레이리스트 i의 정답 곡 집합 (set타입)
        # 정답 중에서 롱테일에 해당하는 곡들
        gt_longtail = set(gt_tracks).intersection(long_tail_tracks)
        
        lt_recall_denom += len(gt_longtail)

        recos_i = recos[i]
        recos_longtail = set(recos_i).intersection(long_tail_tracks)

        # 롱테일 정답 중 추천에 성공한 곡
        hit_longtail = gt_longtail.intersection(recos_longtail)
        lt_recall_numer += len(hit_longtail)

        # 커버리지: 모든 추천된 곡 중 롱테일 곡들을 모아둠
        recommended_longtail_tracks.update(recos_longtail)
    
    # 롱테일 Recall
    if lt_recall_denom > 0:
        lt_recall = lt_recall_numer / lt_recall_denom
    else:
        lt_recall = 0.0

    # 롱테일 Coverage
    if len(long_tail_tracks) > 0:
        lt_coverage = len(recommended_longtail_tracks) / len(long_tail_tracks)
    else:
        lt_coverage = 0.0

    return {
        'LongTail_Recall': lt_recall,
        'LongTail_Coverage': lt_coverage
    }

def compute_metrics(recos, evaluator, long_tail_tracks=None):
    """
    기존 evaluator의 메서드를 이용해 Precision, Recall, R-Precision, NDCG, Clicks를 계산하고,
    선택적으로 롱테일 지표까지 추가 계산하여 반환.
    F1 스코어를 추가 계산 (2PR/(P+R)).
    """
    precisions   = evaluator.compute_all_precisions(recos)
    recalls      = evaluator.compute_all_recalls(recos)
    r_precisions = evaluator.compute_all_R_precisions(recos)
    ndcgs        = evaluator.compute_all_ndcgs(recos)
    clicks       = evaluator.compute_all_clicks(recos)

    avg_precision = precisions.mean()
    avg_recall    = recalls.mean()

    # F1 스코어 계산: 2*P*R / (P+R)
    if (avg_precision + avg_recall) > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0.0

    results = {
        'Precision':   avg_precision,
        'Recall':      avg_recall,
        'F1':          f1_score,
        'R-Precision': r_precisions.mean(),
        'NDCG':        ndcgs.mean(),
        'Clicks':      clicks.mean()
    }

    # 롱테일 집합이 주어지면 롱테일 지표까지 추가
    if long_tail_tracks is not None:
        lt_metrics = compute_longtail_metrics(recos, evaluator, long_tail_tracks)
        results.update(lt_metrics)

    return results

def set_all_seeds(seed_value=42):
    """
    모든 랜덤 시드를 고정하는 함수
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # 다중 GPU용
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    print(f"[INFO] All random seeds set to {seed_value} for reproducibility")

def load_cached_evaluator(data_manager, cache_file_path='resources/cache/test_evaluator.pkl'):
    """
    캐시된 평가자를 로드하거나 새로 생성해서 저장하고 반환
    """
    import pickle
    
    # 캐시 디렉토리 생성
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    
    if os.path.exists(cache_file_path):
        print(f"[INFO] Loading cached evaluator from {cache_file_path}")
        with open(cache_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"[INFO] Creating new evaluator and caching to {cache_file_path}")
        # gt_test 생성
        gt_test = []
        for i in DataManager.N_SEED_SONGS:  # range(1, 11)
            gt_test += data_manager.ground_truths["test"][i]
        
        # 새 evaluator 생성
        test_evaluator = Evaluator(data_manager, gt=np.array(gt_test), n_recos=500)
        
        # 캐시에 저장
        with open(cache_file_path, 'wb') as f:
            pickle.dump(test_evaluator, f)
            
        return test_evaluator


###############################################################################
# 2) 메인 실행부
###############################################################################
if __name__ == "__main__":
    # 모든 랜덤 시드 고정
    set_all_seeds(42)
    
    # ─────────────────────────────────────────────────────────────────────────
    # (A) DataManager 불러오기 (resplit=False로 데이터 분할 고정)
    # ─────────────────────────────────────────────────────────────────────────
    data_manager = DataManager(resplit=False)
    print("[INFO] DataManager loaded with fixed split.")

    # ─────────────────────────────────────────────────────────────────────────
    # (B) track_popularity 계산 결과 캐싱/로딩
    # ─────────────────────────────────────────────────────────────────────────
    popularity_cache = 'resources/cache/track_popularity.npy'
    longtail_cache = 'resources/cache/longtail_tracks.npy'
    
    os.makedirs(os.path.dirname(popularity_cache), exist_ok=True)
    
    if os.path.exists(popularity_cache) and os.path.exists(longtail_cache):
        track_popularity = np.load(popularity_cache)
        long_tail_tracks = set(np.load(longtail_cache))
        print(f"[INFO] Loaded cached track popularity and longtail data")
    else:
        train_csr = data_manager.binary_train_set
        track_popularity = np.array(train_csr.sum(axis=0)).flatten()
        
        # 하위 20%를 롱테일로 정의
        percentile_20 = np.percentile(track_popularity, 20)
        long_tail_mask = (track_popularity <= percentile_20)
        long_tail_tracks = set(np.where(long_tail_mask)[0])
        
        # 결과 캐싱
        np.save(popularity_cache, track_popularity)
        np.save(longtail_cache, np.array(list(long_tail_tracks)))
        print(f"[INFO] Computed and cached track popularity data")
    
    print(f"[INFO] Track popularity shape = {track_popularity.shape}")
    print(f"[INFO] # of Long-tail tracks = {len(long_tail_tracks)} / total {data_manager.n_tracks}")

    # ─────────────────────────────────────────────────────────────────────────
    # (C) 테스트 평가자(evaluator) 준비 (캐싱 적용)
    # ─────────────────────────────────────────────────────────────────────────
    test_evaluator = load_cached_evaluator(data_manager)

    # ─────────────────────────────────────────────────────────────────────────
    # (D) 추천 결과 로드 및 백업
    # ─────────────────────────────────────────────────────────────────────────
    recos_paths = {
        'MF-Transformer': 'resources/recos/MF-Transformer.npy',
        'MUSE':           'resources/recos/MUSE.npy',
        'LARP':           'resources/recos/LARP.npy',
        'PISA':           'resources/recos/PISA.npy'
    }

    # 백업 디렉토리 생성
    backup_dir = 'resources/recos_backup'
    os.makedirs(backup_dir, exist_ok=True)
    backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    recos_dict = {}
    for model_name, path in recos_paths.items():
        if os.path.exists(path):
            # 백업 파일 생성 (최초 실행 시에만)
            backup_path = f"{backup_dir}/{model_name}_{backup_timestamp}.npy"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(path, backup_path)
                print(f"[INFO] Backed up {model_name} recos to {backup_path}")
            
            # 원본 파일 로드
            raw_recos = np.load(path, allow_pickle=True)
            clamped = safe_clamp_recos(raw_recos, data_manager.n_tracks)
            recos_dict[model_name] = clamped
            print(f"[INFO] Loaded {model_name} recos with shape = {clamped.shape}")
        else:
            print(f"[WARNING] Recommendation file not found: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # (E) 모델별 메트릭 계산 (Precision, Recall, F1, NDCG, Clicks + 롱테일 지표)
    # ─────────────────────────────────────────────────────────────────────────
    metrics_results = {}
    for model_name, recos in recos_dict.items():
        metrics = compute_metrics(recos, test_evaluator, long_tail_tracks=long_tail_tracks)
        metrics_results[model_name] = metrics

    # ─────────────────────────────────────────────────────────────────────────
    # (F) 결과 출력 및 CSV 저장
    # ─────────────────────────────────────────────────────────────────────────
    # 1) 콘솔에 출력
    for model_name, metrics in metrics_results.items():
        print(f"\n=== Metrics for {model_name} ===")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")

    # 2) pandas DataFrame으로 정리
    metrics_df = pd.DataFrame(metrics_results).T
    print("\n[Metrics DataFrame]\n", metrics_df)

    # 3) CSV로 저장
    output_folder = 'metrics_results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_time = datetime.now().strftime('%m%d%H%M')
    filename = f'couterpart_results_{current_time}.csv'
    full_path = os.path.join(output_folder, filename)

    metrics_df.to_csv(full_path)
    print(f"[INFO] Metrics saved to: {full_path}")