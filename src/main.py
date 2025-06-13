import json
import argparse
import os
import time
import random
import numpy as np
import pandas as pd
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 기존 import들 유지
from src.data_manager.data_manager import DataManager, EvaluationDataset, pad_collate_eval
from src.rta.utils import get_device
from src.rta.rta_model import RTAModel
from src.baselines.mftf import MFTF

# RTA 컴포넌트들
from src.rta.aggregator.gru import GRUNet
from src.rta.aggregator.cnn import GatedCNN
from src.rta.aggregator.decoder import DecoderModel
from src.rta.aggregator.base import AggregatorBase
from src.rta.representer.base_representer import BaseEmbeddingRepresenter
from src.rta.representer.fm_representer import FMRepresenter
from src.rta.representer.attention_representer import AttentionFMRepresenter

# 베이스라인 모델들
from src.baselines.muse import MUSE
from src.baselines.larp import LARP
from src.baselines.pisa import PISA


# 모델 타입 분류 (하드코딩 제거)
RTA_MODELS = ["MF-GRU", "MF-CNN", "MF-AVG", "MF-Transformer", "FM-Transformer", "NN-Transformer"]
BASELINE_MODELS = ["MUSE", "LARP", "PISA"]


def set_all_seeds(seed_value=42):
    """모든 랜덤 시드를 고정하는 함수"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    print(f"[INFO] 모든 랜덤 시드를 {seed_value}로 고정했습니다.")


def load_config(params_file, model_name):
    """설정 로드 및 training_params 자동 병합"""
    with open(params_file, "r") as f:
        all_params = json.load(f)
    
    if model_name not in all_params:
        available = list(all_params.keys())
        raise KeyError(f"모델 '{model_name}' 설정이 없습니다. 사용 가능: {available}")
    
    config = all_params[model_name].copy()
    
    # training_params 자동 병합
    if 'training_params' in config:
        training_params = config.pop('training_params')
        config.update(training_params)
    
    return config


def create_model(model_name, data_manager, config):
    """모델 생성 - 극도로 단순화"""
    print(f"Initialize: {model_name}")
    
    # RTA 모델들
    if model_name in RTA_MODELS:
        if model_name == "MF-TFIDF":
            return create_mf_tfidf(data_manager, config)
        else:
            return create_rta_model(model_name, data_manager, config)
    
    # 베이스라인 모델들 - 각 모델이 받을 수 있는 파라미터만 필터링
    elif model_name == "MUSE":
        # MUSE가 받을 수 있는 파라미터만 선택
        muse_params = {
            k: v for k, v in config.items() 
            if k in ['k', 'n_items', 'hidden_size', 'lr', 'batch_size', 'alpha', 
                    'inv_coeff', 'var_coeff', 'cov_coeff', 'n_layers', 'maxlen', 
                    'dropout', 'embedding_dim', 'n_sample', 'step', 'training_params']
        }
        # n_items 기본값 설정 (MUSE에서 필요)
        if 'n_items' not in muse_params:
            muse_params['n_items'] = data_manager.n_tracks + 1
        return MUSE(data_manager=data_manager, **muse_params)
    
    elif model_name == "LARP":
        # LARP가 받을 수 있는 파라미터만 선택
        larp_params = {
            k: v for k, v in config.items()
            if k in ['n_sample', 'k', 'hidden_size', 'n_layers', 'num_heads', 
                    'intermediate_size', 'method', 'queue_size', 'momentum', 
                    'lr', 'alpha', 'batch_size', 'training_params']
        }
        return LARP(data_manager=data_manager, **larp_params)
    
    elif model_name == "PISA":
        # PISA가 받을 수 있는 파라미터만 선택
        pisa_params = {
            k: v for k, v in config.items()
            if k in ['n_sample', 'sampling', 'embed_dim', 'queue_size', 'momentum',
                    'session_key', 'item_key', 'time_key', 'training_params']
        }
        return PISA(data_manager=data_manager, device=get_device(), **pisa_params)
    
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")


def fix_transformer_params(d, n_heads):
    """Transformer 파라미터 자동 수정"""
    if d % n_heads == 0:
        return d, n_heads
    
    print(f"[경고] embed_dim({d})이 num_heads({n_heads})로 나누어지지 않습니다.")
    
    # 1. n_heads를 d의 약수로 조정
    possible_heads = [h for h in [1, 2, 4, 8, 16, 32, 64, 128] if d % h == 0 and h <= d]
    if possible_heads:
        new_n_heads = max(possible_heads)
        print(f"[자동수정] n_heads를 {n_heads} -> {new_n_heads}로 변경")
        return d, new_n_heads
    
    # 2. d를 n_heads의 배수로 조정
    new_d = ((d // n_heads) + 1) * n_heads
    print(f"[자동수정] d를 {d} -> {new_d}로 변경")
    return new_d, n_heads


def create_rta_model(model_name, data_manager, config):
    """RTA 모델 생성"""
    # 기본 파라미터 설정
    d = config.get('d', 128)
    
    # Representer 생성
    if model_name.startswith('MF-'):
        representer = BaseEmbeddingRepresenter(data_manager, d)
    elif model_name.startswith('FM-'):
        representer = FMRepresenter(data_manager, d)
    elif model_name.startswith('NN-'):
        # NN-Transformer의 경우 attention heads 검증
        n_att_heads = config.get('n_att_heads', 8)
        d, n_att_heads = fix_transformer_params(d, n_att_heads)
        
        representer = AttentionFMRepresenter(
            data_manager, emb_dim=d,
            n_att_heads=n_att_heads,
            n_att_layers=config.get('n_att_layers', 2),
            dropout_att=config.get('drop_att', 0.1)
        )
    
    # Aggregator 생성  
    if 'AVG' in model_name:
        aggregator = AggregatorBase()
    elif 'CNN' in model_name:
        cnn_params = {k: v for k, v in config.items() 
                     if k in ['n_layers', 'kernel_size', 'conv_size', 'res_block_count', 'k_pool', 'drop_p']}
        aggregator = GatedCNN(d, **cnn_params)
    elif 'GRU' in model_name:
        aggregator = GRUNet(d, config.get('h_dim', 256), d,
                           config.get('n_layers', 2), config.get('drop_p', 0.1))
    elif 'Transformer' in model_name:
        # Transformer 파라미터 검증 및 수정
        n_heads = config.get('n_heads', 8)
        d, n_heads = fix_transformer_params(d, n_heads)
        
        aggregator = DecoderModel(
            embd_size=d,
            max_len=config.get('max_size', 50),
            n_head=n_heads,
            n_layers=config.get('n_layers', 6),
            drop_p=config.get('drop_p', 0.1)
        )
    
    # 수정된 설정 업데이트
    config['d'] = d
    
    return RTAModel(data_manager, representer, aggregator, training_params=config)


def create_mf_tfidf(data_manager, config):
    """MF-TFIDF 모델 생성"""
    representer = BaseEmbeddingRepresenter(data_manager, config['d'])
    aggregator = DecoderModel(config['d'], config.get('max_size', 50),
                             config.get('n_heads', 8), config.get('n_layers', 6),
                             config.get('drop_p', 0.1))
    return MFTF(data_manager, representer, aggregator, training_params=config)


def backup_file(file_path):
    """파일 백업"""
    if os.path.exists(file_path):
        backup_dir = os.path.join(os.path.dirname(file_path), "backup")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(os.path.basename(file_path))
        backup_file = os.path.join(backup_dir, f"{name}_{timestamp}{ext}")
        shutil.copy2(file_path, backup_file)
        print(f"[INFO] 백업: {backup_file}")


def main():
    """메인 함수"""
    set_all_seeds(42)
    
    # 인수 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--params_file", type=str, default="resources/params/best_params_rta.json")
    parser.add_argument("--data_path", type=str, default="resources/data/baselines")
    parser.add_argument("--models_path", type=str, default="resources/models")
    parser.add_argument("--recos_path", type=str, default="resources/recos")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--use_dataparallel", action="store_true")
    parser.add_argument("--use_original_npz", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()

    try:
        # 1. 설정 로드
        config = load_config(args.params_file, args.model_name)
        print(f"[INFO] 설정 로드: {args.model_name}")

        # 2. 데이터 매니저 초기화
        if args.use_original_npz:
            os.environ["USE_ORIGINAL_NPZ"] = "1"
        
        data_manager = DataManager(
            foldername=os.path.dirname(args.data_path), 
            resplit=False, 
            dim=config.get('d', 128)
        )

        # 3. 베이스라인 데이터 로드 (필요 시)
        df_train = None
        if args.model_name in BASELINE_MODELS:
            try:
                df_train = pd.read_hdf(f"{args.data_path}/df_train_for_test")
            except Exception as e:
                print(f"[경고] 베이스라인 데이터 로드 실패: {e}")

        # 4. 모델 생성 (핵심 단순화!)
        model = create_model(args.model_name, data_manager, config)
        
        # 5. GPU 설정
        if torch.cuda.is_available():
            if args.use_dataparallel and "," in args.gpu_ids:
                device_ids = [int(x) for x in args.gpu_ids.split(",")]
                model = nn.DataParallel(model, device_ids=device_ids)
                model = model.to(f"cuda:{device_ids[0]}")
            else:
                device_id = int(args.gpu_ids.split(",")[0])
                model = model.to(f"cuda:{device_id}")

        # 6. 훈련 실행
        save_path = os.path.join(args.models_path, f"{args.model_name}_best.pth")
        
        if not args.eval_only:
            backup_file(save_path)
            print(f"\n[Train] {args.model_name}")
            start_time = time.time()
            
            # 훈련 실행 (모델 타입별 분기)
            if args.model_name in RTA_MODELS:
                if isinstance(model, nn.DataParallel):
                    model.module.run_training(tuning=False, savePath=save_path)
                else:
                    model.run_training(tuning=False, savePath=save_path)
            else:  # BASELINE_MODELS
                if isinstance(model, nn.DataParallel):
                    model.module.run_training(train=df_train, tuning=False, 
                                            savePath=save_path, sample_size=args.sample_size)
                else:
                    model.run_training(train=df_train, tuning=False, 
                                     savePath=save_path, sample_size=args.sample_size)
            
            print(f"[Train] 완료: {time.time() - start_time:.2f}초")
        
        else:
            # 평가 전용: 모델 로드
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(torch.load(save_path))
            else:
                model.load_state_dict(torch.load(save_path))
            print(f"[INFO] 모델 로드: {save_path}")

        # 7. 평가 실행
        print(f"[Inference] {args.model_name}")
        start_time = time.time()

        test_dataset = EvaluationDataset(data_manager, data_manager.test_indices)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                                   num_workers=0, collate_fn=pad_collate_eval)

        # 추론
        model.eval()
        torch.cuda.empty_cache()
        
        try:
            if isinstance(model, nn.DataParallel):
                recos = model.module.compute_recos(test_dataloader, n_recos=500)
            else:
                recos = model.compute_recos(test_dataloader, n_recos=500)
        except Exception as e:
            print(f"[ERROR] 추론 오류: {e}")
            recos = np.zeros((len(test_dataset), 500), dtype=np.int64)

        print(f"[Inference] 완료: {time.time() - start_time:.2f}초")

        # 8. 결과 저장
        os.makedirs(args.recos_path, exist_ok=True)
        recos_path = os.path.join(args.recos_path, f"{args.model_name}.npy")
        backup_file(recos_path)
        np.save(recos_path, recos)
        print(f"결과 저장: {recos_path}")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()