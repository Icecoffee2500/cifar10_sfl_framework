import random
import numpy as np
import torch
import os
import logging
import sys

logger = logging.getLogger(__name__)

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

def set_seed(seed: int = 1234) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    logger.info(f"🌱 Setting random seed to {seed} for reproducibility")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set torch number of threads to 1 for deterministic behavior
    torch.set_num_threads(1)
    
    logger.info("✅ Random seed fixed for reproducible results")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        elapsed_time = elapsed_time / 1000

    return elapsed_time 

def setup_logging(file_name):
    # root 설정 (콘솔 + 파일)
    fmt = "%(asctime)s %(levelname)s\t[%(name)s]\t%(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt
    )  # 기본 콘솔(또는 root) 설정

    # 필요하면 추가 파일 핸들러 (basicConfig에 file_name 주면 콘솔 없어짐 -> 직접 추가 선호)
    root = logging.getLogger()
    file_handler = logging.FileHandler(file_name, mode="w")
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(file_handler)

# # To print in color during test/train 
# def prRed(skk): logger.info("\033[91m {}\033[00m" .format(skk))
# def prGreen(skk): logger.info("\033[92m {}\033[00m" .format(skk))

class StreamColorFormatter(logging.Formatter):
    COLORS = {
        "red": "\x1b[91m",
        "green": "\x1b[92m",
        "yellow": "\x1b[93m",
        "reset": "\x1b[0m",
    }

    def format(self, record):
        # 기본 포맷으로 문자열 생성
        s = super().format(record)
        # record.extra로부터 color 값을 읽음 (없으면 None)
        color = getattr(record, "color", None)
        if color in self.COLORS:
            return f"{self.COLORS[color]}{s}{self.COLORS['reset']}"
        return s

def setup_logging_with_color(file_name):
    fmt = "%(asctime)s %(levelname)s\t[%(name)s]\t%(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)


    # 기존 핸들러 제거 (Hydra 같은 프레임워크가 붙여놨을 수 있음)
    for h in list(root.handlers):
        root.removeHandler(h)

    # 파일 핸들러 (plain)
    file_h = logging.FileHandler(file_name, mode="w")
    file_h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_h)

    # 콘솔 핸들러 (컬러 포맷터 사용)
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(StreamColorFormatter(fmt, datefmt=datefmt))
    root.addHandler(stream_h)

class MessageColorFormatter(logging.Formatter):
    COLORS = {
        "red": "\x1b[91m",
        "green": "\x1b[92m",
        "yellow": "\x1b[93m",
        "reset": "\x1b[0m",
    }

    def __init__(self, prefix_fmt, datefmt=None):
        """
        prefix_fmt: 머리부분(타임/레벨/[name]\t 등)만 포함하는 포맷 문자열.
                    예: "%(asctime)s %(levelname)s [%(name)s]\t"
        """
        super().__init__(prefix_fmt, datefmt=datefmt)
        # 별도 포맷터: prefix만 포맷(메시지 포함하지 않음)
        self.prefix_formatter = logging.Formatter(prefix_fmt, datefmt=datefmt)

    def format(self, record):
        # 1) prefix (asctime level [name]    ) 부분 생성 (여기엔 %(message)s 없음)
        prefix = self.prefix_formatter.format(record)

        # 2) 원래 메시지 얻기 (formatMessage를 쓰지 않음 — 메시지만 순수하게)
        message = record.getMessage()

        # 3) 예외 정보가 있으면 메시지 뒤에 붙이기 (옵션)
        if record.exc_info:
            # logging.Formatter.formatException 사용
            exc_text = self.formatException(record.exc_info)
            message = message + "\n" + exc_text

        # 4) 색 지정 (record.extra 로 color 전달)
        color = getattr(record, "color", None)
        if color in self.COLORS:
            message = f"{self.COLORS[color]}{message}{self.COLORS['reset']}"

        # 5) 합쳐서 반환
        return f"{prefix}{message}"

# 설정 함수
def setup_logging_color_message_only(file_name):
    # prefix 포맷(메시지 제외)
    prefix_fmt = "%(asctime)s %(levelname)s\t[%(name)s]\t"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # fmt = "%(asctime)s %(levelname)s\t[%(name)s]\t%(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S"

    # 기존 핸들러 제거 (Hydra 등에서 붙였을 수 있음)
    for h in list(root.handlers):
        root.removeHandler(h)

    # 파일 핸들러: plain format (타임/레벨/[name]/message 모두 포함)
    file_fmt = "%(asctime)s %(levelname)s\t[%(name)s]\t%(message)s"
    fh = logging.FileHandler(file_name, mode="w")
    fh.setFormatter(logging.Formatter(file_fmt, datefmt=datefmt))
    root.addHandler(fh)

    # 콘솔 핸들러: prefix는 포맷하고 message만 컬러 적용
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(MessageColorFormatter(prefix_fmt, datefmt=datefmt))
    root.addHandler(ch)

# 색 입힌 로그를 남기는 helper
def prRed(msg, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(msg, extra={"color": "red"})

def prGreen(msg, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(msg, extra={"color": "green"})