import time
import math
import numpy as np
# import pandas as pd

map_constraints = np.loadtxt('./data/obstacle_mask.txt', dtype=bool)

# @exec_time_decorator
def map_constraints_check(x: float, y: float) -> bool:
    """
    if the (x,y) impossible return false  
    """
    x_max = map_constraints.shape[0] - 0.5
    y_max = map_constraints.shape[1] - 0.5

    if (0 <= x < x_max) and (0 <= y < y_max):
        return not map_constraints[math.floor(x), math.floor(y)]
    else:
        return False
    
    

def exec_time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.5f}초")
        return result
    return wrapper

def round_angle(angle):
    """
    각도 어림하기 (45도 간격으로 반올림)
    """
    # 기준 각도 (라디안 단위로 변환)
    predefined_angles = [0,
                            math.radians(45),
                            math.radians(90),
                            math.radians(135),
                            math.radians(180),
                            -math.radians(135),
                            -math.radians(90),
                            -math.radians(45)
                        ]

    # 각도를 -π에서 π 사이로 변환
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    
    # 가장 가까운 각도 찾기
    def adjust_angle(angle):
        # 라디안 기준으로 각도를 -π에서 π 사이로 변환
        if angle > math.pi:
            angle -= 2 * math.pi  # -π에서 π 사이로 변환
        return angle
    closest_angle = min(predefined_angles, key=lambda x: abs(adjust_angle(abs(angle - x))))
    # print(math.degrees(closest_angle))

    # return angle
    return closest_angle