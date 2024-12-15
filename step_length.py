from typing import Tuple, Union

import numpy as np
from scipy import signal

def filter_acceleration(acceleration: np.ndarray, frequency: float = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function filters then normalises and centers around 0 the input 3D acceleration

    Parameters
    ----------
    acceleration : N,3 shaped numpy array,
        Contains accelerometer data sequence in m/s^2 un-normalised with gravity offset
    frequency : float, default = 100 hz
        Frequency at which the data has been acquired

    kwargs : dict:
    - gravity : float. Earth's gravity value, default 9.7 m/s^2
    - butter_order : order of the low pass Butterworth filter. default: 2
    - cutoff_step_frequency : float. Represents the number of step one can walk in a second. default: 2 hz
    - filter_all_accelerations : is set to True, will filter along each axis

    Returns
    -------
    acceleration_norm : N,1 np.ndarray
        The filtered acceleration norm

    acceleration_sequence:  N,3 np.ndarray, optional
        The filtered acceleration axis. Set the argument filter_all_accelerations=True to have it.
        else returns the input.
    """

    gravity = kwargs.get("gravity", 9.7)
    delta_time = 1.0 / frequency
    low_pass_cutoff = kwargs.get("cutoff_step_frequency", 2.0)  # hz
    butter_order = kwargs.get("butter_order", 2)  # 2nd order butterworth

    # noinspection PyTupleAssignmentBalance
    lp_numer, lp_denom = signal.butter(butter_order, 2.0 * low_pass_cutoff * delta_time, "low")
    
    # normalise by gravity
    if len(acceleration.shape) > 1:
        acceleration = np.linalg.norm(acceleration, axis=1)
    # A31의 데이터가 Calibration이 되어 있는 것 같아서 아래 코드 비활성화합니다.
    # acceleration /= gravity

    # center the acce norm around 0
    acceleration = signal.filtfilt(lp_numer, lp_denom, acceleration) - np.mean(acceleration)
    return acceleration


def detect_steps(acceleration: np.ndarray, acceleration_threshold: float, frequency: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function calls :func: `filter_acceleration` function by itself.

    Parameters
    ----------
    acceleration : N,3 shaped np.ndarray,
        Contains accelerometer data sequence in m/s^2 un-normalised with an gravity offset
    acceleration_threshold : float, default : 0.07
        The threshold at which the acceleration can be considered as part of a step.
    frequency: float

    Returns
    -------
    acceleration_norm: np.ndarray
        The filtered acceleration norm
    zero_crossing: np.ndarray
        Indices where the normalised centered acceleration crosses 0.
    peaks: np.ndarray
        Indices where the local acceleration peaks are located
    valleys: np.ndarray
        Indices where the local acceleration peaks are located
    """

    acceleration = filter_acceleration(acceleration, frequency=frequency, **kwargs)
    zero_crossing = np.where((acceleration[:-1] * acceleration[1:]) < 0)[0]

    if len(zero_crossing) <= 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    # delete the first index element if the wave is negative to ensure the acceleration goes positive then negative
    if acceleration[zero_crossing[0]] > 0 and acceleration[zero_crossing[0] + 1] < 0:
        zero_crossing = zero_crossing[1:]
    
    # t_plus and t_minus are respectively times where the signal transits from negative, positive to positive, negative
    peaks = []
    valleys = []
    for t_plus, t_minus in zip(zero_crossing[::2], zero_crossing[1::2]):
        max_index = np.argmax(acceleration[t_plus: t_minus])
        peaks.append(max_index + t_plus)

    for t_plus, t_minus in zip(zero_crossing[2::2], zero_crossing[1::2]):
        min_index = np.argmin(acceleration[t_minus: t_plus])
        valleys.append(min_index + t_minus)

    peaks = np.array(peaks, dtype=int)
    valleys = np.array(valleys, dtype=int)

    indices = np.where(acceleration[peaks] > acceleration_threshold)[0]
    peaks = peaks[indices]
    indices = np.where(acceleration[valleys] < -acceleration_threshold)[0]
    valleys = valleys[indices]

    # if len(peaks) == 0:
    #     print("Warning: no acceleration peaks found during sequence")
    # if len(valleys) == 0:
    #     print("Warning: no acceleration valleys found during sequence")

    return acceleration, zero_crossing, peaks, valleys


def compute_step_length(accelerometer_norm: Union[Tuple, np.ndarray], weinberg_gain: float) -> float:
    """
    # Article:  Xinyu Hou 2020 (eq 7, 8, 9, 10, 11)
    Weingberg method (eq 8):
    weinberg_gain is K in the formula

    Parameters
    ----------
    accelerometer_norm : array-like
        The function will find the map and min of this iterable. In m/s^2
    weinberg_gain :
        The gain is the empirical value that is  proportional to the pedestrian leg length

    Returns
    -------
    step_length : float
        The length of the walked step in meters
    """
    step_length = weinberg_gain * np.power(np.max(accelerometer_norm) - np.min(accelerometer_norm), 1.0 / 4)
    return step_length


def compute_step_timestamp(acceleration_threshold: float, weinberg_gain: float, acce, frequency: float):
    acce_norm, zero, _max, _min = detect_steps(acce, acceleration_threshold=acceleration_threshold, frequency=frequency)

    results = []

    for index, (max_idx, min_idx) in enumerate(zip(_max, _min)):
        results.append({
            "step_length": compute_step_length((acce_norm[max_idx], acce_norm[min_idx]), weinberg_gain=weinberg_gain),
            "start": zero[2 * index],
            "middle": zero[2 * index + 1],
            "end": zero[2 * index + 2]
        })

    return results
