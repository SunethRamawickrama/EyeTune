import platform
import cv2
import ctypes
import subprocess

def get_room_color(frame):
    small = cv2.resize(frame, (50, 50))
    avg_color = small.mean(axis=(0,1))  # (B,G,R)
    return avg_color[::-1]  # (R,G,B)

def rgb_to_temp(rgb):
    r, g, b = rgb
    if r > b:  # warmish
        return 3500
    elif b > r:  # coolish
        return 7000
    return 5500  # neutral

def set_temperature(kelvin: int):
    os_type = platform.system().lower()
    if "windows" in os_type:
        _set_gamma_windows(kelvin)
    elif "darwin" in os_type:  # macOS
        subprocess.Popen(["osascript", "-e", f'display dialog "Set temp {kelvin}K (stub)"'])
    elif "linux" in os_type:
        subprocess.Popen(["redshift", "-O", str(kelvin)])
    else:
        print("OS not supported for tint")

def reset_temperature():
    os_type = platform.system().lower()
    if "linux" in os_type:
        subprocess.Popen(["redshift", "-x"])

def _set_gamma_windows(temp: int):
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hdc = user32.GetDC(0)  
    ramp = (ctypes.c_uint16 * 768)()

    def clamp(x): return min(65535, max(0, int(x)))

    for i in range(256):
        val = i / 255.0
        r = val
        g = val * (temp / 6500)
        b = val * (temp / 6500)
        ramp[i] = clamp(r * 65535)
        ramp[i+256] = clamp(g * 65535)
        ramp[i+512] = clamp(b * 65535)

    gdi32.SetDeviceGammaRamp(hdc, ramp)
    user32.ReleaseDC(0, hdc) 

def auto_adjust(frame):
    rgb = get_room_color(frame)
    kelvin = rgb_to_temp(rgb)
    set_temperature(kelvin)
