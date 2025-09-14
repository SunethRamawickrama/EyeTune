import pyautogui
import pywinauto
import time
import platform

# Check for OS
def get_zoom_hotkey():
    """Returns the appropriate zoom hotkey based on the operating system"""
    system = platform.system().lower()
    if system == 'darwin':  # macOs
        return 'command'
    else:  # Windows, Linux, etc.
        return 'ctrl'

def scale():
    print(f"ear is less than zoom in threshold. Squinting eyes detected")
    modifier_key = get_zoom_hotkey()
    pyautogui.hotkey(modifier_key, '+')

def reset():
    print(f"EAR is normal. Back to normal dimensions")
    modifier_key = get_zoom_hotkey()
    pyautogui.hotkey(modifier_key, '-')
