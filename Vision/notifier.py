import platform
from plyer import notification

def show_notification(title: str, message: str, timeout: int = 3):
    """
    Cross-platform notification wrapper.
    - Windows: uses Action Center
    - macOS: uses Notification Center
    - Linux: uses notify-send (desktop environment must support it)
    """
    system = platform.system()

    # Plyer handles Windows, macOS, and Linux already.
    # But we can adjust defaults per OS if needed.
    notification.notify(
        title=title,
        message=message,
        timeout=timeout  # seconds
    )
