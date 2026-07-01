import os
import time
from datetime import datetime
from PIL import ImageGrab
import pyautogui
def take_full_screenshot():
    folder_name = "data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: {folder_name}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"screenshot_{timestamp}.png"
    file_path = os.path.join(folder_name, filename)
    try:
        screenshot = ImageGrab.grab()
        screenshot.save(file_path)
        print(f"Screenshot saved successfully: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return None
if __name__ == "__main__":
    time.sleep(1.2) 
    path = take_full_screenshot()
    if path:
        print(f"File location: {os.path.abspath(path)}")