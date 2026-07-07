import pyautogui
import time
def move_to( x, y, duration=0.5):
    pyautogui.moveTo(x, y, duration=duration)
    return 
def click_at( x, y, button='left', clicks=1, interval=0.1):
    pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
    print(f"Clicked {clicks} time(s) at ({x}, {y})")
    return 
def double_click_at( x, y):
    pyautogui.doubleClick(x=x, y=y)
    print(f"Double clicked at ({x}, {y})")
    return 
def right_click_at( x, y):
    pyautogui.rightClick(x=x, y=y)
    print(f"Right clicked at ({x}, {y})")
    return 
def drag_to( x, y, duration=0.5, button='left'):
    pyautogui.dragTo(x, y, duration=duration, button=button)
    print(f"Dragged to ({x}, {y})")
    return 
def wait(x):
    time.sleep(x)
def type_text(text, interval=0.05):
    pyautogui.write(text, interval=interval)
    print(f"Typed: {text}")
    return
