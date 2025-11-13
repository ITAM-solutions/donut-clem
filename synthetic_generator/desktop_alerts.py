# """
# File name: desktop_alerts
# Author: Fran Moreno
# Last Updated: 01 Aug 2025
# Version: 1.3.5
# Description: This file defines the utility function `show_desktop_alert()`, that will notify the user
# about some process finished by showing an alert depending on the device OS (Windows or macOS), and
# specifically for Windows, will also flash the icon of the app in the toolbar so it is easier to catch
# the user's attention.
# """
# import ctypes
# import platform
#
# import win32con
# import win32gui
#
# from settings import ASSETS_PATH, get_logger
# APP_NAME = "Clem App"
#
# logger = get_logger(__name__)
#
# PLATFORM_NAME = platform.system().lower()
# PLATFORM_RELEASE = platform.uname().release
#
# toast_notifier = None
#
#
# def flash_window(count=5, timeout=0) -> None:
#     """
#     Flashes the app's icon in the system toolbar so the user gets notified about some finished process.
#     Works only on Windows 10 and 11!
#     :param count: maximum number of flashes.
#     :param timeout: Time limit for the flashes.
#     :return: None (flashes the app's icon).
#     """
#     clem_hwnd = []
#
#     def get_clem_window(hwnd, _ctx):
#         if win32gui.GetWindowText(hwnd) == APP_NAME:
#             clem_hwnd.append(hwnd)
#
#     win32gui.EnumWindows(get_clem_window, None)
#
#     if not clem_hwnd:
#         logger.error(f"Could not find window with name {APP_NAME}. Attempt to flash in taskbar failed.")
#         return
#
#     try:
#         win32gui.FlashWindowEx(clem_hwnd[0], win32con.FLASHW_ALL | win32con.FLASHW_TIMERNOFG, count, timeout)
#     except Exception:
#         logger.error("There was an error while trying to flash the App window.")
#         return
#
#
# def show_desktop_alert(title: str, text: str, duration: int = 20) -> None:
#     """
#     Shows a system alert to notify the user about a finished process.
#     The alert is native and works for Windows (10 and 11) and macOS.
#     :param title: Title of the alert.
#     :param text: Text to display in the alert.
#     :param duration: Time that the alert will show up.
#     :return: None (displays a native alert on user's device).
#     """
#     ico_path = str(ASSETS_PATH / 'itam.ico')
#
#     if PLATFORM_NAME == 'windows':
#         if PLATFORM_RELEASE not in ('10', '11'):
#             logger.error(f'Cannot display Windows toast. Windows version is too old (Windows {PLATFORM_RELEASE})')
#             return
#
#         try:
#             from win10toast import ToastNotifier
#
#             ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)
#
#             global toast_notifier
#             if not toast_notifier:
#                 toast_notifier = ToastNotifier()
#
#             toast = toast_notifier
#             toast.show_toast(
#                 title,
#                 text,
#                 duration=duration,
#                 icon_path=ico_path,
#                 threaded=True,
#             )
#
#             # flash_window()
#         except ImportError:
#             logger.error('Error while importing win10toast library. Cannot display Windows toast.')
#
#     # elif platform == 'darwin':
#     #     try:
#     #         from mac_notifications import client
#     #
#     #         client.create_notification(
#     #             title=title,
#     #             subtitle=text,
#     #             icon=ico_path,
#     #             sound="Frog",
#     #         )
#     #     except ImportError:
#     #         logger.error('Error while importing mac_notifications library. Cannot display MacOS toast.')
#     else:
#         logger.warning(f"There is no implementation for desktop alerts on platform type '{PLATFORM_NAME}'")
