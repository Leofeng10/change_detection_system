
from tello_control_ui import TelloUI
import sys


import tello



def main():
    print(sys.path)

    file_name = sys.argv[1]
    drone = tello.Tello('', 8880)
    vplayer = TelloUI(file_name, drone,"./img/")

    # vplayer.root.mainloop()
    vplayer.send_cmd_from_text()

if __name__ == "__main__":
    main()
