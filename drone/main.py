
from tello_control_ui import TelloUI
import sys


import tello



def main():
    print(sys.path)


    drone = tello.Tello('', 8889)
    vplayer = TelloUI(drone,"./img/")

    vplayer.root.mainloop()

if __name__ == "__main__":
    main()
