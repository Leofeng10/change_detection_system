import threading
import time

import matplotlib.pyplot as plt
import pygame as pg
import sys
from simulator_model import *
from camera import Camera
from light import Light
from mesh import Mesh
from scene import Scene
from scene_renderer import SceneRenderer

import cv2
from sklearn.metrics import mean_squared_error
from env import *
import torch
import ast
import argparse
import os
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn
from model_cbam import UC
from utils import get_dataloader, print_and_write_log, set_random_seed
import pygame.image
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.pyplot as pp
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error




class GraphicsEngine:
    def __init__(self, win_size=(1600, 900)):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
        self.angle = 0
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        # detect and use existing opengl context
        self.ctx = mgl.create_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        # create an object to help track time
        self.clock = pg.time.Clock()
        self.file_name = "./VanillaAE/command.txt"
        self.time = 0
        self.delta_time = 0
        # light
        self.light = Light()
        # camera
        self.camera = Camera(self)
        # mesh
        self.mesh = Mesh(self)
        # scene
        self.scene = Scene(self)
        # renderer
        self.scene_renderer = SceneRenderer(self)
        self.curr = 0

        self.abort_flag = False
        self.decoder = None
        self.pos = [0, 0, 0]
        parser = argparse.ArgumentParser()
        # basic
        parser.add_argument('--dataset', default='', help='folderall | filelist | pairfilelist')
        parser.add_argument('--dataroot', default='', help='path to dataset')
        parser.add_argument('--datalist', default='', help='path to dataset file list')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
        parser.add_argument('--imageSize', type=int, default=64,
                            help='the height / width of the input image to network')
        parser.add_argument('--nz', type=int, default=256, help='dimension of the latent layers')
        parser.add_argument('--nblk', type=int, default=2, help='number of blocks')
        parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
        parser.add_argument('--no_cuda', action='store_true', help='not enable cuda (if use CPU only)')
        parser.add_argument('--netG', default='', help='path to netG (to continue training)')
        parser.add_argument('--expf', default='./experiments',
                            help='folder to save visualized images and model checkpoints')
        parser.add_argument('--manualSeed', type=int, help='manual random seed')

        # display and save
        parser.add_argument('--log_iter', type=int, default=50, help='log interval (iterations)')
        parser.add_argument('--visualize_iter', type=int, default=500, help='visualization interval (iterations)')
        parser.add_argument('--ckpt_save_epoch', type=int, default=1, help='checkpoint save interval (epochs)')

        # losses
        parser.add_argument('--mse_w', type=float, default=1.0, help='weight for mse (L2) spatial loss')
        parser.add_argument('--ffl_w', type=float, default=0.0, help='weight for focal frequency loss')
        parser.add_argument('--alpha', type=float, default=1.0,
                            help='the scaling factor alpha of the spectrum weight matrix for flexibility')
        parser.add_argument('--patch_factor', type=int, default=1,
                            help='the factor to crop image patches for patch-based focal frequency loss')
        parser.add_argument('--ave_spectrum', action='store_true', help='whether to use minibatch average spectrum')
        parser.add_argument('--log_matrix', action='store_true',
                            help='whether to adjust the spectrum weight matrix by logarithm')
        parser.add_argument('--batch_matrix', action='store_true',
                            help='whether to calculate the spectrum weight matrix using batch-based statistics')
        parser.add_argument('--freq_start_epoch', type=int, default=1,
                            help='the start epoch to add focal frequency loss')

        opt = parser.parse_args()
        opt.is_train = False

        os.makedirs(os.path.join(opt.expf, 'images'), exist_ok=True)
        os.makedirs(os.path.join(opt.expf, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(opt.expf, 'logs'), exist_ok=True)
        train_log_file = os.path.join(opt.expf, 'logs', 'train_log.txt')
        opt.train_log_file = train_log_file

        cudnn.benchmark = True

        if opt.manualSeed is None:
            opt.manualSeed = random.randint(1, 10000)
        print_and_write_log(train_log_file, "Random Seed: %d" % opt.manualSeed)
        set_random_seed(opt.manualSeed)

        if torch.cuda.is_available() and opt.no_cuda:
            print_and_write_log(train_log_file,
                                "WARNING: You have a CUDA device, so you should probably run without --no_cuda")

        dataloader, nc = get_dataloader(opt)
        opt.nc = nc

        print_and_write_log(train_log_file, opt)

        self.model = UC(opt)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++___________________")

        # DQN
        LEARNING_RATE = 0.00033
        num_episodes = 80000
        space_dim = 42  # n_spaces
        action_dim = 27  # n_actions
        threshold = 200
        env = Env(space_dim, action_dim, LEARNING_RATE)
        check_point_Qlocal = torch.load('./VanillaAE/Qlocal.pth')
        check_point_Qtarget = torch.load('./VanillaAE/Qtarget.pth')
        env.q_target.load_state_dict(check_point_Qtarget['model'])
        env.q_local.load_state_dict(check_point_Qlocal['model'])
        env.optim.load_state_dict(check_point_Qlocal['optimizer'])
        epoch = check_point_Qlocal['epoch']
        env.level = 8
        self.env = env
        self.frame_num = 0
        self.pos = [0, 0, 2]
        self.stop = False
        self.saved_loss = []
        print("Model created")
        self.file_name = "./VanillaAE/command.txt"

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.mesh.destroy()
                self.scene_renderer.destroy()
                pg.quit()
                sys.exit()

    def render(self):
        # clear framebuffer
        self.ctx.clear(color=(0.08, 0.16, 0.18))
        # render scene
        self.scene_renderer.render()
        # swap buffers
        pg.display.flip()

    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001

    def tensor2im(self, input_image, imtype=np.uint8):
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):
                image_tensor = input_image.detach()
            else:
                return input_image
            image_numpy = image_tensor[0].cpu().float().numpy()
            if image_numpy.shape[0] == 1:
                # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            # post-processing: transpose and scaling
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = input_image
        return image_numpy.astype(imtype)

    def detect(self, img):
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        real_cpu = t(img)
        # real_cpu = torch.from_numpy(img)
        real_cpu = real_cpu.to(torch.float)
        real_cpu = real_cpu.unsqueeze(0)
        recon = self.model.sample(real_cpu)
        recon_img = self.tensor2im(recon)
        real_img = self.tensor2im(real_cpu)
        grey_real = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        grey_recon = cv2.cvtColor(recon_img, cv2.COLOR_BGR2GRAY)

        # score, diff = structural_similarity(grey_real, grey_recon, full=True)
        loss = mean_squared_error(grey_real, grey_recon)
        scaled_loss = np.exp(loss / 20)
        return scaled_loss

    def run(self):
        while not self.stop:
            success = self.execute()

            if not success:
                self.re_plan()
            else:
                break


    def re_plan(self):
        print("-----------------")
        print("Re plan path")
        print("-----------------")
        state = self.env.reset_test1(self.pos[0], self.pos[1] + 1, self.pos[2], self.pos[0], self.pos[1], self.pos[2])
        total_reward = 0
        n_test = 1
        cmd = ""
        for i in range(n_test):
            while (1):
                if self.env.uavs[0].done:
                    break
                action = self.env.get_action(FloatTensor(np.array([state[0]])), 0.01)

                next_state, reward, uav_done, info, dx, dy, dz, x, y, z = self.env.step(action.item(), 0)

                print(dx, dy, dz, x, y, z, uav_done)
                cmd += self.env.convert_to_cmd_simulator(dx, dy, dz)
                cmd += "(" + str(x) + "," + str(y) + "," + str(z) + ")\n"

                total_reward += reward
                if uav_done:
                    break

                state[0] = next_state

            with open("VanillaAE/command.txt", 'w') as file:
                file.write(cmd)


    def execute(self):
        count = 0
        f = open(self.file_name, "r")
        commands = f.readlines()
        len_cmd = len(commands)
        curr_cmd = 0
        for command in commands:
            print("command", curr_cmd, "direction: ", command)
            curr_cmd += 1

            if command[0] == "(":
                values = ast.literal_eval(command.strip())
                self.pos = [int(v) for v in values]
                print(self.pos)
                screen = pygame.display.get_surface()
                size = screen.get_size()
                buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
                screen_surf = pygame.image.fromstring(buffer, size, "RGBA")
                pg.image.save(screen_surf, "VanillaAE/screenshot/%d.jpg" % count)
                img = cv2.imread("VanillaAE/screenshot/%d.jpg" % count)
                img = cv2.flip(img, 0)
                img = cv2.resize(img, (256, 256))
                loss = self.detect(img)
                print("Loss: ", loss)
                self.saved_loss.append(loss)

                if loss > 70:
                    print(loss, False)
                    return False
            elif command != '' and command != '\n':

                command = command.rstrip()
                command = command.split(",")
                self.run_command(command)

            if curr_cmd == len_cmd:
                self.stop = True

    def run_command(self, command):
        if int(command[1]) != self.angle:
            turn = int(command[1]) - self.angle
            self.angle = int(command[1])
            for i in range(200):
                self.get_time()
                self.check_events()
                self.camera.update(0, turn / 200)
                self.render()
        if int(command[0]) != 0:
            if int(command[0]) == 1:
                for i in range(125):
                    self.get_time()
                    self.check_events()
                    self.camera.update(int(command[0]), 0)
                    self.render()
            elif int(command[0]) == 4 or int(command[0]) == 5:
                for i in range(375):
                    self.get_time()
                    self.check_events()
                    self.camera.update(int(command[0]), 0)
                    self.render()
            else:

                for i in range(75):
                    self.get_time()
                    self.check_events()
                    self.camera.update(1, 0)
                    self.render()
        else:
            time.sleep(10)




app = GraphicsEngine()
app.run()
print(app.saved_loss)

# losss =[64.93809608890264, 58.21301989805113, 31.971097959334816, 38.335312161262266, 42.76994874609379, 43.094612088005896, 43.708479153731766, 42.76198755209488, 42.66149092197018, 42.92846893094304, 45.48965995464159, 41.58497618494551, 38.10105623281798, 21.686681215075602, 25.51779541320712, 20.726045452930606, 18.83041972807884, 16.648166826647252, 14.17794548055511, 10.737954744265995, 10.68266772271215, 9.62021318043582, 9.616294610863955, 9.462369671002916, 9.536045559459389, 9.887189498868102, 16.027850260197752, 15.816254989813764, 15.943769815605837, 15.545075439658564, 15.704418535710083, 15.666817683887711, 15.773210898658409, 15.684493935743124, 15.824522938875289, 16.009127480233026, 15.840697150763644, 16.106781870366646, 16.023803210773572, 16.237258391053942, 19.41429249730033, 19.49451627254959, 19.82531861700021, 19.921385904236722, 19.584049297965734, 19.596439672826122, 19.569023972275918, 19.48711085793875, 19.50420108871181, 19.53599689722578, 19.174001978694736, 19.842917400669403, 20.95976877952662, 21.25675989045533, 21.52834135822606, 14.765592633276626, 19.321363392484415, 19.434639932049198]
#

# losss = [64.98790673741766, 67.68369276085656, 32.04426001109843, 77.40697116426738, 97.51888792433869, 110.90938665580839, 41.23797638117006, 42.16620665647806, 40.23856019177705, 40.11252130622859, 40.06612276593007, 41.35208962760159, 39.96232616033331, 37.467994594026266, 40.90242163679289, 39.408678299825596, 40.294687442297686, 27.033137980133848, 23.96860484477314, 22.60972162909744, 19.015414485588938,
#          19.201391638296116, 19.077375664922585, 18.810676189359654, 18.828207421646347, 18.828336705130674, 18.616591367830377, 18.68855635578438, 19.26269489813812, 18.980975628247688, 19.06807736169945, 20.12317736357024, 20.855813857561134, 20.670301656783426, 20.25431626891789, 19.7190452052751, 20.11303174373676, 19.87699411388028, 10.51601328228678, 10.436454010637593, 10.394362379989067, 10.168275349489766, 10.141027815167337, 12.168922253776845, 13.154168952830402, 13.314771445848075, 10.096622154229962, 18.36585931784111, 10.052955478155205, 9.679449553854191, 9.318903605756303]
#

#
# def update(frame):
#     x.append(len(x))
#     y.append(frame)
#     line.set_data(x, y)
#     ax.relim()
#     ax.autoscale_view()
#     plt.pause(0.5)
#
#
# x = []
# y = []
# fig, ax = pp.subplots()
# ax.set_ylim(0, 200)
# ax.set_xlim(0, 60)
# line, = ax.plot(x, y)

# ax.set_xlabel("Detection")
# ax.set_ylabel("Loss")
# plt.title("Loss of Change Detection")
#
#
#
#
# def main():
#     for l in losss:
#         # l = np.exp(l / 20)
#         update(l)
#
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()






























