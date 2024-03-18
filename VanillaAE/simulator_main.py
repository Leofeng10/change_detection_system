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

class GraphicsEngine:
    def __init__(self, win_size=(1600, 900)):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
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
        self.directions = [6,0, 6, 1, 6, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]
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
        self.stop = False
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
        state = self.env.reset_test1(self.pos[0], self.pos[1] + 1, self.pos[2], self.pos[0], self.pos[1], self.pos[2])
        total_reward = 0
        n_test = 1
        cmd = "command\n"
        for i in range(n_test):
            while (1):
                if self.env.uavs[0].done:
                    break
                action = self.env.get_action(FloatTensor(np.array([state[0]])), 0.01)

                next_state, reward, uav_done, info, dx, dy, dz, x, y, z = self.env.step(action.item(), 0)

                print(dx, dy, dz, x, y, z, uav_done)
                cmd += self.env.convert_to_cmd(dx, dy, dz)
                cmd += "(" + str(x) + " " + str(y) + " " + str(z) + ")\n"

                total_reward += reward
                if uav_done:
                    break

                state[0] = next_state

            with open("drone/command.txt", 'w') as file:
                file.write(cmd)


    def execute(self):
        count = 0
        while True:
            self.get_time()
            self.check_events()
            self.camera.update()
            self.render()
            self.delta_time = self.clock.tick(60)
            count +=1
            if count % 100 == 0:
                screen = pygame.display.get_surface()
                size = screen.get_size()
                buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
                screen_surf = pygame.image.fromstring(buffer, size, "RGBA")
                pg.image.save(screen_surf, "VanillaAE/screenshot/%d.jpg" % count)
                img = cv2.imread( "VanillaAE/screenshot/%d.jpg" % count)
                img = cv2.flip(img, 0)
                img = cv2.resize(img, (256, 256))
                loss = self.detect(img)
                print("Loss: ", loss)
                if loss > 120:
                    return False









app = GraphicsEngine()
app.run()































# import pygame as pg
# import moderngl as mgl
# import sys
# from model import *
# from camera import Camera
# from light import Light
# from mesh import Mesh
# from scene import Scene
# from scene_renderer import SceneRenderer
# import pygame.camera
# import pygame.image
# from OpenGL.GL import *
# from OpenGL.GLU import *
#
# import time

#
# class GraphicsEngine:
#     def __init__(self, win_size=(1600, 900)):
#         # init pygame modules
#         pg.init()
#         # glEnable(GL_DEPTH_TEST)
#         # pygame.camera.init()
#         # window size
#         self.WIN_SIZE = win_size
#         # set opengl attr
#         pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
#         pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
#         pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
#         # create opengl context
#         self.screen = pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
#         # mouse settings
#         pg.event.set_grab(True)
#         pg.mouse.set_visible(False)
#         # self.capture = pg.camera.Camera(0, (256, 256))
#         # self.capture.start()
#         # detect and use existing opengl context
#         self.ctx = mgl.create_context()
#         # self.ctx.front_face = 'cw'
#         self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
#         # create an object to help track time
#         self.clock = pg.time.Clock()
#         self.time = 0
#         self.delta_time = 0
#         # light
#         self.light = Light()
#         # camera
#         self.camera = Camera(self)
#         # mesh
#         self.mesh = Mesh(self)
#         # scene
#         self.scene = Scene(self)
#         # renderer
#         self.scene_renderer = SceneRenderer(self)
#         self.directions = [6,0, 6, 1, 6, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]
#         self.curr = 0
#
#     def check_events(self):
#         for event in pg.event.get():
#             if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
#                 self.mesh.destroy()
#                 self.scene_renderer.destroy()
#                 pg.quit()
#                 sys.exit()
#
#     def render(self):
#         # clear framebuffer
#         self.ctx.clear(color=(0.08, 0.16, 0.18))
#         # render scene
#         self.scene_renderer.render()
#         # swap buffers
#         pg.display.flip()
#
#     def get_time(self):
#         self.time = pg.time.get_ticks() * 0.001
#
#     def run(self):
#         # for direction in self.directions:
#         #     if direction == 6:
#         #         for i in range(1000):
#         #             self.get_time()
#         #             self.check_events()
#         #             self.camera.update(direction, 0.09)
#         #             self.render()
#         #     else:
#         #         for i in range(1000):
#         #             self.get_time()
#         #             self.check_events()
#         #             self.camera.update(direction, 0)
#         #             self.render()
#         count = 0
#         while True:
#             self.get_time()
#             self.check_events()
#             self.camera.update()
#             self.render()
#             self.delta_time = self.clock.tick(60)
#             count +=1
#             if count % 100 == 0:
#                 screen = pygame.display.get_surface()
#                 size = screen.get_size()
#                 buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
#                 screen_surf = pygame.image.fromstring(buffer, size, "RGBA")
#                 pg.image.save(screen_surf, "VanillaAE/screenshot/%d.jpg" % count)









app = GraphicsEngine()
app.run()






























