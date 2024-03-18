import glm
import pygame as pg

FOV = 50  # deg
NEAR = 0.1
FAR = 100
SPEED = 0.005
SENSITIVITY = 0.04


class Camera:
    def __init__(self, app, position=(-4, 0, 2), yaw=0, pitch=0):
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]
        self.position = glm.vec3(position)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        self.yaw = yaw
        self.pitch = pitch
        # view matrix
        self.m_view = self.get_view_matrix()
        # projection matrix
        self.m_proj = self.get_projection_matrix()


    def rotate(self, yaw):
        self.yaw += yaw

    def update_camera_vectors(self):
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)

        self.forward.x = glm.cos(yaw) * glm.cos(pitch)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.sin(yaw) * glm.cos(pitch)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def update(self, direction, yaw):
        if yaw != 0:
            self.rotate(yaw)
            self.update_camera_vectors()
            self.m_view = self.get_view_matrix()
        if direction != 0:
            self.move(direction)
            self.update_camera_vectors()
            self.m_view = self.get_view_matrix()


    def move(self, direction):
        velocity = 0.004
        if direction == 0:
            self.position += self.forward * velocity
        if direction == 1:
            self.position += self.forward * velocity
        if direction == 2:
            self.position -= self.right * velocity
        if direction == 3:
            self.position += self.right * velocity
        if direction == 4:
            self.position += self.up * velocity
        if direction == 5:
            self.position -= self.up * velocity

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.forward, self.up)

    def get_projection_matrix(self):
        return glm.perspective(glm.radians(FOV), self.aspect_ratio, NEAR, FAR)

