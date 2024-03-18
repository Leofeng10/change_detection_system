from simulator_model import *
import glm


class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        # floor
        n, s = 40, 2
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                add(Cube(app, pos=(x, -s, z)))
        add(Cube(app, pos=(10, 0, 40), tex_id=2))
        # columns for base environment
        add(Cube(app, pos=(10, 0, 10), tex_id=2))
        add(Cube(app, pos=(10, 2, 10), tex_id=2))
        add(Cube(app, pos=(12, 0, 10), tex_id=2))
        add(Cube(app, pos=(12, 2, 10), tex_id=2))
        add(Cube(app, pos=(2, 0, 0), tex_id=2))
        add(Cube(app, pos=(2, 2, 0), tex_id=2))
        add(Cube(app, pos=(2, 4, 0), tex_id=2))
        add(Cube(app, pos=(2, 0, 2), tex_id=2))
        add(Cube(app, pos=(2, 2, 2), tex_id=2))
        add(Cube(app, pos=(2, 4, 2), tex_id=2))
        add(Cube(app, pos=(2, 0, 4), tex_id=2))
        add(Cube(app, pos=(2, 2, 4), tex_id=2))
        add(Cube(app, pos=(2, 4, 4), tex_id=2))

        # columns for changed environment
        # add(Cube(app, pos=(10, 0, 10), tex_id=2))
        # add(Cube(app, pos=(10, 2, 10), tex_id=2))
        # add(Cube(app, pos=(12, 0, 10), tex_id=2))
        # add(Cube(app, pos=(12, 2, 10), tex_id=2))
        # add(Cube(app, pos=(2, 0, 0), tex_id=2))
        # add(Cube(app, pos=(2, 2, 0), tex_id=2))
        # add(Cube(app, pos=(2, 4, 0), tex_id=2))
        # add(Cube(app, pos=(2, 0, 2), tex_id=2))
        # add(Cube(app, pos=(2, 2, 2), tex_id=2))
        # add(Cube(app, pos=(2, 4, 2), tex_id=2))
        # add(Cube(app, pos=(2, 0, 4), tex_id=2))
        # add(Cube(app, pos=(2, 2, 4), tex_id=2))
        # add(Cube(app, pos=(2, 4, 4), tex_id=2))
        # # changes
        # add(Cube(app, pos=(2, 6, 0), tex_id=1))
        # add(Cube(app, pos=(2, 6, 2), tex_id=1))
        # add(Cube(app, pos=(2, 6, 4), tex_id=1))
        # add(Cube(app, pos=(2, 8, 0), tex_id=1))
        # add(Cube(app, pos=(2, 8, 2), tex_id=1))
        # add(Cube(app, pos=(2, 8, 4), tex_id=1))
        # add(Cube(app, pos=(2, 10, 0), tex_id=1))
        # add(Cube(app, pos=(2, 10, 2), tex_id=1))
        # add(Cube(app, pos=(2, 10, 4), tex_id=1))
        # add(Cube(app, pos=(2, 12, 0), tex_id=1))
        # add(Cube(app, pos=(2, 12, 2), tex_id=1))
        # add(Cube(app, pos=(2, 12, 4), tex_id=1))
        # add(Cube(app, pos=(2, 14, 0), tex_id=1))
        # add(Cube(app, pos=(2, 14, 2), tex_id=1))
        # add(Cube(app, pos=(2, 14, 4), tex_id=1))


        # for i in range(9):
        #     add(Cube(app, pos=(15, i * s, -9 + i), tex_id=2))
        #     add(Cube(app, pos=(15, i * s, 5 - i), tex_id=2))

        # cat
        # add(Cat(app, pos=(0, -1, -10)))

        # moving cube
        # self.moving_cube = MovingCube(app, pos=(0, 6, 8), scale=(3, 3, 3), tex_id=1)
        # add(self.moving_cube)

    def update(self):
        # self.moving_cube.rot.xyz = self.app.time
        pass
