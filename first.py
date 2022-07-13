import pygame
import math

import utils
from utils import scale_image, blit_rotate_center

GRASS = scale_image(pygame.image.load("imgs\\grass.jpg"), 2.5)
TRACK = pygame.image.load("imgs\\track.png")
TRACK_BORDER = pygame.image.load("imgs\\track-border.png")
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load("imgs\\finish.png")
FINISH_LINE_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (25, 178)
CAR = scale_image(pygame.image.load("imgs\\red-car.png"), 0.5)
REWARD_GATE = pygame.image.load("imgs\\reward-gate.png")

PI = math.pi
LINE_LENGTH = 300

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("racing go brrrrr")


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.max_boost_vel = 1.5 * max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 180
        self.x, self.y = self.START_POS
        self.acceleration = 0.4
        self.center = (self.x + int(self.img.get_width() / 2), self.y + int(self.img.get_width() / 2))
        self.looking_angles = [0, 30, 45, 60, 75, 90, 105, 160, 135, 150, 180, 270]
        # self.looking_angles = [180]
        # self.looking_angles = [0, 45, 90, 180]
        # self.looking_angles = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90, 180]
        self.collision_points = []
        # self.line_masks = []
        # self.vision_angles = []
        self.line_pixels = []
        self.print = True
        self.WALL = 100
        self.FINISH_FRONT = 101
        self.FINISH_LINE = 102

    def new_center(self):
        if self.center is not None:
            return self.x + int(self.img.get_width() / 2), self.y + int(self.img.get_width() / 2)

    def angle_in_rads(self):
        return PI * self.angle / 180 * -1

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        masks = []
        angels = [] # I know what I wrote
        pixels = []
        for a in self.looking_angles:
            angle = (a + self.angle) % 360
            line = []
            # img = pygame.transform.rotate(VISION_LINE, angle)
            # if angle <= 90:
            #     win.blit(img, (self.center[0], self.center[1] + LINE_LENGTH * math.sin(deg_to_rad(angle))))
            # elif angle <= 180:
            #     win.blit(img, (self.center[0] + LINE_LENGTH * math.cos(deg_to_rad(angle)), self.center[1] + LINE_LENGTH * math.sin(deg_to_rad(angle))))
            # elif angle <= 270:
            #     win.blit(img, (self.center[0] + LINE_LENGTH * math.cos(deg_to_rad(angle)), self.center[1]))
            # else:
            #     win.blit(img, (self.center[0], self.center[1]))

            for i in range(LINE_LENGTH):
                point = (math.floor(self.center[0] + math.cos(deg_to_rad(angle)) * i), math.floor(self.center[1] + math.sin(deg_to_rad(angle)) * i))
                if point[0] < 0 or point[1] < 0 or point[0] >= win.get_width() or point[1] >= win.get_height():
                    break
                line.append(point)
            pixels.append(line)
            # masks.append(pygame.mask.from_surface(img))
            # angels.append(angle)
        # self.line_masks = masks
        # self.vision_angles = angels
        self.line_pixels = pixels
        self.get_sight_points(TRACK_BORDER_MASK)

    def get_sight_points(self, mask):
        sight_points = []
        for line in self.line_pixels:
            found = False
            for pixel in line:
                if mask.get_at(pixel) != 0:
                    sight_points.append(pixel)
                    found = True
                    break
            if not found:
                sight_points.append(None)

        self.collision_points = sight_points
        self.print = False


        # for line in self.line_masks:
        #     angle = self.vision_angles[idx]
        #     if angle == 0:
        #         offset = (int(self.center[0] - 0), int(self.center[1] - 0))
        #     elif angle <= 90:
        #         offset = (int(self.center[0]), int(self.center[1]) + LINE_LENGTH * math.sin(deg_to_rad(angle)))
        #     elif angle <= 180:
        #         offset = (int(self.center[0] + LINE_LENGTH * math.cos(deg_to_rad(angle))), int(self.center[1] + LINE_LENGTH * math.sin(deg_to_rad(angle))))
        #     elif angle < 270:
        #         offset = (int(self.center[0] + LINE_LENGTH * math.cos(deg_to_rad(angle))), int(self.center[1]))
        #     else:
        #         offset = (int(self.center[0]), int(self.center[1]))

            # intersect_mask = mask.overlap_mask(line, offset)
            # # WIN.blit(intersect_mask.to_surface().convert_alpha(), (0, 0))
            # intersect_outline = intersect_mask.outline()
            # if self.print:
            #     print(mask.overlap_area(line, offset))
            #     print(intersect_outline)
            #     print(len(intersect_outline))
            #     print("here we go")
            #
            # if intersect_outline is None or len(intersect_outline) == 0:
            #     poi = None
            # else:
            #     poi = intersect_outline[0]
            #     small = math.dist(self.center, poi)
            #     for x in intersect_outline:
            #         if x == poi:
            #             continue
            #         dist = math.dist(self.center, x)
            #         if dist < small:
            #             small = dist
            #             poi = x

    def move_forward(self):
        if self.vel <= self.max_vel:
            self.vel = min(self.vel + self.acceleration, self.max_vel)
            self.move()
        else:
            self.reduce_speed()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel / 2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def boost_forward(self):
        self.vel = min(self.vel + 3 * self.acceleration, self.max_boost_vel)
        self.move()

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.center = self.new_center()
        self.angle = 180
        self.vel = 0
        return [
            RewardGate(15, 340), RewardGate(16, 491), RewardGate(80, 615 - 83 * math.sin(45), 45),
            RewardGate(200, 737 - 83 * math.sin(45), 45), RewardGate(388, 760, 270), RewardGate(420, 555, -40),
            RewardGate(403, 723), RewardGate(592, 486, 270), RewardGate(615, 648), RewardGate(740, 762, 270),
            RewardGate(768, 666), RewardGate(765, 380, 45), RewardGate(666, 356, 90), RewardGate(453, 354, 60),
            RewardGate(443, 255, -60), RewardGate(612, 237, 270), RewardGate(762, 239, -45), RewardGate(763, 62, 45),
            RewardGate(634, 34, 90), RewardGate(412, 36, 270), RewardGate(262, 187), RewardGate(259, 404, -45),
            RewardGate(147, 388), RewardGate(146, 200), RewardGate(133, 33, 270)
        ]

    def reduce_speed(self):
        if self.vel >= 0:
            self.vel = max(self.vel - self.acceleration / 2, 0)
        else:
            self.vel = min(self.vel + self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel * .6
        self.move()


class Agent(AbstractCar):
    IMG = CAR
    START_POS = utils.START_POS

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.FORWARD = 0
        self.BACKWARD = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.BOOST = 4
        self.FORWARD_LEFT = 5
        self.FORWARD_RIGHT = 6
        self.BACK_LEFT = 7
        self.BACK_RIGHT = 8
        self.BOOST_LEFT = 9
        self.BOOST_RIGHT = 10
        self.SIT = 11

    @property
    def all_actions(self):
        return [
            self.BOOST,
            self.BACKWARD,
            self.LEFT,
            self.RIGHT,
            self.BOOST,
            self.FORWARD_LEFT,
            self.FORWARD_RIGHT,
            self.BACK_LEFT,
            self.BACK_RIGHT,
            self.BOOST_LEFT,
            self.BOOST_RIGHT,
            self.SIT
        ]

    # input dims:
    # vel, angle, collision points,


class RewardGate:
    def __init__(self, x1, y1, angle=0):
        self.color = [0, 0, 0, 0]
        self.line = [(x1, y1)]
        self.angle = angle


def make_gate_mask(gates):
    gate_masks = []
    for g in gates:
        img = pygame.transform.rotate(REWARD_GATE, g.angle)
        gate_masks.append(pygame.mask.from_surface(img))
    return gate_masks


def deg_to_rad(angle):
    return angle * PI / -180


run = True
clock = pygame.time.Clock()
FPS = 60
images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
car = Agent(10, 8)
gates = car.reset()
gate_masks = make_gate_mask(gates)


def draw(win, images, player, gates: list[RewardGate]):
    gate_masks = []
    for img, pos in images:
        win.blit(img, pos)
    for g in gates:
        img = pygame.transform.rotate(REWARD_GATE, g.angle)
        win.blit(img, g.line[0])
        gate_masks.append(pygame.mask.from_surface(img))

    player.draw(win)
    pygame.display.update()
    return gate_masks


def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False

    if keys[pygame.K_KP0] or keys[pygame.K_ESCAPE]:
        pygame.quit()
        return False
    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        moved = True
        if keys[pygame.K_LSHIFT]:
            player_car.boost_forward()
        else:
            player_car.move_forward()
    if keys[pygame.K_s]:
        moved = True
        player_car.move_backward()

    if not moved:
        player_car.reduce_speed()
    player_car.center = player_car.new_center()
    return True


while run:
    clock.tick(FPS)
    draw(WIN, images, car, gates)

    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
        if event.type == pygame.MOUSEBUTTONDOWN:
            print("MOUSE CLICKED!!!!!!!!!!!!!!!!!!!!")
            print(car.collision_points)
            print(pygame.mouse.get_pos())

    run = move_player(car)

    # wall_hit = car.collide(TRACK_BORDER_MASK)
    # if wall_hit is not None:
    #     # car.bounce()
    #     # bad reward -9999
    #     gates = car.reset()
    #     gate_masks = make_gate_mask(gates)

    finish_poi = car.collide(FINISH_LINE_MASK, *FINISH_POSITION)
    if finish_poi is not None:
        if finish_poi[1] == FINISH.get_height() - 1:
            print("don't cheat")
            #     bad reward -9999
            gates = car.reset()
            gate_masks = make_gate_mask(gates)
        else:
            print(":D")
            # best reward +9999
            gates = car.reset()
            gate_masks = make_gate_mask(gates)

    for g in gates:
        gate_hit = car.collide(gate_masks[0], *g.line[0])
        if gate_hit is not None:
            print("cleared")
            # good reward +1000
            gates.remove(g)
            gate_masks.remove(gate_masks[0])

#     reward -0.1 every frame

pygame.quit()
print("\ncya")
