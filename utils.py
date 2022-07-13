import pygame
import math


# from first import RewardGate


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)


def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)


GRASS = scale_image(pygame.image.load("imgs\\grass.jpg"), 2.5)
TRACK = pygame.image.load("imgs\\track.png")
TRACK_BORDER = pygame.image.load("imgs\\track-border.png")
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load("imgs\\finish.png")
FINISH_LINE_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (25, 178)
CAR = scale_image(pygame.image.load("imgs\\red-car.png"), 0.5)
REWARD_GATE = pygame.image.load("imgs\\reward-gate.png")
START_POS = (60, 215)
FINISH_POSITION1 = (50, 278)
START_POS1 = (90, 230)
PI = math.pi
LINE_LENGTH = 300
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
IMAGES = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
FPS = 30
# print(FINISH.get_height(), FINISH.get_width())
# GATES = [
#             RewardGate(146, 146), RewardGate(133, 33, 270), RewardGate(16, 491),
#             RewardGate(270, 811 - 83 * math.sin(45), 45), RewardGate(420, 555, -40),
#             RewardGate(615, 719), RewardGate(768, 510), RewardGate(539, 356, 90),
#             RewardGate(762, 239, -45), RewardGate(534, 34, 90), RewardGate(262, 259)
#         ]
#
# print(True ^ True)
# print(True ^ False)
# print(False ^ True)
# print(False ^ False)
