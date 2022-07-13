import pygame
import sys
from pygame.locals import *

import utils


def draw_level():
    pygame.init()

    WHITE = (255, 255, 255)
    GRAY = (144, 144, 144)
    BLACK = (0, 0, 0)

    mouse_position = (0, 0)
    drawing = False
    screen = pygame.display.set_mode((900, 900), 0, 32)
    screen.fill(GRAY)
    pygame.display.set_caption("ScratchBoard")
    screen.blit(utils.FINISH, utils.FINISH_POSITION1)
    screen.blit(utils.CAR, utils.START_POS1)
    last_pos = None

    done_drawing = False
    while not done_drawing:
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:
                if drawing:
                    mouse_position = pygame.mouse.get_pos()
                    if last_pos is not None:
                        pygame.draw.line(screen, BLACK, last_pos, mouse_position, 5)
                    last_pos = mouse_position
            elif event.type == MOUSEBUTTONUP:
                mouse_position = (0, 0)
                drawing = False
            elif event.type == MOUSEBUTTONDOWN:
                drawing = True
            elif keys[K_SPACE] and not drawing:
                last_pos = None
            elif keys[K_RETURN]:
                done_drawing = True

        pygame.display.update()

    print("done")


