import pygame
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from utils import blit_rotate_center
import utils
from os.path import exists

qtable = "learning\\qtable.txt"
agent_eps = "learning\\eps.txt"
mem_cnt = "learning\\mem.txt"
state_mem = "learning\\state.txt"
action_mem = "learning\\action.txt"
reward_mem = "learning\\reward.txt"
new_state_mem = "learning\\new state.txt"
terminal_mem = "learning\\terminal.txt"


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:@' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, max_vel, rotation_vel, gamma, eps, lr, batch_size, n_actions, max_mem_size=1000000,
                 eps_end=.002, eps_dec=.00025):
        # ML stuff
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.looking_angles = [0, 35, 90, 125, 180, 270]
        self.input_dims = [len(self.looking_angles) + 1]

        self.q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=self.input_dims, fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        # if exists(qtable):
        #     print("LOADING Q-TABLE")
        #     self.q_eval.load_state_dict(T.load(qtable))
        #     self.q_eval.eval()
        # if exists(agent_eps):
        #     file = open(agent_eps)
        #     self.eps = float(file.read())
        #     file.close()
        # if exists(agent_eps):
        #     file = open(agent_eps)
        #     self.eps = float(file.read())
        #     file.close()

        # Game stuff
        self.img = utils.CAR
        self.max_vel = max_vel
        self.max_boost_vel = 1.25 * max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 180
        self.x, self.y = utils.START_POS
        self.acceleration = 0.4
        self.center = (self.x + int(self.img.get_width() / 2), self.y + int(self.img.get_width() / 2))
        self.num_looking_angles = len(self.looking_angles)
        self.collision_points = []
        self.distance_to_collision = []
        self.line_pixels = []
        self.can_see_goal = False
        self.gates_passed = 0

        self.FORWARD = 0
        self.BACKWARD = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.BOOST = 4
        # self.FORWARD_LEFT = 5
        # self.FORWARD_RIGHT = 6
        # self.BACK_LEFT = 7
        # self.BACK_RIGHT = 8
        # self.BOOST_LEFT = 9
        # self.BOOST_RIGHT = 10
        self.SIT = 11

    @property
    def all_actions(self):
        return [
            self.FORWARD,
            self.BACKWARD,
            self.LEFT,
            self.RIGHT,
            self.BOOST,
            self.SIT]

    # more ML stuff
    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_cntr % self.mem_size

        self.state_memory[idx] = state
        self.new_state_memory[idx] = new_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def choose_action(self, observation, learn=True):
        if not learn:
            action = self.action_helper(observation)
        elif np.random.random() > self.eps:
            action = self.action_helper(observation)
        else:  # not learning and random < eps
            action = np.random.choice(self.action_space)

        return action

    def action_helper(self, observation):
        state = T.tensor([observation]).to(self.q_eval.device)
        actions = self.q_eval.forward(state)
        return T.argmax(actions).item()

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_idx = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.q_eval.forward(state_batch)[batch_idx, action_batch]
        q_next = self.q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

    # back to the game
    def new_center(self):
        if self.center is not None:
            return self.x + int(self.img.get_width() / 2), self.y + int(self.img.get_width() / 2)

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        pixels = []
        for a in self.looking_angles:
            angle = (a + self.angle) % 360
            rad = angle * utils.PI / -180
            line = []
            for i in range(utils.LINE_LENGTH):
                point = (math.floor(self.center[0] + math.cos(rad) * i), math.floor(self.center[1] + math.sin(rad) * i))
                if point[0] < 0 or point[1] < 0 or point[0] >= win.get_width() or point[1] >= win.get_height():
                    break
                line.append(point)
            pixels.append(line)
        self.line_pixels = pixels
        self.get_sight_points(utils.TRACK_BORDER_MASK)

    def get_sight_points(self, track_mask):
        # goal =
        sight_points = []
        distance = []
        can_see_goal = False
        for line in self.line_pixels:
            found = False
            for pixel in line:
                if track_mask.get_at(pixel) != 0 or (
                        pixel[1] == utils.FINISH_POSITION[1] and utils.FINISH_POSITION[0] <=
                        pixel[0] <= utils.FINISH_POSITION[0] + utils.FINISH.get_width()):
                    sight_points.append(pixel)
                    distance.append(math.dist(self.center, pixel))
                    found = True
                    break

                elif pixel[1] == utils.FINISH_POSITION[1] + utils.FINISH.get_height() and utils.FINISH_POSITION[0] \
                        <= pixel[0] <= utils.FINISH_POSITION[0] + utils.FINISH.get_width():
                    sight_points.append(pixel)
                    distance.append(math.dist(self.center, pixel))
                    found = True
                    can_see_goal = True
                    break

            if not found:
                sight_points.append(None)
                distance.append(-1)

        self.collision_points = sight_points
        self.distance_to_collision = distance
        self.can_see_goal = can_see_goal

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

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
        self.vel = min(self.vel + 1.5 * self.acceleration, self.max_boost_vel)
        self.move()

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = utils.START_POS
        self.center = self.new_center()
        self.angle = 180
        self.vel = 0
        self.gates_passed = 0

    def reduce_speed(self):
        if self.vel >= 0:
            self.vel = max(self.vel - self.acceleration / 2, 0)
        else:
            self.vel = min(self.vel + self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel * .6
        self.move()

    def do_action(self, move):
        if move == self.FORWARD:
            self.move_forward()

        elif move == self.BACKWARD:
            self.move_backward()

        elif move == self.LEFT:
            self.rotate(left=True)
            self.reduce_speed()

        elif move == self.RIGHT:
            self.rotate(right=True)
            self.reduce_speed()

        elif move == self.BOOST:
            self.boost_forward()

        elif move == self.SIT:
            self.reduce_speed()

        self.center = self.new_center()

    # def do_action(self):
    #     keys = pygame.key.get_pressed()
    #     moved = False
    #     self.move_forward()
    #     if keys[pygame.K_KP0] or keys[pygame.K_ESCAPE]:
    #         pygame.quit()
    #         return False
    #     if keys[pygame.K_a]:
    #         self.rotate(left=True)
    #     if keys[pygame.K_d]:
    #         self.rotate(right=True)
    #     if keys[pygame.K_w]:
    #         moved = True
    #         if keys[pygame.K_LSHIFT]:
    #             self.boost_forward()
    #         else:
    #             self.move_forward()
    #     if keys[pygame.K_s]:
    #         moved = True
    #         self.move_backward()
    #
    #     if not moved:
    #         self.reduce_speed()
    #     self.center = self.new_center()
    #     return True


class RewardGate:
    def __init__(self, x1, y1, angle=0):
        self.color = [0, 0, 0, 0]
        self.line = [(x1, y1)]
        self.angle = angle


class Game:
    def __init__(self, car: Agent, win):
        self.gates = [
            RewardGate(15, 340), RewardGate(16, 491), RewardGate(80, 615 - 83 * math.sin(45), 45),
            RewardGate(200, 737 - 83 * math.sin(45), 45), RewardGate(388, 760, 270), RewardGate(420, 555, -40),
            RewardGate(403, 723), RewardGate(592, 486, 270), RewardGate(615, 648), RewardGate(740, 762, 270),
            RewardGate(768, 666), RewardGate(765, 380, 45), RewardGate(666, 356, 90), RewardGate(453, 354, 60),
            RewardGate(443, 255, -60), RewardGate(612, 237, 270), RewardGate(762, 239, -45), RewardGate(763, 62, 45),
            RewardGate(634, 34, 90), RewardGate(412, 36, 270), RewardGate(262, 187), RewardGate(259, 404, -45),
            RewardGate(147, 388), RewardGate(146, 200), RewardGate(133, 33, 270)
        ]
        self.car = car
        self.win = win
        self.images = utils.IMAGES
        self.gate_masks = []
        self.clock = pygame.time.Clock()
        self.FPS = 60

    def draw(self):
        gate_masks = []
        for img, pos in self.images:
            self.win.blit(img, pos)
        for gate in self.gates:
            img = pygame.transform.rotate(utils.REWARD_GATE, gate.angle)
            self.win.blit(img, gate.line[0])
            gate_masks.append(pygame.mask.from_surface(img))

        self.car.draw(self.win)
        self.gate_masks = gate_masks
        pygame.display.update()

    def reset(self, mask):
        self.gates = [
            RewardGate(15, 300), RewardGate(16, 390), RewardGate(16, 491), RewardGate(80, 615 - 83 * math.sin(45), 45),
            RewardGate(200, 737 - 83 * math.sin(45), 45), RewardGate(388, 760, 270), RewardGate(420, 555, -40),
            RewardGate(403, 723), RewardGate(592, 486, 270), RewardGate(615, 648), RewardGate(740, 762, 270),
            RewardGate(768, 666), RewardGate(765, 380, 45), RewardGate(666, 356, 90), RewardGate(453, 354, 60),
            RewardGate(443, 255, -60), RewardGate(612, 237, 270), RewardGate(762, 239, -45), RewardGate(763, 62, 45),
            RewardGate(634, 34, 90), RewardGate(412, 36, 270), RewardGate(262, 187), RewardGate(259, 404, -45),
            RewardGate(147, 388), RewardGate(146, 200), RewardGate(133, 33, 270)
        ]
        self.car.x, self.car.y = utils.START_POS
        self.car.center = self.car.new_center()
        self.car.vel = 0
        self.car.angle = 180
        self.car.gates_passed = 0
        self.draw()
        self.car.get_sight_points(mask)
        observation = []
        for d in self.car.distance_to_collision:
            observation.append(d)
        observation.append(self.car.vel)
        # observation.append(self.car.angle)
        # observation.append(self.car.can_see_goal)
        # observation.append(self.car.gates_passed)

        return observation

    def check_collisions(self, mask):
        reward = -1
        hit_wall = False
        done = False
        # turn reward -0.1
        wall_hit = self.car.collide(utils.TRACK_BORDER_MASK)
        if wall_hit is not None:
            hit_wall = True
            done = True
            reward -= 50
            # bad reward -99999
            # self.reset()

        finish_poi = self.car.collide(utils.FINISH_LINE_MASK, *utils.FINISH_POSITION)
        if finish_poi is not None:
            if finish_poi[1] != 0:
                done = True
                reward -= 50
                # bad reward -99999
                # self.reset()
            elif not hit_wall:
                reward += 9999
                done = True
                # best reward +99999
                # self.reset()

        if len(self.gates) != 0:
            gate_hit = self.car.collide(self.gate_masks[0], *self.gates[0].line[0])
            if gate_hit is not None:
                self.car.gates_passed += 1
                # good reward +999
                reward += 50 * self.car.gates_passed
                self.gates = self.gates[1:]
                self.gate_masks = self.gate_masks[1:]

        self.car.get_sight_points(mask)
        observation = []
        for d in self.car.distance_to_collision:
            observation.append(d)
        observation.append(self.car.vel)
        # observation.append(self.car.angle)
        # observation.append(self.car.can_see_goal)
        # observation.append(self.car.gates_passed)

        return observation, reward, done

    def run(self, learn=True):
        max_score = -math.inf
        attempts = "learning\\attempts.txt"
        eps_history = []
        tries = 1
        if exists(attempts):
            file = open(attempts, "r")
            tries = int(file.read())
            file.close()
        finished = False
        frames_to_tick = 30
        while not finished:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                T.save(self.car.q_eval.state_dict(), qtable)
                finished = True
            done = False
            score = 0
            observation = self.reset(utils.TRACK_BORDER_MASK)
            frames = 0
            if tries % 25 == 0:
                frames_to_tick += 5
            while not done:
                for event in pygame.event.get():
                    if event == pygame.QUIT:
                        finished = True
                self.draw()
                pygame.display.update()
                # self.clock.tick(self.FPS)
                move = self.car.choose_action(observation)
                self.car.do_action(move)
                new_observation, reward, done = self.check_collisions(utils.TRACK_BORDER_MASK)
                score += reward
                self.car.store_transition(observation, move, reward, new_observation, done)
                # self.check_collisions()
                observation = new_observation
                if learn:
                    self.car.learn()
                frames += 1
            eps_history.append(agent.eps)
            max_score = max(score, max_score)
            print('episode:', tries, 'score %d' % score, 'ending epsilon %.4f' % agent.eps, "max score %d" % max_score)
            # agent.dec_eps()
            tries += 1
        file = open(attempts, "w")
        file.write(str(tries))
        file.close()
        pygame.quit()


if __name__ == "__main__":
    pygame.init()
    WIN = pygame.display.set_mode((utils.WIDTH, utils.HEIGHT))
    # max_vel, rotation_vel, gamma, eps, lr batch_size, n_actions, max_mem_size=1000000,
    #                  eps_end=.01, eps_dec=.0005
    # check input dims
    if not exists(qtable):
        agent = Agent(10, 8, .99, 1.0, .01, 128, 6)
    else:
        agent = Agent(10, 8, .99, .01, .01, 128, 6)
    g = Game(agent, WIN)
    g.run()
# Is the observation distance to collision points, velocity, and angle?




