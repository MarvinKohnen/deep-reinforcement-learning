from io import BytesIO
from time import time
import pygame

from . import settings as s


class Item(object):
    def __init__(self):
        pass

    def avatar(self):
        raise NotImplementedError()

    def render(self, screen, x, y):
        screen.blit(self.avatar, (x, y))

    def get_state(self) -> tuple:
        raise NotImplementedError()


def loadScaledAvatar(path: str | bytes, expected_size: list[int] = None):
    # load
    if isinstance(path, bytes):
        avatar = pygame.image.load(BytesIO(path))
    else:
        avatar = pygame.image.load(path)
    if expected_size is not None:
        assert avatar.get_size() == expected_size
    # scale
    width, height = avatar.get_width(), avatar.get_height()
    width, height = width * s.SCALE, height * s.SCALE
    avatar = pygame.transform.scale(avatar, (width, height))
    return avatar


class Coin(Item):
    avatar = loadScaledAvatar(s.ASSET_DIR / "coin.png")

    def __init__(self, pos, collectable=False):
        super(Coin, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.collectable = collectable

    def get_state(self):
        return self.x, self.y


class Bomb(Item):
    DEFAULT_AVATARS = {
        color: loadScaledAvatar(s.ASSET_DIR / f"bomb_{color}.png")
        for color in s.AGENT_COLORS
    }

    def __init__(self, pos, owner, timer, power, bomb_sprite):
        super(Bomb, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.owner = owner
        self.timer = timer
        self.power = power

        self.active = True
        self.avatar = bomb_sprite

    def get_state(self):
        return (self.x, self.y), self.timer

    def get_blast_coords(self, arena):
        x, y = self.x, self.y
        blast_coords = [(x, y)]

        for i in range(1, self.power + 1):
            if arena[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, self.power + 1):
            if arena[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, self.power + 1):
            if arena[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, self.power + 1):
            if arena[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))

        return blast_coords


class Explosion(Item):
    ASSETS = [
        [loadScaledAvatar(s.ASSET_DIR / f"explosion_{i}.png") for i in range(4)],
        [loadScaledAvatar(s.ASSET_DIR / f"smoke_{i}.png") for i in range(2)],
    ]

    def __init__(self, x, y, owner, timer):
        super().__init__()
        #self.blast_coords = blast_coords
        #self.screen_coords = screen_coords
        self.owner = owner
        self.timer = timer
        self.stage = 0
        self.x, self.y = x, y

    def is_dangerous(self):
        return self.stage == 0

    def get_state(self):
        return (self.x, self.y), self.stage, self.timer

    def next_stage(self):
        try:
            self.stage += 1
            self.timer = len(Explosion.ASSETS[self.stage])
        except IndexError:
            self.stage = None

    def render(self, screen, x, y):
        avatar = pygame.transform.rotate(
            Explosion.ASSETS[self.stage][self.timer - 1], (-50 * time()) % 360
        )
        rect = avatar.get_rect()
        rect.center = x + 15, y + 15
        screen.blit(avatar, rect.topleft)
