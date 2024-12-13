import pygame
import numpy as np
from time import time

from bomberman_rl import settings as s, GUI

class QualificationGUI(GUI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        font_name = s.ASSET_DIR / "OpenSans-Regular.ttf"
        self.fonts = {k: pygame.font.Font(font_name, 40) for k, v in self.fonts.items()}
        self.window = pygame.display.set_mode((s.WIDTH, 2 * s.HEIGHT))
        pygame.display.set_caption("Qualification Standings")
        pygame.display.init()

    def initScreen(self):
        self.screen = pygame.Surface((s.WIDTH, s.HEIGHT))

    # def loop(self):
    #     while True:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 quit()
    #         yield

    def render_leaderboard(self, leaderboard):
        if self.screen is None:
            self.initScreen()
            #self.loop()
            
        self.screen.fill((0, 0, 0))
        self.frame += 1

        # Scores
        y_base = s.GRID_OFFSET[1] + 15 * s.SCALE
        for i, (player, score) in enumerate(leaderboard):
            self.render_text(
                player,
                50 * s.SCALE,
                y_base + 50 * s.SCALE * i,
                (255, 255, 255) if i > 6 else (234,182,118) if i > 2 else (204, 102, 0),
                valign="center",
                size="big",
            )
            self.render_text(
                f"{score:.1f}",
                630 * s.SCALE,
                y_base + 50 * s.SCALE * i,
                (255, 255, 255) if i > 6 else (234,182,118) if i > 2 else (204, 102, 0),
                valign="center",
                halign="right",
                size="big",
            )
        self.window.blit(self.screen, self.screen.get_rect())
        pygame.event.pump()
        pygame.display.update()

    def quit(self):
        pygame.quit()


# screen = None
# WIDTH, HEIGHT = 800, 600

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# GRAY = (200, 200, 200)

# # Fonts
# font = pygame.font.Font(None, 36)

# if not self.world.running:
#             x_center = (
#                 (s.WIDTH - s.GRID_OFFSET[0] - s.COLS * s.GRID_SIZE) / 2
#                 + s.GRID_OFFSET[0]
#                 + s.COLS * s.GRID_SIZE
#             )
#             color = np.int_(
#                 (
#                     255 * (np.sin(3 * time()) / 3 + 0.66),
#                     255 * (np.sin(4 * time() + np.pi / 3) / 3 + 0.66),
#                     255 * (np.sin(5 * time() - np.pi / 3) / 3 + 0.66),
#                 )
#             )
#             self.render_text(
#                 leading.display_name,
#                 x_center,
#                 320 * s.SCALE,
#                 color,
#                 valign="top",
#                 halign="center",
#                 size="huge",
#             )
#             self.render_text(
#                 "has won the round!",
#                 x_center,
#                 350 * s.SCALE,
#                 color,
#                 valign="top",
#                 halign="center",
#                 size="big",
#             )
#             leading_total = max(
#                 self.world.agents, key=lambda a: (a.total_score, a.display_name)
#             )
#             if leading_total is leading:
#                 self.render_text(
#                     f"{leading_total.display_name} is also in the lead.",
#                     x_center,
#                     390 * s.SCALE,
#                     (128, 128, 128),
#                     valign="top",
#                     halign="center",
#                     size="medium",
#                 )
#             else:
#                 self.render_text(
#                     f"But {leading_total.display_name} is in the lead.",
#                     x_center,
#                     390 * s.SCALE,
#                     (128, 128, 128),
#                     valign="top",
#                     halign="center",
#                     size="medium",
#                 )

# def setup():
#     global screen
#     pygame.init()
#     screen = pygame.display.set_mode((WIDTH, HEIGHT))
#     pygame.display.set_caption("Qualification Standings")

# # Function to render the list on the screen
# def render_leaderboard(leaderboard):
#     screen.fill(BLACK)  # Clear the screen
#     title = font.render("Players", True, WHITE)
#     screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 10))

#     y_offset = 60
#     for name, score in leaderboard:
#         text = font.render(name, True, BLACK)
#         pygame.draw.rect(screen, GRAY, (50, y_offset, WIDTH - 100, 40), border_radius=5)
#         screen.blit(text, (60, y_offset + 5))
#         y_offset += 50  # Space out the items
#     pygame.display.flip()

# # Main loop
# def main():
#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 quit()
#         yield
