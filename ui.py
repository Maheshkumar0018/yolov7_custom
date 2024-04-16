# import pygame
# import sys

# # Initialize Pygame
# pygame.init()

# # Constants
# SCREEN_WIDTH = 400
# SCREEN_HEIGHT = 300
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# FONT_SIZE = 24

# # Function to draw text
# def draw_text(surface, text, font, color, x, y):
#     text_surface = font.render(text, True, color)
#     text_rect = text_surface.get_rect()
#     text_rect.topleft = (x, y)
#     surface.blit(text_surface, text_rect)

# # Checkbox class
# class Checkbox:
#     def __init__(self, x, y, text):
#         self.rect = pygame.Rect(x, y, 20, 20)
#         self.checked = False
#         self.text = text

#     def draw(self, surface):
#         pygame.draw.rect(surface, BLACK, self.rect, 2)
#         if self.checked:
#             pygame.draw.line(surface, BLACK, (self.rect.left + 4, self.rect.centery), (self.rect.centerx - 2, self.rect.bottom - 6), 2)
#             pygame.draw.line(surface, BLACK, (self.rect.centerx - 2, self.rect.bottom - 6), (self.rect.right - 4, self.rect.top + 4), 2)
#         draw_text(surface, self.text, font, BLACK, self.rect.right + 10, self.rect.y)

#     def handle_event(self, event):
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             if event.button == 1:
#                 if self.rect.collidepoint(event.pos):
#                     self.checked = not self.checked
#                     if self.text == "Select All":
#                         for checkbox in checkboxes:
#                             checkbox.checked = self.checked
#                     print_selected_options()

# # Create checkboxes
# checkboxes = [
#     Checkbox(50, 50, "Option 1"),
#     Checkbox(50, 100, "Option 2"),
#     Checkbox(50, 150, "Option 3")
# ]

# # "Select All" checkbox
# select_all_checkbox = Checkbox(50, 200, "Select All")

# # Function to print selected options
# def print_selected_options():
#     selected_options = [checkbox.text for checkbox in checkboxes if checkbox.checked]
#     print("Selected options:", selected_options)

# # Set up the Pygame window
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("Checkbox Example")
# clock = pygame.time.Clock()
# font = pygame.font.SysFont(None, FONT_SIZE)

# # Main loop
# running = True
# while running:
#     screen.fill(WHITE)

#     # Event handling
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         for checkbox in checkboxes + [select_all_checkbox]:
#             checkbox.handle_event(event)

#     # Drawing
#     for checkbox in checkboxes + [select_all_checkbox]:
#         checkbox.draw(screen)

#     # Update display
#     pygame.display.flip()
#     clock.tick(60)

# # Quit Pygame
# pygame.quit()
# sys.exit()

import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT_SIZE = 24

# Function to draw text
def draw_text(surface, text, font, color, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.topleft = (x, y)
    surface.blit(text_surface, text_rect)

# Checkbox class
class Checkbox:
    def __init__(self, x, y, text):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.checked = False
        self.text = text

    def draw(self, surface):
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        if self.checked:
            pygame.draw.line(surface, BLACK, (self.rect.left + 4, self.rect.centery), (self.rect.centerx - 2, self.rect.bottom - 6), 2)
            pygame.draw.line(surface, BLACK, (self.rect.centerx - 2, self.rect.bottom - 6), (self.rect.right - 4, self.rect.top + 4), 2)
        draw_text(surface, self.text, font, BLACK, self.rect.right + 10, self.rect.y)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self.checked = not self.checked
                    if self.text == "Select All":
                        for checkbox in checkboxes:
                            checkbox.checked = self.checked
                    print_selected_options()  # Call the function when a checkbox is clicked
                    return True  # Return True if the event was handled

        return False  # Return False if the event was not handled

# Create checkboxes
checkboxes = [
    Checkbox(50, 50, "Option 1"),
    Checkbox(50, 100, "Option 2"),
    Checkbox(50, 150, "Option 3")
]

# "Select All" checkbox
select_all_checkbox = Checkbox(50, 200, "Select All")

# Function to print selected options
def print_selected_options():
    selected_options = [checkbox.text for checkbox in checkboxes if checkbox.checked]
    for i in selected_options:
        print(i)
    #print("Selected options:", selected_options)

# Set up the Pygame window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Checkbox Example")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, FONT_SIZE)

# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Check if the event is handled by any checkbox
        event_handled = False
        for checkbox in checkboxes + [select_all_checkbox]:
            if checkbox.handle_event(event):
                event_handled = True
                break
        
        if not event_handled:
            # Handle other events here if needed
            pass

    # Drawing
    for checkbox in checkboxes + [select_all_checkbox]:
        checkbox.draw(screen)

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
