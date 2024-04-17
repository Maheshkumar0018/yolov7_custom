
# ui.py
import pygame
import sys

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

    def draw(self, surface, font):  # Pass the font as an argument
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
                    return True

        return False

# Create checkboxes
checkboxes = [
    Checkbox(50, 50, "python.exe"),
    Checkbox(50, 100,"Google Chrome"),
    Checkbox(50, 150, "Teams.exe")
]

# "Select All" checkbox
select_all_checkbox = Checkbox(50, 200, "Select All")

# Function to print selected options
def print_selected_options():
    selected_options = [checkbox.text for checkbox in checkboxes if checkbox.checked]
    return selected_options



#### main.py

import pygame
import sys
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import threading
import time
import csv
from datetime import datetime
from ui import Checkbox, checkboxes, select_all_checkbox, print_selected_options

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT_SIZE = 24

# Set up the Pygame window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Process Monitor")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, FONT_SIZE)

# Function to count the total number of relevant PIDs
def count_pids(process_names):
    pid_cnt = 0
    for process in psutil.process_iter():
        try:
            name = process.name()
            if any(name.lower() in p.lower() for p in process_names):
                pid_cnt += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pid_cnt

# Function to get process name for a given PID
def get_process_name(pid):
    for process in psutil.process_iter():
        try:
            if process.pid == pid:
                return process.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return "Unknown"

# Calculate the total number of relevant PIDs
process_names = ["python.exe"]
total_pids = count_pids(process_names)
print("Total PIDs:", total_pids)

# Create dictionaries to store CPU and memory usage data for each PID
cpu_data = defaultdict(list)
mem_data = defaultdict(list)
time_data = defaultdict(list)

# Function to update CPU and memory usage data for a given process
def update_data(process_name, cpu_data, mem_data, time_data):
    current_time = time.time() # Fetch current time outside the loop
    for process in psutil.process_iter():
        try:
            name = process.name()
            pid = process.pid
            if process_name.lower() in name.lower():
                cpu_percent = process.cpu_percent(interval=None)
                mem_percent = process.memory_percent()
                cpu_data[pid].append(cpu_percent)
                mem_data[pid].append(mem_percent)
                time_data[pid].append(current_time) # Store the timestamp
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

# Function to update CPU and memory usage data using threads
def update_data_threads(process_names, cpu_data, mem_data, time_data):
    threads = []
    for process_name in process_names:
        thread = threading.Thread(target=update_data, args=(process_name, cpu_data, mem_data, time_data))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

# Function to update the plot with new data
def animate(i):
    update_data_threads(process_names, cpu_data, mem_data, time_data)
    plt.clf() # Clear the current figure
    selected_options = print_selected_options()
    print("Selected options:", selected_options)  # Print selected options for debugging
    if not selected_options:
        print("selected options are not coming")
        return  # No selected options, no need to plot
    for idx, (pid, data) in enumerate(cpu_data.items()):
        process_name = get_process_name(pid)
        if process_name in selected_options:
            plt.subplot(total_pids, 2, idx + 1)
            plt.subplots_adjust(hspace=0.9)
            plt.plot(time_data[pid], data)
            plt.title(f'CPU Usage for {process_name} (PID: {pid})')
            plt.xlabel('Time')
            plt.xticks(rotation=100)
            plt.ylabel('CPU Usage (%)')
        
    for idx, (pid, data) in enumerate(mem_data.items()):
        process_name = get_process_name(pid)
        if process_name in selected_options:
            plt.subplot(total_pids, 2, idx + 1 + total_pids)
            plt.plot(time_data[pid], data)
            plt.title(f'Memory Usage for {process_name} (PID: {pid})')
            plt.xlabel('Time')
            plt.ylabel('Memory Usage (%)')


    # Save data to CSV file
    with open('process_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PID', 'Process Name', 'Time', 'CPU Usage (%)', 'Memory Usage (%)'])
        for pid in cpu_data:
            process_name = get_process_name(pid)
            if process_name in selected_options:
                for i in range(len(cpu_data[pid])):
                    timestamp = datetime.fromtimestamp(time_data[pid][i]).strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([pid, process_name, timestamp, cpu_data[pid][i], mem_data[pid][i]])

# Function to run Pygame event loop
def pygame_event_loop():
    running = True
    while running:
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

        screen.fill(WHITE)

        # Drawing
        for checkbox in checkboxes + [select_all_checkbox]:
            checkbox.draw(screen, font)

        # Update display
        pygame.display.flip()
        clock.tick(60)

# Start Pygame event loop in a separate thread
pygame_thread = threading.Thread(target=pygame_event_loop)
pygame_thread.start()

# Create a Matplotlib figure
fig = plt.figure(figsize=(4, 5))
# Create the animation
ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False) # Update every 1 second
# Display the plot
plt.show()

# Wait for the Pygame thread to finish
pygame_thread.join()

# Quit Pygame
pygame.quit()
sys.exit()

