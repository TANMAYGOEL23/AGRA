"""
Real-Time Forces and Swarm Trajectory Visualizer.
Implements the 3D/2D Forces Visualization Tool as shown in Paper 1 (Fig 5) & Paper 2 (Fig 3).
- Attractive Force (Green)
- Repulsive Force (Red)
- Target Velocity (Light Blue / Cyan)
- Communication Mode HUD (FAST / RELIABLE)
"""
import sys
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class ForceMonitorVisualizer:
    """
    Interactive 2D/3D projection visualizer for multi-UAV swarms.
    """
    def __init__(self, width: int = 900, height: int = 750, title: str = "Multi-UAV ACACT & DCACS Monitor"):
        self.width = width
        self.height = height
        self.title = title
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.paused = False
        self.fps = 30
        
        # Colors
        self.COLOR_BG = (24, 28, 36)
        self.COLOR_GRID = (40, 48, 62)
        self.COLOR_ATTRACTIVE = (46, 204, 113)  # Green
        self.COLOR_REPULSIVE = (231, 76, 60)    # Red
        self.COLOR_VELOCITY = (52, 152, 219)    # Light Blue / Cyan
        self.COLOR_TRAIL = (120, 140, 160)
        self.COLOR_GOAL = (241, 196, 15)       # Yellow
        self.COLOR_COL_ZONE = (231, 76, 60, 40) # Translucent Red
        self.UAV_COLORS = [
            (52, 152, 219), (230, 126, 34), (155, 89, 182),
            (26, 188, 156), (241, 196, 15), (231, 76, 60),
            (46, 204, 113), (236, 240, 241), (149, 165, 166), (211, 84, 0)
        ]

    def init_display(self):
        if not PYGAME_AVAILABLE:
            print("[Visualizer] Pygame not available, running in headless mode.")
            return False
            
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Helvetica", 14, bold=True)
        self.small_font = pygame.font.SysFont("Helvetica", 11)
        return True

    def world_to_screen(self, pos: np.ndarray, bounds: List[float]) -> tuple:
        """Transforms world coords (x, y) into window pixel coords."""
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
        margin = 60
        arena_w = self.width - 2 * margin
        arena_h = self.height - 2 * margin - 80
        
        norm_x = (pos[0] - xmin) / (xmax - xmin + 1e-6)
        norm_y = (pos[1] - ymin) / (ymax - ymin + 1e-6)
        
        # Invert Y for screen coordinates
        sx = int(margin + norm_x * arena_w)
        sy = int(margin + 50 + (1.0 - norm_y) * arena_h)
        return (sx, sy)

    def handle_events(self) -> bool:
        """Processes keyboard / mouse inputs. Returns False on Quit."""
        if not PYGAME_AVAILABLE or self.screen is None:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
        return True

    def render_frame(self, engine: Any, bounds: List[float]):
        """Renders the entire swarm state, force vectors, and communication HUD."""
        if not PYGAME_AVAILABLE or self.screen is None:
            return

        self.screen.fill(self.COLOR_BG)
        
        # Draw Arena Grid
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
        for gx in np.linspace(xmin, xmax, 9):
            p1 = self.world_to_screen(np.array([gx, ymin]), bounds)
            p2 = self.world_to_screen(np.array([gx, ymax]), bounds)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for gy in np.linspace(ymin, ymax, 9):
            p1 = self.world_to_screen(np.array([xmin, gy]), bounds)
            p2 = self.world_to_screen(np.array([xmax, gy]), bounds)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        scale = (self.width - 120) / (xmax - xmin + 1e-6)

        # Draw UAV trails, goals, and vectors
        for i, uav in enumerate(engine.uavs):
            col = self.UAV_COLORS[i % len(self.UAV_COLORS)]
            
            # 1. Goal marker
            gx, gy = self.world_to_screen(uav.goal, bounds)
            pygame.draw.circle(self.screen, self.COLOR_GOAL, (gx, gy), 6, 2)
            lbl_g = self.small_font.render(f"G{uav.uav_id}", True, self.COLOR_GOAL)
            self.screen.blit(lbl_g, (gx + 8, gy - 8))

            # 2. Trajectory trail
            if len(uav.trajectory) > 1:
                trail_pts = [self.world_to_screen(p, bounds) for p in uav.trajectory[::2]]
                if len(trail_pts) > 1:
                    pygame.draw.lines(self.screen, col, False, trail_pts, 2)

            # 3. Collision Risk Bubble (r = 2.0m)
            ux, uy = self.world_to_screen(uav.position, bounds)
            col_pixel_r = int(uav.config.collision_radius * scale)
            pygame.draw.circle(self.screen, (80, 40, 40), (ux, uy), col_pixel_r, 1)

            # 4. Forces Vectors (Attractive: Green, Repulsive: Red, Velocity: Cyan)
            # Attractive vector
            a_end = uav.position + uav.attractive_force * 0.4
            ax, ay = self.world_to_screen(a_end, bounds)
            pygame.draw.line(self.screen, self.COLOR_ATTRACTIVE, (ux, uy), (ax, ay), 2)
            
            # Repulsive vector
            r_end = uav.position + uav.repulsive_force * 0.4
            rx, ry = self.world_to_screen(r_end, bounds)
            pygame.draw.line(self.screen, self.COLOR_REPULSIVE, (ux, uy), (rx, ry), 2)
            
            # Target Velocity vector
            v_end = uav.position + uav.target_velocity * 0.8
            vx, vy = self.world_to_screen(v_end, bounds)
            pygame.draw.line(self.screen, self.COLOR_VELOCITY, (ux, uy), (vx, vy), 3)

            # 5. UAV Body
            pygame.draw.circle(self.screen, col, (ux, uy), 8)
            pygame.draw.circle(self.screen, (255, 255, 255), (ux, uy), 8, 1)
            
            # UAV Label + Mode Badge
            mode_tag = f"[{uav.comm_mode}]" if uav.comm_mode else ""
            lbl_u = self.small_font.render(f"{uav.name} {mode_tag}", True, (255, 255, 255))
            self.screen.blit(lbl_u, (ux + 10, uy - 12))

        # Top Header & Telemetry Overlay
        title_text = self.font.render(f"Scenario: {engine.scenario.name} | Controller: {engine.controller.name} | Comm: {engine.channel.name}", True, (240, 240, 240))
        self.screen.blit(title_text, (20, 15))
        
        time_text = self.small_font.render(f"Sim Time: {engine.current_time:.2f}s / {engine.scenario.sim_time_limit:.1f}s | Step: {engine.step_count}", True, (180, 200, 220))
        self.screen.blit(time_text, (20, 35))

        # Legend on Bottom
        legend_y = self.height - 40
        pygame.draw.line(self.screen, self.COLOR_ATTRACTIVE, (30, legend_y), (60, legend_y), 3)
        self.screen.blit(self.small_font.render("Attractive Force", True, self.COLOR_ATTRACTIVE), (68, legend_y - 6))
        
        pygame.draw.line(self.screen, self.COLOR_REPULSIVE, (200, legend_y), (230, legend_y), 3)
        self.screen.blit(self.small_font.render("Repulsive Force", True, self.COLOR_REPULSIVE), (238, legend_y - 6))
        
        pygame.draw.line(self.screen, self.COLOR_VELOCITY, (370, legend_y), (400, legend_y), 3)
        self.screen.blit(self.small_font.render("Target Velocity (ACACT)", True, self.COLOR_VELOCITY), (408, legend_y - 6))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if PYGAME_AVAILABLE and self.screen is not None:
            pygame.quit()
