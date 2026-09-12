"""
Real-Time Forces and Swarm Trajectory Visualizer.
Implements the 3D/2D Forces Visualization Tool as shown in Paper 1 (Fig 5) & Paper 2 (Fig 3).
Features:
- Attractive Force (Green), Repulsive Force (Red), Target Velocity (Cyan).
- Dynamic Path Rendering: For large swarms (up to 50 UAVs), only draws 1 or 2 selected representative paths.
- Interactive Click-to-Inspect: Click any UAV on screen to toggle its trajectory path and view focused telemetry!
- 'T' hotkey: Cycle path display mode (Selected / All / None).
"""
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Set

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class ForceMonitorVisualizer:
    """
    Interactive 2D/3D projection visualizer for multi-UAV swarms.
    """
    def __init__(self, width: int = 960, height: int = 780, title: str = "Multi-UAV ACACT & DCACS / FPS-SCTP Monitor"):
        self.width = width
        self.height = height
        self.title = title
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.paused = False
        self.fps = 30
        
        # Interactive Path & Inspection Modes
        # path_mode: 'SELECTED' (1-2 paths), 'ALL' (all paths), 'NONE' (no trails)
        self.path_mode: str = "SELECTED"
        self.selected_uav_ids: Set[int] = {1, 2}
        self.hovered_uav_id: Optional[int] = None
        self.uav_screen_positions: Dict[int, tuple] = {}
        
        # Colors
        self.COLOR_BG = (20, 24, 32)
        self.COLOR_GRID = (36, 44, 58)
        self.COLOR_ATTRACTIVE = (46, 204, 113)  # Green
        self.COLOR_REPULSIVE = (231, 76, 60)    # Red
        self.COLOR_VELOCITY = (52, 152, 219)    # Light Blue / Cyan
        self.COLOR_TRAIL = (120, 140, 160)
        self.COLOR_GOAL = (241, 196, 15)       # Yellow
        self.COLOR_HIGHLIGHT = (255, 255, 0)
        self.COLOR_CARD_BG = (30, 38, 52, 220)
        
        self.UAV_COLORS = [
            (52, 152, 219), (230, 126, 34), (155, 89, 182),
            (26, 188, 156), (241, 196, 15), (231, 76, 60),
            (46, 204, 113), (236, 240, 241), (149, 165, 166), (211, 84, 0),
            (0, 184, 148), (253, 121, 168), (108, 92, 231), (250, 177, 160)
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
        self.font = pygame.font.SysFont("Helvetica", 13, bold=True)
        self.small_font = pygame.font.SysFont("Helvetica", 11)
        return True

    def world_to_screen(self, pos: np.ndarray, bounds: List[float]) -> tuple:
        """Transforms world coords (x, y) into window pixel coords."""
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
        margin = 60
        arena_w = self.width - 2 * margin
        arena_h = self.height - 2 * margin - 90
        
        norm_x = (pos[0] - xmin) / (xmax - xmin + 1e-6)
        norm_y = (pos[1] - ymin) / (ymax - ymin + 1e-6)
        
        # Invert Y for screen coordinates
        sx = int(margin + norm_x * arena_w)
        sy = int(margin + 50 + (1.0 - norm_y) * arena_h)
        return (sx, sy)

    def handle_events(self) -> bool:
        """Processes keyboard / mouse inputs (click drone to inspect path). Returns False on Quit."""
        if not PYGAME_AVAILABLE or self.screen is None:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_t: # Cycle path display mode
                    if self.path_mode == "SELECTED":
                        self.path_mode = "ALL"
                    elif self.path_mode == "ALL":
                        self.path_mode = "NONE"
                    else:
                        self.path_mode = "SELECTED"
                elif event.key == pygame.K_c: # Clear selections
                    self.selected_uav_ids.clear()
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Left click: check if clicked near any UAV
                mx, my = event.pos
                clicked_id = None
                min_click_d = 18.0
                for u_id, (sx, sy) in self.uav_screen_positions.items():
                    d = np.hypot(mx - sx, my - sy)
                    if d < min_click_d:
                        min_click_d = d
                        clicked_id = u_id
                if clicked_id is not None:
                    if clicked_id in self.selected_uav_ids:
                        self.selected_uav_ids.remove(clicked_id)
                    else:
                        self.selected_uav_ids.add(clicked_id)
        return True

    def render_frame(self, engine: Any, bounds: List[float]):
        """Renders the entire swarm state, force vectors, and communication HUD."""
        if not PYGAME_AVAILABLE or self.screen is None:
            return

        self.screen.fill(self.COLOR_BG)
        self.uav_screen_positions.clear()
        
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
        n_uavs = len(engine.uavs)

        # Draw UAV trails, goals, and vectors
        for i, uav in enumerate(engine.uavs):
            col = self.UAV_COLORS[i % len(self.UAV_COLORS)]
            is_selected = (uav.uav_id in self.selected_uav_ids)
            
            # Decide whether to draw trajectory path for this drone
            should_draw_trail = False
            if self.path_mode == "ALL":
                should_draw_trail = True
            elif self.path_mode == "SELECTED":
                # Only draw if selected or if swarm is small (<= 4)
                should_draw_trail = is_selected or (n_uavs <= 4)
                
            # 1. Goal marker
            gx, gy = self.world_to_screen(uav.goal, bounds)
            if should_draw_trail or is_selected:
                pygame.draw.circle(self.screen, self.COLOR_GOAL, (gx, gy), 5, 2)
                lbl_g = self.small_font.render(f"G{uav.uav_id}", True, self.COLOR_GOAL)
                self.screen.blit(lbl_g, (gx + 6, gy - 6))

            # 2. Trajectory trail
            if should_draw_trail and len(uav.trajectory) > 1:
                trail_pts = [self.world_to_screen(p, bounds) for p in uav.trajectory[::2]]
                if len(trail_pts) > 1:
                    pygame.draw.lines(self.screen, col, False, trail_pts, 3 if is_selected else 1)

            ux, uy = self.world_to_screen(uav.position, bounds)
            self.uav_screen_positions[uav.uav_id] = (ux, uy)

            # 3. Collision Risk Bubble (r = 2.0m) - drawn only for selected or small swarm to avoid clutter
            if is_selected or n_uavs <= 8:
                col_pixel_r = int(uav.config.collision_radius * scale)
                pygame.draw.circle(self.screen, (80, 40, 40), (ux, uy), max(4, col_pixel_r), 1)

            # 4. Forces Vectors (Attractive: Green, Repulsive: Red, Velocity: Cyan)
            if is_selected or n_uavs <= 10:
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
                pygame.draw.line(self.screen, self.COLOR_VELOCITY, (ux, uy), (vx, vy), 2)

            # 5. UAV Body
            body_radius = 7 if n_uavs <= 15 else (5 if n_uavs <= 30 else 4)
            pygame.draw.circle(self.screen, col, (ux, uy), body_radius)
            if is_selected:
                # Highlight ring around clicked drone
                pygame.draw.circle(self.screen, self.COLOR_HIGHLIGHT, (ux, uy), body_radius + 4, 2)
            else:
                pygame.draw.circle(self.screen, (255, 255, 255), (ux, uy), body_radius, 1)
            
            # Label
            if n_uavs <= 12 or is_selected:
                lbl_u = self.small_font.render(f"{uav.name}", True, (240, 240, 240))
                self.screen.blit(lbl_u, (ux + 8, uy - 10))

        # Top Header & Telemetry Overlay
        title_text = self.font.render(f"Scenario: {engine.scenario.name} ({n_uavs} UAVs) | Controller: {engine.controller.name} | Comm: {engine.channel.name}", True, (240, 240, 240))
        self.screen.blit(title_text, (20, 12))
        
        mode_hint = f"Path Mode: [{self.path_mode}] (Press 'T' to cycle / Click drone to toggle path)"
        sub_text = self.small_font.render(f"Sim Time: {engine.current_time:.2f}s / {engine.scenario.sim_time_limit:.1f}s | {mode_hint}", True, (180, 200, 220))
        self.screen.blit(sub_text, (20, 32))

        # Bottom Legend
        legend_y = self.height - 35
        pygame.draw.line(self.screen, self.COLOR_ATTRACTIVE, (25, legend_y), (55, legend_y), 3)
        self.screen.blit(self.small_font.render("Attractive Force", True, self.COLOR_ATTRACTIVE), (62, legend_y - 6))
        
        pygame.draw.line(self.screen, self.COLOR_REPULSIVE, (180, legend_y), (210, legend_y), 3)
        self.screen.blit(self.small_font.render("Repulsive Force", True, self.COLOR_REPULSIVE), (218, legend_y - 6))
        
        pygame.draw.line(self.screen, self.COLOR_VELOCITY, (335, legend_y), (365, legend_y), 3)
        self.screen.blit(self.small_font.render("Target Velocity", True, self.COLOR_VELOCITY), (372, legend_y - 6))

        sel_txt = f"Selected Drones: {list(self.selected_uav_ids)}" if self.selected_uav_ids else "Click any drone to view path"
        self.screen.blit(self.small_font.render(sel_txt, True, self.COLOR_HIGHLIGHT), (530, legend_y - 6))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if PYGAME_AVAILABLE and self.screen is not None:
            pygame.quit()
