"""
Micromouse Maze Simulation - Phase 1 (Map + Mouse only)
========================================================
- สร้างสนามเขาวงกต (Maze) ขนาด 30x30 ช่อง (16x16 cm ต่อช่อง)
- ใช้อัลกอริทึม Perfect Maze (Recursive Backtracker) -> ทางเดินแคบ 1 ช่อง
  คดเคี้ยวจริงจัง ยากกว่าการสุ่มวางกำแพงแบบกระจายจุดมาก
  และการันตี 100% ว่ามีเส้นทางจาก Start ไป Finish ได้จริง (ตรวจซ้ำด้วย A*)
- สร้างโครงสร้างตัวหนู (Mouse) รู้ตำแหน่งตัวเองและเป้าหมาย พร้อมคำนวณ
  ระยะทางตรงทางทฤษฎี (Manhattan/Euclidean แบบมองทะลุกำแพง)
- แสดงผลด้วย Matplotlib

หมายเหตุ: เฟสนี้ยังไม่คำนวณ/แสดง Shortest Path หรือให้หนูเดินจริง
          (ฟังก์ชัน A* และ mouse.plan_path() เตรียมไว้พร้อมใช้ในเฟสถัดไป)
"""

import numpy as np
import random
import heapq
import time
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# =========================================================
# 1. ค่าคงที่ของสนาม (Maze Configuration)
# =========================================================
GRID_SIZE = 30          # 30x30 ช่อง
CELL_SIZE_CM = 16       # แต่ละช่อง 16x16 cm (ใช้ตอนคำนวณระยะทางจริงถ้าต้องการ)

START = (0, 0)                              # จุดเริ่มต้น (แถว, คอลัมน์)
FINISH = (GRID_SIZE - 1, GRID_SIZE - 1)     # จุดเป้าหมาย (มีชีสอยู่)

# ค่าคงที่แทนประเภทช่อง (สำหรับเก็บใน grid และ visualization)
FREE = 0
WALL = 1


# =========================================================
# 2. A* Algorithm สำหรับหาเส้นทางที่สั้นที่สุดที่หลบสิ่งกีดขวางจริง
#    (ยังไม่ถูกเรียกใช้แสดงผลในเฟสนี้ แต่เตรียมพร้อมไว้)
# =========================================================
def manhattan_distance(a, b):
    """ระยะทางแบบ Manhattan (เส้นตรงทางทฤษฎี ไม่หลบสิ่งกีดขวาง)"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a, b):
    """ระยะทางแบบเส้นตรง (Euclidean) - ใช้เป็นเส้นนำสายตาทางทฤษฎีเท่านั้น"""
    return np.hypot(a[0] - b[0], a[1] - b[1])


def get_neighbors(pos, grid):
    """คืนช่องข้างเคียงที่เดินได้ (4 ทิศ: บน ล่าง ซ้าย ขวา)"""
    r, c = pos
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    result = []
    for nr, nc in candidates:
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if grid[nr, nc] != WALL:
                result.append((nr, nc))
    return result


def a_star_search(grid, start, goal):
    """
    A* search: หาเส้นทางที่สั้นที่สุดที่หลบกำแพงจริง
    คืนค่า: list ของ (row, col) จาก start -> goal หรือ None ถ้าไม่มีเส้นทาง
    """
    open_set = []
    heapq.heappush(open_set, (0 + manhattan_distance(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in visited:
            continue
        visited.add(current)

        for neighbor in get_neighbors(current, grid):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return None  # ไม่มีเส้นทาง


def count_turns(path):
    """นับจำนวนจุดเลี้ยว (decision points) บนเส้นทาง"""
    if len(path) < 3:
        return 0
    turns = 0
    prev_direction = None
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        direction = (dr, dc)
        if prev_direction is not None and direction != prev_direction:
            turns += 1
        prev_direction = direction
    return turns


# =========================================================
# 3. สร้างเขาวงกต (Maze Generation) แบบ "Perfect Maze"
#    ใช้ Recursive Backtracker (DFS แบบสุ่ม) -> ทางเดินแคบ 1 ช่อง
#    คดเคี้ยวจริงจัง มีทางเดียวเท่านั้นระหว่างจุดสองจุดใด ๆ
#    (ยากกว่าการสุ่มวางกำแพงแบบกระจายจุดมาก และการันตี 100% ว่ามีทางเดินถึงกันเสมอ)
# =========================================================
def generate_perfect_maze(size, rng):
    """
    สร้าง Perfect Maze ด้วย Recursive Backtracker บนโครงสร้าง node แบบ half-resolution
    - node แต่ละตัวอยู่ที่พิกัดคู่ (0,2,4,...) บน grid จริง
    - ผนัง (wall) คือช่องคี่ระหว่าง node ที่ "ไม่ได้เชื่อม" กัน
    - ผลลัพธ์: มีทางเดินเดียวเท่านั้นเชื่อมทุกจุด (การันตี connectivity 100%)
      และทางเดินแคบ 1 ช่อง คดเคี้ยว ทำให้ยากกว่าสิ่งกีดขวางแบบสุ่มมาก
    """
    n_nodes = (size + 1) // 2          # จำนวน node ต่อแกน เช่น size=29 -> 15 nodes
    grid = np.ones((size, size), dtype=int) * WALL   # เริ่มจากเป็นกำแพงทั้งหมด

    visited = [[False] * n_nodes for _ in range(n_nodes)]
    stack = [(0, 0)]
    visited[0][0] = True
    grid[0, 0] = FREE

    node_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack[-1]
        unvisited_neighbors = []
        for dr, dc in node_directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_nodes and 0 <= nc < n_nodes and not visited[nr][nc]:
                unvisited_neighbors.append((nr, nc))

        if unvisited_neighbors:
            nr, nc = rng.choice(unvisited_neighbors)
            # เจาะกำแพงระหว่าง node ปัจจุบันกับ node ใหม่ (ช่องกึ่งกลาง)
            wall_r = r * 2 + (nr - r)
            wall_c = c * 2 + (nc - c)
            grid[wall_r, wall_c] = FREE
            grid[nr * 2, nc * 2] = FREE

            visited[nr][nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()  # backtrack

    return grid


def embed_maze_to_full_grid(maze, target_size, finish):
    """
    Perfect maze ที่สร้างได้จะมีขนาด (2*n_nodes - 1) ซึ่งอาจเล็กกว่า target_size เล็กน้อย
    (เช่น 29 แทนที่จะเป็น 30 พอดี เพราะ 30 เป็นเลขคู่ ไม่ลงตัวกับโครงสร้าง node คู่-คี่)
    ฟังก์ชันนี้ฝัง maze ลงในกริดขนาด target_size x target_size และเจาะทางเชื่อม
    สั้น ๆ ไปยังมุมกริดจริง (Finish) เพื่อให้ยังคงเดินถึงกันได้แบบ 100%
    """
    full = np.ones((target_size, target_size), dtype=int) * WALL
    m = maze.shape[0]
    full[:m, :m] = maze

    fr, fc = finish
    r, c = m - 1, m - 1
    while r < fr:
        full[r + 1, c] = FREE
        r += 1
    while c < fc:
        full[r, c + 1] = FREE
        c += 1

    return full


def generate_maze(size=GRID_SIZE, start=START, goal=FINISH, seed=None):
    """
    สร้างเขาวงกตแบบ Perfect Maze (ยาก คดเคี้ยว ทางแคบ) และการันตีว่ามีเส้นทาง
    จาก start ไป goal ได้จริงเสมอ (ตรวจสอบซ้ำด้วย A* เพื่อความชัวร์)

    คืนค่า: grid (numpy array ขนาด size x size)
    """
    rng = random.Random(seed)

    core_size = size if size % 2 == 1 else size - 1   # ต้องเป็นเลขคี่สำหรับ node grid
    maze = generate_perfect_maze(core_size, rng)
    grid = embed_maze_to_full_grid(maze, size, goal)

    check_path = a_star_search(grid, start, goal)
    if check_path is None:
        raise RuntimeError("เกิดข้อผิดพลาด: ไม่สามารถการันตีเส้นทางได้ กรุณาลอง seed อื่น")

    print("[Maze Generator] สร้าง Perfect Maze สำเร็จ "
          "(ทางเดินแคบ 1 ช่อง, การันตีเส้นทางเชื่อมถึงกัน 100%)")

    return grid


# =========================================================
# 4. โครงสร้างตัวหนู (Mouse)
# =========================================================
class Mouse:
    """
    ตัวแทนหนู (Micromouse)
    - รู้พิกัดปัจจุบันและพิกัดเป้าหมาย (Finish/Cheese)
    - คำนวณระยะทางตรงทางทฤษฎี (Manhattan/Euclidean) ได้ทันที
      แต่ระยะนี้ "มองทะลุกำแพง" ใช้เป็นแค่เส้นนำสายตา (heuristic) เท่านั้น
    - หาเส้นทางจริงที่หลบสิ่งกีดขวางด้วย A* ผ่าน plan_path()
    """

    def __init__(self, start_pos, goal_pos, grid):
        self.start_pos = start_pos
        self.position = start_pos          # ตำแหน่งปัจจุบันของหนู
        self.goal_pos = goal_pos           # ตำแหน่งเป้าหมาย (ชีส)
        self.grid = grid                   # แผนที่ที่หนู "รู้"
        self.planned_path = None
        self.direction = "NONE"

    def theoretical_distance(self, method="manhattan"):
        """ระยะทางตรงทางทฤษฎีจากตำแหน่งปัจจุบันไปเป้าหมาย (ไม่หลบกำแพง)"""
        if method == "euclidean":
            return euclidean_distance(self.position, self.goal_pos)
        return manhattan_distance(self.position, self.goal_pos)

    def plan_path(self):
        """ใช้ A* คำนวณเส้นทางจริงที่หลบสิ่งกีดขวาง (เตรียมไว้สำหรับเฟสถัดไป)"""
        path = a_star_search(self.grid, self.position, self.goal_pos)
        self.planned_path = path
        return path

    def __repr__(self):
        return (f"Mouse(pos={self.position}, goal={self.goal_pos}, "
                f"planned_path_len={len(self.planned_path) if self.planned_path else None})")


# =========================================================
# 5. Visualization ด้วย Matplotlib
# =========================================================
def visualize_maze(grid, start, goal, mouse_pos=None, path=None,
                    show_path=False, title="Micromouse Maze - Initial State"):
    """
    วาดสนามเขาวงกตแบบ Grid Map ด้วย Matplotlib
    - ช่องว่าง: สีขาว | กำแพง: สีดำ
    - Start: สีเขียว | Finish (มีชีส): สีแดง
    - หนู: วงกลมสีน้ำเงิน
    """
    size = grid.shape[0]
    fig, ax = plt.subplots(figsize=(9, 9))

    cmap = ListedColormap(["white", "black"])
    ax.imshow(grid, cmap=cmap, origin="upper")

    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.4)
    ax.tick_params(which="both", bottom=False, left=False,
                    labelbottom=False, labelleft=False)

    if show_path and path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="orange", linewidth=2.5, alpha=0.8, zorder=2,
                 label="Shortest Path (A*)")

    ax.add_patch(patches.Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1,
                                    facecolor="limegreen", edgecolor="darkgreen", zorder=3))
    ax.text(start[1], start[0], "S", ha="center", va="center",
            fontsize=10, fontweight="bold", color="black", zorder=4)

    ax.add_patch(patches.Rectangle((goal[1] - 0.5, goal[0] - 0.5), 1, 1,
                                    facecolor="tomato", edgecolor="darkred", zorder=3))
    ax.text(goal[1], goal[0], "F", ha="center", va="center",
            fontsize=10, fontweight="bold", color="black", zorder=4)

    if mouse_pos is not None:
        ax.plot(mouse_pos[1], mouse_pos[0], marker="o", markersize=16,
                 markerfacecolor="royalblue", markeredgecolor="navy", zorder=5)
        ax.text(mouse_pos[1], mouse_pos[0], "M", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=6)

    legend_elements = [
        patches.Patch(facecolor="white", edgecolor="gray", label="Free Space"),
        patches.Patch(facecolor="black", label="Wall / Obstacle"),
        patches.Patch(facecolor="limegreen", edgecolor="darkgreen", label="Start"),
        patches.Patch(facecolor="tomato", edgecolor="darkred", label="Finish (Cheese)"),
    ]
    if mouse_pos is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue",
                       markersize=10, label="Mouse"))
    if show_path and path:
        legend_elements.append(
            plt.Line2D([0], [0], color="orange", linewidth=2.5, label="Shortest Path (A*)"))

    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig, ax


# =========================================================
# 6. GUI Class (Tkinter implementation)
# =========================================================
class MazeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Micromouse Maze Simulator - CMU 271401")
        self.root.configure(bg="#0f172a") # Slate 900
        
        # Grid settings
        self.cell_size = 20 # 20x20 pixels per cell -> 600x600 total canvas
        self.canvas_size = GRID_SIZE * self.cell_size
        
        # Initial maze config
        self.seed_val = 70
        self.show_path = False
        
        # Initialize maze and mouse
        self.generate_new_maze(init=True)
        
        # Build layout
        self.setup_layout()
        self.draw_maze()

    def generate_new_maze(self, init=False):
        if not init:
            try:
                val = self.seed_entry.get().strip()
                self.seed_val = int(val) if val else None
            except ValueError:
                messagebox.showerror("Error", "กรุณากรอก Seed เป็นตัวเลขจำนวนเต็ม")
                return

        # Generate maze & initialize mouse
        self.grid = generate_maze(size=GRID_SIZE, start=START, goal=FINISH, seed=self.seed_val)
        self.mouse = Mouse(start_pos=START, goal_pos=FINISH, grid=self.grid)
        self.planned_path = None
        
        # Pre-plan path
        self.planned_path = self.mouse.plan_path()
        
        if not init:
            self.draw_maze()
            self.log_output(f"\n[System] สร้างเขาวงกตสำเร็จ (Seed: {self.seed_val})")
            self.log_output(f" - กำแพง: {np.sum(self.grid == WALL)} ช่อง ({np.sum(self.grid == WALL)/(GRID_SIZE*GRID_SIZE)*100:.1f}%)")

    def measure_computation_time(self):
        self.log_output("\n=== เริ่มวัดเวลาประมวลผล (Computation Time) 5 ครั้ง ===")
        times = []
        path = None
        for i in range(5):
            start = time.perf_counter()
            path = self.mouse.plan_path()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)
            self.log_output(f"ครั้งที่ {i+1}: {elapsed_ms:.4f} ms | ความยาวเส้นทาง: {len(path)} ช่อง")
        
        avg_time = sum(times) / len(times)
        self.log_output(f"เวลาประมวลผลเฉลี่ย: {avg_time:.4f} ms")
        self.log_output("=====================================================")
        
        self.planned_path = path
        self.draw_maze()

    def toggle_path(self):
        self.show_path = not self.show_path
        self.draw_maze()
        state = "แสดง" if self.show_path else "ซ่อน"
        self.log_output(f"[System] เปลี่ยนสถานะเส้นทาง: {state}")

    def setup_layout(self):
        # Create a container frame
        self.main_frame = tk.Frame(self.root, bg="#0f172a", padx=15, pady=15)
        self.main_frame.pack(fill="both", expand=True)

        # Canvas Frame
        canvas_frame = tk.LabelFrame(
            self.main_frame, text=" แผนที่เขาวงกต (30x30 ช่อง, ช่องละ 16 cm) ",
            bg="#0f172a", fg="#94a3b8", font=("Helvetica", 10, "bold"),
            bd=2, relief="groove", labelanchor="nw"
        )
        canvas_frame.pack(side="left", padx=(0, 15), fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_frame, width=self.canvas_size, height=self.canvas_size,
            bg="#1e293b", bd=0, highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)

        # Sidebar Frame
        self.sidebar = tk.Frame(self.main_frame, bg="#1e293b", width=300, padx=15, pady=15)
        self.sidebar.pack(side="right", fill="both", expand=False)
        self.sidebar.pack_propagate(False) # Keep width fixed

        # Title
        title_lbl = tk.Label(
            self.sidebar, text="Micromouse Maze",
            bg="#1e293b", fg="#f8fafc", font=("Helvetica", 16, "bold")
        )
        title_lbl.pack(anchor="w", pady=(0, 2))
        
        subtitle_lbl = tk.Label(
            self.sidebar, text="Phase 1: Simulation & GUI",
            bg="#1e293b", fg="#64748b", font=("Helvetica", 9, "italic")
        )
        subtitle_lbl.pack(anchor="w", pady=(0, 20))

        # Seed configuration
        seed_frame = tk.Frame(self.sidebar, bg="#1e293b")
        seed_frame.pack(fill="x", pady=(0, 15))
        
        seed_lbl = tk.Label(
            seed_frame, text="Seed เขาวงกต:",
            bg="#1e293b", fg="#cbd5e1", font=("Helvetica", 10, "bold")
        )
        seed_lbl.pack(side="left", padx=(0, 5))

        self.seed_entry = tk.Entry(
            seed_frame, bg="#0f172a", fg="#f8fafc",
            insertbackground="white", bd=1, relief="flat",
            font=("Helvetica", 10), width=10
        )
        self.seed_entry.insert(0, str(self.seed_val))
        self.seed_entry.pack(side="left", padx=5, ipady=3)

        # Helper buttons function
        def create_btn(text, bg, active_bg, cmd):
            btn = tk.Button(
                self.sidebar, text=text, bg=bg, fg="white",
                activebackground=active_bg, activeforeground="white",
                relief="flat", bd=0, font=("Helvetica", 10, "bold"),
                cursor="hand2", command=cmd, height=2
            )
            # Hover animations
            btn.bind("<Enter>", lambda e: btn.configure(bg=active_bg))
            btn.bind("<Leave>", lambda e: btn.configure(bg=bg))
            btn.pack(fill="x", pady=5)
            return btn

        # Buttons
        create_btn("สร้างเขาวงกตใหม่", "#2563eb", "#1d4ed8", self.generate_new_maze)
        create_btn("วัดเวลา A* (5 ครั้ง)", "#059669", "#047857", self.measure_computation_time)
        create_btn("แสดง/ซ่อน เส้นทาง A*", "#d97706", "#b45309", self.toggle_path)

        # Output Log Box
        log_lbl = tk.Label(
            self.sidebar, text="ประวัติการประมวลผล:",
            bg="#1e293b", fg="#cbd5e1", font=("Helvetica", 10, "bold")
        )
        log_lbl.pack(anchor="w", pady=(15, 5))

        # Text area with scrollbar
        self.log_text = tk.Text(
            self.sidebar, bg="#0f172a", fg="#10b981", # Green terminal text
            insertbackground="white", bd=0, relief="flat",
            font=("Consolas", 9), height=15
        )
        self.log_text.pack(fill="both", expand=True, pady=(0, 5))
        
        # Scrollbar for log text
        scrollbar = tk.Scrollbar(self.log_text, bg="#0f172a")
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

        # Initial info
        self.log_output("ระบบเริ่มต้นสำเร็จ\n"
                        f"เขาวงกตขนาด {GRID_SIZE}x{GRID_SIZE} ช่อง (ช่องละ {CELL_SIZE_CM} cm)")

    def log_output(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def draw_maze(self):
        self.canvas.delete("all")
        cs = self.cell_size
        
        # Draw cells
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1 = c * cs
                y1 = r * cs
                x2 = x1 + cs
                y2 = y1 + cs
                
                # Determine cell color
                if self.grid[r, c] == WALL:
                    color = "#1e293b" # Slate 800 for walls
                    outline_color = "#334155"
                else:
                    color = "#ffffff" # White for paths
                    outline_color = "#e2e8f0"
                
                # Custom colors for START and FINISH
                if (r, c) == START:
                    color = "#2ecc71" # Neon Green
                    outline_color = "#27ae60"
                elif (r, c) == FINISH:
                    color = "#e74c3c" # Tomato Red
                    outline_color = "#c0392b"
                    
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color, outline=outline_color, width=1
                )
                
                # Draw text identifier for Start/Finish
                if (r, c) == START:
                    self.canvas.create_text(
                        x1 + cs/2, y1 + cs/2,
                        text="S", fill="#ffffff",
                        font=("Helvetica", 10, "bold")
                    )
                elif (r, c) == FINISH:
                    self.canvas.create_text(
                        x1 + cs/2, y1 + cs/2,
                        text="F", fill="#ffffff",
                        font=("Helvetica", 10, "bold")
                    )

        # Draw Planned A* Path if toggle is on
        if self.show_path and self.planned_path:
            for i in range(len(self.planned_path) - 1):
                r1, c1 = self.planned_path[i]
                r2, c2 = self.planned_path[i+1]
                # Line coordinates (centers of cells)
                lx1 = c1 * cs + cs/2
                ly1 = r1 * cs + cs/2
                lx2 = c2 * cs + cs/2
                ly2 = r2 * cs + cs/2
                self.canvas.create_line(
                    lx1, ly1, lx2, ly2,
                    fill="#fd9644", width=3, capstyle="round" # Vibrant orange path
                )

        # Draw Mouse
        mr, mc = self.mouse.position
        mx = mc * cs + cs/2
        my = mr * cs + cs/2
        r_offset = cs * 0.35 # Mouse radius
        self.canvas.create_oval(
            mx - r_offset, my - r_offset,
            mx + r_offset, my + r_offset,
            fill="#3b82f6", outline="#1d4ed8", width=1.5 # Blue mouse
        )
        self.canvas.create_text(
            mx, my,
            text="M", fill="#ffffff",
            font=("Helvetica", 8, "bold")
        )


# =========================================================
# 7. Main - รันระบบ GUI
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = MazeGUI(root)
    root.mainloop()